"""NPU 专用导出模块：针对 Qualcomm QNN TFLite Delegate 契约。

依据《AI辅助瞄准 模型条件逆向报告》实现硬契约检查与专用导出：
  - 输入：INT8 NHWC [1,H,W,3]，静态形状
  - 输出：INT8 [1,C,N] 或 [1,N,C]（单组合，C=4+类别数，5≤C≤256）
         或双拆分 [1,4,N]+[1,cls,N]
  - 语义：anchor-free cx,cy,w,h + 类别分数，不能内置 NMS/TopK
  - 算子黑名单：TopK / GatherElements / NonMaxSuppression / Mod 等 QNN 不支持算子
  - 全 INT8，LOGISTIC 输出 scale 应为 1/256

合规检查通过才允许导出；不合规明确拒绝并给出原因。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Optional

import numpy as np
import onnx
from onnx import helper

from engine import _tolerant_temp_cleanup, preprocess_for_onnx2tf
from calibration import list_images

# QNN TFLite Delegate 已知不支持/高风险算子
_OP_BLACKLIST = {
    "TopK": "QNN delegate 不支持 TopK，需在后处理中做",
    "GatherElements": "QNN delegate 不支持 GatherElements",
    "NonMaxSuppression": "NMS 不能放进模型图，APK native 层自己做",
    "Mod": "QNN delegate 对 Mod 支持不稳定",
    "ScatterElements": "QNN delegate 不支持 ScatterElements",
    "ScatterND": "QNN delegate 不支持 ScatterND",
    "Range": "QNN delegate 对动态 Range 支持有限",
    "Where": "QNN delegate 对 Where 支持有限",
    "If": "控制流算子不支持",
    "Loop": "控制流算子不支持",
}

# 输出候选数 NMS 后格式的特征（[1,300,6] 等）
_NMS_LIKE_SIZES = {300, 8400, 25200}  # 常见 YOLO NMS 后候选数


def _input_shape(vi) -> list:
    tt = getattr(vi.type, "tensor_type", None)
    if tt is None:
        return []
    return [d.dim_value if d.dim_value > 0 else -1 for d in tt.shape.dim]


def _is_static(shape: list) -> bool:
    return all(d > 0 for d in shape)


def check_npu_compliance(onnx_path: str) -> dict:
    """检查 ONNX 模型是否符合 QNN NPU TFLite 契约。

    返回 {
        compliant: bool,
        issues: [str],          # 不合规原因
        warnings: [str],        # 警告（不阻止导出但需注意）
        info: {                 # 模型信息
            input, outputs, ops, num_classes, output_mode, ...
        }
    }
    """
    model = onnx.load(onnx_path, load_external_data=False)
    issues = []
    warnings = []

    inputs = [vi for vi in model.graph.input
              if vi.name not in {i.name for i in model.graph.initializer}]
    outputs = list(model.graph.output)
    ops = [n.op_type for n in model.graph.node]

    info = {
        "filename": os.path.basename(onnx_path),
        "num_inputs": len(inputs),
        "num_outputs": len(outputs),
        "ops": sorted(set(ops)),
        "op_counts": {op: ops.count(op) for op in sorted(set(ops))},
    }

    # ---- 1. 输入检查 ----
    if len(inputs) != 1:
        issues.append(f"输入数量应为 1，实际 {len(inputs)}（NPU 要求单图像输入）")
    else:
        inp = inputs[0]
        shape = _input_shape(inp)
        info["input"] = {"name": inp.name, "shape": shape}
        # 通道数检查（NCHW: dim1=3, NHWC: dim3=3）
        if len(shape) != 4:
            issues.append(f"输入应为 4 维 [N,C,H,W] 或 [N,H,W,C]，实际 {len(shape)} 维")
        else:
            # 判断布局
            if shape[1] == 3:
                layout = "NCHW"
                channels = shape[1]
            elif shape[3] == 3:
                layout = "NHWC"
                channels = shape[3]
                warnings.append("输入已是 NHWC，onnx2tf 会保持布局")
            else:
                layout = "unknown"
                channels = -1
                issues.append(f"输入通道数不为 3（shape={shape}），NPU 要求 3 通道 RGB")
            info["input"]["layout"] = layout
            info["input"]["channels"] = channels
            if channels != 3:
                if channels != -1:
                    issues.append(f"输入通道数 {channels}，NPU 要求 3")
            # 静态形状检查
            if not _is_static(shape):
                issues.append(f"输入形状含动态维度 {shape}，NPU 要求静态形状")
            else:
                info["input"]["h"] = shape[2] if layout == "NCHW" else shape[1]
                info["input"]["w"] = shape[3] if layout == "NCHW" else shape[2]

    # ---- 2. 输出检查 ----
    if len(outputs) not in (1, 2):
        issues.append(f"输出数量应为 1 或 2，实际 {len(outputs)}（其他数量会初始化失败）")
    else:
        out_shapes = []
        for o in outputs:
            s = _input_shape(o)
            out_shapes.append({"name": o.name, "shape": s})
            if not _is_static(s):
                issues.append(f"输出 '{o.name}' 含动态维度 {s}，NPU 要求静态形状")
            if len(s) != 3:
                issues.append(f"输出 '{o.name}' 应为 3 维 [1,C,N] 或 [1,N,C]，实际 {len(s)} 维")
        info["outputs"] = out_shapes

        if len(outputs) == 1 and len(out_shapes[0]["shape"]) == 3:
            # 单组合输出 [1, C, N] 或 [1, N, C]
            s = out_shapes[0]["shape"]
            dim1, dim2 = s[1], s[2]
            # 判断哪个是通道维（C 在 5-256），哪个是候选数 N
            if 5 <= dim1 <= 256 and dim2 >= 1:
                c_dim, n_dim = dim1, dim2
                layout = "[1,C,N]"
            elif 5 <= dim2 <= 256 and dim1 >= 1:
                c_dim, n_dim = dim2, dim1
                layout = "[1,N,C]"
            else:
                c_dim, n_dim, layout = -1, -1, "unknown"
                issues.append(
                    f"单组合输出 {s} 无法确定通道维（需 5≤C≤256）"
                )
            if c_dim > 0:
                num_classes = c_dim - 4
                info["num_classes"] = num_classes
                info["output_mode"] = "single"
                info["output_layout"] = layout
                if num_classes < 1:
                    issues.append(f"通道维 {c_dim} - 4 = {num_classes} 类，类别数应 ≥1")
                # NMS 后格式检测：[1, 300, 6] 类似
                if dim1 in _NMS_LIKE_SIZES and dim2 == 6:
                    issues.append(
                        f"输出 {s} 疑似已做 NMS 的结果列表（如 [1,300,6]），"
                        f"APK 要求 anchor-free 原始检测头，不能内置 NMS"
                    )

        elif len(outputs) == 2 and all(len(o["shape"]) == 3 for o in out_shapes):
            # 双拆分：一个通道维=4（框），另一个=类别数
            s0, s1 = out_shapes[0]["shape"], out_shapes[1]["shape"]
            # 找哪个是框（通道维=4）
            def _channel_dim(s):
                if 5 <= s[1] <= 256:
                    return s[1], s[2], "[1,C,N]"
                if 5 <= s[2] <= 256:
                    return s[2], s[1], "[1,N,C]"
                return -1, -1, "?"

            c0, n0, l0 = _channel_dim(s0)
            c1, n1, l1 = _channel_dim(s1)
            # 一个应该是 4（框）
            if c0 == 4 and c1 > 4:
                num_classes = c1 - 4
                info["num_classes"] = num_classes
                info["output_mode"] = "split"
                if n0 != n1:
                    issues.append(f"双输出候选数不一致：{n0} vs {n1}，必须相同")
            elif c1 == 4 and c0 > 4:
                num_classes = c0 - 4
                info["num_classes"] = num_classes
                info["output_mode"] = "split"
                if n0 != n1:
                    issues.append(f"双输出候选数不一致：{n0} vs {n1}，必须相同")
            else:
                issues.append(
                    f"双输出需一个通道维=4（框），另一个=4+类别数。"
                    f"实际 {s0} / {s1}"
                )

    # ---- 3. 算子黑名单检查 ----
    found_black = []
    for op in ops:
        if op in _OP_BLACKLIST:
            found_black.append((op, _OP_BLACKLIST[op]))
    if found_black:
        # 去重
        seen = set()
        for op, reason in found_black:
            if op not in seen:
                seen.add(op)
                issues.append(f"含不支持算子 {op}：{reason}")

    # ---- 4. LOGISTIC/Sigmoid 提示 ----
    sigmoid_count = ops.count("Sigmoid")
    if sigmoid_count > 0:
        info["sigmoid_count"] = sigmoid_count
        warnings.append(
            f"模型含 {sigmoid_count} 个 Sigmoid(LOGISTIC) 算子。"
            f"INT8 量化后 QNN delegate 要求其输出 scale=1/256、zero_point=-128，"
            f"否则会报 'LOGISTIC failed to prepare'。导出后需验证。"
        )

    # ---- 5. Resize/动态 shape 警告 ----
    if "Resize" in ops:
        warnings.append("含 Resize 算子，QNN delegate 可能有限支持，建议实测")

    compliant = len(issues) == 0
    return {
        "compliant": compliant,
        "issues": issues,
        "warnings": warnings,
        "info": info,
    }


def _build_rep_dataset(onnx_path: str, calibration: str,
                       num_samples: int, images_dir: Optional[str],
                       work_dir: str):
    """构造 TFLite 全整数量化的代表性数据集生成器（yield NHWC 输入）。"""
    from PIL import Image as PILImage

    model = onnx.load(onnx_path, load_external_data=False)
    inputs_meta = [vi for vi in model.graph.input
                   if vi.name not in {i.name for i in model.graph.initializer}]
    inp = inputs_meta[0]
    shape = _input_shape(inp)
    # onnx2tf 转 NHWC 后，shape 变为 [1, H, W, 3]
    nhwc_shape = [shape[0], shape[2] if len(shape) >= 3 else 1,
                  shape[3] if len(shape) >= 4 else shape[2], shape[1]]

    rng = np.random.default_rng(0)
    imgs = list_images(images_dir) if (calibration == "image" and images_dir) else []

    def gen():
        for i in range(num_samples):
            if imgs:
                img = PILImage.open(imgs[i % len(imgs)]).convert("RGB")
                h, w = nhwc_shape[1], nhwc_shape[2]
                img = img.resize((w, h), PILImage.BILINEAR)
                arr = np.asarray(img, dtype=np.float32) / 255.0
            else:
                arr = rng.random(nhwc_shape, dtype=np.float32)
            yield [arr.astype(np.float32)]
    return gen


def split_merged_output(onnx_path: str, work_dir: str) -> tuple:
    """把 YOLO 单输出 [1,C,N] 拆成 reg[1,4,N] + cls[1,Nc,N] 双输出。

    参考 yezijinn/onnx_to_int8.tflite 项目的设计：
    在 ONNX 图层面用 Slice + Identity 源头拆分（禁止用 TFLiteConverter 的 Split，
    因为 Split 会继承父联合范围压扁 cls 量化）。
    拆分后 cls 有独立量化范围，TFLiteConverter 自动算出 scale≈1/256（满足 LOGISTIC 约束）。

    返回 (split_model_path, num_classes)；
    如果模型不是单输出原料型（已是双输出/无法识别通道维），返回 (onnx_path, None)。
    """
    model = onnx.load(onnx_path, load_external_data=False)
    outputs = list(model.graph.output)
    if len(outputs) != 1:
        return onnx_path, None

    out_vi = outputs[0]
    shape = _input_shape(out_vi)
    if len(shape) != 3:
        return onnx_path, None

    dim1, dim2 = shape[1], shape[2]
    # 判断哪个是通道维（C 在 5-256，即 4+类别数）
    if 5 <= dim1 <= 256:
        c_dim, n_dim, layout = dim1, dim2, "CN"   # [1,C,N]
    elif 5 <= dim2 <= 256:
        c_dim, n_dim, layout = dim2, dim1, "NC"   # [1,N,C]
    else:
        return onnx_path, None

    nc = c_dim - 4
    if nc < 1:
        return onnx_path, None

    out_name = out_vi.name
    from onnx import TensorProto, numpy_helper

    # Slice 参数（按通道维切分：前4=reg框，剩余=cls类别分数）
    if layout == "CN":
        reg_starts, reg_ends = [0, 0, 0], [1, 4, n_dim]
        cls_starts, cls_ends = [0, 4, 0], [1, c_dim, n_dim]
        reg_shape, cls_shape = [1, 4, n_dim], [1, nc, n_dim]
    else:
        reg_starts, reg_ends = [0, 0, 0], [1, n_dim, 4]
        cls_starts, cls_ends = [0, 0, 4], [1, n_dim, c_dim]
        reg_shape, cls_shape = [1, n_dim, 4], [1, n_dim, nc]

    axes = [0, 1, 2]
    # Slice 节点（opset 10+ 用输入而非属性）
    reg_slice = helper.make_node("Slice",
        inputs=[out_name, "reg_starts", "reg_ends", "split_axes"],
        outputs=["reg_mid"], name="split_reg")
    cls_slice = helper.make_node("Slice",
        inputs=[out_name, "cls_starts", "cls_ends", "split_axes"],
        outputs=["cls_mid"], name="split_cls")
    # Identity 节点（创建独立输出张量，确保 TFLiteConverter 独立量化）
    reg_identity = helper.make_node("Identity", ["reg_mid"], ["reg_out"], name="reg_identity")
    cls_identity = helper.make_node("Identity", ["cls_mid"], ["cls_out"], name="cls_identity")

    inits = [
        numpy_helper.from_array(np.array(reg_starts, dtype=np.int64), "reg_starts"),
        numpy_helper.from_array(np.array(reg_ends, dtype=np.int64), "reg_ends"),
        numpy_helper.from_array(np.array(cls_starts, dtype=np.int64), "cls_starts"),
        numpy_helper.from_array(np.array(cls_ends, dtype=np.int64), "cls_ends"),
        numpy_helper.from_array(np.array(axes, dtype=np.int64), "split_axes"),
    ]
    reg_vi = helper.make_tensor_value_info("reg_out", TensorProto.FLOAT, reg_shape)
    cls_vi = helper.make_tensor_value_info("cls_out", TensorProto.FLOAT, cls_shape)

    # 修改图：删原输出，加双输出 + 节点 + 初始化器
    model.graph.output.remove(out_vi)
    model.graph.output.extend([reg_vi, cls_vi])
    model.graph.node.extend([reg_slice, cls_slice, reg_identity, cls_identity])
    model.graph.initializer.extend(inits)

    out_path = os.path.join(work_dir, "split_" + os.path.basename(onnx_path))
    onnx.save(model, out_path)
    return out_path, nc


def export_npu_tflite(onnx_path: str, out_dir: str,
                      calibration: str = "random", num_samples: int = 8,
                      images_dir: Optional[str] = None,
                      original_name: Optional[str] = None) -> dict:
    """导出 QNN NPU 专用的全 INT8 TFLite（输入输出均为 INT8）。

    前置：必须先通过 check_npu_compliance 检查。
    流程：onnx2tf tf_converter 生成 SavedModel → TFLiteConverter 全整数量化。

    性能：onnx2tf tf_converter 是主要瓶颈（占 95%），校准样本数影响极小
    （16/8/4 样本耗时差异 <0.03s），默认用 8 样本平衡精度与内存。
    onnxsim 预处理已按需跳过（无 Constant 节点时直接返回原模型）。

    original_name: 原始模型文件名（含扩展名），用于生成输出文件名。
                   如 original_name="resnet50.onnx" → 输出 "resnet50_npumotified.tflite"。
                   未提供则用 onnx_path 的文件名。
    """
    from tflite_export import _check_deps, _size_str
    import onnx2tf
    import tensorflow as tf

    ok, missing = _check_deps()
    if not ok:
        raise RuntimeError(f"缺少依赖：{', '.join(missing)}")
    try:
        import tf_keras  # noqa: F401
    except Exception:
        raise RuntimeError("INT8 量化需要 tf_keras，请执行：pip install tf_keras --no-deps")

    # 导出前再次检查合规
    check = check_npu_compliance(onnx_path)
    if not check["compliant"]:
        raise RuntimeError(
            "模型不合规，无法导出 NPU TFLite。原因：\n- " +
            "\n- ".join(check["issues"])
        )

    # 输出文件名：原始模型名（去扩展名）+ _npumotified.tflite
    src_name = original_name or os.path.basename(onnx_path)
    base = os.path.splitext(src_name)[0]
    out_filename = f"{base}_npumotified.tflite"

    os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="ort_npu_")
    try:
        # 0) onnxsim 预处理：仅当模型含 Constant 时才简化（绕过 onnx2tf 的 Constant bug）
        model_for_convert = preprocess_for_onnx2tf(onnx_path, tmp)

        # 0.5) 双输出源头拆分：单输出 [1,C,N] → reg[1,4,N] + cls[1,Nc,N]
        #   用 Slice+Identity 在 ONNX 图层面拆分（禁止 TFLiteConverter 的 Split），
        #   让 cls 独立量化（scale≈1/256 满足 LOGISTIC 约束）。
        #   参考 yezijinn/onnx_to_int8.tflite 项目设计。
        model_for_convert, split_nc = split_merged_output(model_for_convert, tmp)
        if split_nc is not None:
            check["info"]["split_to_dual"] = True
            check["info"]["num_classes"] = split_nc

        # 1) onnx2tf tf_converter 后端 → SavedModel（主要瓶颈，占 ~95% 耗时）
        with _tolerant_temp_cleanup():
            onnx2tf.convert(
                input_onnx_file_path=model_for_convert,
                output_folder_path=tmp,
                non_verbose=True,
                tflite_backend="tf_converter",
                batch_size=1,
            )

        sm_path = None
        if os.path.isfile(os.path.join(tmp, "saved_model.pb")):
            sm_path = tmp
        else:
            for f in os.listdir(tmp):
                p = os.path.join(tmp, f)
                if os.path.isdir(p) and os.path.isfile(os.path.join(p, "saved_model.pb")):
                    sm_path = p
                    break
        if not sm_path:
            raise RuntimeError("onnx2tf 未产出 SavedModel，无法继续全整数量化")

        # 2) 全整数量化：输入输出强制 INT8
        rep = _build_rep_dataset(onnx_path, calibration, num_samples,
                                 images_dir, tmp)
        converter = tf.lite.TFLiteConverter.from_saved_model(sm_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # INT8 为主，SELECT_TF_OPS 兜底（部分算子需 TF ops fallback）
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.representative_dataset = rep
        tflite_model = converter.convert()

        out_path = os.path.join(out_dir, out_filename)
        with open(out_path, "wb") as f:
            f.write(tflite_model)

        # 3) 验证输入输出确实是 int8
        verify_detail = _verify_int8(out_path)
        if not verify_detail["ok"]:
            raise RuntimeError(
                f"全整数量化未生效：{verify_detail['detail']}"
                f"（输入输出应为 INT8，可能模型含不支持 INT8 的算子，QNN delegate 无法运行）"
            )

        result = {
            "path": out_path,
            "filename": os.path.basename(out_path),
            "size": _size_str(os.path.getsize(out_path)),
            "size_bytes": os.path.getsize(out_path),
            "precision": "int8",
            "target": "qualcomm_qnn_npu",
            "verified_int8": True,
            "int8_detail": verify_detail["detail"],
            "warnings": check["warnings"],
        }
        result["compliance"] = {
            "num_classes": check["info"].get("num_classes"),
            "output_mode": check["info"].get("output_mode"),
            "input": check["info"].get("input"),
        }
        return result
    finally:
        with _tolerant_temp_cleanup():
            shutil.rmtree(tmp, ignore_errors=True)


def _verify_int8(tflite_path: str) -> dict:
    """检查 tflite 输入输出张量是否为 INT8，返回 {ok, detail}。"""
    detail = []
    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=tflite_path)
        interp.allocate_tensors()
        ok = True
        for d in interp.get_input_details():
            is_i8 = d["dtype"] in (tf.int8,)
            ok = ok and is_i8
            detail.append(f"输入 {d['name']}: {d['dtype'].__name__} shape={list(d['shape'])}")
        for d in interp.get_output_details():
            is_i8 = d["dtype"] in (tf.int8,)
            ok = ok and is_i8
            detail.append(f"输出 {d['name']}: {d['dtype'].__name__} shape={list(d['shape'])}")
        return {"ok": ok, "detail": "; ".join(detail)}
    except Exception as e:
        return {"ok": False, "detail": f"验证异常: {e}"}


def describe_contract() -> dict:
    """返回 NPU 契约描述（供前端展示）。"""
    return {
        "title": "Qualcomm QNN NPU 契约",
        "input": "INT8 NHWC [1,H,W,3]，静态形状，3通道RGB",
        "output_single": "[1,C,N] 或 [1,N,C]，C=4+类别数(5~256)，anchor-free",
        "output_split": "[1,4,N] 框 + [1,cls,N] 分数，候选数N相同",
        "semantics": "cx,cy,w,h + 类别分数，不能内置 NMS/TopK",
        "blacklist": list(_OP_BLACKLIST.keys()),
        "int8": "全 INT8，输入输出均为 int8",
        "logistic_note": "Sigmoid(LOGISTIC) 输出 scale 需为 1/256, zero_point=-128",
    }
