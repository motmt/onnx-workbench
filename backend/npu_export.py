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
    recovered = False

    # ---- 0. end2end NMS 模型：自动还原原始检测头 ----
    # 含 TopK/NonMaxSuppression 的模型（如输出 [1,300,6]）不能直接上 QNN NPU，
    # 但原始 anchor-free 检测头信息还在图里。参考 yezijinn 项目的
    # recover_head_from_end2end：剪掉 NMS 子图、把 NMS 前的检测头输出设为图输出，
    # 再按还原后的模型继续检查。还原失败则按原模型检查（NMS 会被黑名单拒绝）。
    _nms_ops = {"TopK", "NonMaxSuppression"}
    if any(n.op_type in _nms_ops for n in model.graph.node):
        work_dir = tempfile.mkdtemp(prefix="ort_npu_chk_")
        try:
            rec_path, rec_info = recover_head_from_end2end(onnx_path, work_dir)
            if rec_info and rec_info.get("recovered"):
                model = onnx.load(rec_path, load_external_data=False)
                recovered = True
                warnings.append(
                    f"检测到 end2end(NMS) 模型，已自动还原原始检测头"
                    f"（输出 {rec_info.get('output_shape')}，"
                    f"NMS 剪除: {rec_info.get('nms_removed')}）"
                )
        except Exception as e:
            warnings.append(f"end2end NMS 还原失败（{e}），按原模型检查")
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    inputs = [vi for vi in model.graph.input
              if vi.name not in {i.name for i in model.graph.initializer}]
    outputs = list(model.graph.output)
    ops = [n.op_type for n in model.graph.node]

    info = {
        "filename": os.path.basename(onnx_path),
        "num_inputs": len(inputs),
        "num_outputs": len(outputs),
        "recovered": recovered,
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
            # 双拆分：一个通道维=4（框 reg），另一个=类别数 cls
            # （源头拆分：reg[1,4,N] + cls[1,Nc,N]，cls 通道=纯类别数）
            s0, s1 = out_shapes[0]["shape"], out_shapes[1]["shape"]
            # 找哪个是框（通道维=4）
            def _channel_dim(s):
                # 通道维：4（框）或 1~256（类别数）
                if s[1] == 4 or 5 <= s[1] <= 256:
                    return s[1], s[2], "[1,C,N]"
                if s[2] == 4 or 5 <= s[2] <= 256:
                    return s[2], s[1], "[1,N,C]"
                return -1, -1, "?"

            c0, n0, l0 = _channel_dim(s0)
            c1, n1, l1 = _channel_dim(s1)
            # 一个应该是 4（框），另一个是类别数
            if c0 == 4 and 1 <= c1 <= 256:
                num_classes = c1
                info["num_classes"] = num_classes
                info["output_mode"] = "split"
                if n0 != n1:
                    issues.append(f"双输出候选数不一致：{n0} vs {n1}，必须相同")
            elif c1 == 4 and 1 <= c0 <= 256:
                num_classes = c0
                info["num_classes"] = num_classes
                info["output_mode"] = "split"
                if n0 != n1:
                    issues.append(f"双输出候选数不一致：{n0} vs {n1}，必须相同")
            else:
                issues.append(
                    f"双输出需一个通道维=4（框），另一个=类别数。"
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


def recover_head_from_end2end(onnx_path: str, work_dir: str) -> tuple:
    """检测并还原含 NMS 的 end2end 模型。

    参考 yezijinn/onnx_to_int8.tflite 项目的 recover_head_from_end2end 设计：
    1. 检测 NMS 算子（TopK/NonMaxSuppression，含 TRT 风格 FastNMS）
    2. 从 NMS 输入反向遍历图，收集"NMS 祖先"节点（NMS 之前产生的依赖子图）
    3. 在祖先中找 3 维检测头输出 [1,C,A]（C=4+类别数，5≤C≤256），
       取最接近 NMS 的（节点序号最大）——避免误选 NMS 之后的
       [1,300,6] 后处理张量（那样 NMS 子图剪不掉）
    4. 把检测头张量设为图输出，onnxsim 剪掉 NMS 悬空子图

    返回 (recovered_model_path, info_dict)；
    无 NMS 或还原失败时返回 (onnx_path, None)。
    """
    model = onnx.load(onnx_path, load_external_data=False)

    # 1. 检测 NMS 相关算子（含 TRT 风格 TopK FastNMS）
    nms_ops = {"TopK", "NonMaxSuppression"}
    nms_idx = [i for i, n in enumerate(model.graph.node) if n.op_type in nms_ops]
    if not nms_idx:
        return onnx_path, None

    # 2. shape inference（非严格模式，尽可能推断中间张量形状）
    try:
        model_inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        model_inferred = model
    vi_map = {vi.name: vi for vi in model_inferred.graph.value_info}
    for o in model_inferred.graph.output:
        vi_map.setdefault(o.name, o)

    # 3. 从 NMS 输入反向遍历，收集祖先节点（NMS 之前的依赖子图）
    graph_inputs = {i.name for i in model.graph.input}
    producer = {}  # tensor_name -> 产生它的节点下标
    for idx, node in enumerate(model.graph.node):
        for out in node.output:
            if out:
                producer[out] = idx

    ancestors = set()
    stack = []
    for i in nms_idx:
        for inp in model.graph.node[i].input:
            if inp in producer:
                stack.append(producer[inp])
    while stack:
        idx = stack.pop()
        if idx in ancestors:
            continue
        ancestors.add(idx)
        for inp in model.graph.node[idx].input:
            if inp in producer:
                stack.append(producer[inp])

    # 4a. 优先找 reg/cls 分离（Split）或合并（Concat）结构的【完整检测头】：
    #     - Split: 通道维按 [4, Nc] 或 [Nc, 4] 拆分 → 输入即完整检测头 [1,A,4+Nc]
    #     - Concat: 输入含 reg[1,4,N] + cls[1,Nc,N] → 输出即完整检测头
    #     这类张量是 NMS 后处理链之前的原始检测头，选它做输出才能让
    #     onnxsim 剪掉整个 NMS 子图（含 TopK/GatherElements/ReduceMax 等）。
    #     否则会误选 NMS 链中间张量（如 GatherElements 输出 [1,300,8]），
    #     导致黑名单算子残留、模型不合规。
    head_name = None
    head_shape = None
    for idx in sorted(ancestors, reverse=True):  # 从最接近 NMS 的开始
        node = model.graph.node[idx]
        if node.op_type == "Split":
            axis = next((a.i for a in node.attribute if a.name == "axis"), 1)
            sizes = [a.ints for a in node.attribute if a.name == "split"]
            if not sizes or len(sizes[0]) != 2 or 4 not in sizes[0]:
                continue
            if sizes[0][0] + sizes[0][1] <= 4:
                continue
            inp = node.input[0]
            vi = vi_map.get(inp)
            if vi is None:
                continue
            s = _input_shape(vi)
            if len(s) != 3 or not all(d > 0 for d in s):
                continue
            c_axis = s[axis] if axis in (1, -2) else s[-1]
            if c_axis != sizes[0][0] + sizes[0][1]:
                continue
            head_name, head_shape = inp, s
            break
        elif node.op_type == "Concat":
            axis = next((a.i for a in node.attribute if a.name == "axis"), 1)
            if axis not in (1, -2):
                continue
            for out in node.output:
                if not out:
                    continue
                vi = vi_map.get(out)
                if vi is None:
                    continue
                s = _input_shape(vi)
                if len(s) != 3 or not all(d > 0 for d in s):
                    continue
                c_dim = s[1] if axis == 1 else s[-1]
                nc = c_dim - 4
                if not (5 <= c_dim <= 256 and nc >= 1):
                    continue
                in_shapes = [_input_shape(vi_map.get(i)) for i in node.input
                             if i in vi_map]
                if any(len(sh) == 3 and sh[1] == 4 for sh in in_shapes) and \
                   any(len(sh) == 3 and sh[1] == nc for sh in in_shapes):
                    head_name, head_shape = out, s
                    break
            if head_name:
                break

    # 4b. 回退：在祖先节点输出中找 3 维候选（某维在 5~256 = 4+类别数），
    #     取节点序号最大者（最接近 NMS）
    if head_name is None:
        candidates = []
        for idx in ancestors:
            node = model.graph.node[idx]
            for out in node.output:
                if not out or out in graph_inputs:
                    continue
                vi = vi_map.get(out)
                if vi is None:
                    continue
                shape = _input_shape(vi)
                if len(shape) != 3 or not all(d > 0 for d in shape):
                    continue
                d1, d2 = shape[1], shape[2]
                if (5 <= d1 <= 256 and d2 >= 1) or (5 <= d2 <= 256 and d1 >= 1):
                    candidates.append((idx, out, shape))
        if not candidates:
            return onnx_path, None
        candidates.sort(key=lambda x: x[0])
        head_idx, head_name, head_shape = candidates[-1]

    # 5. 替换图输出：删除原输出，用检测头张量作为新输出
    #    （保留原张量名，ONNX 输出 ValueInfo 的 name 必须匹配图中张量）
    new_outputs = list(model.graph.output)
    while model.graph.output:
        model.graph.output.pop()
    from onnx import ValueInfoProto
    new_vi = ValueInfoProto()
    new_vi.CopyFrom(vi_map[head_name])
    model.graph.output.extend([new_vi])

    # 6. 保存并跑 onnxsim 剪掉 NMS 悬空子图
    pre_path = os.path.join(work_dir, "pre_recover.onnx")
    onnx.save(model, pre_path)

    try:
        import onnxsim
        simplified, ok = onnxsim.simplify(pre_path)
        if ok:
            out_path = os.path.join(work_dir, "recovered.onnx")
            onnx.save(simplified, out_path)
            sm = onnx.load(out_path, load_external_data=False)
            out_shape = _input_shape(sm.graph.output[0]) if sm.graph.output else []
            # 验证剪枝后还有 TopK 吗（应该被剪掉了）
            remaining_nms = [n for n in sm.graph.node if n.op_type in nms_ops]
            return out_path, {
                "recovered": True,
                "output_shape": out_shape,
                "nms_removed": len(remaining_nms) == 0,
                "head_tensor": head_name,
                "original_outputs": [o.name for o in new_outputs],
            }
    except Exception:
        pass

    # onnxsim 失败，返回预处理模型（可能含 NMS 子图但输出已改）
    return pre_path, {"recovered": True, "note": "onnxsim partial, output redirected"}


def _try_source_split(model, out_vi, c_dim: int, n_dim: int,
                      work_dir: str) -> tuple:
    """源头拆分：回溯到 Concat 合并点，在 reg/cls 分支【合并之前】拆分。

    参考 yezijinn/onnx_to_int8.tflite 项目设计（DM4 契约）：
    - reg 输出取自 reg 分支末端（Mul/Add/Sub/Conv 等算术算子输出，像素值域大）
    - cls 输出取自 cls 分支末端（Sigmoid/Softmax 输出，值域 [0,1]）
    这样 TFLiteConverter 量化时两个输出【各自独立校准】：
    cls 自动得到 scale≈1/256（LOGISTIC 约束），reg 得到像素级大 scale。
    而 Slice/Split 拆分（对合并后联合张量）会让 cls 继承父联合范围被压扁。

    对 4 类模型（reg/cls 形状相同 [1,4,N]），用 producer 算子类型区分：
    Sigmoid/Softmax 产出的输入是 cls，另一个是 reg。

    返回 (split_model_path, num_classes)；找不到合并点返回 None。
    """
    try:
        mi = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        return None
    vi_map = {vi.name: vi for vi in mi.graph.value_info}
    for o in mi.graph.output:
        vi_map.setdefault(o.name, o)

    producer = {}
    for idx, n in enumerate(model.graph.node):
        for out in n.output:
            if out:
                producer[out] = idx

    def tshape(name):
        vi = vi_map.get(name)
        if vi is None:
            return None
        s = _input_shape(vi)
        return s if all(d > 0 for d in s) else None

    nc = c_dim - 4

    def _emit(reg, cls, reg_shape, cls_shape, nc_out):
        """在 reg/cls 分支末端加 Identity 改名输出（reg_out/cls_out）。

        - 输出名必须合法（不含 '/'），否则 onnx2tf 生成 TF op name 时
          报 "OP name does not match pattern"
        - Identity 不改变量化传播：reg_out 传播自上游算术算子（像素大 scale
          独立校准），cls_out 传播自 Sigmoid（输出 [0,1] → scale=1/256）
        """
        from onnx import TensorProto
        reg_id = helper.make_node("Identity", [reg], ["reg_out"], name="src_split_reg")
        cls_id = helper.make_node("Identity", [cls], ["cls_out"], name="src_split_cls")
        reg_vi = helper.make_tensor_value_info("reg_out", TensorProto.FLOAT, reg_shape)
        cls_vi = helper.make_tensor_value_info("cls_out", TensorProto.FLOAT, cls_shape)
        old_outputs = list(model.graph.output)
        while model.graph.output:
            model.graph.output.pop()
        model.graph.output.extend([reg_vi, cls_vi])
        model.graph.node.extend([reg_id, cls_id])
        out_path = os.path.join(work_dir, "split_src.onnx")
        try:
            onnx.save(model, out_path)
        except Exception:
            # 保存失败：恢复原输出和节点，回退 Slice 方案
            model.graph.node.remove(reg_id)
            model.graph.node.remove(cls_id)
            while model.graph.output:
                model.graph.output.pop()
            model.graph.output.extend(old_outputs)
            return None
        return out_path, nc_out

    # 从输出张量向上回溯：数据移动算子（Transpose/Reshape/Identity）继续，
    # 遇到 Concat（reg/cls 合并型）或 Split（reg/cls 分离型）直接拆分
    cur = out_vi.name
    visited = set()
    while cur in producer and cur not in visited:
        visited.add(cur)
        idx = producer[cur]
        node = model.graph.node[idx]
        op = node.op_type
        if op == "Split":
            # reg/cls 分离型：Split 把完整检测头 [1,A,4+Nc] 拆成 reg + cls
            axis = next((a.i for a in node.attribute if a.name == "axis"), 1)
            sizes = [a.ints for a in node.attribute if a.name == "split"]
            if not sizes or len(sizes[0]) != 2 or 4 not in sizes[0]:
                return None
            if len(node.output) < 2:
                return None
            s0, s1 = tshape(node.output[0]), tshape(node.output[1])
            if not s0 or not s1 or len(s0) != 3 or len(s1) != 3:
                return None
            # 通道维（CN: axis=1；NC: axis=-1/最后一维）
            if axis in (1, -2):
                c0, c1, n0, n1 = s0[1], s1[1], s0[2], s1[2]
            else:
                c0, c1, n0, n1 = s0[-1], s1[-1], s0[1], s1[1]
            if n0 != n1:
                return None
            reg = cls = None
            reg_shape = cls_shape = None
            for o, c, s in ((node.output[0], c0, s0), (node.output[1], c1, s1)):
                if c == 4:
                    reg, reg_shape = o, s
                elif 1 <= c <= 256:
                    cls, cls_shape = o, s
            if not (reg and cls):
                return None
            nc_out = cls_shape[1] if axis in (1, -2) else cls_shape[-1]
            return _emit(reg, cls, reg_shape, cls_shape, nc_out)
        if op == "Concat":
            axis = 1
            for attr in node.attribute:
                if attr.name == "axis":
                    axis = attr.i
            if axis not in (1, -2):
                return None
            # 收集形状匹配的候选输入（reg[1,4,N] 或 cls[1,nc,N]）
            candidates = []
            for inp in node.input:
                s = tshape(inp)
                if not s or len(s) != 3 or s[2] != n_dim or s[1] not in (4, nc):
                    continue
                candidates.append(inp)
            if len(candidates) < 2:
                return None
            # 形状区分：reg 通道维=4，cls 通道维=nc（nc!=4 时形状可区分）
            reg = cls = None
            for inp in candidates:
                s = tshape(inp)
                if s[1] == 4 and reg is None:
                    reg = inp
                elif s[1] == nc and nc != 4 and cls is None:
                    cls = inp
            # 形状无法区分（nc==4）时，用 producer 算子类型区分：Sigmoid/Softmax→cls
            if reg and not cls:
                for inp in candidates:
                    if inp == reg:
                        continue
                    pidx = producer.get(inp)
                    if pidx is not None and \
                            model.graph.node[pidx].op_type in {"Sigmoid", "Softmax"}:
                        cls = inp
                        break
            if not (reg and cls):
                return None
            # 找到合并点：在分支末端加 Identity 输出（见 _emit 注释）
            return _emit(reg, cls, [1, 4, n_dim], [1, nc, n_dim], nc)
        elif op in {"Transpose", "Reshape", "Identity", "Squeeze",
                    "Unsqueeze", "Flatten"}:
            cur = node.input[0]
        else:
            return None
    return None


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

    # 1. 源头拆分：回溯 Concat 合并点，在 reg/cls 分支合并前拆分。
    #    cls 分支末端是 Sigmoid → TFLite 独立量化 scale=1/256（满足 LOGISTIC）；
    #    reg 分支末端是算术算子 → 像素级大 scale 独立校准。
    try:
        result = _try_source_split(model, out_vi, c_dim, n_dim, work_dir)
        if result:
            return result
    except Exception:
        pass

    # 2. 回退：Slice + Identity 拆分（联合量化，cls 精度稍差但可用）
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

        # 0.3) end2end NMS 检测头还原（参考 yezijinn 项目）：
        #   合规检查已判定为 NMS 模型时，剪掉 NMS 子图、还原原始检测头输出。
        #   注意：这里重新还原一次拿模型文件（check 阶段的还原结果在临时目录已清理）。
        if check["info"].get("recovered"):
            model_for_convert, rec_info = recover_head_from_end2end(model_for_convert, tmp)
            if not (rec_info and rec_info.get("recovered")):
                raise RuntimeError("end2end NMS 检测头还原失败，无法继续导出")
            check["info"]["recover_info"] = rec_info

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
