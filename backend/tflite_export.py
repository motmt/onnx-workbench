"""TFLite 导出模块：把 ONNX 模型转成 TensorFlow Lite，供移动端部署。

使用 onnx2tf（v2.6+）直接生成 .tflite 文件：
  - float32  onnx2tf 默认输出
  - float16  onnx2tf 默认输出
  - int8     开启 output_integer_quantized_tflite + 校准数据

onnx2tf 内部会做 ONNX 简化(shape inference)并删除临时文件，
需用 _tolerant_temp_cleanup 包裹以兼容沙箱环境。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Optional

import numpy as np
import onnx

# 复用 engine 里的临时文件清理（onnx2tf/onnxsim 也会删临时 inferred 文件）
from engine import _tolerant_temp_cleanup, preprocess_for_onnx2tf
from calibration import RandomCalibrationReader, ImageCalibrationReader, list_images


def _check_deps() -> tuple:
    """检查 TFLite 导出依赖是否就绪，返回 (ok, missing)。"""
    missing = []
    try:
        import onnx2tf  # noqa: F401
    except Exception:
        missing.append("onnx2tf")
    try:
        import tensorflow  # noqa: F401
    except Exception:
        missing.append("tensorflow")
    return (len(missing) == 0, missing)


def _size_str(num_bytes: int) -> str:
    mb = num_bytes / (1024 * 1024)
    if mb >= 1:
        return f"{mb:.2f} MB"
    return f"{num_bytes / 1024:.1f} KB"


def _model_base(onnx_path: str) -> str:
    return os.path.splitext(os.path.basename(onnx_path))[0]


def _find_tflite(out_dir: str, precision: str) -> Optional[str]:
    """在 onnx2tf 输出目录里查找匹配精度的 .tflite 文件。"""
    precision = precision.lower()
    # onnx2tf 的命名约定：{base}_float32.tflite / {base}_float16.tflite
    # int8: {base}_quantized_int8.tflite 或 {base}_int8.tflite
    candidates = []
    if precision == "float32":
        candidates = ["_float32.tflite"]
    elif precision == "float16":
        candidates = ["_float16.tflite"]
    elif precision == "int8":
        candidates = ["_quantized_int8.tflite", "_int8.tflite",
                      "_full_integer_quantized.tflite"]
    for fn in sorted(os.listdir(out_dir)):
        low = fn.lower()
        if low.endswith(".tflite"):
            for c in candidates:
                if low.endswith(c):
                    return os.path.join(out_dir, fn)
            # int8 的兜底匹配
            if precision == "int8" and ("int8" in low or "quant" in low):
                return os.path.join(out_dir, fn)
    return None


def _input_shape(vi) -> list:
    tt = getattr(vi.type, "tensor_type", None)
    if tt is None:
        return []
    return [d.dim_value if d.dim_value > 0 else -1 for d in tt.shape.dim]


def _resolve_shape(shape: list, batch: int = 1) -> list:
    out = []
    for i, v in enumerate(shape):
        out.append(batch if (v <= 0 and i == 0) else (1 if v <= 0 else v))
    return out


def _build_calibration_npy(onnx_path: str, calibration: str,
                           num_samples: int, images_dir: Optional[str],
                           work_dir: str) -> list:
    """生成 int8 校准数据，保存成 .npy。

    onnx2tf 的 -oiqt 要求每条校准数据为 4 元素：
        [input_op_name, numpy_file_path, mean, std]
    且校准数据需预归一化到 [0, 1] 范围。
    返回 [[op_name, npy_path, mean_str, std_str], ...]
    """
    from PIL import Image as PILImage

    model = onnx.load(onnx_path, load_external_data=False)
    inputs_meta = [vi for vi in model.graph.input
                   if vi.name not in {i.name for i in model.graph.initializer}]

    rng = np.random.default_rng(0)
    result = []
    for m in inputs_meta:
        name = m.name
        shape = _resolve_shape(_input_shape(m) or [1])
        is_image = len(shape) >= 3  # 含通道维的输入视为图像

        if calibration == "image" and is_image and images_dir:
            imgs = list_images(images_dir)
            if imgs:
                # 用真实图片，缩放到 [0,1]
                img = PILImage.open(imgs[0]).convert("RGB")
                h, w = shape[-2], shape[-1] if len(shape) >= 4 else (224, 224)
                if len(shape) >= 4:
                    h, w = shape[-2], shape[-1]
                else:
                    h, w = 224, 224
                img = img.resize((w, h), PILImage.BILINEAR)
                arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC [0,1]
                arr = np.transpose(arr, (2, 0, 1))  # CHW
                arr = np.expand_dims(arr, 0)  # NCHW
                arr = np.broadcast_to(arr, tuple(shape)).copy()
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            else:
                arr = rng.random(shape, dtype=np.float32)  # [0,1)
                mean = np.array(0.0, dtype=np.float32)
                std = np.array(1.0, dtype=np.float32)
        else:
            # 随机数据，归一化到 [0,1]
            arr = rng.random(shape, dtype=np.float32)
            mean = np.array(0.0, dtype=np.float32)
            std = np.array(1.0, dtype=np.float32)

        npy_path = os.path.join(work_dir, f"calib_{name}.npy")
        np.save(npy_path, arr.astype(np.float32))
        result.append([name, npy_path, mean, std])
    return result


def export_tflite(onnx_path: str, out_dir: str, precision: str = "float32",
                  calibration: str = "random", num_samples: int = 16,
                  images_dir: Optional[str] = None) -> dict:
    """把 ONNX 模型导出为 TFLite。

    precision: "float32" | "float16" | "int8"
    calibration: "random" | "image"（仅 int8 有效）
    返回 {path, filename, size, size_bytes, precision}
    """
    ok, missing = _check_deps()
    if not ok:
        raise RuntimeError(
            f"TFLite 导出缺少依赖：{', '.join(missing)}。"
            f"请在 venv 中执行：pip install onnx2tf tensorflow-cpu"
        )

    import onnx2tf

    os.makedirs(out_dir, exist_ok=True)
    base = _model_base(onnx_path)
    precision = precision.lower()

    tmp_tf = tempfile.mkdtemp(prefix="ort_tflite_")
    try:
        # 0) onnxsim 预处理：折叠 Constant，绕过 onnx2tf 的 Constant bug
        model_for_convert = preprocess_for_onnx2tf(onnx_path, tmp_tf)

        # 构造 onnx2tf 参数
        kwargs = dict(
            input_onnx_file_path=model_for_convert,
            output_folder_path=tmp_tf,
            non_verbose=True,
        )

        if precision == "int8":
            # int8 量化需要 tf_converter 后端，它依赖 tensorflow + tf_keras
            try:
                import tf_keras  # noqa: F401
            except Exception:
                raise RuntimeError(
                    "INT8 TFLite 量化需要 tf_keras 依赖。请在 venv 中执行："
                    "pip install tf_keras"
                )
            # 生成校准数据
            calib = _build_calibration_npy(
                onnx_path, calibration, num_samples, images_dir, tmp_tf
            )
            if calib:
                kwargs["output_integer_quantized_tflite"] = True
                kwargs["custom_input_op_name_np_data_path"] = calib
                kwargs["input_quant_dtype"] = "int8"
                kwargs["output_quant_dtype"] = "int8"
                # int8 量化需要 tf_converter 后端（flatbuffer_direct 不支持量化）
                kwargs["tflite_backend"] = "tf_converter"

        # onnx2tf 内部会删 onnxsim 临时文件，用 tolerant cleanup 包裹
        with _tolerant_temp_cleanup():
            onnx2tf.convert(**kwargs)

        # 在输出目录里找对应精度的 tflite
        tflite_src = _find_tflite(tmp_tf, precision)
        if not tflite_src:
            # 列出实际产出的文件，便于诊断
            produced = [f for f in os.listdir(tmp_tf) if f.endswith(".tflite")]
            # 检查是否是 tf_keras 缺失导致的 int8 失败
            hint = ""
            if precision == "int8":
                try:
                    import tf_keras  # noqa: F401
                except Exception:
                    hint = "（int8 量化需要 tf_keras，请执行：pip install tf_keras）"
            raise RuntimeError(
                f"未找到 {precision} 精度的 tflite 文件。{hint}"
                f"实际产出：{produced or '无 tflite 文件'}"
            )

        out_path = os.path.join(out_dir, f"{base}_{precision}.tflite")
        shutil.copyfile(tflite_src, out_path)

        return {
            "path": out_path,
            "filename": os.path.basename(out_path),
            "size": _size_str(os.path.getsize(out_path)),
            "size_bytes": os.path.getsize(out_path),
            "precision": precision,
            "calibration": calibration if precision == "int8" else None,
        }
    finally:
        # 清理临时目录（用 tolerant 方式，避免沙箱拦截）
        with _tolerant_temp_cleanup():
            shutil.rmtree(tmp_tf, ignore_errors=True)


def available_precisions() -> list:
    """返回当前可导出的精度列表。"""
    return [
        {"id": "float32", "name": "float32", "desc": "原始精度，最大但兼容性最好"},
        {"id": "float16", "name": "float16", "desc": "半精度，体积减半，精度基本无损"},
        {"id": "int8", "name": "INT8 量化", "desc": "全整数量化，体积最小，适合移动端"},
    ]
