"""校准数据读取器：用于静态量化(INT8)的校准数据生成。

提供两种校准方式（对应 UI 里的两个单选项）：
  - RandomCalibrationReader   随机数据
  - ImageCalibrationReader    真实图片（推荐）
两者都实现 onnxruntime.quantization.CalibrationDataReader 接口。
"""

from __future__ import annotations

import os
import glob
from typing import Iterator, Optional

import numpy as np
from PIL import Image
from onnxruntime.quantization import CalibrationDataReader


def _coerce_dtype(onnx_dtype: int) -> np.dtype:
    """把 ONNX 的 dtype 枚举值映射成 numpy dtype，未知则回退到 float32。"""
    mapping = {
        1: np.float32,   # FLOAT
        2: np.uint8,     # UINT8
        3: np.int8,      # INT8
        6: np.int32,     # INT32
        7: np.int64,     # INT64
        9: np.bool_,     # BOOL
        11: np.float64,  # DOUBLE
        12: np.uint32,   # UINT32
        13: np.uint64,   # UINT64
        16: np.float16,  # FLOAT16
    }
    return mapping.get(onnx_dtype, np.float32)


def _input_shape(meta) -> Optional[list]:
    """从 ONNX ValueInfoExp proto 解析出形状，动态维度返回为 -1。"""
    shape = []
    tt = meta.type
    tensor_type = getattr(tt, "tensor_type", None)
    if tensor_type is None:
        return None
    for dim in tensor_type.shape.dim:
        v = dim.dim_value if dim.dim_value > 0 else -1
        shape.append(v)
    return shape


def _resolve_shape(shape: list, batch: int = 1) -> list:
    """把动态维度(-1)替换成可用的具体值。"""
    out = []
    for i, v in enumerate(shape):
        if v <= 0:
            # 第 0 维用 batch，其余维度给一个合理小值
            out.append(batch if i == 0 else 1)
        else:
            out.append(v)
    return out


class RandomCalibrationReader(CalibrationDataReader):
    """用随机数据做校准——无需任何额外输入，速度快但精度一般。"""

    def __init__(self, inputs_meta, num_samples: int = 16, seed: int = 0):
        self._samples = []
        rng = np.random.default_rng(seed)
        for _ in range(num_samples):
            feed = {}
            for m in inputs_meta:
                name = m.name
                shape = _resolve_shape(_input_shape(m) or [1])
                dtype = _coerce_dtype(getattr(m.type.tensor_type, "elem_type", 1))
                if np.issubdtype(dtype, np.floating):
                    arr = rng.standard_normal(shape).astype(dtype)
                else:
                    arr = rng.integers(0, 10, size=shape).astype(dtype)
                feed[name] = arr
            self._samples.append(feed)
        self._iter: Iterator = iter(self._samples)

    def get_next(self):
        return next(self._iter, None)


class ImageCalibrationReader(CalibrationDataReader):
    """用真实图片做校准——把图片 resize/normalize 成模型输入形状，精度更好。"""

    def __init__(self, inputs_meta, image_paths, num_samples: int = 16):
        self._samples = []
        paths = list(image_paths)[:num_samples]
        if not paths:
            raise ValueError("没有可用的校准图片，请先上传图片或改用随机数据校准。")
        for p in paths:
            feed = {}
            for m in inputs_meta:
                name = m.name
                shape = _resolve_shape(_input_shape(m) or [1, 3, 224, 224])
                dtype = _coerce_dtype(getattr(m.type.tensor_type, "elem_type", 1))
                arr = _load_image_as_array(p, shape, dtype)
                feed[name] = arr
            self._samples.append(feed)
        self._iter = iter(self._samples)

    def get_next(self):
        return next(self._iter, None)


def _load_image_as_array(path: str, shape: list, dtype: np.dtype) -> np.ndarray:
    """读一张图片，resize 成目标 H/W，归一化到 [0,1]，再转成目标形状与 dtype。"""
    img = Image.open(path).convert("RGB")
    # shape 约定为 [N, C, H, W]（NCHW）
    if len(shape) >= 4:
        h, w = shape[-2], shape[-1]
    else:
        h, w = 224, 224
    img = img.resize((w, h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
    # 归一化（ImageNet 均值方差，常见做法）
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    # HWC -> CHW
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)  # 加 batch 维
    # 广播/裁剪到目标形状
    target = tuple(_resolve_shape(shape))
    try:
        arr = np.broadcast_to(arr, target).copy()
    except ValueError:
        # 形状不兼容时直接 resize
        arr = np.resize(arr, target)
    return arr.astype(dtype)


def list_images(folder: str) -> list:
    """枚举一个目录下的图片文件。"""
    if not folder or not os.path.isdir(folder):
        return []
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    out = []
    for e in exts:
        out.extend(glob.glob(os.path.join(folder, e)))
        out.extend(glob.glob(os.path.join(folder, "**", e), recursive=True))
    # 去重保序
    seen = set()
    res = []
    for p in out:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            res.append(p)
    return res
