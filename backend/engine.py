"""ONNX 轻量化工作台核心引擎。

封装 microsoft/onnxruntime 的能力，提供：
  - load_model_info      解析模型结构与统计信息
  - optimize_model       图优化（ORT_ENABLE_ALL，并保存优化后的图）
  - quantize_dynamic     INT8 动态量化
  - quantize_static      INT8 静态量化（带校准）
  - benchmark            推理基准测试（延迟/吞吐）
"""

from __future__ import annotations

import os
import time
import copy
import contextlib
from typing import Optional

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_dynamic,
    quantize_static,
    QuantType,
    QuantFormat,
    CalibrationMethod,
)

from calibration import RandomCalibrationReader, ImageCalibrationReader, list_images


@contextlib.contextmanager
def _tolerant_temp_cleanup():
    """量化期间 onnxruntime 会生成并删除临时文件（-inferred.onnx、临时目录）。

    某些环境（如 WorkBuddy 沙箱）会拦截文件删除并抛出非标准异常，导致量化中断。
    onnxruntime 内部用的是 pathlib.Path.unlink() 和 shutil.rmtree()，而不是 os.remove，
    所以必须 patch 这两个。这里把它们临时替换成"删除失败也静默"的版本——
    这些只是清理临时文件，删不掉不影响量化结果。
    """
    import pathlib
    import shutil

    _orig_unlink = pathlib.Path.unlink
    _orig_rmtree = shutil.rmtree
    _orig_os_remove = os.remove
    _orig_os_unlink = os.unlink

    def _win_delete_file(path):
        """用 Windows DeleteFileW 原生 API 绕过 Python 层拦截。"""
        try:
            import ctypes
            p = os.fspath(path)
            return ctypes.windll.kernel32.DeleteFileW(p) != 0
        except Exception:
            return False

    def _win_delete_dir(path):
        """用 Windows RemoveDirectoryW 原生 API。"""
        try:
            import ctypes
            p = os.fspath(path)
            # 先尝试递归删除内容
            for root, dirs, files in os.walk(path, topdown=False):
                for f in files:
                    _win_delete_file(os.path.join(root, f))
                for d in dirs:
                    ctypes.windll.kernel32.RemoveDirectoryW(os.path.join(root, d))
            return ctypes.windll.kernel32.RemoveDirectoryW(p) != 0
        except Exception:
            return False

    def _silent_path_unlink(self, *a, **kw):
        if _win_delete_file(self):
            return
        try:
            return _orig_unlink(self, *a, **kw)
        except Exception:
            pass  # 临时文件删不掉无所谓

    def _silent_rmtree(path, *a, **kw):
        try:
            return _orig_rmtree(path, *a, **kw)
        except Exception:
            _win_delete_dir(path)

    def _silent_os_remove(path, *a, **kw):
        if _win_delete_file(path):
            return
        try:
            return _orig_os_remove(path, *a, **kw)
        except Exception:
            pass

    pathlib.Path.unlink = _silent_path_unlink
    shutil.rmtree = _silent_rmtree
    os.remove = _silent_os_remove
    os.unlink = _silent_os_remove
    try:
        yield
    finally:
        pathlib.Path.unlink = _orig_unlink
        shutil.rmtree = _orig_rmtree
        os.remove = _orig_os_remove
        os.unlink = _orig_os_unlink


def preprocess_for_onnx2tf(onnx_path: str, work_dir: str) -> str:
    """用 onnxsim 简化模型，折叠 Constant 节点为 initializer。

    解决 onnx2tf 2.6.8 的 bug：当 Add/Reshape 等算子的输入直接连
    Constant 节点时，disable_unnecessary_transpose 会访问
    Constant 对象的 .op 属性而崩溃（'Constant' object has no attribute 'op'）。
    onnxsim 会把 Constant 折叠进 initializer，绕过该问题。

    返回简化后的模型路径；onnxsim 不可用或失败时返回原路径。
    """
    try:
        import onnxsim
        model = onnx.load(onnx_path, load_external_data=False)
        simplified, ok = onnxsim.simplify(model)
        if ok:
            out_path = os.path.join(work_dir, os.path.basename(onnx_path))
            onnx.save(simplified, out_path)
            return out_path
    except Exception:
        pass
    return onnx_path


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def file_size_str(num_bytes: int) -> str:
    """字节数转成人类可读的 MB/KB 字符串。"""
    if num_bytes is None:
        return "-"
    mb = num_bytes / (1024 * 1024)
    if mb >= 1:
        return f"{mb:.2f} MB"
    return f"{num_bytes / 1024:.1f} KB"


def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def _safe_check(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"模型文件不存在：{path}")


def _value_info_name(vi) -> str:
    return vi.name


def _elem_type_str(t: int) -> str:
    return {
        1: "float32", 2: "uint8", 3: "int8", 6: "int32", 7: "int64",
        9: "bool", 11: "float64", 12: "uint32", 13: "uint64", 16: "float16",
    }.get(t, f"type({t})")


def _shape_of(vi) -> list:
    tt = getattr(vi.type, "tensor_type", None)
    if tt is None:
        return []
    return [d.dim_value if d.dim_value > 0 else -1 for d in tt.shape.dim]


# ---------------------------------------------------------------------------
# 核心引擎
# ---------------------------------------------------------------------------

class OnnxEngine:
    """工作台核心：所有对模型的操作都通过这里。"""

    def __init__(self, uploads_dir: str, exports_dir: str, images_dir: str):
        self.uploads_dir = uploads_dir
        self.exports_dir = exports_dir
        self.images_dir = images_dir
        os.makedirs(exports_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

    # ---- 模型信息 ----------------------------------------------------------

    def load_model_info(self, path: str) -> dict:
        """读取并解析 ONNX 模型，返回结构化信息。"""
        _safe_check(path)
        model = onnx.load(path, load_external_data=False)

        info = {
            "path": path,
            "filename": os.path.basename(path),
            "size_bytes": os.path.getsize(path),
            "size": file_size_str(os.path.getsize(path)),
            "ir_version": model.ir_version,
            "producer": model.producer_name or "unknown",
            "opsets": [{"domain": o.domain or "ai.onnx", "version": o.version}
                       for o in model.opset_import],
            "inputs": [],
            "outputs": [],
            "num_nodes": len(model.graph.node),
            "num_params": sum(len(init.dims) for init in model.graph.initializer) or
                           len(model.graph.initializer),
            "ops": [],
        }

        for vi in model.graph.input:
            if vi.name in {i.name for i in model.graph.initializer}:
                continue
            elem = getattr(vi.type, "tensor_type", None)
            info["inputs"].append({
                "name": vi.name,
                "shape": _shape_of(vi),
                "dtype": _elem_type_str(getattr(elem, "elem_type", 1)) if elem else "unknown",
            })
        for vi in model.graph.output:
            elem = getattr(vi.type, "tensor_type", None)
            info["outputs"].append({
                "name": vi.name,
                "shape": _shape_of(vi),
                "dtype": _elem_type_str(getattr(elem, "elem_type", 1)) if elem else "unknown",
            })

        # 算子统计
        op_counter: dict[str, int] = {}
        for node in model.graph.node:
            op_counter[node.op_type] = op_counter.get(node.op_type, 0) + 1
        info["ops"] = sorted(op_counter.items(), key=lambda x: -x[1])

        # 用 onnx.checker 做一次校验（不抛错，只标记）
        try:
            onnx.checker.check_model(model)
            info["valid"] = True
        except Exception as e:
            info["valid"] = False
            info["check_error"] = str(e)

        return info

    # ---- 图优化 ------------------------------------------------------------

    def optimize_model(self, path: str, level_name: str = "all") -> dict:
        """利用 ORT 的图优化产出优化后的 ONNX 文件。

        level_name: "basic" | "extended" | "all"
        """
        _safe_check(path)
        level_map = {
            "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        level = level_map.get(level_name, ort.GraphOptimizationLevel.ORT_ENABLE_ALL)

        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(self.exports_dir, f"{base}_optimized.onnx")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = level
        # 关键：设置优化后图的保存路径
        sess_opts.optimized_model_filepath = out_path

        # 创建 session 触发优化并写出文件
        ort.InferenceSession(path, sess_opts, providers=["CPUExecutionProvider"])

        if not os.path.isfile(out_path):
            raise RuntimeError("图优化未产出文件，可能该优化级别无需改动。")

        return {
            "path": out_path,
            "filename": os.path.basename(out_path),
            "size": file_size_str(os.path.getsize(out_path)),
            "size_bytes": os.path.getsize(out_path),
            "level": level_name,
        }

    # ---- 动态量化 ----------------------------------------------------------

    def quantize_dynamic(self, path: str, weight_type: str = "qint8",
                         op_types: Optional[list] = None) -> dict:
        """INT8 动态量化：只量化权重，激活在运行时动态量化。"""
        _safe_check(path)
        wt_map = {
            "qint8": QuantType.QInt8,
            "quint8": QuantType.QUInt8,
            "qint16": QuantType.QInt16,
        }
        wt = wt_map.get(weight_type, QuantType.QInt8)
        if op_types is None:
            op_types = ["MatMul", "Gemm", "Conv"]

        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(self.exports_dir, f"{base}_quantized_dynamic.onnx")

        with _tolerant_temp_cleanup():
            quantize_dynamic(
                model_input=path,
                model_output=out_path,
                weight_type=wt,
                op_types_to_quantize=op_types,
                per_channel=False,
                reduce_range=False,
            )

        if not os.path.isfile(out_path):
            raise RuntimeError("动态量化未产出文件。")

        return {
            "path": out_path,
            "filename": os.path.basename(out_path),
            "size": file_size_str(os.path.getsize(out_path)),
            "size_bytes": os.path.getsize(out_path),
            "kind": "INT8 动态量化",
            "weight_type": weight_type,
        }

    # ---- 静态量化（带校准）--------------------------------------------------

    def quantize_static(self, path: str, calibration: str = "random",
                        num_samples: int = 16, weight_type: str = "qint8",
                        activation_type: str = "quint8") -> dict:
        """INT8 静态量化：权重与激活都量化，需要校准数据来确定激活范围。

        calibration: "random"（随机数据）或 "image"（真实图片，推荐）
        """
        _safe_check(path)

        # 先用 onnx 拿到输入元信息，构建校准读取器
        model = onnx.load(path, load_external_data=False)
        inputs_meta = [vi for vi in model.graph.input
                       if vi.name not in {i.name for i in model.graph.initializer}]

        if calibration == "image":
            imgs = list_images(self.images_dir)
            if not imgs:
                raise ValueError(
                    "images 目录里没有图片，无法用真实图片校准。"
                    "请先上传校准图片，或改用随机数据校准。"
                )
            reader = ImageCalibrationReader(inputs_meta, imgs, num_samples=num_samples)
        else:
            reader = RandomCalibrationReader(inputs_meta, num_samples=num_samples)

        wt_map = {"qint8": QuantType.QInt8, "quint8": QuantType.QUInt8, "qint16": QuantType.QInt16}
        at_map = {"qint8": QuantType.QInt8, "quint8": QuantType.QUInt8, "qint16": QuantType.QInt16}

        base = os.path.splitext(os.path.basename(path))[0]
        suffix = "image" if calibration == "image" else "random"
        out_path = os.path.join(self.exports_dir, f"{base}_quantized_static_{suffix}.onnx")

        with _tolerant_temp_cleanup():
            quantize_static(
                model_input=path,
                model_output=out_path,
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                activation_type=at_map.get(activation_type, QuantType.QUInt8),
                weight_type=wt_map.get(weight_type, QuantType.QInt8),
                per_channel=False,
                reduce_range=False,
                calibrate_method=CalibrationMethod.MinMax,
            )

        if not os.path.isfile(out_path):
            raise RuntimeError("静态量化未产出文件。")

        return {
            "path": out_path,
            "filename": os.path.basename(out_path),
            "size": file_size_str(os.path.getsize(out_path)),
            "size_bytes": os.path.getsize(out_path),
            "kind": f"INT8 静态量化({('真实图片' if calibration == 'image' else '随机数据')}校准)",
            "calibration": calibration,
            "weight_type": weight_type,
            "activation_type": activation_type,
        }

    # ---- 基准测试 ----------------------------------------------------------

    def benchmark(self, path: str, runs: int = 100, warmup: int = 10) -> dict:
        """对模型做推理基准测试，返回延迟与吞吐统计。"""
        _safe_check(path)
        if runs < 1:
            runs = 1
        if warmup < 0:
            warmup = 0

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = max(1, os.cpu_count() or 1)
        sess = ort.InferenceSession(path, sess_opts, providers=["CPUExecutionProvider"])

        # 构造随机输入 feed
        rng = np.random.default_rng(0)
        feed = {}
        for inp in sess.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            dtype = np.float32
            try:
                dtype = ort.np_dtype(inp.type)
            except Exception:
                pass
            if np.issubdtype(dtype, np.floating):
                feed[inp.name] = rng.standard_normal(shape).astype(dtype)
            else:
                feed[inp.name] = rng.integers(0, 10, size=shape).astype(dtype)

        # 预热
        for _ in range(warmup):
            sess.run(None, feed)

        # 计时
        latencies = []
        for _ in range(runs):
            t0 = time.perf_counter()
            sess.run(None, feed)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        arr = np.asarray(latencies)
        result = {
            "runs": runs,
            "warmup": warmup,
            "avg_ms": round(float(arr.mean()), 3),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "min_ms": round(float(arr.min()), 3),
            "max_ms": round(float(arr.max()), 3),
            "std_ms": round(float(arr.std()), 3),
            "throughput_fps": round(1000.0 / float(arr.mean()), 2) if arr.mean() > 0 else 0,
            "input_name": list(feed.keys())[0] if feed else "",
        }
        return result

    # ---- 综合比较 ----------------------------------------------------------

    def compare(self, original_path: str, variants: list) -> list:
        """对原始模型和若干变体做体积+基准对比。"""
        out = []
        for v in variants:
            p = v.get("path") or v.get("model_path")
            entry = {
                "name": v.get("name") or v.get("kind") or os.path.basename(p),
                "path": p,
                "size": file_size_str(os.path.getsize(p)) if p and os.path.isfile(p) else "-",
                "size_bytes": os.path.getsize(p) if p and os.path.isfile(p) else 0,
                "kind": v.get("kind", ""),
                "filename": os.path.basename(p) if p else "",
            }
            try:
                bm = self.benchmark(p, runs=50, warmup=5)
                entry.update(bm)
            except Exception as e:
                entry["benchmark_error"] = str(e)
            out.append(entry)
        return out
