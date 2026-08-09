"""Flask API 层：ONNX 轻量化工作台后端。

端点一览：
  GET  /                          前端页面
  GET  /api/health                健康检查 + 依赖状态
  POST /api/upload                上传 ONNX 模型（或校准图片）
  GET  /api/models                列出已上传模型
  GET  /api/model/<id>            模型详情
  POST /api/optimize              图优化
  POST /api/quantize/dynamic      动态量化
  POST /api/quantize/static       静态量化（带校准）
  POST /api/benchmark             基准测试
  POST /api/compare               多模型对比
  POST /api/tflite/export         导出 TFLite
  GET  /api/exports               列出已导出产物
  GET  /api/download?path=...     下载产物文件
  POST /api/sample/create         生成一个内置样例模型（方便直接试用）
"""

from __future__ import annotations

import os
import json
import uuid
import time
import ctypes
import hashlib
import threading

from flask import Flask, request, jsonify, send_file, send_from_directory, abort

import engine as engine_mod
from engine import OnnxEngine
import tflite_export
import npu_export

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
UPLOADS_DIR = os.path.join(PROJECT_DIR, "uploads")
EXPORTS_DIR = os.path.join(PROJECT_DIR, "exports")
IMAGES_DIR = os.path.join(UPLOADS_DIR, "images")
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")
SAMPLE_DIR = os.path.join(PROJECT_DIR, "sample_models")

for d in (UPLOADS_DIR, EXPORTS_DIR, IMAGES_DIR, FRONTEND_DIR, SAMPLE_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512MB

eng = OnnxEngine(uploads_dir=UPLOADS_DIR, exports_dir=EXPORTS_DIR, images_dir=IMAGES_DIR)

# 简单的内存模型注册表（id -> {path, filename, original_name, ...}）
_MODELS: dict[str, dict] = {}
_LOCK = threading.Lock()
# 注册表持久化文件：重启后恢复模型列表和原始文件名
_REGISTRY_FILE = os.path.join(UPLOADS_DIR, ".registry.json")

ALLOWED_MODEL_EXT = {".onnx"}
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _save_registry():
    try:
        with open(_REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(_MODELS, f, ensure_ascii=False)
    except Exception:
        pass


def _register(path: str, kind: str = "onnx",
              original_name: str = None, sha256: str = None) -> dict:
    mid = uuid.uuid4().hex[:12]
    rec = {"id": mid, "path": path, "filename": os.path.basename(path),
           "kind": kind, "size_bytes": os.path.getsize(path)}
    if original_name:
        rec["original_name"] = original_name
    if sha256:
        rec["sha256"] = sha256
    with _LOCK:
        _MODELS[mid] = rec
    _save_registry()
    return rec


def _file_sha256(stream) -> str:
    """对文件流计算 SHA-256（用于上传去重）。"""
    h = hashlib.sha256()
    while True:
        b = stream.read(1 << 20)
        if not b:
            break
        h.update(b)
    return h.hexdigest()


def _load_registry():
    """启动时恢复注册表：先读 JSON，再扫描上传目录补齐缺失模型。"""
    if os.path.isfile(_REGISTRY_FILE):
        try:
            with open(_REGISTRY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for mid, rec in data.items():
                if rec.get("path") and os.path.isfile(rec["path"]):
                    _MODELS[mid] = rec
        except Exception:
            pass
    for f in sorted(os.listdir(UPLOADS_DIR)):
        if f.endswith(".onnx"):
            p = os.path.join(UPLOADS_DIR, f)
            if not any(r["path"] == p for r in _MODELS.values()):
                _register(p)


_load_registry()
_save_registry()


def _ok(data=None, **kw):
    payload = {"ok": True}
    if data is not None:
        payload["data"] = data
    payload.update(kw)
    return jsonify(payload)


def _err(msg: str, code: int = 400, **kw):
    payload = {"ok": False, "error": msg}
    payload.update(kw)
    return jsonify(payload), code


def _safe_path(base_dir: str, target: str) -> str:
    """防止路径穿越，确保文件在指定目录下。"""
    base = os.path.realpath(base_dir)
    real = os.path.realpath(target)
    if not real.startswith(base + os.sep) and real != base:
        abort(403)
    return real


# ---------------------------------------------------------------------------
# 前端
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    _safe_path(FRONTEND_DIR, os.path.join(FRONTEND_DIR, filename))
    return send_from_directory(FRONTEND_DIR, filename)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    import importlib
    deps = {}
    for m in ("onnxruntime", "onnx", "numpy", "PIL"):
        try:
            mod = importlib.import_module(m)
            deps[m] = getattr(mod, "__version__", "?")
        except Exception:
            deps[m] = None
    tflite_ok, tflite_missing = tflite_export._check_deps()
    return _ok({
        "status": "running",
        "deps": deps,
        "tflite_available": tflite_ok,
        "tflite_missing": tflite_missing,
        "providers": __import__("onnxruntime").get_available_providers(),
    })


@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return _err("未收到文件")
    orig_name = f.filename
    ext = os.path.splitext(orig_name)[1].lower()
    if ext in ALLOWED_IMAGE_EXT:
        # 图片用 ASCII 名保存，避免中文路径问题
        save_name = f"img_{uuid.uuid4().hex[:8]}{ext}"
        save_path = os.path.join(IMAGES_DIR, save_name)
        f.save(save_path)
        return _ok({"type": "image", "filename": orig_name, "saved_name": save_name,
                    "path": save_path, "size_bytes": os.path.getsize(save_path)})
    if ext in ALLOWED_MODEL_EXT:
        # 上传去重：相同内容（SHA-256）不新增副本，直接复用已有记录
        f.stream.seek(0)
        sha = _file_sha256(f.stream)
        dup = None
        with _LOCK:
            for rec in _MODELS.values():
                if rec.get("sha256") == sha:
                    dup = rec
                    break
        if dup:
            return _ok({"type": "onnx", "model": dup, "duplicate": True,
                        "original_name": dup.get("original_name") or dup.get("filename")})
        # 模型用 ASCII 名保存，避免中文路径在 onnx2tf/tensorflow 层打不开
        f.stream.seek(0)
        save_name = f"model_{uuid.uuid4().hex[:8]}{ext}"
        save_path = os.path.join(UPLOADS_DIR, save_name)
        f.save(save_path)
        rec = _register(save_path, kind="onnx", original_name=orig_name, sha256=sha)
        return _ok({"type": "onnx", "model": rec, "original_name": orig_name})
    return _err(f"不支持的文件类型：{ext}")


@app.get("/api/models")
def list_models():
    with _LOCK:
        return _ok({"models": list(_MODELS.values())})


@app.get("/api/model/<mid>")
def model_detail(mid):
    with _LOCK:
        rec = _MODELS.get(mid)
    if not rec:
        return _err("模型不存在", 404)
    try:
        info = eng.load_model_info(rec["path"])
        info["id"] = mid
        # 带上原始文件名（上传时的原名），前端表格/下拉框用
        info["original_name"] = rec.get("original_name") or rec.get("filename")
        return _ok(info)
    except Exception as e:
        return _err(f"解析模型失败：{e}", 500)


@app.post("/api/optimize")
def optimize():
    data = request.get_json(force=True) or {}
    mid = data.get("model_id") or data.get("id")
    level = data.get("level", "all")
    path = _resolve_path(mid, data.get("path"))
    if not path:
        return _err("缺少 model_id 或 path")
    try:
        res = eng.optimize_model(path, level_name=level)
        return _ok(res)
    except Exception as e:
        return _err(f"图优化失败：{e}", 500)


@app.post("/api/quantize/dynamic")
def quantize_dynamic_api():
    data = request.get_json(force=True) or {}
    path = _resolve_path(data.get("model_id"), data.get("path"))
    if not path:
        return _err("缺少 model_id 或 path")
    weight_type = data.get("weight_type", "qint8")
    op_types = data.get("op_types") or None
    try:
        res = eng.quantize_dynamic(path, weight_type=weight_type, op_types=op_types)
        return _ok(res)
    except Exception as e:
        return _err(f"动态量化失败：{e}", 500)


@app.post("/api/quantize/static")
def quantize_static_api():
    data = request.get_json(force=True) or {}
    path = _resolve_path(data.get("model_id"), data.get("path"))
    if not path:
        return _err("缺少 model_id 或 path")
    calibration = data.get("calibration", "random")
    num_samples = int(data.get("num_samples", 16))
    weight_type = data.get("weight_type", "qint8")
    activation_type = data.get("activation_type", "quint8")
    try:
        res = eng.quantize_static(
            path, calibration=calibration, num_samples=num_samples,
            weight_type=weight_type, activation_type=activation_type,
        )
        return _ok(res)
    except Exception as e:
        return _err(f"静态量化失败：{e}", 500)


@app.post("/api/benchmark")
def benchmark_api():
    data = request.get_json(force=True) or {}
    path = _resolve_path(data.get("model_id"), data.get("path"))
    if not path:
        return _err("缺少 model_id 或 path")
    runs = int(data.get("runs", 100))
    warmup = int(data.get("warmup", 10))
    try:
        res = eng.benchmark(path, runs=runs, warmup=warmup)
        return _ok(res)
    except Exception as e:
        return _err(f"基准测试失败：{e}", 500)


@app.post("/api/compare")
def compare_api():
    data = request.get_json(force=True) or {}
    variants = data.get("variants", [])
    if not variants:
        return _err("缺少 variants")
    try:
        res = eng.compare(data.get("original_path", ""), variants)
        return _ok({"results": res})
    except Exception as e:
        return _err(f"对比失败：{e}", 500)


@app.post("/api/tflite/export")
def tflite_export_api():
    data = request.get_json(force=True) or {}
    path = _resolve_path(data.get("model_id"), data.get("path"))
    if not path:
        return _err("缺少 model_id 或 path")
    precision = data.get("precision", "float32")
    calibration = data.get("calibration", "random")
    num_samples = int(data.get("num_samples", 16))
    try:
        res = tflite_export.export_tflite(
            path, EXPORTS_DIR, precision=precision,
            calibration=calibration, num_samples=num_samples,
            images_dir=IMAGES_DIR,
        )
        return _ok(res)
    except RuntimeError as e:
        # 依赖缺失类错误，返回 501 + 友好提示
        return _err(str(e), 501, kind="dependency")
    except Exception as e:
        return _err(f"TFLite 导出失败：{e}", 500)


@app.get("/api/npu/contract")
def npu_contract():
    """返回 NPU 契约描述。"""
    return _ok(npu_export.describe_contract())


@app.post("/api/npu/check")
def npu_check():
    """检查模型是否符合 QNN NPU 契约。"""
    data = request.get_json(force=True) or {}
    path = _resolve_path(data.get("model_id"), data.get("path"))
    if not path:
        return _err("缺少 model_id 或 path")
    try:
        res = npu_export.check_npu_compliance(path)
        return _ok(res)
    except Exception as e:
        return _err(f"NPU 检查失败：{e}", 500)


@app.post("/api/npu/export")
def npu_export_api():
    """导出 QNN NPU 专用全 INT8 TFLite（需先通过合规检查）。"""
    data = request.get_json(force=True) or {}
    mid = data.get("model_id")
    path = _resolve_path(mid, data.get("path"))
    if not path:
        return _err("缺少 model_id 或 path")
    # 取原始模型名用于输出文件命名（resnet50.onnx → resnet50_npumotified.tflite）
    original_name = None
    if mid:
        with _LOCK:
            rec = _MODELS.get(mid)
        if rec:
            original_name = rec.get("original_name") or rec.get("filename")
    calibration = data.get("calibration", "random")
    num_samples = int(data.get("num_samples", 8))
    if calibration == "image":
        from calibration import list_images
        if not list_images(IMAGES_DIR):
            return _err("已选择真实图片校准，但尚未上传校准图片（请先上传 50~200 张）", 400)
    try:
        res = npu_export.export_npu_tflite(
            path, EXPORTS_DIR, calibration=calibration,
            num_samples=num_samples, images_dir=IMAGES_DIR,
            original_name=original_name,
        )
        return _ok(res)
    except RuntimeError as e:
        return _err(str(e), 501, kind="npu_invalid")
    except Exception as e:
        return _err(f"NPU 导出失败：{e}", 500)


@app.get("/api/exports")
def list_exports():
    out = []
    if os.path.isdir(EXPORTS_DIR):
        for fn in sorted(os.listdir(EXPORTS_DIR)):
            p = os.path.join(EXPORTS_DIR, fn)
            if os.path.isfile(p):
                out.append({
                    "filename": fn,
                    "path": p,
                    "size_bytes": os.path.getsize(p),
                    "ext": os.path.splitext(fn)[1],
                })
    # 也列出 uploads 下的原始模型
    if os.path.isdir(UPLOADS_DIR):
        for fn in sorted(os.listdir(UPLOADS_DIR)):
            if fn.endswith(".onnx"):
                p = os.path.join(UPLOADS_DIR, fn)
                out.append({"filename": fn, "path": p,
                            "size_bytes": os.path.getsize(p), "ext": ".onnx"})
    return _ok({"files": out})


@app.get("/api/download")
def download():
    path = request.args.get("path", "")
    if not path or not os.path.isfile(path):
        return _err("文件不存在", 404)
    # 只允许下载 uploads / exports / sample_models 下的文件
    allowed_roots = (UPLOADS_DIR, EXPORTS_DIR, SAMPLE_DIR)
    real = os.path.realpath(path)
    if not any(real.startswith(os.path.realpath(r) + os.sep) for r in allowed_roots):
        return _err("非法下载路径", 403)
    return send_file(real, as_attachment=True, download_name=os.path.basename(path))


@app.post("/api/sample/create")
def create_sample():
    """生成一个内置样例模型（小型 MLP），方便用户直接试用全部功能。"""
    name = request.get_json(silent=True) or {}
    fname = name.get("name", "sample_mlp.onnx")
    out_path = os.path.join(SAMPLE_DIR, fname)
    try:
        _make_sample_mlp(out_path)
        rec = _register(out_path, kind="sample")
        return _ok({"model": rec})
    except Exception as e:
        return _err(f"生成样例失败：{e}", 500)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _resolve_path(mid, path):
    if mid:
        with _LOCK:
            rec = _MODELS.get(mid)
        if rec:
            return rec["path"]
    if path and os.path.isfile(path):
        return path
    return None


def _make_sample_mlp(out_path: str):
    """用 onnx 原生 API 构造一个 3 层 MLP 样例模型。"""
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    import numpy as np

    # 输入 [N,64] -> 64 -> 32 -> 10
    rng = np.random.default_rng(0)
    w1 = rng.standard_normal((64, 64)).astype(np.float32)
    b1 = rng.standard_normal((64,)).astype(np.float32)
    w2 = rng.standard_normal((64, 32)).astype(np.float32)
    b2 = rng.standard_normal((32,)).astype(np.float32)
    w3 = rng.standard_normal((32, 10)).astype(np.float32)
    b3 = rng.standard_normal((10,)).astype(np.float32)

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 64])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

    nodes = [
        helper.make_node("Gemm", ["input", "w1", "b1"], ["h1"], name="Gemm1"),
        helper.make_node("Relu", ["h1"], ["h2"], name="Relu1"),
        helper.make_node("Gemm", ["h2", "w2", "b2"], ["h3"], name="Gemm2"),
        helper.make_node("Relu", ["h3"], ["h4"], name="Relu2"),
        helper.make_node("Gemm", ["h4", "w3", "b3"], ["output"], name="Gemm3"),
    ]
    inits = [
        numpy_helper.from_array(w1, name="w1"),
        numpy_helper.from_array(b1, name="b1"),
        numpy_helper.from_array(w2, name="w2"),
        numpy_helper.from_array(b2, name="b2"),
        numpy_helper.from_array(w3, name="w3"),
        numpy_helper.from_array(b3, name="b3"),
    ]
    graph = helper.make_graph(nodes, "sample_mlp", [inp], [out], inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)],
                          producer_name="onnx-workbench")
    m.ir_version = 9
    onnx.checker.check_model(m)
    onnx.save(m, out_path)


if __name__ == "__main__":
    print("ONNX 轻量化工作台启动中...")
    print(f"项目目录: {PROJECT_DIR}")
    print(f"上传目录: {UPLOADS_DIR}")
    print(f"导出目录: {EXPORTS_DIR}")
    app.run(host="127.0.0.1", port=5000, debug=False)
