# ONNX 轻量化工作台

基于 [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) 的可视化模型轻量化工具。
提供模型加载、图优化、INT8 量化、基准测试、TFLite 导出与高通骁龙 NPU 专用导出等功能，带 Web UI。

## 功能

| 功能 | 说明 |
|------|------|
| 模型加载与解析 | 解析 ONNX 模型的输入/输出/算子/IR/opset，内置样例模型一键试用 |
| 图优化 | onnxruntime ORT_ENABLE_ALL，产出优化后的 ONNX 图 |
| INT8 动态量化 | `quantize_dynamic`，仅量化权重 |
| INT8 静态量化 | `quantize_static` + 校准，支持随机数据 / 真实图片两种校准 |
| 基准测试 | 延迟 p50/p95/p99、吞吐 fps |
| TFLite 导出 | float32 / float16 / INT8 三精度，用于移动端部署 |
| 高通骁龙 NPU 专用导出 | 按 QNN TFLite Delegate 契约做合规检查 + 全 INT8 导出 |

## 快速开始

```bash
# 1. 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/macOS

# 2. 安装核心依赖
pip install -r requirements.txt

# 3.（可选）安装 TFLite/NPU 导出依赖
pip install -r requirements-tflite.txt
pip install tf_keras --no-deps    # 避免依赖解析卡顿

# 4. 启动
python backend/app.py
# 浏览器打开 http://127.0.0.1:5000
```

Windows 用户也可双击项目根目录的 `启动工作台.bat`。

## 项目结构

```
.
├── backend/
│   ├── app.py             Flask API 层
│   ├── engine.py          核心引擎（加载/优化/量化/基准）
│   ├── calibration.py     校准数据读取器
│   ├── tflite_export.py   TFLite 导出
│   └── npu_export.py      高通 QNN NPU 合规检查与专用导出
├── frontend/
│   └── index.html         Web UI（单页）
├── sample_models/         内置样例模型
├── requirements.txt       核心依赖
└── requirements-tflite.txt TFLite/NPU 可选依赖
```

## NPU 专用导出说明

针对 Qualcomm QNN TFLite Delegate 的硬契约：

- **输入**：INT8 NHWC `[1,H,W,3]`，静态形状
- **输出**：INT8 `[1,C,N]` 或 `[1,N,C]`（`C = 4 + 类别数`，anchor-free）
- **全 INT8**，输入输出均为 int8
- **禁用算子**：TopK / GatherElements / NonMaxSuppression / Mod 等 QNN 不支持算子
- **LOGISTIC**：Sigmoid 输出 scale 自动满足 `1/256`、`zero_point=-128`

不合规的模型会被明确拒绝并列出原因；合规模型导出后会验证输入输出确实为 INT8。

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 + 依赖状态 |
| POST | `/api/upload` | 上传 ONNX 模型或校准图片 |
| POST | `/api/sample/create` | 生成内置样例模型 |
| GET | `/api/model/<id>` | 模型详情 |
| POST | `/api/optimize` | 图优化 |
| POST | `/api/quantize/dynamic` | 动态量化 |
| POST | `/api/quantize/static` | 静态量化 |
| POST | `/api/benchmark` | 基准测试 |
| POST | `/api/tflite/export` | TFLite 导出 |
| POST | `/api/npu/check` | NPU 合规检查 |
| POST | `/api/npu/export` | NPU 专用全 INT8 导出 |
| GET | `/api/download?path=` | 下载产物 |

## 技术栈

- **后端**：Flask + onnxruntime + onnx + onnxsim + onnx2tf + tensorflow
- **前端**：原生 HTML / CSS / JavaScript（单页，无构建步骤）

## License

MIT
