# 模型侧里程碑说明

状态：`runtime-ready / real-probe-quality-failed`  
目标环境：M5 MacBook Air，16GB 统一内存  
默认场景：视觉合规审查

## 本里程碑交付

- `src/inference/contracts.py`：与 API、数据实现解耦的请求、结果、证据、拒答和溯源协议；
- `src/inference/qwen3_vl.py`：Qwen3-VL 统一适配器，负责提示、JSON 解析、像素坐标和证据一致性；
- `src/inference/mlx_backend.py`：惰性 MLX-VLM 后端，构造时不导入 MLX、不加载权重；
- `src/inference/service_bridge.py`：对现有异步服务协议的薄转换层；
- `src/inference/diagnostics.py`：无网络、无权重加载的环境/缓存/readiness 诊断 CLI；
- `src/inference/performance.py`：预热、P50/P95 和峰值 RSS 的结构化采样接口；
- `src/inference/real_probe.py`：真实模型单/多图探针，保留原始输出、图片哈希、失败和性能证据；
- `src/inference/fusion.py`：主 VLM 与 specialist 的审慎融合、冲突拒答和来源追踪；
- `src/inference/export.py`：批量样本到 evaluator 兼容 JSONL 的原子导出入口；
- `src/inference/mock_backend.py`：无需权重的确定性闭环；
- `src/inference/specialists.py`：Florence-2、RF-DETR 的协同接口占位；
- `src/training/`：LoRA/QLoRA 契约、`ExperimentSpec`/`RunManifest` 与 24 组 dry-run 实验矩阵，不包含训练执行器；
- `configs/models/`：4-bit MLX、M5/16GB QLoRA 探针和 specialist 注册配置；
- `tests/model/`：模型核心测试仅依赖标准库；服务桥接测试复用服务侧 Pydantic 协议。

## 快速验证

在项目根目录执行：

```bash
env PYTHONDONTWRITEBYTECODE=1 uv run --no-project \
  --with 'pydantic>=2,<3' --with typing-extensions \
  --with pillow --with pyyaml --with fastapi \
  python -m unittest discover -s tests/model -p 'test_*.py' -v
```

直接运行无权重拒答闭环：

```bash
python3 -m src.inference.smoke \
  --query '图像看不清，证据不足' \
  --seed 23
```

查看真实运行 readiness；固定 revision 快照完整时当前本机输出 ready：

```bash
python3 -m src.inference.diagnostics --compact --require-ready
```

mock 调用示例：

```python
from pathlib import Path

from src.inference import MockBackend, ModelRequest, Qwen3VLAdapter, load_model_config

config = load_model_config(Path("configs/models/qwen3_vl_2b_mlx_4bit.json"))
adapter = Qwen3VLAdapter(config, MockBackend())
result = adapter.analyze(
    ModelRequest(
        image=b"opaque-to-model-layer",
        image_width=1280,
        image_height=720,
        query="找出违规区域；证据不足时拒答",
    )
)
```

## 真实 MLX 接入

当前配置禁止隐式下载。第五阶段已将许可和大小核验后的固定 revision 放入标准 Hugging Face 缓存，因此默认配置可离线解析它。如改用另一个受控的本地目录，复制配置并设置：

```json
{
  "allow_download": false,
  "local_model_path": "/absolute/path/to/Qwen3-VL-2B-Instruct-4bit"
}
```

然后调用 `create_adapter(config_path)` 和 `adapter.load()`。后端仅接受本地图片路径；图片解码、文件签名、尺寸变换不属于模型目录职责。

真实单图探针调用面：

```python
from pathlib import Path

from PIL import Image

from src.inference import ModelRequest, create_adapter

config_path = Path("configs/models/qwen3_vl_2b_mlx_4bit.local.json")
image_path = Path("/absolute/path/to/licensed_probe.jpg")
with Image.open(image_path) as image:
    width, height = image.size

adapter = create_adapter(config_path)
adapter.load()
result = adapter.analyze(
    ModelRequest(
        image=image_path,
        image_width=width,
        image_height=height,
        query="找出不符合要求的区域；证据不足时拒答",
    )
)
```

首次真实探针前必须确认模型配置指向固定 revision 的本地快照，并补录本地权重 SHA-256；不要把 `allow_download` 改成 `true` 后直接用于正式评测。

上游应把异常稳定映射如下：

| 模型异常 | 建议服务错误 |
|---|---|
| `ModelNotReadyError` | `MODEL_NOT_READY` |
| `ModelOutputError` | `OUTPUT_VALIDATION_FAILED` |
| `ValueError`（请求或配置上限） | `INVALID_IMAGE` / `INVALID_QUERY` |

现有服务侧协议是异步的 `analyze(AdapterRequest) -> ModelOutput`；
`AsyncServiceAdapterBridge` 用工作线程调用同步模型核心并完成字段转换。API Schema
1.0 暂无独立 `refusal` 字段，因此桥接层把拒答码保留为
`warnings=["refusal:<code>"]`；模型核心中的 `RefusalSignal` 仍是强类型字段。

## 输出与拒答契约

`ModelResult` 保证：

- `violation` 至少有一个原图像素坐标证据框；
- `compliant` 不得同时出现正例框；
- `uncertain` 必须携带 `RefusalSignal`，且不得编造正例框；
- 每个结果携带模型 ID、不可变 revision、配置指纹、随机种子和是否确定性生成；
- 模型代码围栏只进行一次确定性剥离；其余无效 JSON 不猜测修复，直接抛出稳定错误。

## 当前证据

2026-08-08 第五阶段在 `.venv` 执行 53 个模型侧测试，53 个全部通过且无跳过；相关 API 桥接 36/36 通过，evaluator 全量 27/27 通过。真实 MLX 诊断 ready，权重哈希、单图与 5 图输出、延迟、进程峰值 RSS 和失败原因见第五阶段报告。

真实探针暴露了结论误判、JSON 截断、越界坐标、字段缺失和负例幻觉；因此真实模型质量状态为 failed。100 请求稳定性、正式精度和量化精度损失仍为 `not_run`。

## 风险与后续门槛

| 风险 | 当前控制 | 下一门槛 |
|---|---|---|
| MLX-VLM API/算子兼容性 | mlx-vlm 0.6.10 真实加载和生成已通过 | 锁定依赖并增加真实 smoke 门禁 |
| 16GB 内存与无主动散热 | 4-bit、512 输出 token 硬上限、串行 MLX 生成 | 用系统级统一内存/交换采样补充进程 RSS |
| 权重漂移 | revision 固定，safetensors SHA-256 已登记 | 每次发布前重算哈希 |
| JSON 不稳定 | 强提示、一次围栏修复、严格解析 | 回归集计算 Schema 有效率与修复率 |
| 定位坐标误差 | 强制原图整数坐标与越界拒绝 | 用标注图验证预处理坐标还原 |
| QLoRA 可行性未知 | 只提供 batch=1 的探针配置与训练协议 | 小样本最小训练探针，不直接跑正式实验 |
| specialist 实际置信度校准未完成 | 冲突默认拒答，来源和匹配数可追踪 | 独立基线后在冻结集标定置信度/IoU 阈值 |
| 超时/取消未实现 | 未在模型目录伪造服务能力 | 由推理编排层提供可取消执行边界 |
| 根运行环境依赖不完整 | 使用 `uv run --with` 的隔离环境已通过桥接测试 | 根依赖所有者锁定 Pydantic 与 `typing_extensions` |
| 服务超时不能停止工作线程中的 MLX 算子 | 串行后端限制并发内存压力 | 真实探针后引入可取消进程边界或后端取消能力 |
| 模型异常尚未被服务层逐类捕获 | `ModelNotReadyError`/`ModelOutputError` 类型稳定 | 服务编排层分别映射 `MODEL_NOT_READY`/`OUTPUT_VALIDATION_FAILED` |
| API 1.0 无结构化拒答字段 | 临时通过 `warnings` 保留拒答码 | 下一兼容版本增加可选 `refusal` 对象 |

## 参考接口

- Qwen3-VL 原始模型卡：<https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct>
- MLX 4-bit 转换模型：<https://huggingface.co/mlx-community/Qwen3-VL-2B-Instruct-4bit>
- MLX-VLM 官方用法：<https://github.com/Blaizzy/mlx-vlm/blob/main/docs/usage.md>
- 下一阶段实验计划：[`next_experiment_plan.md`](next_experiment_plan.md)
- 后端兼容性矩阵：[`backend_integration_matrix.md`](backend_integration_matrix.md)
- 运行诊断与性能采样：[`runtime_diagnostics.md`](runtime_diagnostics.md)
- 算法实验闭环：[`experiment_workflow.md`](experiment_workflow.md)
- 第五阶段真实模型报告：[`phase5_real_model_report.md`](phase5_real_model_report.md)
