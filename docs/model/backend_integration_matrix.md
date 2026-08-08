# 算法—后端兼容性矩阵

检查日期：2026-08-08  
后端契约位置：`src/core/model_registry.py`、`src/core/schemas.py`  
模型桥接位置：`src/inference/service_bridge.py`

## 契约矩阵

| 后端要求或场景 | 桥接行为 | 验证结果 |
|---|---|---|
| `ModelAdapter` runtime Protocol | 提供 `identity`、`ready`、异步 `analyze` | `isinstance(..., ModelAdapter)` 通过 |
| registry 直接注入 | `create_service_adapter` 返回可直接写入 `ModelRegistration.adapter` 的对象 | active alias 可解析 |
| `AdapterRequest.image_bytes` | mock 直接使用 bytes；MLX 才写入请求生命周期内的安全临时文件 | mock 成功；MLX 缓存缺失未运行 |
| `AnalyzeOptions` | 转换为 `GenerationConfig`，保留 seed、temperature、top-p、max tokens | 成功路径通过 |
| 正常违规输出 | 转为后端 `ModelOutput` 和 `ObjectSource.VLM` | 通过 |
| 不确定/拒答 | 保留 `uncertain=true`；API 1.0 暂以 `warnings=refusal:<code>` 携带拒答码 | 通过 |
| 非法模型 JSON | 不泄露 raw text，返回 provider-neutral 非法 Mapping，复用后端 Schema 校验 | 422 `OUTPUT_VALIDATION_FAILED` |
| 越界 bbox | 模型核心先拒绝，桥接进入相同 Schema 失败路径 | 422 `OUTPUT_VALIDATION_FAILED` |
| 模型运行时未就绪 | 转为后端 `ServiceError(MODEL_NOT_READY)` | 接口已实现 |
| 请求生成参数超配置上限 | 转为 `ServiceError(INVALID_QUERY)` | 接口已实现 |
| 后端推理超时 | 桥接 await 可由 `AnalyzeService.wait_for` 取消 | 504 `INFERENCE_TIMEOUT` |
| 上游任务取消 | 不捕获 `CancelledError`，保持取消语义 | 通过 |
| 无模型可选依赖环境 | `src.inference` 与 mock 不导入 MLX、Pydantic、FastAPI；仅导入 `service_bridge` 时需要后端依赖 | 标准 Python smoke 通过 |

## 后端最小注入方式

mock 或测试环境：

```python
from src.core.model_registry import ModelRegistration, ModelRegistry, ModelState
from src.inference import MockBackend
from src.inference.service_bridge import create_service_adapter

adapter = create_service_adapter(
    "configs/models/qwen3_vl_2b_mlx_4bit.json",
    backend=MockBackend(),
)
registry = ModelRegistry()
registry.register(
    ModelRegistration(
        model_id="qwen3-vl-2b-instruct-4bit",
        adapter=adapter,
        state=ModelState.ACTIVE,
        source="local-model-config",
        quantization="4-bit",
    )
)
registry.set_runtime_status(
    requested_mode="qwen3-vl-mlx",
    selected_mode="mock",
    degraded=False,
)
```

真实 MLX 环境必须先复制配置、填写已存在的 `local_model_path`，然后：

```python
adapter = create_service_adapter("configs/models/qwen3_vl_2b_mlx_4bit.local.json")
adapter.load()
assert adapter.ready
```

只有 `load()` 成功后才能把真实适配器登记为 `ACTIVE`。构造函数不会导入 MLX 或加载权重；默认配置的 `allow_download=false` 会阻止缺少本地路径时的隐式下载。

## 超时与取消边界

桥接使用 `asyncio.to_thread`，因此不会阻塞事件循环，且等待方能及时收到取消或 504。但是 Python 取消不能中断已经在线程中执行的同步 MLX 算子；该算子会继续到当前生成结束。M5/16GB 环境保持 registry 并发数为 1，下一阶段应评估可终止子进程或 MLX 原生取消能力。

## 本机缓存检查

第五阶段已将 `mlx-community/Qwen3-VL-2B-Instruct-4bit` 的固定 revision `9c4f5209e57b31f4b9dfba735de3fb983739c9cc` 下载到 Hugging Face 缓存。快照包含 `config.json` 和 `model.safetensors`，诊断状态为 complete/ready；权重 SHA-256 见第五阶段报告。

## 测试证据

- `tests/model`：53/53 通过，无跳过；
- `tests/api/test_analyze.py`：28/28 通过；
- registry/readiness 相关测试：3/3 通过；
- 全量相关 API 文件：36/36 通过；执行期间曾捕获 `/version` 的并行契约更新窗口，后端补齐 `runtime` 后复测已恢复全绿；
- evaluator 全量回归：27/27 通过，包含模型结果 JSONL 契约；
- `PYTHONNOUSERSITE=1` 的标准 Python 环境中 mock smoke 成功，证明基础模型包不依赖 MLX、Pydantic 或 FastAPI。
