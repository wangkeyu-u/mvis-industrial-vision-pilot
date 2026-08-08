# 算法运行诊断与性能采样

## 诊断 CLI

诊断命令只读取环境、包元数据、配置和本地缓存，不导入 MLX 模型、不连接网络、不加载权重：

```bash
python3 -m src.inference.diagnostics
```

紧凑 JSON 与 readiness 门禁：

```bash
python3 -m src.inference.diagnostics --compact --require-ready
```

退出码：

- `0`：诊断完成；未使用 `--require-ready` 时允许状态为 unavailable；
- `2`：使用了 `--require-ready`，但真实运行条件不完整；
- 其他非零：配置文件损坏或诊断本身失败。

输出包含：

- Python、操作系统、machine、总物理内存；
- MLX、MLX-VLM、Hugging Face Hub、Pydantic、Pillow、FastAPI 的安装状态和版本；
- 模型 ID、固定 revision、配置指纹、下载和 remote-code 策略；
- 量化 bits、group size、mode；
- 固定生成参数与输入/输出资源上限；
- 精确 revision 对应的本地 snapshot 路径、配置文件和 safetensors 状态；
- readiness 原因和结构化性能状态。

2026-08-08 本机结果为 `unavailable`：MLX 与 MLX-VLM 未安装，且固定 revision 的 MLX 4-bit snapshot 不存在。诊断正确返回 exit code 2，性能字段为：

```json
{
  "available": false,
  "measured_runs": 0,
  "p50_latency_ms": null,
  "p95_latency_ms": null,
  "peak_memory_mb": null,
  "samples": []
}
```

这不是测试失败，而是防止把 mock 或不完整缓存的结果冒充真实模型性能。

## 缓存门禁

未配置 `local_model_path` 时，缓存解析只接受：

```text
<HF cache>/models--mlx-community--Qwen3-VL-2B-Instruct-4bit/
  snapshots/9c4f5209e57b31f4b9dfba735de3fb983739c9cc/
```

目录必须同时存在 `config.json` 和至少一个 `*.safetensors`。`allow_download=false` 时，MLX 后端只会使用显式本地目录或上述固定 revision 缓存；两者都不存在就返回 `ModelNotReadyError`。

## 性能采样接口

只有真实 adapter 已加载、真实探针图片已登记时才能采样：

```python
from src.inference import PerformanceSampler

sampler = PerformanceSampler()
summary = sampler.run(
    lambda: adapter.analyze(request),
    warmup_runs=2,
    measured_runs=30,
)
payload = summary.as_dict()
```

输出记录每次 latency 和当时进程峰值 RSS，并汇总 nearest-rank P50/P95、最小/最大延迟和全程峰值内存。`peak_memory_mb` 是进程累计峰值 RSS，不等同于 MLX 张量独占内存或系统总统一内存；正式报告还需配合系统级内存/交换采样。

不具备真实运行条件时必须使用：

```python
from src.inference import unavailable_performance

summary = unavailable_performance("pinned_model_snapshot_unavailable")
```

不得对 mock 调用 `PerformanceSampler` 后把结果放入模型性能报告。
