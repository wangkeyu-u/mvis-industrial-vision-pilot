# 本地 API 部署手册（M5 MacBook Air 16GB）

本里程提供可运行的 FastAPI 服务边界，默认接入确定性 `MockModelAdapter`，用于验证文件安全、Schema、超时、错误码、注册表和可观测性。Mock 结果不是算法结果，不得用于业务判断。服务不下载模型，也不保存上传图片。

## 环境与启动

建议 Python 3.11–3.13。在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "fastapi>=0.115,<1" "pydantic>=2.10,<3" \
  "uvicorn[standard]>=0.34,<1" "python-multipart>=0.0.20,<1" \
  "Pillow>=11,<13" "PyYAML>=6,<7"
MVIS_MODEL_MODE=auto PYTHONPATH=. \
  python -m src.api serve --host 127.0.0.1 --port 8001
```

统一入口会在启动 Uvicorn 前执行必需的 preflight；required 检查失败时退出码为 2，不会误启动一个看似可用的服务。`auto` 的显式 Mock 降级允许启动，但 `production_ready=false`。

## Preflight 与机器报告

只运行启动检查：

```bash
MVIS_MODEL_MODE=auto PYTHONPATH=. python -m src.api preflight
```

检查项包括：版本化 service/model 配置存在性、实现与配置的 API Schema 版本、active 模型 readiness、降级状态、工作区磁盘余量、12GB 进程内存预算和设备物理内存。输出是单个 JSON 文档，`can_serve` 表示是否可以启动，`production_ready` 只有非降级真实模型才可能为 true。

执行 preflight 加 100 次顺序 ASGI 稳定性采样，并原子写入 JSON：

```bash
MVIS_MODEL_MODE=auto MVIS_CODE_VERSION=stage-4 PYTHONPATH=. \
  python -m src.api validate --requests 100 \
  --output docs/deployment/stage4_validation_report.json
```

报告包含状态码分布、成功率、唯一 request_id 数、墙钟/API 延迟摘要、采样前后进程历史峰值内存及预算。所有采样都带 `kpi_eligible`：

- Mock 必须为 `false`，原因是 `mock_latency_is_not_real_model_kpi`；
- 进程内真实模型冒烟也不是同机正式基准，原因是 `in_process_smoke_sample_is_not_hardware_benchmark`。

因此该报告只能证明服务稳定性、Schema 和调用链，不替代 M5 上 30 次正式热/冷启动性能测试及 100 次真实模型无 OOM 验收。

## ModelOps 生命周期

注册项遵循受门禁的 `candidate → validated → active → retired` 生命周期。候选项只有在 adapter `ready=true` 且配置指纹、模型指纹均存在时才能进入 `validated`；激活只接受 validated 项，并用 `expected_active_model_id`、`expected_active_fingerprint` 做比较并交换。激活会原子设置新 `active`，把原 active 降为 validated 并保存为 `previous`。回滚会原子恢复 ready 的 previous，并把被回滚版本转为 retired。

不依赖 MLX 或真实权重的完整演示：

```bash
PYTHONPATH=. python -m src.api lifecycle-demo
```

ModelOps 写 API 默认关闭。令牌只能通过环境变量注入，至少 16 字符，不允许写入 YAML，也不会进入日志或状态响应：

```bash
export MVIS_MODELOPS_TOKEN='replace-with-a-secret-from-your-secret-store'
```

所有写操作还要求安全的 `X-ModelOps-Actor`，操作原因、request_id、前后状态、active/previous、配置指纹和模型指纹进入内存审计链及结构化日志。示例：

```bash
TOKEN="$MVIS_MODELOPS_TOKEN"
ACTOR='release-operator'

curl -s -X POST http://127.0.0.1:8001/v1/models/model-v2/validate \
  -H "X-ModelOps-Token: $TOKEN" \
  -H "X-ModelOps-Actor: $ACTOR" \
  -H 'Content-Type: application/json' \
  -d '{"reason":"offline validation suite passed"}'

curl -s -X POST http://127.0.0.1:8001/v1/models/model-v2/activate \
  -H "X-ModelOps-Token: $TOKEN" \
  -H "X-ModelOps-Actor: $ACTOR" \
  -H 'Content-Type: application/json' \
  -d '{
    "reason":"controlled rollout",
    "expected_active_model_id":"model-v1",
    "expected_active_fingerprint":"<64-lowercase-hex>"
  }'

curl -s -X POST http://127.0.0.1:8001/v1/models/rollback \
  -H "X-ModelOps-Token: $TOKEN" \
  -H "X-ModelOps-Actor: $ACTOR" \
  -H 'Content-Type: application/json' \
  -d '{"reason":"rollback after validation alarm","expected_active_model_id":"model-v2"}'

curl -s http://127.0.0.1:8001/v1/models/audit?limit=100 \
  -H "X-ModelOps-Token: $TOKEN" \
  -H "X-ModelOps-Actor: $ACTOR"
```

错误门禁为：未配置令牌 `MODELOPS_DISABLED`/503、认证失败 `MODELOPS_UNAUTHORIZED`/401、状态或 CAS 冲突 `MODELOPS_CONFLICT`/409。管理令牌头不在开发 CORS allowlist 中，浏览器跨域 UI 不能调用 ModelOps 写接口。

开发测试额外需要：

```bash
python -m pip install "pytest>=8,<10" "httpx>=0.28,<1"
PYTHONPATH=. pytest -q tests/api
```

快速验证：

```bash
curl -s http://127.0.0.1:8001/health/live
curl -s http://127.0.0.1:8001/health/ready
curl -s http://127.0.0.1:8001/version
curl -s http://127.0.0.1:8001/v1/models
curl -s -X POST http://127.0.0.1:8001/v1/analyze \
  -F 'image=@sample.jpg;type=image/jpeg' \
  -F 'query=找出不符合要求的区域' \
  -F 'task=inspect' \
  -F 'model=active' \
  -F 'use_specialist=false' \
  -F 'options={"temperature":0,"seed":42}'
```

同一路径也接受 `application/json`，供浏览器客户端发送 data URL。`image_width` 和 `image_height` 仅作为客户端提示，服务端坐标始终以安全解码得到的真实宽高为准：

```json
{
  "image": "data:image/png;base64,iVBORw0KGgo...",
  "image_width": 1280,
  "image_height": 720,
  "query": "找出不符合要求的区域",
  "task": "inspect",
  "model": "active",
  "use_specialist": false,
  "options": {"temperature": 0, "seed": 42}
}
```

JSON 请求体采用流式大小上限；base64 在解码前后分别检查长度，解码后继续走与 multipart 相同的签名、完整解码、单帧、尺寸和像素校验。OpenAPI `/docs` 同时公开两种 transport。

## 模型运行模式与降级

`MVIS_MODEL_MODE` 支持三种明确模式：

- `mock`：只注册 `mock-vlm-0`，用于 API、并发和稳定性验证；
- `real`：加载模型配置并通过 `AsyncServiceAdapterBridge` 预加载 MLX；运行时或权重不可用时 `/health/ready` 返回 503，绝不切换 Mock；
- `auto`：优先加载真实桥；失败后注册不可用的真实 candidate，并将明确标识为 Mock 的 adapter 设为 active。

`auto` 降级时，`/health/ready`、`/version` 和 `/v1/models` 都包含类似以下状态，API 响应中的模型仍显示 `mock-vlm-0`，不会伪装成真实 Qwen：

```json
{
  "requested_mode": "auto",
  "selected_mode": "mock",
  "degraded": true,
  "fallback_reason": "mlx_runtime_unavailable"
}
```

常用环境变量：

```bash
export MVIS_MODEL_MODE=auto
export MVIS_MODEL_CONFIG=configs/models/qwen3_vl_2b_mlx_4bit.json
export MVIS_CODE_VERSION="$(git rev-parse --short HEAD)"
export MVIS_CORS_ORIGINS=http://127.0.0.1:8000,http://localhost:8000
```

服务本身不会打开模型下载。当前登记配置要求本地 MLX checkpoint；缺少 `mlx`/`mlx-vlm`、本地权重、适配器或加载条件时只返回安全原因码，不通过健康 API 暴露本地绝对路径或底层异常正文。

## 前端开发 URL 与 CORS

先在 8000 端口启动现有 UI，再在 8001 端口启动本服务。前端可直接访问：

```text
http://127.0.0.1:8000/?client=api&endpoint=http%3A%2F%2F127.0.0.1%3A8001%2Fv1%2Fanalyze
```

默认 CORS 只允许配置文件中列出的本地开发 origin，允许 `GET/POST/OPTIONS`，暴露 `X-Request-ID`，不启用 credentials，也不接受通配符 origin。生产部署推荐同源反向代理 `/v1`、`/health` 和 `/version`；同源请求不需要 CORS。

## 配置与 M5 资源边界

默认配置位于 `configs/service/default.yaml`。通过 `MVIS_SERVICE_CONFIG` 指向另一个受版本控制的 YAML；通过 `MVIS_CODE_VERSION` 注入 Git commit 或构建版本。

针对 16GB 统一内存的默认值：

- 单图最大 10MiB、25MP、单边 8192px；
- 推理并发为 1，避免多份大张量同时常驻；
- 推理超时 8 秒，容量等待 100ms；
- 每 50ms 检查客户端断连；可取消的 Mock/异步 adapter 会立即收到取消；
- 图片全程内存处理，不使用用户文件名构造路径。
- JPEG/PNG/WebP 容器必须精确结束，拒绝追加脚本、ZIP 等 polyglot 尾载荷。

实际 4-bit 适配器接入后，应在同一 M5 设备上重新执行 30 次热身/非热身延迟测试和 100 请求稳定性测试，不能用 Mock 证据替代 KPI-06–08。

## 适配器接口

真实算法只需在其它算法目录实现 `src.core.model_registry.ModelAdapter`：

- `identity`：返回基础模型和适配器版本；
- `ready`：反映权重、适配器和必要资源状态；
- `analyze(AdapterRequest)`：异步返回统一 `ModelOutput`或可由其校验的 mapping。

适配器必须响应 `asyncio` 取消，不得吞掉 `CancelledError`。如果底层推理库是不可中断的阻塞调用，API 超时只能停止等待，不能保证立即释放该底层计算；部署前必须用真实适配器验证取消语义。

## 错误、健康与日志

`/health/live` 只表示进程存活；`/health/ready` 要求 active 注册项和适配器均 ready；`/version` 返回服务、代码、API、Schema 和当前模型版本。`/v1/models` 暴露别名、生命周期、来源和权重哈希登记状态，不返回本地绝对路径。

所有请求都返回 `X-Request-ID`，错误体中返回同一 ID。JSON 日志包含 SRS 规定的模型、任务、图像形状、延迟分解、进程峰值内存、状态和错误码。日志不包含图片、base64、文件名或完整查询正文。

响应以 `latency` 返回预处理、推理、校验分解，并同时提供内容相同的 `timing` 兼容字段；`latency_ms` 是端到端总耗时。错误体在顶层和 `error.request_id` 均提供同一个追踪 ID，以兼容现有 HTTP 客户端。

## 已知风险

- 当前 active 注册项是 mock，健康仅证明服务链路可用，不证明算法就绪。
- ModelOps 审计链当前驻留进程内，重启后不会保留；生产部署需把同一结构化日志送入不可变外部日志存储。
- `auto` 降级后的 ready=200 表示 active Mock 链路可服务；必须同时检查 `runtime.degraded`。`real` 模式加载失败时 ready=503。
- `AsyncServiceAdapterBridge` 使用 `asyncio.to_thread`；HTTP 任务可以停止等待，但已经进入底层 MLX 的同步生成不保证被抢占。严格资源隔离需要进程级 worker/终止机制。
- Pillow 校验拒绝动图/多帧文件；这符合 MVP “单图”边界。
- 进程 `ru_maxrss` 是历史峰值，不是单请求增量；正式性能报告需配合 M5 系统采样工具。
- 依赖锁文件不在本工作目录权限内；发布前必须由根目录负责人将上述范围固化为已审计的 lockfile。
