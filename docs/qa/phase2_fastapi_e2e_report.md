# 第二阶段 FastAPI 联调与端到端报告

执行日期：2026-08-08  
结论：UI 已通过同源代理实际调用当前 FastAPI `/v1/analyze`，multipart 与 base64 JSON 均通过；当前运行时降级到已登记的 `mock-vlm-0`，因此网络/API 证据有效，但不能作为真实 Qwen3-VL 效果证据。

## 自动化结果

| 层级 | 覆盖 | 结果 |
|---|---|---:|
| app/server 集成 | 离线 Mock、代理、CORS、JSON/multipart、request_id、超时、不可达、错误透传 | 10/10 |
| UI 契约 | 控件、状态、框、导出、API 传输和取消入口 | 5/5 |
| JS 客户端 | Mock、multipart/JSON、取消、超时、网络错误、拒答映射、响应校验 | 11/11 |
| uvicorn/FastAPI 网络 E2E | 成功、不确定、拒答、非法图片、MODEL_NOT_READY、双传输 | 5/5 |
| 构建检查 | `node --check` 两个模块、Python 服务导入 | 通过 |

功能与端到端测试合计 31/31 通过。

## 真实浏览器联调证据

实际启动：FastAPI `127.0.0.1:19101`，UI/代理 `127.0.0.1:19100`。

| 场景 | 观察结果 |
|---|---|
| multipart | 页头 `FastAPI · multipart`；模型 `mock-vlm-0`；适配器 `mock-compliance-v0`；50%；框 `[187.5,95,562.5,285]` |
| request_id | UI 生成 `req_ui_*`，FastAPI 日志、响应头、响应体和页面一致 |
| JSON/base64 | `?transport=json` 成功；TRACE 显示 `fastapi-proxy` |
| 后端停止 | 同一页面再次提交返回 `MODEL_NOT_READY`；页面保留对应 `req_ui_*` |
| 陈旧结果 | 后端停止后证据数为 0，导出 href 移除，复制禁用，Canvas 清空 |

FastAPI `/health/ready` 同源代理结果显示：服务 ready，但 `runtime.selected_mode=mock`、`degraded=true`、`fallback_reason=mlx_runtime_unavailable`；active 是 `mock-compliance-v0`，4-bit Qwen3-VL 是未就绪 candidate。

## 接口行为

- multipart 字段：`image`、`query`、`task`、`model`、`use_specialist`、JSON 字符串 `options`。
- JSON 字段：base64 data URL `image`、原图尺寸、查询、任务、模型、协同开关和 options。
- 请求头：客户端生成安全 `X-Request-ID`；代理原样转发；FastAPI 响应头与结构化错误体返回同一 ID。
- CORS：默认走同源代理；仅对 localhost/127.0.0.1/::1 的显式 Origin 回显，不使用 `*`。
- 取消：客户端合并调用方 AbortSignal 与超时控制器；用户取消映射 `AbortError`，超时映射 `INFERENCE_TIMEOUT`。
- 后端不可达：代理返回 503 `MODEL_NOT_READY`；FastAPI 自身 4xx/5xx 错误体和 request_id 原样透传。
- 拒答：API Schema 以 `result=uncertain` 加 `warnings=["refusal:<code>"]` 表达时，UI 策略态显示“已拒答”，JSON 保留原始 result。

## 已知限制

1. 当前浏览器联调的 active 模型是后端测试替身；真实 Qwen3-VL candidate 因 MLX runtime 不可用而未就绪。
2. 同源代理在浏览器取消后会关闭下游连接，但 Python 标准库无法保证已进入模型执行的上游工作立即停止；真正的服务端取消传播仍需 ASGI 任务级验证。
3. 代理采用本地开发用 `ThreadingHTTPServer`，不是生产反向代理；发布部署应使用受支持的 ASGI/反向代理方案。
4. 客户端 timeout 包含请求发送与等待响应；代理另有独立 timeout，二者应满足客户端略大于后端推理阈值的配置关系。

