# 第三阶段一键演示与 readiness 验收报告

执行日期：2026-08-08  
结论：一键启动、readiness 预检、运行时真值标识、验收证据导出和真实 FastAPI mock-adapter 浏览器回归已打通。

## 启动证据

实际执行：

```bash
python3 app/run_demo.py --backend-port 19201 --ui-port 19200 --startup-timeout 120
```

启动器自动使用 `uv` 的 Python 3.13 隔离环境，FastAPI `/health/ready` 返回 200 后才启动 UI。记录的真值为：

```text
mode=mock-adapter degraded=True
```

这证明真实 uvicorn/FastAPI/代理链路可用，不是模型效果证据。

## 自动化与构建结果

| 层级 | 覆盖 | 结果 |
|---|---|---:|
| Python 集成 | 本地服务、代理、CORS、超时、不可达、一键启动命令/readiness | 14/14 |
| UI 契约 | 控件、五种结果、四种运行真值、验收证据 | 6/6 |
| JS 客户端 | 图片、双传输、request_id、取消/超时、错误体、readiness | 14/14 |
| uvicorn/FastAPI 网络 E2E | 成功、不确定、拒答、非法图片、服务未就绪、双传输 | 5/5 |
| 构建检查 | 两个 JS 模块语法、两个启动器 `--help`、一键 `--dry-run` | 通过 |

功能与端到端自动化合计 **39/39 通过**。隔离 `uv` 命令需使用 `PYTHONPATH="$PWD"`，启动与测试文档已固化该前置条件。

## 浏览器级回归

| 场景 | 页面状态 | 证据数 | 结果导出 | 验收证据 |
|---|---|---:|---|---|
| multipart 违规 | `violation` | 1，82% | 有 | `readiness=degraded`，request_id 匹配 |
| 合规 | `compliant` | 0 | 有 | `objects=[]` |
| 不确定 | `uncertain` | 0 | 有 | `objects=[]` |
| 拒答 | `refused` | 0 | 有 | 原始 `result=uncertain`，策略态 `refused` |
| JSON 违规 | `violation` | 1 | 有 | `transport=json`，`trace=fastapi-proxy` |
| 用户取消 | `error` | 0 | 无 | `REQUEST_CANCELLED`，`result=null` |
| 300 ms 客户端超时 | `error` | 0 | 无 | `INFERENCE_TIMEOUT`，request_id 匹配 |
| 成功后停止 FastAPI | `error` | 0 | 无 | `MODEL_NOT_READY`，`readiness=unavailable` |
| 离线页 | 页头 `mock` | — | — | 明确显示“离线 Mock” |

陈旧结果专项：在已有违规框和结果 JSON 的页面停止 FastAPI，再次提交后证据卡数从 1 变为 0，结果导出 href 移除，验收证据记录 `displayed_evidence_count=0`、`result_export_available=false`、`overlay_object_count=0`。

## 验收证据格式

`mvis_demo_acceptance_evidence` 包含：

- client mode、endpoint、transport 和 timeout；
- readiness 分类、runtime 和 active model；
- 文件尺寸、MIME、字节数、任务和模型别名；
- 成功/错误/取消 outcome，request_id 和当前 result；
- 陈旧结果防护状态。

出于隐私，证据不嵌入图片字节或查询原文。

## 已知限制

1. 默认演示 adapter 是确定性测试替身；不得将 82% 或其耗时纳入算法/性能报告。
2. 本地代理是开发演示用 `ThreadingHTTPServer`，不是生产反向代理。
3. 浏览器取消会立即清理 UI 并关闭下游请求；已经进入标准库代理的上游模型工作不保证立即停止，生产部署需使用支持取消传播的 ASGI 代理。
4. “真实模型就绪”的 UI 分类已自动化测试，但本轮未加载真实 Qwen3-VL 权重，因此没有真实模型浏览器效果证据。
