# 第二阶段：FastAPI 双服务启动说明

> 第三阶段首选一键启动：`python3 app/run_demo.py --open`。详见 `docs/demo/one_click_demo.md`。本文保留手动双进程方式，用于单独排查后端或代理。

本阶段区分三类证据，演示时禁止混淆：

| 页面/服务 | 网络链路 | 模型证据含义 |
|---|---|---|
| `/` | 浏览器内 `MockAnalysisClient` | 仅证明 UI、状态和离线演示可用 |
| `/?client=api` | 浏览器 → 同源代理 → FastAPI，multipart | 证明真实 HTTP/FastAPI 链路；模型以 API 返回的 `model` 和警告为准 |
| `/?client=api&transport=json` | 浏览器 → 同源代理 → FastAPI，base64 JSON | 证明 JSON 传输兼容性；同样不等于真实模型权重已加载 |

当前默认 FastAPI 会根据运行环境选择模型。若 `/health/ready` 显示 `selected_mode=mock` 或结果警告包含 `mock_adapter`，这是真实 FastAPI 服务层加后端测试替身，不是 Qwen3-VL 推理结果。

## 1. 启动 FastAPI

已有完整 Python 环境时：

```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8011
```

仓库尚无统一依赖环境时，可用隔离的临时环境启动：

```bash
PYTHONPATH="$PWD" uv run --no-project \
  --with fastapi --with pydantic --with python-multipart \
  --with pyyaml --with pillow --with uvicorn \
  uvicorn src.api.app:app --host 127.0.0.1 --port 8011
```

## 2. 启动 UI 与同源代理

另开终端：

```bash
python3 app/server.py \
  --host 127.0.0.1 \
  --port 8000 \
  --backend-url http://127.0.0.1:8011 \
  --proxy-timeout 8
```

浏览器访问 `http://127.0.0.1:8000/?client=api`。页头会先显示“检查 FastAPI readiness”，然后按实际状态显示以下之一：

```text
FastAPI 就绪 · 模型替身
真实模型就绪
FastAPI 不可用
```

同源代理解决浏览器 CORS：页面与 `/v1/analyze` 都来自 `127.0.0.1:8000`，代理再把原始请求体、Content-Type 和 `X-Request-ID` 转发给 8011。不要直接把 `endpoint` 指向另一个端口，除非 FastAPI 本身配置了精确 CORS 来源。

## 3. 就绪检查

```bash
curl -i http://127.0.0.1:8000/health/ready
curl -i http://127.0.0.1:8000/version
```

通过标准：响应包含 `X-MVIS-Backend: fastapi-proxy`、`X-Request-ID`，且 `/health/ready` 为 200。读取 `details.runtime` 和模型登记状态，确认当前究竟是 mock、MLX 还是其他运行模式。

## 4. 可选客户端配置

| 参数 | 默认 | 说明 |
|---|---|---|
| `client` | `mock` | `api` 才调用 FastAPI 路径 |
| `transport` | API 模式下 `multipart` | 可设为 `json` |
| `timeout_ms` | `10000` | 客户端超时，范围最高 120000ms |
| `endpoint` | `/v1/analyze` | 推荐保持同源相对路径 |

示例：

```text
http://127.0.0.1:8000/?client=api&transport=json&timeout_ms=8000
```

请求进行中时主按钮变为“取消请求”，再次点击或按 `Esc` 会终止浏览器请求。每次新请求开始前就会清空旧边界框和结果导出；成功、错误和取消都会生成仅对应当前尝试的“验收证据” JSON。

## 5. 端到端测试

无第三方依赖的 UI/代理测试：

```bash
python3 -m unittest discover -s tests/integration -p 'test_*.py' -v
python3 -m unittest discover -s tests/ui -p 'test_*.py' -v
node --test tests/ui/*.test.mjs
```

真实 uvicorn/FastAPI 网络测试：

```bash
PYTHONPATH="$PWD" uv run --no-project \
  --with pytest --with fastapi --with pydantic --with python-multipart \
  --with httpx --with pyyaml --with pillow --with uvicorn \
  pytest -q tests/integration/fastapi_proxy_e2e_test.py
```
