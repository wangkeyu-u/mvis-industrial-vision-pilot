# 第三阶段：一键演示启动

## 最稳定的面试演示

在项目根目录执行：

```bash
python3 app/run_demo.py --open
```

启动器会依次：

1. 启动真实 uvicorn/FastAPI 服务；
2. 轮询 `/health/ready`，未就绪时不启动 UI；
3. 启动同源反向代理 UI；
4. 打开 `http://127.0.0.1:8000/?client=api`。

首次运行如果当前 Python 依赖不完整，启动器会使用 `uv` 建立隔离的 Python 3.13 临时运行环境。按 `Ctrl+C` 同时停止由它启动的 UI 和 FastAPI 子进程。

## 证据真值

默认的 `app.demo_backend:app` 是“真实 FastAPI 服务层 + 确定性模型替身”。它覆盖图片校验、multipart/JSON、Schema、request_id、取消、超时和错误体，但不能用于证明模型精度。页头必须显示黄色：

```text
FastAPI 就绪 · 模型替身
```

运行时标识的含义：

| 标识 | 含义 | 可以声称的证据 |
|---|---|---|
| 离线 Mock | 浏览器内置客户端 | 仅 UI/交互/导出 |
| FastAPI 就绪 · 模型替身 | 真实 HTTP/FastAPI，但模型降级 | 服务契约和端到端链路 |
| 真实模型就绪 | readiness 显示非降级真实模型 | 还需冻结集指标才能证明效果 |
| FastAPI 不可用 | 网络不可达或模型未就绪 | 错误闭环证据 |

## 与当前真实模型配置联调

```bash
python3 app/run_demo.py \
  --backend-app src.api.app:app \
  --open
```

实际是真实模型还是回退模式，以页头和 `/health/ready` 的 `details.runtime` 为准，不以启动命令推断。

## 演示快速路径

1. 点击“载入内置演示图”；
2. 运行“违规”，展示证据框、置信度、模型、耗时和 request_id；
3. 依次运行“合规 / 不确定 / 拒答”，确认都不保留旧框；
4. 运行“错误”后点“取消请求”，确认返回 `REQUEST_CANCELLED`；
5. 点“导出验收证据”。证据文件包含客户端配置、readiness、输入元数据、结果/错误和陈旧结果防护状态；不嵌入图片字节或查询原文。

JSON 传输页：`http://127.0.0.1:8000/?client=api&transport=json`。

## 常用参数

```bash
python3 app/run_demo.py --help
python3 app/run_demo.py --dry-run
python3 app/run_demo.py --backend-port 9011 --ui-port 9000 --open
```

