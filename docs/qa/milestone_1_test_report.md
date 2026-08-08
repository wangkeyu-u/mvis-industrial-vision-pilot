# 首个里程碑 QA 测试报告

> 第二阶段真实 FastAPI 联调已完成；最新网络端到端证据见 `docs/qa/phase2_fastapi_e2e_report.md`。本文件保留第一阶段产品层基线结果。

执行日期：2026-08-08  
范围：`app`、`ui`、`src/client`、`tests/integration`、`tests/ui`、`docs/demo`、`docs/qa`  
结论：产品演示层可实际启动并通过自动化、真实 API 契约与核心浏览器验收；真实模型推理、M5 性能与少量人工可用性项目尚未验收。

## 自动化结果

| 套件 | 命令 | 结果 |
|---|---|---:|
| 服务集成 | `python3 -m unittest discover -s tests/integration -p 'test_*.py' -v` | 5/5 通过 |
| UI 静态契约 | `python3 -m unittest discover -s tests/ui -p 'test_*.py' -v` | 4/4 通过 |
| 客户端行为 | `node --test tests/ui/client.test.mjs` | 8/8 通过 |
| JS 构建/语法 | `node --check ui/app.mjs` 与客户端 | 2/2 通过 |
| 产品层合计 | 三套功能测试 | 17/17 通过，0 失败 |
| 真实 API 契约 | 隔离依赖环境运行 3 个 `tests/api/test_analyze.py` 用例 | 3/3 通过，1 个上游弃用警告 |

功能与 API 契约测试合计 20/20 通过。构建/语法检查另外 2/2 通过。真实 API 契约覆盖 JSON/base64 传输、版本化响应、原图像素框和 OpenAPI 双传输声明。

覆盖证据包括：健康与版本端点、静态资源、路径逃逸拒绝、base64 图片签名、原图像素框、四种业务结果、稳定错误码、客户端可替换性、成功响应失败关闭、模型/耗时/警告/trace、显式置信度和 JSON 导出。

## 浏览器实测证据

| 检查项 | 实际结果 |
|---|---|
| 初始渲染 | 上传、查询、模板、任务、模型、协同开关、结果区均可见；修复后无破图 |
| 图片上传 | 点击上传 `assets/system_architecture.png` 成功，识别为 750×380、65.3KB |
| 违规与框 | 状态“需复核”；框 `[420,68,615,274]`；Canvas 与可读证据卡均显示 91% 和 `fusion`；画布随视口重绘 |
| 合规 | 状态“通过”，`objects=[]`，没有遗留证据框 |
| 不确定 | 状态“不确定”，提示证据不足与人工补图，未生成框 |
| 拒答 | 状态“已拒答”，明确高风险/超用途并交人工，未生成框 |
| 错误 | 状态“错误”，展示 `INFERENCE_TIMEOUT` 与行动建议 |
| HTTP 客户端 | `?client=api` 显示 `API · /v1/analyze`；请求成功并展示 `mock-api` 与 request_id |
| 键盘提交 | 上传后在查询框按 `⌘+Enter` 成功运行审查 |
| JSON 导出 | 原生链接文件名为 `<request_id>.json`；data URI 可解析，request_id、模型、结论、0.91 置信度与页面一致 |
| 陈旧结果保护 | 成功后运行超时场景，旧导出链接、复制按钮和证据卡立即撤销 |
| 1024×768 | 双栏布局，无横向溢出：`scrollWidth=1024` |
| 390×844 | 单栏布局，上传入口可见，无横向溢出：`scrollWidth=390` |
| 控制台 | 核心流程未发现 error/warn 日志 |

## 浏览器实测发现并修复

1. **空状态破图**：作者样式覆盖了原生 `hidden` 规则，导致未设置 `src` 的预览图片出现破图图标。已增加统一 `[hidden]` 规则，并在重新加载后复核。
2. **API 模式 fetch 绑定**：原生 `fetch` 作为实例属性调用时丢失宿主上下文，浏览器报 `Illegal invocation`。已在 `HttpAnalysisClient` 构造时绑定 `globalThis`，加入回归断言，并复核真实 HTTP 路径。
3. **置信度仅存在于 Canvas**：视觉上可见但辅助技术与文本验收无法读取。已增加证据卡，显示标签、置信度、原图框和来源。
4. **失败后残留旧导出**：一次成功后若下一次请求失败，旧 JSON 可能仍可导出。现在新图片、错误和清空操作均立即撤销导出与证据。
5. **临时 Blob 下载兼容性**：内嵌浏览器不报告临时 Blob 下载事件。已改为页面内原生 `download` 链接与 JSON data URI，并直接解析链接内容验证完整性。

## 当前接口

客户端统一接口：

```text
analyze({ image, imageWidth, imageHeight, query, task, model, useSpecialist }, { signal })
  -> Promise<AnalysisResponse>
```

- `MockAnalysisClient`：浏览器内确定性场景，用于 UI 并行开发；始终显示 Mock 警告。
- `HttpAnalysisClient`：发送 `/v1/analyze` JSON/base64 请求；结构化映射 HTTP 错误。
- 成功响应会校验 request_id、模型、结果、解释、对象、合法框、0–1 置信度和来源；缺失或非法时映射为 `OUTPUT_VALIDATION_FAILED`。
- 后端只有 `latency` 而无 `latency_ms/timing` 时，客户端会归一化并计算总耗时；因此与当前真实 API Schema 兼容。
- 选择方式：默认 `mock`；URL 加 `?client=api` 切换 HTTP；`?endpoint=...` 可替换端点。
- 当前本地服务的 `/v1/analyze` 仍是契约 Mock，不代表模型已接入。

## 未关闭风险

| 风险 | 影响 | 下一步证据 |
|---|---|---|
| 真实 API 已通过契约测试但未进行双进程实时联调 | 同源部署、反向代理和取消传播仍可能有差异 | 启动真实 FastAPI、配置同源 UI/代理并跑浏览器回归 |
| 框来自 Mock | 只证明坐标映射，不证明定位准确 | 冻结测试集 Acc@IoU 0.5 / mAP50 |
| 386ms 为合成耗时 | 不可用于 KPI-06 | M5 目标机预热后至少 30 次 P50/P95 |
| 内存与稳定性未测试 | KPI-07/08 未知 | M5 16GB 峰值内存与连续 100 请求 |
| 拖放、纯键盘文件选择和目标浏览器“保存文件”落盘尚未完成整轮人工验收 | 可用性仍有小缺口；内嵌浏览器不暴露 data URI 下载事件 | 在目标机 Safari/Chrome 按 M-01/M-04/M-06 复测并保存文件证据 |
| 当前服务只实现部分错误码 | 真实故障覆盖不完整 | API 就绪后覆盖需求基线全部 8 类错误码 |

发布判断：可作为“首个里程碑产品演示层”交付；不得标记为算法 MVP、M5 性能达标或生产就绪。
