# 第四阶段评测作品集 QA 报告

执行日期：2026-08-08  
范围：产品演示和质量专属目录。未修改 evaluator、训练或数据实现。

## 实现结论

页面已可导入 evaluator `report.json`、`metrics.json` 或完整 package JSON，展示基线/候选对比、置信区间、KPI、切片和失败案例。fixture/mock 水印由 package 多处真值字段合并判定，任一来源表明 fixture/mock 时都不允许进入模型验收资格。

## 自动化证据

| 测试层 | 命令/范围 | 结果 |
|---|---|---|
| 服务与同源代理集成 | `unittest tests/integration` | 14/14 通过 |
| UI 静态契约 | `unittest tests/ui` | 7/7 通过 |
| 客户端与 evaluator 导入 | `node --test tests/ui/*.test.mjs` | 19/19 通过 |
| 真实 FastAPI 网络 E2E | `fastapi_proxy_e2e_test.py` | 5/5 通过 |
| 构建检查 | 4 个 JS `node --check`、fixture `json.tool`、启动器 dry-run/help | 全部通过 |

功能与端到端自动化合计 **45/45 通过**。真实 FastAPI E2E 由隔离 `uv` 环境启动当前后端并覆盖成功、不确定、拒答、非法图片、未就绪、超时和两种传输；其中一个测试可同时覆盖多个状态。

## 浏览器回归证据

实际启动了 `app.demo_backend:app` 的 uvicorn/FastAPI 与同源 UI：

| 检查 | 结果 |
|---|---|
| 主页复杂度 | 评测面板默认 hidden，主分析控件未改变 |
| 运行诊断 | 显示 `mock-adapter`、`active`、READY、别名、模型/adapter/source/量化和明确降级原因 |
| readiness 刷新 | request_id 发生变化，状态与生命周期重新渲染 |
| fixture 水印 | 红色黏附水印“FIXTURE / MOCK · 不可用于模型验收”持续可见 |
| 实验对比 | 5 行零样本/候选/变化，同时显示绝对 CI、配对改善 CI 和分母 |
| KPI | 5 个 KPI 均为 `fixture_only`，验收资格为 false，原因可见 |
| 切片 | 3 个切片按失败率展示，包含 N、失败数、F1 和 IoU |
| 失败案例 | 2 个案例，sample_id、truth→prediction 和 4 类 failure code 可读 |
| 键盘 | `Escape` 关闭面板，`aria-expanded=false`，焦点回到“评测证据” |
| 桌面响应式 | 当前浏览器视口无页面横向溢出；表格自身局部可滚动 |
| 主流程回归 | 关闭面板后违规分析仍成功：1 个框、结果 JSON 和验收证据正常 |

## 安全与失败关闭

- 单文件限制 5 MB，一次限制 12 个 JSON；
- 所有数值必须为有限数；五项比率与绝对区间边界必须在 0–1，配对变化与改善区间必须在 -1–1；
- report Schema 仅接受 `1.0.0`；
- metrics/report 总体指标和 comparison candidate 不一致时拒绝展示；
- `fair_comparison` 不为 true 时拒绝配对比较；
- package JSON 哈希缺失或不匹配时拒绝；
- 导入异常先清空旧评测 DOM，再显示稳定错误码。

## 证据限制

1. 内置 fixture 是评测管线契约演示，其 88.9% Macro-F1 等数值不得引用为模型成绩。
2. package 哈希验证只证明导入的 JSON 与 manifest 一致，不证明数据标注或试验设计无偏。
3. 切片差异是描述性证据；当前 evaluator 明确不做多重比较校正。
4. 真实模型的 KPI 和候选对比仍需正式冻结集 package；UI 不生成或修改评测数值。
