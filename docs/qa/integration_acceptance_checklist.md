# 首个里程碑集成验收清单

适用范围：视觉合规审查产品演示层。基线来源为[历史需求规格](https://github.com/wangkeyu-u/mvis-industrial-vision-pilot/blob/c601fc9860dc/AI%E5%A4%9A%E6%A8%A1%E6%80%81%E8%A7%86%E8%A7%89%E7%AE%97%E6%B3%95%E7%B3%BB%E7%BB%9F_%E9%9C%80%E6%B1%82%E8%A7%84%E6%A0%BC%E8%AF%B4%E6%98%8E%E4%B9%A6_v1.0.md)。该规格保留在对应提交中；本清单记录当时的验收范围，不替代当前实现说明或冻结测试集上的算法验收。

## 执行前提

- 设备：M5 MacBook Air / 16GB，或在证据中明确记录替代环境。
- 启动：`python3 app/server.py --host 127.0.0.1 --port 8000`。
- 浏览器 Mock：访问 `http://127.0.0.1:8000/`。
- API 客户端链路：访问 `http://127.0.0.1:8000/?client=api`。
- 使用有授权、无不必要个人信息的测试图片；不要把 Mock 结果用于实际业务判断。

## 自动化准入

| 编号 | 关联需求 | 验收项 | 预期结果 | 自动化证据 |
|---|---|---|---|---|
| A-01 | FR-IN-001/002 | JPG/PNG/WEBP 与文件签名校验 | 合法图片通过；伪装 MIME 被 `INVALID_IMAGE` 拒绝 | `tests/ui/client.test.mjs` |
| A-02 | FR-IN-003 | 中文/英文查询字段透传 | 1–1000 字符进入统一请求；空查询返回 `INVALID_QUERY` | `tests/integration/test_demo_server.py` |
| A-03 | FR-UI-001 | 上传、查询、任务、模型、协同开关 | 控件完整且可操作 | `tests/ui/test_ui_contract.py` |
| A-04 | FR-GRD-001/004 | 原图像素框与叠加映射 | API 返回原图像素框；画布按渲染尺寸缩放 | 两组 UI 测试 |
| A-05 | FR-UI-002 | 结论、解释、模型、耗时、追踪标识 | 字段均来自同一次响应 | 两组 UI 测试 |
| A-05b | FR-UI-002 | 证据置信度可读展示 | Canvas 标签和 DOM 证据卡均显示置信度、框与来源 | UI 契约 + 浏览器实测 |
| A-06 | FR-UI-003 | JSON 复制与导出 | 导出内容为完整响应，文件名使用 `request_id` | `tests/ui/test_ui_contract.py` |
| A-07 | FR-INF-004 | 不确定与拒答 | 无证据时 `objects=[]`，不伪造目标框 | JS 与服务集成测试 |
| A-08 | NFR-REL-002 | 稳定错误码和 request_id | 错误展示码、行动建议与追踪标识 | 服务集成测试 |
| A-09 | 12.2 隐私 | 默认不保存原图 | 响应 `image_persisted=false`，服务无上传落盘路径 | 服务集成测试 + 代码审查 |
| A-10 | 11.3 API 兼容性 | Mock/API 客户端可替换 | `?client=api` 使用同一 UI 和 `/v1/analyze` 契约 | JS 客户端测试 |

自动化命令：

```bash
python3 -m unittest discover -s tests/integration -p 'test_*.py' -v
python3 -m unittest discover -s tests/ui -p 'test_*.py' -v
node --test tests/ui/*.test.mjs
node --check ui/app.mjs
node --check ui/evaluation-panel.mjs
node --check src/client/api-client.mjs
node --check src/client/evaluation-report.mjs
node --check src/client/real-acceptance.mjs
python3 -m json.tool ui/fixtures/evaluation-package.fixture.json >/dev/null
```

## 人工浏览器验收

| 编号 | 操作 | 通过标准 | 状态/证据 |
|---|---|---|---|
| M-01 | 拖放图片、点击上传、键盘 Enter/Space 上传各一次 | 三种入口均可用，文件名/尺寸/大小正确 | 部分通过：浏览器实测点击上传；拖放与纯键盘待测 |
| M-02 | 运行“违规”场景并缩放窗口 | 边界框始终贴合图片；标签含类别、置信度、来源 | 通过：750×380 图片输出 `[420,68,615,274]` 并正确叠加 |
| M-03 | 依次运行合规、不确定、拒答、错误快捷场景 | 状态色、解释、警告与证据框策略一致 | 通过：四类状态均浏览器实测 |
| M-04 | 导出 JSON 并用 JSON 解析器打开 | JSON 可解析，含模型、耗时、警告、trace、request_id | 通过：浏览器生成原生 `.json` 下载链接，链接内容已解析并与页面 request_id/模型/置信度一致 |
| M-05 | 在 1440×900、1024×768、390×844 下检查布局 | 无横向溢出，主要操作和结果无需隐藏菜单即可到达 | 部分通过：1024×768、390×844 无横向溢出；1440×900 待精确复测 |
| M-06 | 仅用键盘完成上传后的查询、模型切换和提交 | 焦点可见、顺序合理、快捷键可用 | 部分通过：`⌘+Enter` 提交实测通过；纯键盘文件选择与模型切换待目标浏览器复测 |
| M-07 | 使用 `?client=api` 重跑违规与超时场景 | 成功和错误均由 HTTP 契约返回，UI 不需改代码 | 部分通过：浏览器实测成功链路；HTTP 超时由集成测试覆盖 |

## 算法/API 接入门槛

真实推理 API 接入前必须确认：

1. `/v1/analyze` 支持 base64 图片，或在 `HttpAnalysisClient` 内切换为 multipart；公共 UI 不感知模型私有格式。
2. `result` 至少稳定提供 `violation | compliant | uncertain | refused`；失败使用需求基线错误码，不以 200 包装错误。
3. `bbox` 为原图 `[x1,y1,x2,y2]` 像素坐标且合法；“未发现”与不确定/拒答不得带正例框。
4. `model`、`latency_ms`、`timing`、`warnings`、`request_id` 可追踪；新增字段允许，现有字段类型不可破坏。
5. UI 明确区分 Mock 与真实 API；关闭 Mock 警告必须与真实后端切换同时发生。
6. 成功响应缺少必需字段、非法框或越界置信度时，客户端必须以 `OUTPUT_VALIDATION_FAILED` 失败关闭，不渲染部分结果。

完整执行证据、发现缺陷与剩余风险见 `docs/qa/milestone_1_test_report.md`。

## 第二阶段 FastAPI 联调准入

| 编号 | 验收项 | 通过标准 | 当前证据 |
|---|---|---|---|
| P2-01 | 同源代理与 CORS | 浏览器不跨域；本地预检精确回显 Origin，不使用通配符 | 代理测试 + 浏览器通过 |
| P2-02 | multipart | 图片和全部表单字段原样进入 FastAPI | 网络 E2E + 浏览器通过 |
| P2-03 | JSON/base64 | FastAPI JSON transport 返回相同 Schema | 网络 E2E + 浏览器通过 |
| P2-04 | request_id | UI、代理、FastAPI 日志、响应头/体一致 | 浏览器与 E2E 通过 |
| P2-05 | 取消/超时 | 用户取消为 AbortError；客户端/代理超时为 INFERENCE_TIMEOUT | JS + 代理测试通过 |
| P2-06 | 不确定/拒答 | 不伪造对象；机器可读 refusal 映射已拒答 | 网络 E2E + JS 通过 |
| P2-07 | 非法图片 | FastAPI 返回 400 INVALID_IMAGE 和 request_id | 网络 E2E 通过 |
| P2-08 | 服务未就绪 | 健康检查或分析返回 503 MODEL_NOT_READY | 网络 E2E + 浏览器停服通过 |
| P2-09 | 陈旧结果 | 任一错误/取消后旧框、证据、复制、导出均清空 | 浏览器停服 + UI 逻辑通过 |

第二阶段详细证据见 `docs/qa/phase2_fastapi_e2e_report.md`；双服务启动见 `docs/demo/fastapi_live_start.md`。

## 第三阶段演示可运行性准入

| 编号 | 验收项 | 通过标准 | 当前证据 |
|---|---|---|---|
| P3-01 | 一键启动 | FastAPI readiness 200 后才启动 UI；`Ctrl+C` 清理子进程 | 实机通过 + `test_run_demo.py` |
| P3-02 | 运行真值标识 | 离线 Mock / FastAPI 降级 / 真实模型 / 不可用四态不混淆 | JS + 浏览器通过 |
| P3-03 | 内置演示样本 | 一次点击完成合法 PNG 加载，仍经过客户端校验 | 浏览器通过 |
| P3-04 | 验收证据导出 | 成功、错误、取消均可导出当前尝试证据；不嵌入图片/查询原文 | 浏览器解析通过 |
| P3-05 | 陈旧结果防护 | 新请求开始即清空旧框；超时/取消/停服后 result 为 null | 三个浏览器场景通过 |

第三阶段详细证据见 `docs/qa/phase3_demo_readiness_report.md`；启动说明见 `docs/demo/one_click_demo.md`。

## 第四阶段算法作品集准入

| 编号 | 验收项 | 通过标准 | 当前证据 |
|---|---|---|---|
| P4-01 | report/package 导入 | 支持 report、metrics 和完整 package JSON | parser 测试 + 浏览器 fixture |
| P4-02 | package 完整性 | manifest 中所有 JSON 组件 SHA-256 通过才标记 verified | 哈希正/反例自动化通过 |
| P4-03 | 评测真值 | fixture/mock/unverified 都不可进入模型验收，水印持续可见 | 单测 + 浏览器通过 |
| P4-04 | 实验对比 | 零样本/候选/变化、绝对 CI、配对改善 CI 和分母可读 | 5/5 指标浏览器通过 |
| P4-05 | KPI 状态 | KPI-01–05 状态、阈值/原因和顶层资格一致 | parser + 浏览器通过 |
| P4-06 | 切片/失败案例 | 切片 N/失败率/指标可读；失败案例不被平均值隐藏 | 浏览器通过 |
| P4-07 | 生命周期/诊断 | 模型状态、ready、source、量化、hash、别名、降级原因可刷新 | 真实 FastAPI readiness 浏览器通过 |
| P4-08 | 可用性 | 默认折叠、Escape 关闭/焦点返还、局部表格滚动、导入错误清空旧 DOM | 契约 + 浏览器通过 |

详细使用见 `docs/demo/evaluation_portfolio_demo.md`，QA 证据见 `docs/qa/phase4_evaluation_portfolio_report.md`。

## 第五阶段真实链路准入

| 编号 | 验收项 | 通过标准 | 当前证据 |
|---|---|---|---|
| P5-01 | 真实标记闸门 | 仅 selected_mode=real、degraded=false、active ready 时显示真实标记 | 单测 + 真实浏览器通过 |
| P5-02 | 降级截图阻断 | mock/degraded/test-double 覆盖醒目水印，导出 screenshot_eligible=false | 浏览器截图 + 导出解析通过 |
| P5-03 | 真实请求绑定 | readiness request_id、分析 request_id、模型/指纹、耗时进入独立证据 | 浏览器导出解析通过 |
| P5-04 | JSON 导出 | 真实结果 JSON request_id/模型/耗时与页面一致 | compliant/refused 通过 |
| P5-05 | 不确定失败关闭 | 超时或结构失败不保留旧框/JSON/真实绑定 | 504/422 浏览器通过 |
| P5-06 | 评测真值隔离 | real readiness 不会移除 fixture/mock 水印 | 浏览器通过 |
| P5-07 | 合法样例与定位 | manifest/许可/SHA 通过后运行真实定位样例，预测框必须来自模型 | 样例准入通过；定位阻塞：3 个缺陷 Probe 为 422 或拒答，无模型框 |
| P5-08 | revision 绑定 | revision 必须来自 readiness 或分析响应 | 阻塞：当前 API 未返回 |
| P5-09 | 正式模型成绩 | 非 fixture evaluator package 哈希、provenance、样本门槛均通过 | 阻塞：仅 fixture；KSDD pilot test=56，低于 300 |

完整真实证据、request_id 和阻塞项见 `docs/qa/phase5_real_e2e_report.md`。

## 第六阶段零样本 / LoRA Pilot 准入

| 编号 | 验收项 | 通过标准 | 当前证据 |
|---|---|---|---|
| P6-01 | 双包真实性 | 两侧均为非 fixture/mock、`pilot_only=true` 且 package JSON 哈希通过 | 零样本通过；LoRA 包缺失 |
| P6-02 | 公平总体 | data manifest SHA-256、唯一 sample ID 集合、prompt、基础 revision 全部相同 | 客户端门禁完成；真实 pair 待 LoRA |
| P6-03 | adapter 身份 | 零样本无 adapter hash；LoRA 必须提供 adapter SHA-256 | 零样本通过；LoRA 待验 |
| P6-04 | 五项指标与 CI | 两侧五项点估计、各自 CI、delta 和方向均可读 | UI 完成；真实 pair 待验 |
| P6-05 | 失败切片 | 两侧切片 N/失败率/失败码并排展示 | UI 完成；真实 pair 待验 |
| P6-06 | 延迟/内存 | 两侧 P50/P95、process RSS peak、MLX allocator peak 分口径显示 | 零样本真实证据通过；LoRA 待验 |
| P6-07 | 运行/质量分离 | `runtime_ready` 不推导 `quality_accepted` | Mock 浏览器实测 + UI 状态逻辑 |
| P6-08 | Pilot 标签 | 仅 `pilot_failed` / `pilot_candidate`；56 样本永久锁定正式 KPI | 客户端门禁 + 真实零样本浏览器通过 |
| P6-09 | 不匹配失败关闭 | manifest/sample ID 任一不一致时隐藏对比并清空可验收 portfolio | 解析器反例通过；真实不匹配包不伪造 |
| P6-10 | 真实浏览器对比 | 实际导入算法零样本与 LoRA 正式 pilot 包并记录证据 | 阻塞：LoRA smoke `resource_limit_exceeded`，adapter/package 未生成 |

完整状态与真实零样本证据见 `docs/qa/phase6_baseline_lora_report.md`。

## 第七阶段集成验收

| 编号 | 验收项 | 通过标准 | 当前证据 |
|---|---|---|---|
| P7-01 | 三模式契约 | 客户端显式发送 `vlm_only/specialist_only/fused` | multipart/JSON 客户端 + 浏览器 Mock 通过 |
| P7-02 | 证据链分层 | specialist 热力图/框来源、VLM 解释、fusion 冲突、复核原因分开 | fused/specialist_only 浏览器通过 |
| P7-03 | 真实定位门禁 | 只有真实 specialist 或一致性门禁通过的 fused 可标记真实 | Mock/VLM/conflict 均 fail closed；真实 runtime 尚阻塞 |
| P7-04 | 运行/质量分离 | VLM、specialist、analysis mode 的 runtime_ready/quality_status 分开 | 真实 readiness 浏览器通过 |
| P7-05 | 真实包配对 | 两包哈希、KSDD manifest、56 个 sample ID 一致 | 真实 VLM/PatchCore package 浏览器通过 |
| P7-06 | 提升与回退 | 五指标、CI、延迟/内存、切片和失败案例同时可读 | Macro-F1 +78.1 pp；IoU 0；FPR 回退，状态 `pilot_failed` |
| P7-07 | Pilot 防误导 | 56 张结果不得展示为正式 KPI | `PILOT ONLY / FORMAL KPI LOCKED` 永久可见 |
| P7-08 | 陈旧结果 | 不确定、拒答、超时、readiness 阻断和 422 后旧框/JSON 为空 | Mock + 真实 API 浏览器通过 |
| P7-09 | 响应式/键盘 | 390×844 单列可读，模式选择、主操作和关闭可键盘达 | 浏览器 + UI 契约通过 |
| P7-10 | 真实在线定位 | 真实 specialist/fused 返回框、heatmap、request_id/revision/耗时 | **阻塞**：specialist/fused `runtime_ready=false`，未伪造截图 |

最终判断见 `docs/qa/phase7_final_acceptance.md`。
