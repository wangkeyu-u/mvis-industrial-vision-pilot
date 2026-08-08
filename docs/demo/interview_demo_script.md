# 视觉合规审查：最终 8–10 分钟面试演示脚本

目标：证明真实模型、Mock、评测 fixture 和正式模型成绩是四类不同证据；任何一类都不能冒充另一类。

## 演示前检查（不计时）

1. 按 `docs/qa/phase5_real_e2e_report.md` 启动 real FastAPI 和同源 UI。
2. 打开 `/?client=api&acceptance=real&timeout_ms=120000`。
3. 只有页面显示绿色 `REAL MODEL EVIDENCE` 才继续真实链路；出现红色阻断层时，直接进入“阻塞演示”，不要切到 Mock 冒充。
4. 确认 `data/processed/ksdd_v0/probe_manifest.json` 存在；“载入许可 Probe”会在浏览器端复核许可证、归属和 SHA-256。若按钮不可用，直接展示阻塞报告，不手工绕过。

## 0:00–0:50｜问题、设备与证据原则

> 这是面向视觉合规审查的端侧多模态系统：单张图片加自然语言要求，输出结论、原图像素证据框、解释、不确定性和结构化 JSON。目标设备是 16GB Apple Silicon，候选路线是 Qwen3-VL-2B 4-bit。今天重点不只是“模型能回答”，而是每个结果能否证明来自哪条链路。

指出“图片不持久化”、Schema 和 readiness 状态。

## 0:50–1:45｜真实验收闸门

指向绿色验收条：

> 这个标记不是 URL 参数直接点亮的。页面重新请求 `/health/ready`，严格要求 selected_mode=real、degraded=false、active 模型 ready，且来源不能带 Mock/Test Double。readiness request_id、模型指纹、量化和 revision 都进入证据；后端没有返回 revision 时明确写 not_reported。

如当前未解锁：展示覆盖式 `REAL SCORE CAPTURE BLOCKED` 和导出的阻塞 JSON，然后跳到 7:20。

## 1:45–3:20｜许可图片、请求与 JSON

操作：点击“载入许可 Probe”。页面显示 `ksdd_kos10_part3`、`cc-by-nc-sa-4.0` 和 500×1273；再次点击可在三个缺陷 Probe 间循环。提交一次审查。

> 演示服务只暴露 manifest allowlist。浏览器先校验文件类型、签名和 SHA-256，再以 multipart 经同源代理进入真实 FastAPI。sample_id、license、attribution 和 hash 进入验收证据；图片字节不进入证据 JSON。结果中的 request_id 同时出现在页面、JSON、HTTP 响应和后端日志。

本次实测如实展示：三个缺陷 Probe 分别出现 422 或拒答/不确定，`objects=[]`。点击“导出 JSON”和“导出验收证据”。

> manifest 里有真值框，不代表模型返回了框。产品层绝不把真值框画成预测框，所以当前定位验收是阻塞，而不是通过。

## 3:20–4:25｜拒答、不确定与陈旧结果防护

指出许可 Probe 的 `refused`、`uncertain=true` 和无对象框；然后加载下一 Probe，展示新请求开始时旧结果立即清空。说明独立不确定回归结果：

> 拒答/不确定链路已真实通过；另一个样例暴露结构化输出 422。UI 清空旧框、旧 JSON 和 readiness 绑定，没有用 Mock 填补。失败本身也是可导出的验收证据。

## 4:25–5:10｜模型与 revision 证据边界

> readiness 已记录 active 模型、4-bit、config/model fingerprint，但当前 API 没有返回 revision。只读配置知道固定 revision，不等于本次响应携带 revision，所以导出文件保留 not_reported。这是待后端补齐的契约缺口。

展示证据 JSON 中的 readiness request_id、分析 request_id、latency 和 `model_revision`。

## 5:10–6:35｜实验对比与失败切片

打开“评测证据”，先展示生命周期诊断，再加载 fixture。

> 即使当前是真实模型 readiness，fixture 仍然是红色 `FIXTURE / MOCK · 不可用于模型验收`。界面展示零样本/候选、绝对和配对置信区间、KPI、切片与失败案例，但不会因为数字好看就获得模型验收资格。证据导出明确写 `verified=false`、`model_performance_claim_allowed=false`。

快速指向 5 个 KPI、最高失败率切片和 failure code。

## 6:35–7:20｜数据许可、pilot 边界与定位阻塞

> KSDD V0 已冻结 383 个去重样例，并提供样例级许可、哈希和实体隔离切分。本轮确实使用了登记 Probe。但 test 只有 56 个样例，低于正式 KPI 的 300；更关键的是，三个缺陷 Probe 都没有得到模型框。因此数据准入已通过，模型定位仍阻塞，不能报告 Acc@IoU 或 mAP50。

## 7:20–8:10｜降级链路反证

切到 FastAPI mock-adapter 的 `acceptance=real` 页面或展示阻断截图。

> 同一个前端面对 mock-adapter 会识别 selected_mode 非 real、degraded 和 test-double，覆盖整页阻断水印，导出 screenshot_eligible=false。Mock 仍能用于离线演示，但不能留下看起来像真实成绩的截图。

## 8:10–9:00｜系统工程能力

> 主流程支持 multipart/JSON、request_id、取消、超时、错误体和陈旧结果防护；评测附录支持哈希校验与失败关闭；真实模式把 readiness 和每次结果绑定。它们共同解决的是“模型输出能不能被审计”，而不只是页面能不能显示答案。

## 9:00–10:00｜结论与下一步

> 当前真实 MLX 模型、合法样例上传、拒答/不确定、JSON/证据导出和 fixture 导入都已端到端运行。许可 Probe 拒答约 1.5 秒，但这只是少量观测，不是 P95。最终模型验收还差三件事：修复结构化定位输出并产出真实框、在 API 中绑定 revision、生成非 fixture 的 evaluator 成绩包并达到正式样本门槛。

补充自动化证据：产品/质量范围 52/52 通过，包括 15 个服务/代理/许可路由集成、8 个 UI 契约、24 个客户端/评测/闸门用例和 5 个真实网络 E2E。

收尾：

> 这套系统最重要的能力，是让真实、降级、Mock、fixture、失败和正式成绩在界面与导出证据中始终不可混淆。

## 面试官追问速答

- 为什么选 Qwen3-VL-2B？轻量、多任务与定位能力适合端侧候选，但最终选择必须由零样本、微调、量化和兼容性实验共同决定。
- 为什么还要专用模型？VLM 提供语义与解释，RF-DETR/Florence 提供定位对照或协同；冲突时降置信或拒答，而非无条件融合。
- 如何证明框正确？当前没有真实模型框，所以明确阻塞；UI 坐标映射通过不等于模型定位通过，最终要用冻结集 Acc@IoU 0.5、mAP50 和失败切片证明。
- 如何避免泄漏？测试集训练前冻结并按实体去重；失败案例进入隔离复核池，不能直接回流训练。
- Mock 有什么价值？允许接口、状态、可用性和错误流程先被验收；它不提供任何模型效果证据。
