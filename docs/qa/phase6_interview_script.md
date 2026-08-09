# 第六阶段 8–10 分钟面试演示脚本

目标：展示一个可审计的真实模型实验工作台，而不是只展示“漂亮分数”。全程把运行可用性、pilot 质量判断和正式 KPI 资格分开陈述。

## 演示前准备

- 离线保底：`python3 app/server.py`，打开 `http://127.0.0.1:8000/`。
- 真实运行联调：沿用 Phase 5 的 FastAPI + 同源代理启动方式，打开 `?client=api&acceptance=real`。
- 准备零样本与 LoRA 两个完整 evaluator 包。每个选择框一次选中 `package_manifest.json` 和清单内全部 JSON；若性能/运行身份单独提供，再同时选中对应 `run_manifest.json` 或 `performance.json`。
- 演示前先核对 LoRA 包确实存在。若不存在，走文末“诚实阻塞路径”，不要使用 fixture 代替。

## 0:00–0:50｜产品定位与主流程

操作：打开“视证台”，快速指向上传、查询、证据框、结果/解释/置信度、request_id 和 JSON 导出。

讲述：

> 这是面向视觉合规审查的本地演示系统，目标设备是 16GB Apple Silicon。主分析流程处理单次图片审查；评测证据是独立的审计附件，不会因为导入离线成绩而伪造在线结果。

## 0:50–1:40｜运行 readiness 与质量状态分离

操作：打开“评测证据”，指向顶部诊断和 Phase 6 三个状态标识。

讲述：

> `runtime_ready` 只回答模型服务能否运行；`quality_accepted` 回答质量证据是否达到验收。真实、非降级 readiness 才能显示运行 ready，但它不会自动把 pilot 变成正式成绩。最右侧的 `FORMAL KPI LOCKED` 对 56 张测试集始终成立。

若当前是 Mock，明确说：

> 现在运行态是 `RUNTIME_BLOCKED`，因为这是离线 Mock 页面；下面导入的仍可是真实离线评测包，两类证据互不洗白。

## 1:40–2:50｜导入真实零样本包

操作：点击“A / ZERO-SHOT”，一次选中完整零样本 JSON 包；等待显示 `SHA verified`、`N=56` 和 `pilot_failed`。

讲述：

> 这里先验证 package manifest 对每个 JSON 组件的 SHA-256，再读取固定总体。这个真实零样本 pilot 的 Macro-F1 和 Acc@IoU 都是 0，JSON 有效率和证据一致率约 98.2%。我保留这个负结果，因为作品集的价值在可复现和不掩盖失败。

补充资源证据：

> 零样本实测 P50 约 1.88 秒、P95 约 2.31 秒；process RSS 峰值约 707MB，MLX allocator 峰值约 2.76GB，两种口径分开显示。

## 2:50–4:10｜导入 LoRA 包并解释公平性

操作：点击“B / LORA”，导入真实 LoRA 包。

讲述：

> 系统不会看到两个包就直接算差值。它要求相同 data manifest、完全相同的 56 个 sample ID、相同 prompt、相同基础 revision；零样本必须没有 adapter，LoRA 必须给出 adapter SHA-256。这样变化才可以归因到 adapter，而不是数据或提示词漂移。

屏幕应出现 `PAIR VERIFIED`。若没有，停在阻断码并说明，不继续讲指标。

## 4:10–5:35｜五项指标与置信区间

操作：逐行指向 Macro-F1、Acc@IoU、JSON validity、hard-negative FPR、evidence–conclusion consistency。

讲述：

> 每行同时展示零样本和 LoRA 点估计、各自 95% bootstrap 区间与 delta。方向改善只是实验信号；最终状态仍按 LoRA 的绝对 pilot 目标判断。`pilot_candidate` 表示值得扩大验证，不等于质量验收；任一目标未达到就是 `pilot_failed`。

避免说“显著提升”，除非正式 paired report 明确给出足够分母和结论强度；56 样本默认只说“pilot 观察”。

## 5:35–6:45｜失败切片而非只看平均数

操作：滚动到左右两列 failure slices，选择 2–3 个最高失败率切片。

讲述：

> 平均指标会隐藏工业表面细缺陷和困难负例。这里并排展示两侧切片的 N、失败率和失败码；失败案例保留 sample ID、truth 和 prediction，可以直接定位 LoRA 改善了什么、又退化了什么。

## 6:45–7:35｜延迟与内存代价

操作：指向 ZERO/LORA 的 P50、P95、RSS peak、MLX peak 卡片。

讲述：

> 质量提升需要一起看本地设备代价。这里不把 MLX allocator 和进程 RSS 相加，也不把一次运行峰值包装成系统级统一内存压力。候选即使质量更好，只要延迟或内存不适合 16GB 设备，也还不能进入发布。

## 7:35–8:30｜现场演示 fail-closed 门禁

操作：如有算法提供的真实不匹配包，导入并展示 `DATA_MANIFEST_MISMATCH` 或 `SAMPLE_ID_MISMATCH`；没有则只解释已自动化验证的反例，不现场修改产物。

讲述：

> 数据 manifest 或 sample ID 任何一个不同，页面会清空对比内容并阻止 delta。它不会让观众误读一组不公平的数字，也不会保留上一次成功对比的陈旧结果。

## 8:30–9:20｜收束与工程判断

讲述：

> 这个阶段交付的不只是两个分数，而是完整证据链：哈希验签、运行身份、公平总体、五项指标和 CI、失败切片、延迟/内存，以及明确的质量状态。56 张数据只能形成 `pilot_failed` 或 `pilot_candidate`，正式 KPI 必须换成满足样本门槛的独立评测。

最后给出当前真实结论，例如：

> 当前零样本真实 pilot 是 `pilot_failed`；LoRA 为【读取屏幕状态】。无论候选结果如何，`quality_accepted=false`，下一步是扩大独立测试集并完成正式验收。

## LoRA 包缺失时的诚实阻塞路径（约 6 分钟）

如果演示当天 LoRA 包仍未生成：

1. 正常展示 runtime/quality 分离和真实零样本包导入；
2. 展示零样本真实指标、CI、失败切片、P50/P95 与内存；
3. 指向 B 槽和 `PAIR_INCOMPLETE`，明确“没有 adapter hash 和 LoRA evaluator package”；
4. 打开 `docs/qa/phase6_baseline_lora_report.md` 的阻塞章节；本轮可展示 `resource_limit_exceeded`、MLX peak 15997.066MB、12GB budget 和 adapter hash `null`；
5. 结束语使用：

> 产品和质量门禁已经就绪，但 LoRA smoke 在 16GB 目标设备上触发内存超限，没有生成 adapter，所以我没有生成对比截图，也没有声称微调提升。这是一个有 run manifest 和资源峰值的可追踪阻塞，不是用 Mock 填平的空白。

该路径可以证明系统诚实失败关闭，但不能标记“真实零样本 × LoRA 对比完成”。
