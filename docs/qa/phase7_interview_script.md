# 第七阶段 8–10 分钟面试演示脚本

主线：不把“能跑”、“指标改善”和“模型验收通过”混为一件事。演示一个在 M5 / 16GB 约束下可运行、可追踪、会说“不够好”的视觉合规审查系统。

## 0:00–0:45 问题与设计约束

话术：

> 这是一个本地视觉合规审查台。我把产品问题拆成三层：VLM 负责语义结论和解释，specialist 负责异常检测与定位，fusion 只在两者一致时放行；冲突时转人工复核。所有真实标记都有 readiness 和证据门禁，Mock 只用于离线演示。

指向顶部 `离线 Mock / 真实模型 / 降级`标识，说明图片不持久化。

## 0:45–2:15 主流程：fused 与 specialist_only

1. 离线入口启动：

   ```bash
   .venv/bin/python app/server.py --host 127.0.0.1 --port 8000
   ```

2. 载入内置演示图，保持 `fused`，点击“开始审查”。
3. 指出原图像素坐标框、置信度、模型、总耗时/分段耗时和 request_id。
4. 讲“模型协作证据链”：
   - specialist 框来源和异常热力区；
   - VLM 语义解释；
   - fusion 状态与冲突；
   - 人工复核原因。
5. 特意指出 `REAL LOCALIZATION BLOCKED`：这个框是 Mock 交互证据，不是模型验收证据。
6. 切换 `specialist_only`再跑一次：没有独立 VLM 解释，界面不会用 fusion 文案填充它。

## 2:15–3:15 失败关闭与证据导出

1. 依次点“不确定”、“拒答”、“错误”。
2. 说明每次新请求会先撤销旧框、旧 JSON URL 和旧 readiness 绑定。错误后导出的是本次错误证据，不是上一次成功结果。
3. 展示“导出 JSON”与“导出验收证据”的区别：前者是 API 结果，后者还包含 readiness、模式、输入元数据和 stale-result 防护。

## 3:15–4:45 真实链路与诚实阻断

1. 打开 `?client=api&acceptance=real`。
2. 顶部显示真实 Qwen VLM：`selected_mode=real`、`degraded=false`、revision `9c4f...c9cc`。
3. 打开评测证据的运行诊断：active VLM ready，LoRA candidate not ready，PatchCore specialist `runtime_blocked / unvalidated`，三个 analysis mode 分开显示。
4. 载入许可 Probe，选 `fused`并提交。界面在请求前以 `MODEL_NOT_READY` 阻断。
5. 关闭 specialist 后跑真实 VLM-only；这个样例返回 `OUTPUT_VALIDATION_FAILED`，分析 request_id 为 `req_ui_1786235524032_c185f0eaf63543ca`。

话术：

> 我没有为了截一张好看的图而把 manifest 真值框、Mock 框或离线预测框冒充在线结果。这里真实的产品能力是：它知道哪一段没准备好，而且不会保留陈旧证据。

## 4:45–7:00 真实评测包：提升和失败一起讲

1. 零样本槽导入 Phase 6 VLM package 的 7 个 JSON 和 `run_manifest.json`。
2. 候选槽导入 Phase 7 PatchCore package 的 7 个 JSON 和 `run_manifest.json`。
3. 指向 `PAIR VERIFIED`：两侧都是 non-mock/non-fixture，KSDD manifest SHA-256 一致，56 个 sample ID 完全一致。
4. 讲五个数：
   - Macro-F1：0.0% → 78.1%；
   - Acc@IoU 0.5：0.0% → 0.0%；
   - JSON 有效率：98.2% → 100.0%；
   - 困难负例 FPR：0.0% → 4.3%，是回退；
   - 证据—结论一致率：98.2% → 100.0%。
5. 讲资源：VLM P50 1882 ms，PatchCore P50 7.3 ms；PatchCore process RSS peak 1915 MB，仍在 16GB 设备预算内。
6. 指向 `PILOT_FAILED / FORMAL KPI LOCKED`：56 张小于正式样本门槛 300；定位没有达标，不能只讲 Macro-F1 提升。
7. 打开失败切片：9 个 pixel-mask 正例定位全部失败；47 个 hard negative 中 2 个误报。

## 7:00–8:15 工程亮点

- **真值分层**：Mock、real runtime、quality accepted、formal KPI 是四个独立状态。
- **定位所有权**：fused 只保留 specialist 框；VLM 框在无 specialist 支持时被丢弃。
- **严格对比**：包 SHA、manifest、sample ID、prompt/revision/adapter 根据候选类型强制检查。
- **可观测与导出**：request_id、revision、耗时、状态和阻断理由可导出，但不嵌入图片字节和查询原文。
- **资源感知**：把 16GB 统一内存当作架构约束，LoRA smoke 超预算就停，不继续制造虚假成绩。

## 8:15–9:15 失败复盘与下一步

1. VLM 零样本在细粒度灰度缺陷上大量不确定/拒答，且有一次真实输出结构校验失败。
2. PatchCore 图片级分类明显改善，但 bbox 阈值/热力图几何校准失败，IoU 没有改善。
3. LoRA smoke 超出 16GB 机器的 12GB 预算，没有生成 adapter 或成绩包。
4. specialist checkpoint 已有，但 FastAPI bridge 还没有就绪；同源代理也还需要放行短期 heatmap URI。
5. 下一步按风险排序：完成 specialist adapter 与 heatmap 代理 → 重做 bbox 校准与失败切片 → 获得签名评测 attestation → 扩大冻结测试集后再谈正式 KPI。

## 9:15–10:00 收尾

> 这个作品集最想表达的不是“我有一个很高的准确率”，而是“我能把数据、模型、服务、界面和验收证据做成同一条不会撒谎的链路”。当定位或 runtime 没有达标时，系统会告诉你究竟是哪一层失败。

# 常见追问

## 为什么 Macro-F1 提高 78.1 pp 还是 pilot_failed？

因为验收是多指标门禁，不是单指标排名。Acc@IoU 仍为 0，hard-negative FPR 回退，而且 N=56 小于正式门槛 300。

## 为什么 fused 不直接合并两边的框？

VLM 的语义能力不等于可验收几何定位。定位框由 specialist 独占；fusion 只决定是否放行，不凭空生成第三套框。

## 为什么 real readiness 是绿色，fused 却被阻止？

顶部 real 标记证明 active VLM runtime 是真实且非降级；`analysis_modes.fused.runtime_ready=false`证明 specialist 分支没就绪。runtime 真实与某个复合模式可用是两个不同问题。

## 如何防止错误后截到上一次成功框？

新请求一开始就清空 result、overlay、JSON URL、evidence presentation 和 readiness binding；错误/取消只会导出当次 outcome。浏览器回归会在每个错误态后检查证据数和框数都为 0。

## 为什么需要包哈希和 sample ID？

相同指标名不代表相同实验。manifest SHA 保证数据版本，sample ID 集合保证总体，prompt/revision/adapter 保证变量边界，package component SHA 保证文件没被篡改。

## M5 / 16GB 的关键取舍是什么？

VLM 4-bit 推理可运行，但微调 smoke 超预算；PatchCore 的延迟很低，但需要更好的定位校准。因此工程上选择 VLM 语义 + 轻量 specialist + 保守 fusion，而不是将所有能力都塞进一个大模型。

## 如果再有两周，最先做什么？

先完成 PatchCore service adapter 和 heatmap 同源代理，因为它们阻断真实在线观测；然后用当前 9 个定位失败样例重做阈值/几何校准，再扩充测试集和签名评测证明。
