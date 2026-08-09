# 第六阶段零样本 / LoRA Pilot 对比报告

执行日期：2026-08-09  
目标设备：Apple Silicon / 16 GB unified memory  
当前结论：产品侧的双包导入、身份核对、五项指标/置信区间、失败切片、延迟/内存与质量门禁已经可运行。真实零样本包已在浏览器导入并通过组件 SHA-256；LoRA smoke 因 16GB 设备内存约束失败，没有生成 adapter 或评测包，因此公平对比保持阻断，未制作也未宣称 LoRA 成绩截图。

## 结论先行

- `runtime_ready` 与 `quality_accepted` 分离：运行时是否可服务不代表模型质量通过；离线 Mock 下显示 `RUNTIME_BLOCKED`，导入真实离线评测包也不会改变该状态。
- 任何 pilot 包都永久显示 `PILOT ONLY · 56 TEST SAMPLES · 禁止包装为正式 KPI`，并固定 `quality_accepted=false`、`formal_kpi_claim_allowed=false`。
- 零样本与 LoRA 只有在数据 manifest SHA-256、完整 sample ID 集合、prompt 身份及基础模型 revision 全部相同，且 LoRA adapter SHA-256 存在时，才展示指标对比。
- 当前唯一真实包的质量判断为 `pilot_failed`。它可作为实验诊断证据，不能作为正式 KPI 或模型验收通过证据。

## 已导入的真实零样本证据

包目录：`artifacts/model/phase6/zero_shot/evaluation_package`  
运行证据：`artifacts/model/phase6/zero_shot/run_manifest.json`

该包是来自真实 Qwen3-VL 推理的 `ksdd-0.1.0` 零样本 pilot，可验证真实导入、指标与资源展示；但它不是 `sft_fair_cli` 应输出的 `ksdd_sft-1.0.0` 公平配对零样本包。后者必须和 LoRA 包共同以 `sft_manifest.json` 为 data manifest。若直接把当前包与未来 SFT fair LoRA 包配对，manifest SHA 不同会被 UI 正确阻断。因此“真实零样本已导入”不等于“正式 fair pair 的 A 侧已到达”。

浏览器一次选择 `package_manifest.json` 及其 6 个 JSON 组件，并附加 `run_manifest.json`。页面完成组件哈希校验，显示 `evaluation-package · N=56 · pilot_failed · SHA verified`。主分析工作区保持为空，证明离线评测导入没有伪造在线分析结果。

| 身份/边界 | 实测值 |
|---|---|
| 数据集 | `ksdd-0.1.0`，frozen test |
| 数据 manifest SHA-256 | `fe9983728d7aa830f11a1369ff1a4eb353be8e8403685a1cf45ea23db128085a` |
| sample IDs | 56 个唯一 ID；由 `report.samples` 提取并排序 |
| prompt SHA-256 | `cba1c2914ce8d7e2eff0abb2ad188fe1103f34919846b49fa9a6b191754d28fc` |
| 基础模型 revision | `9c4f5209e57b31f4b9dfba735de3fb983739c9cc` |
| run / experiment | `ksdd_v0_qwen3_vl_2b_zero_shot_prompt_v1` |
| adapter | 禁用；hash 为 `null`，符合零样本约束 |
| 样本门槛 | 56；正式 KPI 最低 300，故 `formal_kpi_eligible=false` |
| 完成情况 | 55 completed / 1 unavailable；不可用样本仍保留在固定 56 ID 总体中 |

### 五项 Pilot 指标

| 指标 | 点估计 | 95% bootstrap CI | 观察数 | Pilot 判断 |
|---|---:|---:|---:|---|
| Macro-F1 | 0.0000 | [0.0000, 0.0000] | 56 | 未达到 0.82 |
| Acc@IoU 0.5 | 0.0000 | [0.0000, 0.0000] | 9 | 未达到 0.70 |
| JSON schema validity | 0.9821 | [0.9464, 1.0000] | 56 | 未达到 0.99 |
| Hard-negative FPR | 0.0000 | [0.0000, 0.0000] | 47 | 达到不高于 0.10 |
| Evidence–conclusion consistency | 0.9821 | [0.9464, 1.0000] | 56 | 点估计达到 0.97 |

以上数字是 56 张冻结测试集上的真实 pilot 结果。页面与报告都禁止把它们表述成正式 KPI；`pilot_failed` 由绝对 pilot 目标计算，不会因个别指标通过而升级为 `quality_accepted`。

### 失败与资源证据

- 失败案例 56/56；`CLASSIFICATION_ERROR` 56，`LOCALIZATION_MISS` 9，`INVALID_JSON` 1，`EVIDENCE_CONCLUSION_INCONSISTENT` 1。
- 页面展示按失败率排序的切片，并保留 sample ID、truth、prediction 与 failure code，避免平均数隐藏失败模式。
- 推理延迟：P50 `1882.164 ms`，P95 `2307.581 ms`，范围 `1676.247–2425.169 ms`。
- 内存：process peak RSS `706.781 MB`；MLX allocator peak `2763.019 MB`。两者口径不同，页面分卡显示，不相加，也不解释为系统统一内存压力。

## 公平对比门禁

双包都通过 SHA-256 后仍须逐项满足：

1. `pilot_only=true` 且不是 fixture/mock；
2. `data_provenance.manifest_sha256` 完全一致；
3. `report.samples[*].sample_id` 非空、唯一，排序后的集合完全一致；
4. prompt SHA/fingerprint/content/version 身份一致；
5. 基础模型 revision 一致；
6. 零样本包没有 adapter hash；LoRA 包必须有 adapter SHA-256。

任一失败都设置 `COMPARISON_BLOCKED`，隐藏身份、指标、资源和切片对比 DOM，且清空可被上层验收导出的 portfolio。解析器反例已经覆盖 `DATA_MANIFEST_MISMATCH` 与 `SAMPLE_ID_MISMATCH`；不会用两个不公平总体计算 delta。

即使全部匹配，输出也只可能是 `pilot_failed` 或 `pilot_candidate`：

- `pilot_failed`：LoRA 任一绝对 pilot 指标未达到目标；
- `pilot_candidate`：五项点估计达到 pilot 目标，仅表示值得继续扩大样本验证；
- `quality_accepted`：对 56 样本 pilot 永远为 `false`；
- 正式 KPI：永远锁定，必须改用满足正式样本门槛的独立评测。

## 当前阻塞与禁止项

截至本报告生成时，工作区没有 LoRA `evaluation_package`、adapter hash 或公平配对 `fair_report.json`。最新受控训练证据为 `artifacts/model/phase6/qlora_smoke_16_retry1/run_manifest.json`：

| 训练字段 | 实测值 |
|---|---|
| run kind / steps | `smoke` / 16 |
| status | `resource_limit_exceeded` |
| base revision | `9c4f5209e57b31f4b9dfba735de3fb983739c9cc` |
| train / validation | 271 / 56 |
| trainable parameters | 8.716288M（0.410%） |
| wall time | 324.516 s |
| process peak RSS | 1869.938 MB |
| MLX allocator peak | 15997.066 MB |
| configured MLX budget | 12288 MB |
| runtime error | Metal command buffer insufficient memory |
| adapter / adapter hash | 未生成 / `null` |

训练入口已对 `LazyRow` 数据转换问题做过修复，本次进入实际训练并完成初始验证，但在后续 step 超出内存预算。该失败来自算法线程的真实 `run_manifest`，不由 UI 推断。由于目标设备是 16GB，不能通过放宽为接近全机内存的假预算来标记成功；后续需要算法侧降低序列长度/累积策略/可训练模块内存，或明确改用更大内存设备并重新定义运行目标。

因此：

- 未完成真实双包浏览器导入；
- 未生成 Phase 6 对比截图；
- 未宣称 LoRA 优于零样本；
- 未把现有 56 张结果包装为正式 KPI；
- 未用复制、修改或合成包替代算法线程的真实输出；失败训练目录也没有被当作 LoRA package。

算法侧生成 LoRA 包后，按下节步骤复验并把本节状态改为“已完成”；在此之前发布判断为“UI/门禁可用，真实公平对比阻塞”。

## LoRA 包到达后的浏览器复验步骤

1. 启动 `python3 app/server.py`，访问 `http://127.0.0.1:8000/`；若要同时验证运行 readiness，改用既有真实 API 启动方式和 `?client=api&acceptance=real`。
2. 打开“评测证据”，在 A 槽一次导入零样本包的全部 JSON；在 B 槽一次导入 LoRA 包的全部 JSON。
3. 确认状态栏分别显示 `RUNTIME_READY` 或 `RUNTIME_BLOCKED`、`PILOT_FAILED` 或 `PILOT_CANDIDATE`、以及永久的 `FORMAL KPI LOCKED`。
4. 核对 manifest、56 个 exact sample ID、prompt、两个基础 revision、LoRA adapter hash 与两个 run ID。
5. 核对五项点估计、两侧 CI、delta、P50/P95、RSS/MLX peak 和两侧失败切片。
6. 只在 `PAIR VERIFIED` 时记录浏览器证据；若出现任何 mismatch，记录阻断码而不是指标截图。

## 已执行的产品侧证据

| 检查 | 当前结果 |
|---|---:|
| 服务、同源代理与启动集成 | 15/15 通过 |
| JavaScript UI/client 测试 | 24/24 通过 |
| Python UI 合同测试 | 8/8 通过 |
| 真实 FastAPI 网络 E2E | 5/5 通过 |
| 模块语法与 diff whitespace | 通过 |
| 真实零样本包浏览器导入 | 通过，SHA verified，N=56，pilot_failed |
| 390×844 响应式语义检查 | 通过 |
| 真实零样本 × LoRA 浏览器对比 | 阻塞：LoRA smoke 内存超限，包缺失 |

本轮产品/网络专项合计 **52/52 通过**；5 个 JavaScript 模块语法检查、fixture JSON 解析和 diff whitespace 检查均通过。真实双包完成状态仍必须以算法产物实际出现为前提。

仓库级 `uv run --python 3.13 pytest -q` 另有 **214 passed、12 subtests passed**；仅出现 Starlette `TestClient` 关于未来 `httpx2` 的弃用提示，不影响本轮结果。
