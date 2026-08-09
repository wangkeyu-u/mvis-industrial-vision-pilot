# 第八阶段算法报告：缺陷定位修复与实体分组评测

执行日期：2026-08-09
设备：Apple Silicon / 16 GB 统一内存（单进程内存门禁 12 GB）
目标：解决第七阶段"图片级分类有效、定位 Acc@IoU = 0"的核心缺陷。
结论先行：**定位从 0/9 修复到 6/9（test）与 0.745（实体分组 CV 汇集），监督 U-Net 取代 PatchCore 成为新 pilot_candidate；production_ready 保持 false。**

> 修订提示：本文第5节保留的是 v1 历史数字。第 8.1 阶段发现候选配置的间接选择污染；当前主结果以 [第 8.1 阶段方法学修订报告](phase8_1_methodology_report.md) 为准。

## 1. 复现与失败机理（步骤一）

第七阶段 PatchCore checkpoint 重推理 56 张 test，异常分与第七阶段**逐分一致**（max abs diff = 0.0，工件 `artifacts/model/phase8/patchcore_reinfer/`）。9 个正例逐样本分析（证据：`artifacts/model/phase8/failure_analysis/`）：

- 5 个分类 TP 全部 `low_overlap`/`position_offset`（IoU 0.117–0.350）：热力图只高亮划痕的一端/最强响应点，预测框无法覆盖缺陷全长。
- 4 个 `classification_miss`（kos38/kos39，分数 11.84–12.06 < 阈值 13.10）。
- 根因：整图 500×1265 压到 256×256，垂直方向约 5 倍分辨率损失，细划痕退化为单点 blob。

详细报告：[docs/qa/phase8_localization_failure_report.md](../qa/phase8_localization_failure_report.md)。

## 2. 后处理实验（步骤二，全程 validation-only 选参）

实验框架 `src/evaluation/localization_postprocess.py` + `phase8_experiments.py`；每组实验保存完整配置、seed（20260809）、数据/模型指纹、全套指标与逐样本预测（`artifacts/model/phase8/postprocess_experiments/`、`tiled_postprocess_experiments/`）。

| 路线 | 候选数 | validation Acc@IoU | test Acc@IoU | test Pixel Dice | test Box F1 |
|---|---:|---:|---:|---:|---:|
| 整图 256² 热力图 + 后处理网格 | 1152 | 0.125 | 0.000 | 0.017 | 0.000 |
| tile PatchCore（500² 三 tile，mean 融合）+ 网格 | 5184 | 0.375 | 0.222 | 0.009 | 0.006 |
| 监督 U-Net + 网格 | 5184 | 0.875 | **0.667** | **0.537** | **0.364** |

后处理维度均按要求覆盖：绝对/分位数像素阈值、多连通域（k=1/3/全部）、opening/closing/dilation 组合、最小连通域面积、细长缺陷纵横比放宽与共线合并、tile→原图纯平移无损回映射、重叠区 max/mean/加权融合。**负面结论同样重要：整图 256² 热力图上任何后处理都无法修复定位**——分辨率损失是物理性的，只能换路线。

## 3. 监督分割基线（步骤三）

`src/training/unet_segmentation.py`：U-Net，timm ResNet-18 编码器（复用第七阶段钉住缓存的同一权重文件，SHA-256 `80c49d…c612`，**零新增下载**），14.3M 参数。训练用现有 256² tile（train 324 / 正例 53），真实像素 mask，BCE+Dice 损失（各 1.0），正例加权采样，AdamW 1e-4 wd，batch 8，MPS fp32。

- 17 epoch 早停（patience 12），validation tile Dice 0.8396，训练 2 分钟。
- 推理：256² 密集网格（stride 128）+ weighted 融合回原图；P50 147 ms/图，进程峰值 RSS 1517 MB，MPS 峰值 1187 MB —— 远低于 12 GB 门禁。
- 配置：`configs/models/ksdd_unet_resnet18_phase8.json`；运行工件 `artifacts/model/phase8/unet_resnet18/`。

## 4. PatchCore 与 U-Net 公平对比（同一 test 56 图，同一评测器）

| 指标 | PatchCore（第七阶段） | tile PatchCore（第八阶段） | U-Net（第八阶段） |
|---|---:|---:|---:|
| 图像 Macro-F1 | 0.78125 | 0.78125 | **0.93913** |
| 图像 AUROC | — | — | 1.0 |
| Acc@IoU 0.5 | 0.000（0/9） | 0.222（2/9） | **0.667（6/9）** |
| Pixel Dice | — | 0.009 | **0.537** |
| Pixel IoU | — | 0.005 | **0.367** |
| Box P/R/F1 | — | 0.003/0.222/0.006 | 0.250/0.667/**0.364** |
| 困难负例 FPR | 0.0426 | 0.0426 | 0.0426 |
| P50 / P95 延迟 | 7.3 / 15.1 ms | 28.7 / 63.1 ms | 147.3 / 154.6 ms |
| 进程峰值 RSS | 1915 MB | 3485 MB | 1517 MB |
| 角色 | 少样本/无监督异常检测 | 同左 + tile 化 | 监督像素级分割 |

代价如实记录：U-Net 更慢（147 ms vs 7.3 ms，仍远低于任何交互预算）；其负例出框率 0.19（9/47）拉低 box precision 至 0.25；图像级困难负例 FPR 三条路线同为 2/47。

## 5. 实体分组交叉验证（步骤四，修正评测协议）

协议：[docs/evaluation/phase8_entity_grouped_protocol.md](../evaluation/phase8_entity_grouped_protocol.md)（`ksdd_entity_grouped_cv_v1`，48 实体 Group K-Fold，实体零跨折）。所有数字为 **internal pilot validation**，不是 external holdout，不是 production acceptance。

**汇集折外预测（383 图，每图恰被预测一次）+ bootstrap 95% CI：**

| 模型 | Acc@IoU 0.5 | 图像 Macro-F1 | Pixel Dice | Box F1 |
|---|---|---|---|---|
| PatchCore tiled（nested 5×4） | 0.039 [0.000, 0.096] | 0.762 [0.690, 0.823] | 0.006 [0.004, 0.008] | 0.001 [0.000, 0.002] |
| U-Net（5 折 + 实体内层 holdout） | **0.745 [0.611, 0.870]** | **0.972 [0.945, 0.994]** | **0.453 [0.336, 0.572]** | **0.400 [0.303, 0.497]** |

U-Net 每折 Acc@IoU：0.800 / 0.700 / 0.800 / 0.900 / 0.545（折间方差如实公开）；PatchCore 每折：0 / 0 / 0 / 0 / 0.182。这组 v1 CV 提供了 U-Net 跨实体改善的内部信号，但第 8.1 阶段独立审计发现：候选后处理列表与 fusion 曾由原 validation 全局结果预筛，而这些实体后来会进入外层折；因此 v1 不应表述为“完全消除选择偏差”。修订数字以 `ksdd_entity_grouped_nested_cv_v2` 报告为准。

## 6. 系统集成（步骤五）

- 新 specialist manifest `artifacts/model/phase8/specialist_manifest_unet.json`（checkpoint 双 SHA 校验、数据/编码器身份校验、融合模式与后处理配置冻结、`test_labels_used_for_selection=false` 强制）。
- `create_specialist_service_adapter` 按 `algorithm` 分派：`unet` → 新 `UnetServiceAdapterBridge`；`patchcore` → 原加载路径**逐字节保留**（回滚路径，`tests/model/test_phase8_service_bridge.py` 锁定）。
- 三模式 `vlm_only / specialist_only / fused` 不变；定位框 `source=unet`；冲突时保守拒答逻辑不变；`quality_status=pilot_candidate`、`quality_accepted=false`、`serving_tier=pilot`、`production_ready=false`。
- 真实 API 冒烟：`specialist_only` 对 kos15/Part3.jpg 返回 violation + bbox [240,825,500,945]（与离线 IoU 0.935 样本一致）+ 真实 heatmap artifact，HTTP 200，延迟 338 ms（含冷启动）。

## 7. 成功、失败、局限与下一步

**成功**：定位 0/9 → 6/9（test）、CV 0.745；图像 Macro-F1 0.781 → 0.939；全流程 validation-only 选参、实体隔离、bootstrap CI、逐样本可复查。
**失败**：整图 PatchCore 后处理路线被证伪（保留负面结果）；3/9 正例仍未命中（掩码只覆盖大缺陷一段，碎裂为多个小框）；负例出框率偏高（0.19）。
**局限**：test 已参与选择，其数字只是选择证据；CV 是 internal pilot validation；KSDD 单域 + CC BY-NC-SA 4.0；U-Net 内层为单 holdout 而非完整嵌套；48 实体样本量决定 CI 仍宽。
**下一步**：① 扩充正例实体数（采集或公开数据集并查许可证）后重跑冻结协议冲击 formal KPI（≥300 图门槛）；② 对碎裂掩码做实例级合并/CRF 精化；③ 负例出框率优化（阈值与面积的 precision 侧折中）；④ 若回到无监督路线，tile 化 PatchCore + 更大 memory bank 是已验证可行的起点；⑤ 许可审查完成前保持 non-commercial pilot。

## 8. 复现命令

```bash
.venv/bin/python -m src.inference.patchcore_reinfer
.venv/bin/python -m src.evaluation.phase8_failure_analysis
.venv/bin/python -m src.evaluation.phase8_experiments
.venv/bin/python -m src.inference.patchcore_tiled
.venv/bin/python -m src.evaluation.phase8_tiled_experiments
.venv/bin/python -m src.training.unet_segmentation
.venv/bin/python -m src.evaluation.phase8_tiled_experiments \
  --tiled-dir artifacts/model/phase8/unet_resnet18 \
  --output-dir artifacts/model/phase8/unet_postprocess_experiments
.venv/bin/python -m src.evaluation.phase8_cv_runner
.venv/bin/python -m src.evaluation.phase8_cv_unet
.venv/bin/python -m src.inference.phase8_specialist_manifest
.venv/bin/python -m pytest -q
```
