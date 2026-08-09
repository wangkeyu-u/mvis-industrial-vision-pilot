# 第八阶段定位失败案例报告（PatchCore 逐样本分析）

执行日期：2026-08-09
输入：第七阶段冻结 PatchCore checkpoint（SHA-256 `1ad6587c…a681`），重推理 56 张 test 的 float32 热力图（`artifacts/model/phase8/patchcore_reinfer/`，与第七阶段逐分一致，max abs diff = 0.0）。
约束：本报告为只读诊断，未使用任何 test 标签调参。

## 总体结论

第七阶段冻结配置（图像阈值 13.0969、bbox 阈值 0.8、只保留最大连通域）在 9 个缺陷正例上 **Acc@IoU 0.5 = 0/9**，逐样本失败如下：

| 样本 | 异常分 | 分类 | IoU | 失败类别 | 真值框面积 | 预测框面积 |
|---|---:|---|---:|---|---:|---:|
| ksdd_kos10_part3 | 15.03 | TP | 0.117 | position_offset | 23814 | 17820 |
| ksdd_kos15_part3 | 16.71 | TP | 0.269 | low_overlap | 29184 | 10074 |
| ksdd_kos16_part5 | 16.17 | TP | 0.281 | low_overlap | 18873 | 17346 |
| ksdd_kos27_part0 | 13.76 | TP | 0.194 | low_overlap | 14384 | 9581 |
| ksdd_kos38_part0 | 11.84 | FN | 0.000 | classification_miss | 28466 | — |
| ksdd_kos38_part1 | 11.97 | FN | 0.000 | classification_miss | 13761 | — |
| ksdd_kos39_part6 | 11.90 | FN | 0.000 | classification_miss | 22960 | — |
| ksdd_kos39_part7 | 12.06 | FN | 0.000 | classification_miss | 13806 | — |
| ksdd_kos43_part1 | 14.39 | TP | 0.350 | low_overlap | 32296 | 47470 |

困难负例假阳性 2/47（与第七阶段一致）。

## 失败机理（可视化证据）

全部 9 张正例的 overlay 见 `artifacts/model/phase8/failure_analysis/overlays/`（左：原图 + 绿真值/红预测；右：256² 热力图 + 同框）。两类根因：

1. **热点≠缺陷全长（5 个分类 TP 全部如此）**。KSDD 缺陷是横跨数百像素的细长沙划痕；整图压到 256×256 后垂直方向损失约 5 倍分辨率，划痕在热力图上退化为单个亮点，且亮点通常只落在划痕的一端/最强响应处。预测框因此只盖住缺陷的一小段：IoU 上限被压到 0.1–0.35，永远达不到 0.5。
2. **分类漏检（4 个 FN）**。kos38/kos39 两个实体的 4 张正例异常分 11.84–12.06，略低于 validation 阈值 13.10。这属于图像级分类失败，后处理无法补救。

失败类别计数：`classification_miss` 4、`low_overlap` 4、`position_offset` 1。`heatmap_resolution_loss` 标志位未触发，因为该标志按"真值框映射到热力图后高度 < 3px"判定，而 KSDD 真值框是整条划痕的外接框（高度方向不窄）；真正的分辨率损失体现在**划痕宽度方向**（划痕宽 5–15px → 热力图上 < 1–3px，响应被周围正常纹理平均掉，只剩最强点）。

## 后处理天花板实验（validation-only 选参）

对 256² 整图热力图跑了 1152 组后处理配置（阈值模式 × 多连通域 × 形态学 × 最小面积 × 细长合并），validation 最优 `quantile0.95 + open3 + 单连通域`：

- validation Acc@IoU 0.125（1/8）→ test 0.0（0/9），pixel Dice 0.017。

**结论：在 256² 整图热力图上，任何后处理组合都无法修复定位。** 这是第八阶段转向 tile 化与监督分割的直接证据。

## 修复后对照

tile 版 PatchCore（500×500 三 tile + mean 融合）+ 同款后处理网格：validation Acc@IoU 0.375 → test 0.222（2/9）。
监督 U-Net（ResNet-18 编码器，BCE+Dice）：validation Acc@IoU 0.875 → test **0.667（6/9）**，pixel Dice 0.537，预测掩码完整刻画划痕全长（对照 overlay 见 `artifacts/model/phase8/unet_test_overlays/`）。U-Net 还把 PatchCore 的 4 个分类漏检全部救回（9/9 分类为正）。

U-Net test 逐样本（命中 = IoU≥0.5）：

| 样本 | IoU | 命中 | 备注 |
|---|---:|:---:|---|
| ksdd_kos10_part3 | 0.762 | ✓ | 掩码略宽于真值（fp 3931） |
| ksdd_kos15_part3 | 0.935 | ✓ | 近乎完美 |
| ksdd_kos16_part5 | 0.845 | ✓ | |
| ksdd_kos27_part0 | 0.302 | ✗ | 掩码只覆盖缺陷一段，预测碎成 2 框（fn 1901） |
| ksdd_kos38_part0 | 0.051 | ✗ | 只盖住缺陷边缘一小段（fn 4740） |
| ksdd_kos38_part1 | 0.533 | ✓ | 边界命中 |
| ksdd_kos39_part6 | 0.067 | ✗ | 只盖住缺陷一端（fn 1314） |
| ksdd_kos39_part7 | 0.596 | ✓ | |
| ksdd_kos43_part1 | 0.848 | ✓ | |

剩余 3 个未命中的统一模式：**掩码只覆盖大缺陷的一部分（预测碎裂/偏小）**，不再是分类漏检。代价也要诚实记录：U-Net 在 47 个困难负例上有 9 张产生预测框（localization_negative_rate 0.19），box precision 0.25 因此被拉低；图像级困难负例 FPR 维持 0.0426（2/47）。
