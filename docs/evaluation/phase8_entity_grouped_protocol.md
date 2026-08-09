# 第八阶段冻结评测协议：实体分组交叉验证（internal pilot validation）

> 历史协议：本文记录 v1。第 8.1 阶段审计发现全局 validation 候选预筛的间接污染；当前协议见 [phase8_1_entity_grouped_protocol.md](phase8_1_entity_grouped_protocol.md)。

协议 ID：`ksdd_entity_grouped_cv_v1`
冻结日期：2026-08-09
实现：`src/evaluation/entity_cv.py`（协议原语）、`src/evaluation/phase8_cv_runner.py`（PatchCore 嵌套 CV）、`src/evaluation/phase8_cv_unet.py`（U-Net CV）

## 为什么需要这个协议

第七/八阶段的模型路线选择与后处理选参都使用了 56 张冻结 test 的标签。根据已确立的结论，**该 test 不能再包装成独立无偏验收集**。KSDD 没有第二个真正未使用的外部数据集，因此本协议的产物一律标记为 **`internal_pilot_validation`**：不得称为 external holdout，不得称为 production acceptance，`production_ready` 保持 `false`。

## 分组单元与泄漏约束

1. **分组单元 = physical entity**（`entity_id`，48 个，对应 48 个物理换向器）。同一实体的所有图像（含相同场景不同曝光的近似重复图）必须落在同一折，**任何实体绝不跨训练/验证/测试折**。
2. 精确重复图已在数据版本 `ksdd-0.1.0` 构建期跨目录合并（`source_sample_ids` 记录了被合并的同图不同目录样本），因此不存在跨实体的精确重复；近似重复由实体分组天然覆盖。
3. 折分配确定性：实体按 `sha256(seed:entity_id)` 排序后按正例数贪心均衡到 K 折；同一 seed 重跑结果逐位一致（`tests/evaluation/test_entity_cv.py` 锁定）。

## 外层与内层

- **外层 Group K-Fold，K=5**：每折约 9–10 个实体（~76 张图）作为该折测试集，其余 38–39 个实体为训练池。
- **内层（仅用于选参，绝不碰外层测试折）**：
  - PatchCore：完整 **nested 5×4**。内层 4 折各自重新拟合 memory bank，候选后处理配置在汇集的内层折外热力图上打分，图像阈值用内层折外分数选取。
  - U-Net：受训练成本约束，内层为**单次实体分组 holdout**（训练池的 1/4 实体），用于早停、像素阈值与后处理候选选择。内层划分与外层使用不同 seed 派生。
- 每折流程：拟合/训练（仅训练池实体）→ 内层选参 → 冻结 → 对外层测试折评一次。

## 候选集合

后处理候选不是为 CV 重新发明的：PatchCore 取其 tile 实验 validation 网格中获胜融合模式下 validation 排名前 8 的配置加冠军配置；U-Net 同理（其自身实验的 validation 前 8 + 冠军）。CV 内部只允许从这组冻结候选中按内层指标选择。

## 报告规则

1. **汇集折外预测**（383 张图，每张恰好被其外层折模型预测一次）计算 pooled 指标：Acc@IoU 0.5、图像 Macro-F1、Pixel Dice、Box F1，附 **bootstrap 95% 置信区间**（2000 次重采样，seed 冻结，逐图重采样）。
2. **每折指标全部公开**，不允许只报最好一折；折间方差本身就是结论的一部分。
3. 每折记录：测试实体清单、图像阈值、所选配置指纹、训练 epoch（U-Net）、耗时。
4. 所有产物标记 `internal_pilot_validation`、`external_holdout=false`、`production_acceptance=false`。

## 已知局限

- 48 个实体、50 张正例的总量决定了置信区间仍宽；协议消除的是**选择偏差**，不能凭空增加样本量。
- U-Net 内层是单 holdout 而非完整 4 折嵌套，阈值/配置选择方差略高；已在报告中标注。
- KSDD 单一受控域 + CC BY-NC-SA 4.0；跨域与商业可用性不在本协议覆盖范围。
