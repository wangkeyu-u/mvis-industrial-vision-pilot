# 第 8.1 阶段冻结评测协议

协议 ID：`ksdd_entity_grouped_nested_cv_v2`

状态：`internal_pilot_validation`，不是 external holdout，不是 production acceptance。

## 修订原因

v1 虽然保证了外层实体隔离，但候选后处理配置与 fusion mode 由原始 validation 的全局排名预先筛选。新的外层 Group K-Fold 覆盖全部 383 张图，所以原 validation 的7个实体会进入各外层测试折，导致候选集存在间接选择污染。

v2 不读取任何全局 validation/test 排名工件。候选空间由 `src/evaluation/phase8_1_protocol.py` 中的代码常量冻结并生成 SHA-256 指纹。

## 外层与内层

- 外层：48个 physical entity 的5折 Group K-Fold，每张图恰好生成一次折外预测。
- PatchCore：每个外层折内运行完整4折内层，重新拟合 memory bank；在汇集内层折外结果上选择图像阈值、fusion 与后处理。
- U-Net：外层训练池再做一次实体分组 holdout；fit 实体只用于梯度更新，inner holdout 只用于早停、图像阈值、fusion 和后处理选择。
- 外层测试实体不得进入 fit、early stop、阈值、fusion 或后处理选择。运行时与自动测试都会对该约束 fail closed。

## 静态候选空间

- fusion：`max / mean / weighted`，必须在每个外层折的内层数据上重新选择。
- U-Net：18个静态配置，覆盖绝对阈值、连通域数、最小面积、dilation 和细长缺陷合并。
- PatchCore：18个静态配置，覆盖分位数阈值、多/全连通域、面积、opening 和细长缺陷合并。

候选空间的数量受笔记本资源门禁约束，但它的定义不依赖任何样本指标。

## 置信区间

主区间使用 entity-cluster bootstrap：每次以48个实体为单位有放回抽样，被抽中的实体携带其全部图像。这保留同一换向器多视图之间的相关性。

旧的逐图 bootstrap 仅作为 `naive_image_bootstrap` 对照，不再是主置信区间。

## 永久限制

- KSDD 是单域、48实体、50张正例的小样本数据。
- v2 减少了内部选择偏差，但不会将内部 CV 变成外部 holdout。
- U-Net 内层仍是单 holdout，不是完整嵌套4折。
- 项目路线比较历史上看过原 test 结果，溯源字段必须保留 `test_labels_used_for_route_comparison=true`。
- `external_holdout=false`、`production_ready=false`。
