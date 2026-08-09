# 第 8.1 阶段方法学修订报告

执行日期：2026-08-10

结论：第八阶段的 U-Net 定位提升仍然存在，但 v1 对像素轮廓和 Box F1 的估计偏乐观。修订后的主结果为：U-Net `Acc@IoU=0.7451`，entity-cluster bootstrap 95% CI `[0.6226, 0.8627]`；`Pixel Dice=0.2337`、`Box F1=0.3016`。它仍只是 `internal_pilot_validation`，`production_ready=false`。

## 1. 发现的问题

v1 外层 Group K-Fold 的实体隔离是真实的，383张图也都有唯一折外预测。但两个实现细节会让结果偏乐观：

1. `load_cv_candidates()` 从原 validation 的全局实验结果选取 Top 8 后处理配置和单一 fusion mode。原 validation 的7个实体后来分布在全部5个外层测试折，因此候选集有间接污染。
2. v1 的 bootstrap 以图像为单位，忽略同一 physical entity 的多张图相关性。

## 2. 修订实现

- 协议升级为 `ksdd_entity_grouped_nested_cv_v2`。
- U-Net 与 PatchCore 各自使用18个静态后处理候选；候选由代码常量生成并记录指纹，不读取任何全局指标工件。
- `max / mean / weighted` 三种 fusion 在每个外层折的内层数据上重新选择。
- 图像阈值、后处理、fusion 和 U-Net 早停都不得读取外层实体。运行时断言和自动测试均 fail closed。
- 主置信区间改为48实体的 cluster bootstrap；旧图片级结果仅作 `naive_image_bootstrap` 对照。
- 临时热力图在每折选择完成后删除。PatchCore 报告工件约 444KB，U-Net 保留5个折 checkpoint 后约274MB，符合笔记本可复现目标。

完整协议见 [phase8_1_entity_grouped_protocol.md](../evaluation/phase8_1_entity_grouped_protocol.md)。

## 3. 修订结果

汇集383条唯一折外预测，48实体在外层折间零重叠，五折选择审计中外层实体泄漏数为0。

| 模型 | 指标 | v1 点估计 | v2 点估计 | v2 entity-cluster 95% CI |
|---|---|---:|---:|---:|
| PatchCore tiled | Acc@IoU 0.5 | 0.0392 | **0.0196** | [0.0000, 0.0612] |
| PatchCore tiled | Macro-F1 | 0.7618 | **0.7618** | [0.6978, 0.8245] |
| U-Net | Acc@IoU 0.5 | 0.7451 | **0.7451** | [0.6226, 0.8627] |
| U-Net | Macro-F1 | 0.9724 | **0.9724** | [0.9421, 0.9944] |
| U-Net | Box F1 | 0.4000 | **0.3016** | [0.2426, 0.3772] |
| U-Net | Pixel Dice | 0.4525 | **0.2337** | [0.1414, 0.4197] |

U-Net v2 每折 Acc@IoU：`0.900 / 0.800 / 0.800 / 0.600 / 0.636`；所选 fusion：`max / max / max / weighted / max`；最佳 epoch：`9 / 6 / 6 / 13 / 9`。

PatchCore v2 每折 Acc@IoU：`0 / 0 / 0 / 0 / 0.091`。该结果进一步支持“PatchCore 图像级异常分类有信号，但不适合当前精确定位要求”。

## 4. 如何解读

- `Acc@IoU` 在修订后保持 0.745，说明 U-Net 对“框是否命中缺陷”的改善较稳健。
- Pixel Dice 下降到 0.234，说明像素边界质量跨实体不稳定；不得再用 v1 的 0.452 宣称稳定分割。
- Box F1 下降反映误出框和多框碎片仍是主要问题。
- 历史 test `6/9` 可作为路线选择证据，但不是新 holdout 成绩。

## 5. 溯源与发布状态

修订 specialist manifest 将原含糊字段拆分为：

- `test_labels_used_for_training=false`
- `test_labels_used_for_postprocess_selection=false`
- `test_labels_used_for_route_comparison=true`
- `external_holdout=false`
- `production_ready=false`

这些字段由 bridge 强制验证。路线比较历史被如实保留，但不会阻止 pilot 运行；任何 test 训练/后处理选参、伪造 external holdout 或将该 manifest 标记为 production ready 都会 fail closed。

## 6. 尚未解决

- 没有真正未参与项目决策的外部 holdout。
- KSDD 仅48实体、50张正例，且是单一受控域。
- KSDD 许可证为 CC BY-NC-SA 4.0，当前只能用于非商业 pilot。
- 正式上线仍需新数据、独立验收、签名 evaluator attestation 和许可审查。
