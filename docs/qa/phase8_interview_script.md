# 第八阶段 8–10 分钟面试讲稿：定位修复与评测协议修正

主线：先承认上一阶段的失败，再展示一条完整的"失败分析 → 假设 → 实验证伪 → 换路线 → 诚实复验"的算法工程闭环。

## 0:00–1:00 上一阶段留下的问题

> 第七阶段 PatchCore 把图像级 Macro-F1 做到 0.78，但 9 个缺陷正例的定位 Acc@IoU 是 0。第八阶段我只回答一个问题：定位为什么坏、怎么修、修完之后怎么证明不是运气。

## 1:00–3:00 失败分析：先复现，再归类

1. 重推理 56 张 test，分数与第七阶段逐分一致（max diff 0.0）——先证明基线可复现。
2. 打开 `artifacts/model/phase8/failure_analysis/overlays/ksdd_kos10_part3.png`：绿是真值、红是预测。KSDD 的缺陷是几百像素长的细划痕，但整图压到 256×256 后垂直损失约 5 倍分辨率，划痕在热力图上只剩一个亮点，预测框永远只盖住缺陷的一端。IoU 上限 0.35，物理上不可能到 0.5。
3. 逐样本分类：5 个 low_overlap/position_offset + 4 个 classification_miss（分数距阈值只差 1 左右）。

## 3:00–4:30 实验证伪：后处理救不了低分辨率

> 我跑了 1152 组后处理组合——分位数阈值、多连通域、形态学、最小面积、细长先验合并——全部只用 validation 选参。validation 最优也只有 Acc@IoU 0.125，test 回到 0。这个负面结果很关键：它证明问题不在后处理，在热力图分辨率，逼我换路线。

接着展示 tile 化 PatchCore（500×500 三 tile、无损平移回映射、max/mean/加权融合）：定位 0 → 2/9，方向对但不够。

## 4:30–6:00 监督基线与公平对比

1. U-Net：同一个钉住 SHA 的 ResNet-18 编码器（零新增下载）、14.3M 参数、256 tile、真实 mask、BCE+Dice、17 epoch 早停、峰值内存 1.5 GB——笔记本上 2 分钟训完。
2. 对比表（同一 56 图、同一评测器）：Acc@IoU 0 → 0.667；Macro-F1 0.781 → 0.939；pixel Dice 0.537。
3. 打开 `artifacts/model/phase8/unet_test_overlays/ksdd_kos15_part3.png`：预测掩码完整刻画划痕全长，IoU 0.935。
4. 诚实讲代价：P50 从 7.3 ms 涨到 147 ms；负例出框率 0.19 拉低 box precision 到 0.25；3/9 仍未命中（掩码碎裂只盖住大缺陷一段）。

## 6:00–8:00 评测协议修正：数字怎么让人信

> 这个 test 已经参与过路线比较，我不再把它叫"无偏验收"。我还对第一版 CV 做了二次审计：外层实体虽然隔离，但候选配置曾由原 validation 全局排名预筛，仍有间接选择污染。v2 将候选网格写成与数据无关的代码常量，fusion、阈值和后处理都在每个外层折内重新选择。

- U-Net 严格 v2 汇集 Acc@IoU 0.745，实体级 95% CI [0.623, 0.863]；但 Pixel Dice 从 v1 的 0.452 回落到 0.234，说明“框命中”比“像素轮廓”更稳健；
- PatchCore 严格 v2 汇集 Acc@IoU 0.020，95% CI [0, 0.061]——它的分类有信号，精确定位不行；
- 全部标记 `internal_pilot_validation`，`production_ready=false`。

## 8:00–9:00 系统集成与回滚

- 新 specialist manifest 按 `algorithm=unet` 分派到新 bridge；第七阶段 PatchCore manifest 仍可加载——回滚就是换一个环境变量。
- 真实 API 冒烟：specialist_only 返回 violation + 原图坐标框 + 真实 heatmap artifact，quality `pilot_candidate`，三模式与冲突保守拒答逻辑不变。

## 9:00–10:00 收尾

> 这一阶段我最想展示的不是 6/9 这个数字，而是方法论：失败先逐样本归因，便宜的假设先用实验证伪，选参不碰外层测试实体，还要审计候选空间从哪里来。我保留了 v1 偏乐观的结果，用 v2 给出修订数字。

## 常见追问

**为什么不用更大的分割模型？** 设备预算是 16GB 统一内存、单进程 12GB 门禁，而且 50 张正例图训大模型只会过拟合；14.3M 参数已经是这个数据量的合理上限。

**为什么 U-Net 阈值那么高（0.999）？** sigmoid 输出饱和是 Dice 损失的常见现象；阈值只用 validation 分数选取，test 只评一次。

**CV 为什么不报最好一折？** v2 的 Acc@IoU 每折为 0.900 / 0.800 / 0.800 / 0.600 / 0.636，折间方差本身就是结论；只报最好一折等于又一次选择偏差。

**什么时候能转 production？** 需要：新的未参与选择的冻结 holdout、≥300 图的 formal KPI 门槛、签名 evaluator attestation、KSDD 与 ImageNet 权重的许可审查。当前四条都不满足。
