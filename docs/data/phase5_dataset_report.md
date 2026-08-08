# 第五阶段：真实公开数据 V0 报告

## 结论

本阶段冻结 `Kolektor Surface-Defect Dataset (KSDD)` 的精细标注版本为
`ksdd-0.1.0`。它是来自真实受控工业环境的电机换向器表面缺陷数据，具备像素级掩码，
能够验证“是否违规/缺陷 + bbox 定位”的最小闭环。它不含自然语言问答标注，因此本版本仅是
**工业异常定位 pilot**，不能证明通用视觉合规问答能力。

冻结测试集只有 56 张，低于 300 张正式 KPI 样本下限。manifest 明确记录
`formal_kpi_eligible=false`；评测包读取该 manifest 时会把 KPI-01～KPI-05 全部强制标记为
`pilot_only`，即使某次模型探针数值达到阈值，也不得声称正式验收。

## 来源选择与 MVTec AD 评估

优先候选 MVTec AD 与工业异常检测、分类和像素级定位高度匹配。其[官方页面](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
说明数据包含 15 个类别、5,000 余张高分辨率图、无缺陷训练图、含缺陷及无缺陷测试图，并提供
像素级标注；许可同样为 CC BY-NC-SA 4.0。但当前官方分发入口要求填写下载表单，旧逐类别
下载入口又不能作为可验证的免身份自动下载源，因此不满足本阶段“无需登录/身份提交、可重复下载”的约束。
此外，MVTec AD 也不提供自然语言问答标注，即使下载成功也只能作为工业异常定位基线。

最终选择 [ViCoS Lab 原始发布页上的 KSDD](https://www.vicos.si/resources/kolektorsdd/)：官方页面明确列出
399 张图，其中 52 张有可见缺陷、347 张无缺陷，图像来自真实受控工业环境，并提供精细标注直链。
原始发布页同时明确许可证为
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)。官方直链无需登录，
归档只有约 97.1 MiB，适合 M5 MacBook Air 16GB 本地处理。

## 下载与许可登记

| 字段 | 冻结值 |
|---|---|
| 原始发布页 | `https://www.vicos.si/resources/kolektorsdd/` |
| 官方归档 | `https://data.vicos.si/datasets/KSDD/KolektorSDD.zip` |
| 下载时间（UTC） | `2026-08-08T03:36:37Z` |
| 归档字节数 | `101831129` |
| 归档 SHA-256 | `65dc621693418585de9c4467d1340ea7958a6181816f0dc2883a1e8b61f9d4dc` |
| 许可证 | `CC BY-NC-SA 4.0` |
| 商业使用 | 不允许，除非另行取得权利方许可 |
| 署名 | Kolektor Group d.o.o.；Domen Tabernik、Samo Sela、Jure Skvarc、Danijel Skocaj；ViCoS Lab / University of Ljubljana |

本数据只能用于符合许可证的非商业研究与验证。转换后的 bbox、并集掩码和 Schema 标注属于改编内容，
继续受相同许可证的署名、非商业和相同方式共享要求约束。它不能直接进入商业产品训练集或生产探针库；
商业使用必须先取得权利方许可。

本地原始归档与解压目录是 `data/raw/ksdd_v0/`，转换结果是
`data/processed/ksdd_v0/`。两者均受仓库 `/data/` ignore 规则保护，不会进入源码提交。
登记文件为 `source_record.json`，冻结清单为 `manifest.json`，其独立哈希写入
`manifest.sha256`，追踪锁为 `configs/data/ksdd_v0.lock.json`。

## Schema 转换

每张 JPG 转换为一个 `DataSample`：

- `sample_id`：`ksdd_{kosNN}_{partN}`；
- `entity_id`：官方物理件目录 `kosNN`，同一物理件的约 8 个非重叠视图不可跨划分；
- `instruction`：固定双语工业表面检查指令，不把它伪装成原始问答标注；
- `response.result`：空掩码映射为 `compliant`，非空掩码映射为 `violation`；
- `response.objects`：非零掩码所有像素的最紧外接框，坐标为原图 `[x1,y1,x2,y2]`；
- `metadata.mask`：保留转换后的像素掩码相对路径；
- `license`：每个样本显式携带 CC BY-NC-SA 4.0 权限、限制与署名；
- 测试集无缺陷图额外标为 `hard_negative`，含义是“来自训练未见物理件的同域正常表面”。

像素掩码转 bbox 会丢失精细形状与多连通区域结构；模型若输出多个框，现有 Acc@IoU0.5
仍可评测，但本 V0 真值框是每张图所有缺陷像素的联合外接框。

## 去重、实体隔离和泄漏

原始 399 张图中发现两对完整重复物理件目录：`kos01 ↔ kos26` 和 `kos27 ↔ kos34`，
共 16 张逐字节重复图像。两张重复正例的官方掩码边界存在轻微差异。冻结规则为：

1. 用图像 SHA-256 识别精确重复；
2. 对标注一致的重复图保留字典序首个来源；
3. 对同一图像的标注差异做官方掩码逐像素并集，再生成联合 bbox；
4. 在样本 metadata 中记录全部来源别名、冲突标志和合并策略；
5. 精确去重后再计算 64-bit dHash，汉明距离不超过 5 的样本进入同一近重复连通分量；
6. 同一 `entity_id` 或同一近重复分量只能进入一个划分。

去重后保留 383 张、48 个独立物理实体、50 个正例和 333 个负例。另检测到 8 个近重复组，
涉及 16 张图；实体泄漏为 0，近重复跨划分泄漏率为 0.0。

## 冻结划分与切片统计

划分使用固定 seed `42` 和目标比例 70/15/15；实体与近重复分量不可拆分，所以报告实际数量：

| Split | 实体 | 总样本 | 缺陷正例 | 正常负例 | 困难负例 |
|---|---:|---:|---:|---:|---:|
| train | 34 | 271 | 34 | 237 | 0 |
| validation | 7 | 56 | 7 | 49 | 0 |
| test（冻结） | 7 | 56 | 9 | 47 | 47 |
| 合计 | 48 | 383 | 50 | 333 | 47 |

关键切片还包括 `domain:industrial`、`object:commutator`、`capture:controlled`、
`annotation:pixel_mask`、`positive/negative`、`mask_derived_bbox` 和
`negative:unseen_physical_item`。完整机器统计见 `dataset_summary.json`。

## Pilot 门禁与指标声明

- `test=56 < 300`，状态固定为 `pilot`；
- `formal_kpi_eligible=false`；
- KPI-01～KPI-05 只能输出探索性数值和置信区间，验收状态强制为 `pilot_only`；
- 本阶段没有运行模型，`dataset_summary.json` 中 `model_metrics=null`；
- 样本数、正负例数、哈希数和泄漏率都是数据质量统计，不是模型成绩；
- 不得把合成 fixture 指标、mock 输出或本数据集统计冒充真实模型成绩。

## 五张真实模型探针

以下文件均来自冻结 test，许可证和署名记录在 `probe_manifest.json`。前三张为缺陷正例，后两张为
同域困难负例：

| Sample | 结果 | bbox | 本地文件 |
|---|---|---|---|
| `ksdd_kos10_part3` | violation | `[59,595,500,649]` | `probes/ksdd_kos10_part3.jpg` |
| `ksdd_kos15_part3` | violation | `[244,826,500,940]` | `probes/ksdd_kos15_part3.jpg` |
| `ksdd_kos16_part5` | violation | `[267,604,500,685]` | `probes/ksdd_kos16_part5.jpg` |
| `ksdd_kos10_part0` | compliant | `[]` | `probes/ksdd_kos10_part0.jpg` |
| `ksdd_kos10_part1` | compliant | `[]` | `probes/ksdd_kos10_part1.jpg` |

探针是许可证允许的非商业研究样例，不代表可无限制再分发或商业使用。

## 可重复命令和稳定接口

下载与校验：

```bash
mkdir -p data/raw/ksdd_v0
curl --fail --location --retry 3 \
  --output data/raw/ksdd_v0/KolektorSDD.zip \
  https://data.vicos.si/datasets/KSDD/KolektorSDD.zip
shasum -a 256 data/raw/ksdd_v0/KolektorSDD.zip
unzip data/raw/ksdd_v0/KolektorSDD.zip -d data/raw/ksdd_v0/extracted
```

转换和冻结：

```bash
.venv/bin/python -m src.data.ksdd_cli \
  --source-root data/raw/ksdd_v0/extracted \
  --archive data/raw/ksdd_v0/KolektorSDD.zip \
  --output-root data/processed/ksdd_v0 \
  --license-policy configs/data/license_policy.yaml \
  --downloaded-at 2026-08-08T03:36:37Z \
  --created-at 2026-08-08T03:36:37Z
```

Python 调用稳定门面为：

```python
from src.data import KSDDSourceRecord, LicensePolicy, prepare_ksdd_dataset
```

算法读取 `samples.jsonl` 与 `manifest.json`，只处理 manifest 中 `split=test` 的 ID。后端或模型将
逐样本输出保存为预测 JSONL 后，可直接运行现有离线评测；`evaluation_ground_truth.jsonl` 已带原图尺寸、
困难负例和切片字段：

```bash
.venv/bin/python -m src.evaluation.cli \
  --ground-truth data/processed/ksdd_v0/evaluation_ground_truth.jsonl \
  --predictions predictions.jsonl \
  --output evaluation-report.json \
  --package-dir evaluation-package \
  --data-manifest data/processed/ksdd_v0/manifest.json \
  --created-at 2026-08-08T03:36:37Z
```

评测包会从 manifest 自动继承 `pilot_only`，不要传 `--fixture-only`；这是实际公开数据 pilot，
不是合成 fixture。

## 验证证据

```text
.venv/bin/ruff check src/data src/evaluation tests/data tests/evaluation
All checks passed!

.venv/bin/python -m pytest tests/data tests/evaluation -q
49 passed in 0.24s
```

真实数据复核：383 个样本全部完成图片解码、扩展名/签名、尺寸、bbox 边界和许可证验证；
manifest SHA-256 为
`fe9983728d7aa830f11a1369ff1a4eb353be8e8403685a1cf45ea23db128085a`，
归档与转换结果合计本地占用约 762 MiB。

## 已知风险

- 非商业许可证与未来商业产品目标冲突；获得单独商业授权前只能用于研究/内部非商业验证。
- 数据仅覆盖单一工业部件、灰度图、受控光照和局部表面缺陷，域外泛化未知。
- 固定双语指令是项目适配层生成的，不是原始数据的人类问答标注。
- 50 个正例和 56 个测试样本都偏少；类别、定位与困难负例指标区间会很宽。
- dHash 只适合快速近重复召回，可能漏掉裁剪、旋转、拼接或强编辑后的泄漏。
- 两组重复实体和两处掩码冲突说明原始发布也需要质量审计；V0 保留了来源与合并证据，后续版本不得静默改变规则。
