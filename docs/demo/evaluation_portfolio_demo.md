# 算法作品集评测证据演示

第四阶段在主分析工作台之外增加了可折叠的“评测证据”审计附录。不导入评测产物时，主流程的上传、查询、边界框和结果展示不变。

## 启动与入口

```bash
python3 app/run_demo.py --open
```

在页头点击“评测证据”。面板中先显示当前 `/health/ready` 运行诊断，再按需导入 evaluator 产物。

## 支持的导入

### 单独 report.json

选择 evaluator 生成的 `report.json`。UI 展示五项总体指标、切片和失败案例。由于单独 report 不含 package provenance、置信区间和哈希清单，界面固定显示：

```text
UNVERIFIED REPORT · 不可用于模型验收
```

### 完整 evaluation package

点击“选择 JSON”，一次选中 package 目录中的全部 JSON：

- `package_manifest.json`
- `config.json`
- `data_provenance.json`
- `environment.json`
- `metrics.json`
- `failures.json`
- `report.json`
- 可选 `comparison.json`

UI 对 manifest 中的所有 JSON 组件逐一校验 SHA-256。`report.md` 和 `report.html` 是人类报告，无需在浏览器 JSON 导入时选择。缺少 JSON 组件或哈希不匹配时 fail-closed，并清空上一份指标。

## 证据真值和水印

| 来源 | 标识 | 模型验收资格 |
|---|---|---|
| `fixture_only=true` | `FIXTURE / MOCK · 不可用于模型验收` | 永久不可用 |
| `mock_only=true` | `MOCK OUTPUT · 不可用于模型验收` | 永久不可用 |
| 单独 report/metrics | `UNVERIFIED REPORT` | 不可用 |
| 哈希通过的非 fixture/mock package | `PACKAGE VERIFIED` | 仍需 `kpi_acceptance.eligible_for_model_acceptance=true` |

界面不会因数值看起来很高而移除 fixture/mock 水印。

## 展示内容

- 零样本基线、候选模型、百分点变化；
- 五项核心指标的 bootstrap 置信区间和有效分母；
- `comparison.json` 的配对改善区间、样本充足性与 significance hint；
- KPI-01–05 状态、阈值和阻断原因；
- 按失败率排序的切片；
- 失败 sample_id、真值/预测结论、输出来源与 failure codes；
- active/candidate/validated/retired 模型状态、ready、source、量化、weight hash、别名、降级原因和 readiness request_id。

## 键盘与错误

- 入口、导入、刷新、清空和关闭均为原生 button；
- 面板打开后获得焦点；按 `Escape` 关闭并把焦点返回入口；
- JSON 解析、Schema、指标边界、区间、package 缺件和哈希错误都显示稳定错误码；
- 任一导入失败都隐藏旧评测内容，避免陈旧指标被误认为当前报告。

