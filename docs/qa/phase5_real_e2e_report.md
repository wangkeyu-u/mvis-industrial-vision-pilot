# 第五阶段真实链路端到端报告

执行日期：2026-08-08  
设备：Apple Silicon / 16 GB unified memory  
结论：真实 MLX 模型、同源代理、浏览器上传、拒答/不确定、JSON/验收证据导出和评测包导入均有真实运行证据。数据样例已经完成许可与哈希登记；但三个合法缺陷 Probe 都没有产生可验收定位框，API 也没有返回模型 revision，且当前没有非 fixture 的 evaluator 模型成绩包。因此本阶段是“真实链路可运行、真实模型定位/成绩验收阻塞”，不是模型 KPI 通过。

## 真实性闸门

页面通过 `?client=api&acceptance=real` 进入真实模型验收模式。只有以下条件全部满足才显示绿色 `REAL MODEL EVIDENCE`：

1. 使用 HTTP API 客户端；
2. `/health/ready` 返回 ready；
3. `runtime.selected_mode` 严格等于 `real`；
4. `runtime.degraded` 严格等于 `false`；
5. active 模型存在、ready 且生命周期为 active；
6. 模型 source/base 不带 Mock、Test Double 或 built-in 标识。

任一条件不满足时，页面覆盖 `REAL SCORE CAPTURE BLOCKED`，并可导出 `mvis_real_model_acceptance_gate` 阻塞报告。普通 Mock 演示入口保持可用，但不能获得真实标记。

## 当前 real readiness 证据

使用 `MVIS_MODEL_MODE=real` 启动当前 `src.api.app:app`：

| 字段 | 实测值 |
|---|---|
| readiness request_id（启动探测） | `req_phase5_probe` |
| 浏览器请求前 readiness request_id（许可样例） | `req_ui_1786161421710_3ed9b02f89da4d6a` |
| requested / selected mode | `real / real` |
| degraded | `false` |
| active model | `qwen3-vl-2b-instruct-4bit` |
| lifecycle / ready | `active / true` |
| source | `model-config:qwen3_vl_2b_mlx_4bit.json` |
| quantization | `4-bit` |
| config fingerprint | `ef76aa6b9a49d5a1ea1066907c31e85cb914bb2c5726096042c89a019ed45cf1` |
| model fingerprint | `3817a1f046d88b69acfbd05b2b7e9caf5c2c4d6383984bbd60f238f472c4df83` |
| startup observed memory peak | `2042–2045 MB` |

只读本地固定配置含 revision `9c4f5209e57b31f4b9dfba735de3fb983739c9cc`，但 `/health/ready`、`/version` 和 `/v1/analyze` 均未返回 revision。浏览器证据因此记录 `model_revision=not_reported`，没有把配置值冒充响应绑定字段。

## 合法样例与数据边界

数据线程已冻结 `ksdd-0.1.0`：383 个去重样例，train/validation/test 为 271/56/56，全部登记为 `cc-by-nc-sa-4.0`。测试集只有 56 个样例，低于正式 KPI 最低 300，因此 `formal_kpi_eligible=false`，只能作为 pilot。

UI 的“载入许可 Probe”只在真实闸门通过时启用。演示服务仅公开 `probe_manifest.json` 中 allowlist 的文件；浏览器在创建 `File` 前校验 SHA-256，并把 sample_id、license、attribution、SHA-256 和期望对象写入验收证据，图片字节与查询文本不嵌入导出文件。路径逃逸与未登记文件均返回 404。

## 真实浏览器证据

浏览器通过同源代理和 multipart 调用真实 FastAPI。

### 链路烟测

| 场景 | request_id | 结果 | 耗时 | 对象 | 结论 |
|---|---|---|---:|---:|---|
| 项目自有图合规烟测 | `req_ui_1786160520097_ed1c5606dfa94420` | compliant | 1531 ms | 0 | JSON 与真实证据导出通过 |
| 高风险/超用途 | `req_ui_1786160557529_3ec3986f3cbc418d` | refused | 4520 ms | 0 | 未伪造框，导出通过 |
| 最终链路截图 | `req_ui_1786160643764_72a95f4bf2634309` | compliant | 1420 ms | 0 | 闸门持续 verified |

截图：[真实链路烟测](phase5_real_chain_smoke.jpg)。它只证明真实 API、模型加载、UI 和导出链路，不证明模型准确率。

### KSDD 许可缺陷 Probe

| sample_id | manifest 真值框 | request_id | 实际结果 | 耗时 | 模型框 |
|---|---|---|---|---:|---:|
| `ksdd_kos10_part3` | `[59,595,500,649]` | `req_ui_1786161298129_d8ca4c3a3a0545c4` | 422 `OUTPUT_VALIDATION_FAILED` | 未返回 | 0 |
| `ksdd_kos15_part3` | `[244,826,500,940]` | `req_ui_1786161327450_7e1c45a109a14d91` | refused / uncertain | 1547 ms | 0 |
| `ksdd_kos16_part5` | `[267,604,500,685]` | `req_ui_1786161345633_6e059827d9b74a25` | refused / uncertain | 1521 ms | 0 |
| `ksdd_kos10_part3`（ground 诊断） | `[59,595,500,649]` | `req_ui_1786161396845_47aaf1cd955f4288` | 422 `OUTPUT_VALIDATION_FAILED` | 未返回 | 0 |
| `ksdd_kos15_part3`（最终证据） | `[244,826,500,940]` | `req_ui_1786161421715_c47368f139454d0e` | refused / uncertain | 1530 ms | 0 |

截图：[许可 Probe 真实拒答](phase5_real_licensed_probe_refused.jpg)。它证明许可样例、真实模型、拒答/不确定和 JSON/证据导出联通；由于 `objects=[]`，不构成定位成功证据。产品层没有把 manifest 真值框绘制成“模型预测框”。

机器可读导出：[许可 Probe 真实验收证据](phase5_real_licensed_probe_evidence.json)。关键字段为 `status=observed`、`screenshot_eligible=true`、`model_performance_claim_allowed=false`、`model_revision=not_reported`；它允许保存真实链路截图，但明确禁止把 fixture 数字声明为模型成绩。

第一个 422 后，页面的结果 JSON、真实请求绑定、证据卡与 overlay 都为空，后续请求也先清空前一结果，验证了错误状态不会保留陈旧证据。

## 评测包导入证据

- 浏览器在真实 readiness 下成功导入 evaluator fixture：5 个指标、5 个 KPI、3 个切片、2 个失败案例。
- 真实 readiness 不会洗白 fixture；页面持续显示 `FIXTURE / MOCK · 不可用于模型验收`。
- 导出的真实闸门证据记录 `truth_state=fixture`、`verified=false`、`eligible_for_model_acceptance=false`，因此 `model_performance_claim_allowed=false`。
- 工作区当前没有带真实模型预测的 `package_manifest.json/report.json/metrics.json`。KSDD 数据包本身不是模型成绩包；`model_metrics=null`，且 pilot 测试集低于正式样本门槛。

## 降级阻断证据

使用真实 FastAPI mock-adapter 打开相同 `acceptance=real` 页面时，`selected_mode=mock-adapter`、`degraded=true` 且来源为 test double。页面覆盖阻断水印，导出 `status=blocked`、`screenshot_eligible=false`。

截图：[降级阻断](phase5_real_gate_blocked.jpg)，readiness request_id 为 `req_ui_1786160694847_7e5e428d107d47af`。

## 自动化与构建证据

| 测试层 | 结果 |
|---|---:|
| 服务、同源代理、许可 Probe allowlist | 15/15 |
| UI 静态、状态与可访问性契约 | 8/8 |
| 客户端、评测导入与真实闸门 | 24/24 |
| 真实 FastAPI 网络 E2E | 5/5 |

合计 **52/52 通过**。另外 5 个 JavaScript 模块 `node --check`、fixture JSON、两个启动器 help、一键启动 dry-run 和 3 张浏览器证据 JPEG 格式检查均通过。

网络 E2E 覆盖 multipart 成功/不确定/拒答、JSON 传输、非法图片、服务未就绪、request_id 与结构化错误。集成测试覆盖 CORS、本地同源代理、取消所需 AbortSignal 客户端契约、代理超时、不可达后端和 allowlist 路径逃逸。

## 启动方式

终端一：

```bash
MVIS_MODEL_MODE=real PYTHONPATH="$PWD" \
  uv run --python 3.13 \
  --with fastapi --with pydantic --with python-multipart \
  --with pyyaml --with pillow --with uvicorn \
  uvicorn src.api.app:app --host 127.0.0.1 --port 19401
```

终端二：

```bash
python3 app/server.py --host 127.0.0.1 --port 19400 \
  --backend-url http://127.0.0.1:19401 --proxy-timeout 125
```

浏览器：`http://127.0.0.1:19400/?client=api&acceptance=real&timeout_ms=120000`

离线 Mock 仍使用默认入口：`python3 app/server.py`，访问 `http://127.0.0.1:8000/`。

## 产品层接口契约

```text
GET /health/ready
  -> runtime.selected_mode, runtime.degraded, active_model, request_id

POST /v1/analyze
  multipart: image, query, task, model, use_specialist, options
  JSON: image(data URL), query, task, model, use_specialist, options
  -> request_id, model, result, objects[], reason, uncertain,
     latency_ms/timing, warnings, policy_state
```

浏览器分析前重新请求 readiness，并把该 readiness 与结果绑定。request_id 由客户端生成并经同源代理转发；取消使用 `AbortController`，超时、网络不可达、4xx/5xx 错误体映射为稳定代码。每次开始请求、换图、错误或取消都会撤销旧 JSON、框与结果绑定。

许可 Probe 辅助接口：

```text
GET /demo/real-probes/manifest.json
GET /demo/real-probes/<allowlisted-basename>
```

该接口只服务 manifest 中登记的演示副本，不开放数据目录浏览。

## 未关闭风险与验收判断

1. **真实定位阻塞**：合法缺陷样例产生 422 或拒答/不确定，没有模型框；不能报告 Acc@IoU 或展示定位通过截图。
2. **revision 未绑定**：API 未返回 revision，证据只能写 `not_reported`。
3. **正式成绩包缺失**：fixture 导入通过，但没有真实 evaluator 模型成绩包；KSDD pilot test 也低于 300 样本门槛。
4. **性能不是 KPI**：1.4–4.5 秒是少量观测，不是预热后 30 次 P50/P95；内存只记录启动峰值。
5. **specialist 未就绪**：请求 specialist 时会警告 `specialist_requested_but_unavailable`；当前结果来自通用 VLM。

发布判断：第五阶段产品与质量链路可实际演示；真实模型成绩、定位 KPI 和正式验收仍为阻塞状态，禁止用 Mock 框、manifest 真值框或 fixture 数字替代。
