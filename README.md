# MVIS 轻量多模态视觉系统

本项目实现面向视觉合规审查的本地优先 API、模型适配、数据处理、评测与 Web UI。目标设备为 M5 MacBook Air 16GB。当前发布线严格区分运行就绪、pilot 质量状态和 production 接受；Mock 只用于契约、安全和稳定性测试，不是真实性能或质量结果。

## 快速开始

```bash
uv sync --group dev
docs/deployment/mvis.sh test
docs/deployment/mvis.sh serve mock vlm_only
```

服务默认地址为 `http://127.0.0.1:8001`：

```bash
curl -s http://127.0.0.1:8001/health/live
curl -s http://127.0.0.1:8001/health/ready
curl -s http://127.0.0.1:8001/version
curl -s http://127.0.0.1:8001/v1/models
curl -s -X POST http://127.0.0.1:8001/v1/analyze \
  -F 'image=@data/processed/ksdd_v0/images/kos10/Part0.jpg;type=image/jpeg' \
  -F 'query=检查视觉合规异常并返回有根据的定位' \
  -F 'analysis_mode=fused' \
  -F 'options={"temperature":0,"seed":20260808,"max_tokens":128}'
```

`analysis_mode` 只接受 `vlm_only`、`specialist_only`、`fused`。在 `fused` 冲突中，最终 bbox 只能来自 specialist，文本结论标记 `uncertain` 并要求人工复核。

## 真实模式与发布验证

真实模式不会回退到 Mock。Specialist 还需显式指向本地、可核验的运行 manifest：

```bash
export MVIS_MODEL_MODE=real
export MVIS_ANALYSIS_MODE=fused
export MVIS_SPECIALIST_MANIFEST=artifacts/model/phase7/patchcore_resnet18/run_manifest.json
export MVIS_SPECIALIST_QUALITY_STATUS=pilot_failed
docs/deployment/mvis.sh preflight real fused
docs/deployment/mvis.sh release-validate fused \
  data/processed/ksdd_v0/images/kos10/Part0.jpg
```

`pilot_failed` 是当前 PatchCore 冻结评测的诚实标记：分类有正向信号，但定位指标未达标。只有与 model/config/revision/data/prompt provenance 完全绑定的签名 evaluator attestation 才能使 `quality_accepted=true`；否则 production active 始终被拒绝。

详细的配置、错误码、CORS、ModelOps 与资源边界见 [部署手册](docs/deployment/README.md)；最终证据见 [Phase 7 发布报告](docs/deployment/phase7_release_report.md)。
