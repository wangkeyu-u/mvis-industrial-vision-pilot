# Contributing

感谢你关注 MVIS。这个仓库把“实验数字是否可信”放在功能数量之前，因此任何改动都应保留可复现性和 fail-closed 行为。

## 本地开发

```bash
uv sync --locked --group dev --extra specialist
docs/deployment/mvis.sh test-all
```

需要训练 U-Net 或运行完整视觉实验时，改用：

```bash
uv sync --locked --group dev --extra training
```

## 提交要求

- 不提交 `data/`、`models/`、`artifacts/`、`runs/`、checkpoint 或第三方权重；
- 不提交 API token、签名密钥、本机绝对路径或含个人数据的截图；
- 模型指标必须绑定数据版本、实体划分、配置、seed 和代码 revision；
- Mock、fixture、pilot 和 external holdout 的结果必须显式区分；
- 任何训练、后处理或阈值选择不得读取外层测试实体；
- 修改 API、schema 或质量门禁时，必须补充回归测试和向后兼容说明。

## Pull Request 检查

PR 描述应包含：问题、改动、验证命令、实验数据口径、风险与回滚方式。提交前运行：

```bash
docs/deployment/mvis.sh test-all
git diff --check
```
