# Security Policy

## Supported scope

安全修复以当前默认分支为准。该项目是本地优先的 internal pilot，不应直接暴露到不受信任的公网，也不构成生产安全承诺。

## Reporting a vulnerability

请不要在公开 Issue 中披露可利用细节、真实密钥、数据样本或模型工件。优先使用 GitHub 仓库的 **Security → Report a vulnerability** 私密报告入口；若仓库未启用该功能，请只提交不含利用细节的 Issue，请求维护者建立私密沟通渠道。

报告建议包含受影响版本、复现条件、影响范围和最小化 PoC。请勿上传受许可证限制或含个人信息的数据。

## Secrets and artifacts

- 管理 token 与 evaluator signing key 只能通过环境变量或密钥存储注入；
- `data/`、`models/`、`artifacts/`、`runs/` 和日志默认不进入 Git；
- Mock 或 fixture 结果不得用于真实模型质量验收；
- 真实模式在缺少可验证 manifest、hash 或 attestation 时必须拒绝启动。
