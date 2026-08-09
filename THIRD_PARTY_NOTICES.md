# Third-party notices and data boundaries

本仓库不分发 KSDD 原始数据、第三方模型权重或训练 checkpoint。使用者需自行从官方来源获取并独立确认适用条款。

## Dataset

- **Kolektor Surface-Defect Dataset (KSDD)**：项目记录的许可标识为 `CC BY-NC-SA 4.0`。当前实验仅按非商业 internal pilot 使用；该数据集不能因为本仓库代码的许可而获得额外授权。

## Models and libraries

- **Qwen3-VL / MLX-VLM**：实际模型与运行库适用各自上游仓库和模型卡条款；本仓库只保存适配代码与配置，不保存权重。
- **PyTorch、timm、Anomalib、OpenCV、FastAPI 及其他依赖**：分别适用各自上游许可证。精确版本由 `uv.lock` 固定。
- **ResNet-18 pretrained weights**：配置记录来源 revision、SHA-256 与字节数；权重本身不进入 Git，使用者需审核其来源和授权。

本文件用于说明边界，不替代任何上游许可证文本或法律意见。如果用于商业场景，必须重新完成数据、模型、权重和依赖的许可审查。
