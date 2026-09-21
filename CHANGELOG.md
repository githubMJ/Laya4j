# Changelog

Laya4j 遵循 [语义化版本](https://semver.org/)。

## [0.1.0] - 2024-XX-XX

### Added
- 初始发布
- 支持 `laya-multilingual-mmbert-base-v0.3.4-1c5edc1` 模型(1.2 GB FP32)
- 三种决策原语:`choice` / `score` / `noul`
- 自动脚本/语言检测(8 种语言)
- LayaRouter 多 checkpoint 路由
- 内置 4 个 preset:`triage` / `guard` / `moderation` / `modelRouter`
- HuggingFace Hub 自动下载(零 token)
- 项目内 `models/` 离线加载
- 与 Python ONNX Runtime 完全对齐(bit-级 diff=0.0000)
- 57 个单元测试(34 个核心 + 23 个中英文混合)
- ONNX Runtime 1.30.0(性能比 1.19.2 提升 49% p50)

### Known Issues
- ONNX 模型(1.2 GB)不在 Maven Central 包内,需从 HuggingFace Hub 或 GitHub Release 单独获取
- base 模型在中文威胁检测(50%)/难度分级(0%)/领域识别(40%)准确率低,生产前必须 fine-tune

[0.1.0]: https://github.com/your-org/Laya4j/releases/tag/v0.1.0