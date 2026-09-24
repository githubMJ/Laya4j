# Changelog

Laya4j 遵循 [语义化版本](https://semver.org/)。

## [0.2.0] - 2026-09-21

面向开源组件发布的重构版本。

### Added
- 新增 `LayaModel` 接口抽象:推理后端可插拔,`LayaRouter`/`LayaPredictor` 面向接口编程,支持注入远程服务/量化运行时/测试桩
- 异常层级细化:`LayaException` 之下新增 `ModelLoadException` / `InferenceException` / `ModelFetchException`,支持编程式区分处理
- `HuggingFaceFetcher.fetch` 支持显式指定本地模型目录;新增系统属性 `laya4j.models.dir` 覆盖项目内 `models/` 目录(摆脱对进程工作目录的依赖)
- `LayaRouter.Builder.repoBase()` 可配置 `RoutingDecision.repo()` 的 HuggingFace 仓库前缀
- `LayaPredictor.Builder` 支持直接注册预构建的 `LayaModel` 实例(`english()`/`multilingual()`/`typedDecisions()`)
- `LayaPredictor.autoLoad()` / `autoLoad(boolean)` 便捷工厂(旧 `autoLoad(Path, boolean)` 标记 `@Deprecated`)
- `LayaConfig` 重构为可校验的不可变配置:移除未使用的 `maxPrefixes` 死字段,新增 `builder()`、`temperatureFor(DecisionType)`,以及 `fromRlAgentConfig(Path)` 直接从模型自带的 `rl_agent_config.json` 加载配置
- 新增 `laya_fine_tune/export_onnx.py`:把上游 [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) 的 PyTorch 权重导出为 Laya4j 可直接加载的 ONNX(含 Split→Slice 兼容性修复、PyTorch/ONNX 数值对齐校验),产物直接写入项目 `models/` 目录;权重同步升级到上游 0.3.5(HF commit `1c5edc1` 权重未变,仅代码版本号同步)
- 新增 `laya4j-example/` 模块存放独立可运行的示例工程(Spring Boot 俄罗斯方块 demo,不参与主库构建/发布)

### Fixed
- **资源泄漏**:`LayaRouter.close()` 此前不关闭 `defaultModel`(单模型场景 ONNX session 永不释放);现在关闭所有已注册模型,且同一实例只关闭一次
- **进程级单例误关**:`LayaOnnxModel.close()` 不再关闭共享的 `OrtEnvironment`(此前多模型场景下第一个 close 会破坏其余 session)
- `ScoreDecision.levels()` / `Question.choices()` / `Question.levels()` 返回防御性拷贝,与 `probabilities()`/`distribution()` 行为一致
- 下载失败等 I/O 错误不再包装为误导性异常类型

### Changed
- **破坏性**:示例代码与压测工具移出发布 jar — `MainApplication` → `src/test/.../examples/QuickStartDemo`,`com.laya4j.benchmark.Benchmark` → `com.laya4j.examples.Benchmark`;运行 demo 改为 `mvn -Pdemo test-compile exec:java`
- **破坏性**:`LayaRouter`/`LayaPredictor.Builder` 的模型参数类型由 `LayaOnnxModel` 放宽为 `LayaModel` 接口
- 日志改用 slf4j-api(`HuggingFaceFetcher` 不再直接 `System.out`);依赖从 `slf4j-nop` 改为 `slf4j-api`,日志绑定由使用方选择(测试环境用 `slf4j-simple`)
- 版本号升至 0.2.0,pom 元数据(scm/url/developer)更新为 laya4j 组织

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

[0.2.0]: https://github.com/aidenma/Laya4j/releases/tag/v0.2.0
[0.1.0]: https://github.com/aidenma/Laya4j/releases/tag/v0.1.0