# laya_fine_tune/ —— 微调与权重导出工具

本目录存放针对上游 [Laya](https://github.com/NandhaKishorM/laya)(Python/PyTorch)
的微调、评估与权重导出脚本(把微调/上游权重导出为 Laya4j(本项目,Java)可直接
加载的 ONNX 文件),**不参与 Java 构建**,也不会打进 Maven Central 发布产物。

## 何时需要重新导出

- 上游 Laya 发布了新版本(修复了 bug 或更新了训练权重)
- 需要接入自己微调(fine-tune)后的 checkpoint
- 需要导出 `english` / `typed-decisions` 等其他 subfolder 的 checkpoint

## 用法

```bash
cd laya_fine_tune
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/NandhaKishorM/laya.git   # 上游 Python 包

python export_onnx.py                                   # 默认导出 multilingual
python export_onnx.py --subfolder english                # 导出 english checkpoint
python export_onnx.py --repo your-org/laya-finetuned      # 自己微调后的仓库
```

产物默认写到 `../models/`(即项目根目录的 `models/`,与 Java 端
`HuggingFaceFetcher.projectModelsDir()` 的默认查找路径一致):

```
models/
├── laya-decision-multilingual-mmbert-base-v{laya版本}-{commit短哈希}.onnx
├── tokenizer/
└── rl_agent_config.json
```

## 导出脚本做了什么

1. 从 HuggingFace Hub 下载/定位权重快照(`safetensors` + `rl_agent_config.json` + tokenizer)
2. 用上游 `laya.common.build_sequence` 构造一个覆盖 choice/score/noul
   三种类型的示例输入,并跑一次 **PyTorch 前向**作为数值基准
3. 用 `torch.onnx.export`(dynamo exporter,opset 17)导出 ONNX 计算图
4. **修复 `Split` 节点**:PyTorch 2.11 导出的 `Split` 算子带
   `num_outputs` 属性,但 `onnxruntime>=1.18` 已不支持该属性,脚本用
   `onnx-graphsurgeon` 把每个 `Split` 重写成等价的多个 `Slice` 节点
5. 用导出后的 ONNX 跑一次 **ONNX Runtime 前向**,与步骤 2 的 PyTorch
   基准比较 `logits`/`act_logits` 的最大绝对误差(阈值 `1e-3`,超过则
   直接终止并报错),确保导出没有引入数值偏差——这是保证 Java 端推理
   结果与 Python 端 **bit 级对齐**的关键步骤
6. 保存 tokenizer 与精简版 `rl_agent_config.json` 到 `models/`,供 Java
   端 `LayaConfig.fromRlAgentConfig(Path)` 直接读取加载

## 导出完成后,如何接入 Java 端

```java
// com.laya4j.model.HuggingFaceFetcher.MODEL_VERSION 更新为脚本打印的版本号,
// 例如 "0.3.5-1c5edc1"

// 运行时也可以完全不依赖 HuggingFaceFetcher 的硬编码版本,直接扫描 models/ 目录:
Path onnx = Files.list(Path.of("models"))
        .filter(p -> p.toString().endsWith(".onnx"))
        .findFirst().orElseThrow();
LayaConfig cfg = LayaConfig.fromRlAgentConfig(Path.of("models/rl_agent_config.json"));
LayaPredictor p = LayaPredictor.builder()
        .multilingualOnnx(onnx)
        .multilingualTokenizer(Path.of("models/tokenizer"))
        .config(cfg)
        .build();
```

同时更新 `models/VERSION.md` 记录新版本的来源信息(Laya 版本号、
HuggingFace commit SHA、训练信息等),保持可追溯性。
