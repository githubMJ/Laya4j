# Model Version

## 当前打包的 ONNX 模型

**文件名**: `laya-decision-multilingual-mmbert-base-v0.3.21-e4e9ddf.onnx`(FP32,约 1.29 GB)

### 来源

| 字段 | 值 |
|---|---|
| Laya Python 包版本 | **0.3.21**(上游 main 分支最新,editable 安装自本地 clone) |
| HuggingFace commit SHA | `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67`(短哈希 `e4e9ddf`) |
| 模型仓库 | `convaiinnovations/laya-multilingual`(**独立仓库**,文件在根目录,无 subfolder) |
| 导出脚本 | [`laya_fine_tune/export_onnx.py`](../laya_fine_tune/export_onnx.py) |
| 上游代码 | https://github.com/NandhaKishorM/laya (v0.3.21) |
| 上游权重 | https://huggingface.co/convaiinnovations/laya-multilingual |

> **注意**:上游 0.3.19+ 把 multilingual 权重从 `convaiinnovations/laya` 的
> `multilingual/` 子目录拆分到独立仓库 `convaiinnovations/laya-multilingual`
> (旧仓库现仅存 English/ModernBERT-large checkpoint)。新仓库布局:
> `model.safetensors` + `rl_agent_config.json` + `encoder/config.json` + `tokenizer/`。
> 导出产物通过 PyTorch/ONNX 数值对齐校验(logits max diff 9.06e-06,
> softmax 后概率 atol=1e-3 与上游 tests/test_onnx.py 同标准)。

### 历史版本

| 版本 | Laya | HF commit | 备注 |
|---|---|---|---|
| **0.3.21-e4e9ddf**(当前) | 0.3.21 | `e4e9ddf`(laya-multilingual 仓库) | 新独立仓库布局;上游 ONNX batch/long parity、opt-in abstention |
| 0.3.5-1c5edc1 | 0.3.5 | `1c5edc1`(旧 laya 仓库 multilingual/ 子目录) | 旧仓库布局;权重与 0.3.4 相同 |

### Encoder: mmBERT-base

### Encoder: mmBERT-base

| 字段 | 值 |
|---|---|
| HF repo | `jhu-clsp/mmBERT-base` |
| 来源 | Johns Hopkins CLSP lab |
| 架构 | `ModernBertForMaskedLM`(ModernBERT 系) |
| 参数量 | ~322M |
| Hidden size | 768 |
| Intermediate size | 1152 |
| Hidden layers | 22 |
| Vocab size | **256,000**(byte-level BPE) |
| Max position embeddings | **8192**(RoPE) |
| 默认上下文 | 1024 (mmBERT 多语种版设置) |
| Head max length | 256 |

**注意**:Laya multilingual 用的不是标准 BERT,是 Johns Hopkins 的 **mmBERT-base**(multilingual ModernBERT,byte-level BPE)。特殊 token:
- `<bos>` = 2 (CLS 位置)
- `<eos>` = 1 (SEP 位置)
- `<mask>` = 4 (MASK 位置)
- `<pad>` = 0

### Decision Head (Laya 自定义)

| 字段 | 值 |
|---|---|
| `head_layers` | 2(2 层 Transformer encoder block) |
| `max_len` | 1024 |
| `head_max_len` | 256(留给 schema + options 的预算) |
| `temperature` | `[1.0, 1.0, 1.0]` (基座未校准,对应 `LayaConfig.defaults()`) |
| 决策头结构 | `TransformerEncoder × 2 + type_emb(3×768) + scorer(LayerNorm→Linear→GELU→Linear→1) + act_head(Linear→256→GELU→Linear→2)` |

### 文件大小

| 文件 | 大小 | 说明 |
|---|---|---|
| `laya-decision-multilingual-mmbert-base-v0.3.5-1c5edc1.onnx` | **1.2 GB** | FP32 满血版(未量化);**不进 git 仓库**,见根目录 `.gitignore` |
| `tokenizer/tokenizer.json` | 33 MB | mmBERT byte-level BPE vocab |
| `tokenizer/tokenizer_config.json` | 637 B | tokenizer 配置 |
| `rl_agent_config.json` | 472 B | decision head 配置,供 `LayaConfig.fromRlAgentConfig()` 读取 |

### 训练信息

| 字段 | 值 |
|---|---|
| updates | 15,987 |
| epochs | 4 |
| 训练时长 | 4.97 小时 |
| world_size | 1 |
| 训练数据 | `laya` 仓库 `data/train.jsonl`(2,676 条) |
| 训练目标 | RLCD(Reinforcement Learning with Calibrated Decisions) |

---

## 版本历史

| 版本 | HF commit | 变更 |
|---|---|---|
| 0.3.5-1c5edc1(当前) | `1c5edc1` | 上游代码修复(路由器线程安全等),权重未变 |
| 0.3.4-1c5edc1 | `1c5edc1` | 初始导出版本 |

---

## 怎么重新生成

导出脚本已内置在本项目 [`laya_fine_tune/export_onnx.py`](../laya_fine_tune/export_onnx.py),
无需再依赖外部/临时目录中的脚本:

```bash
cd tools
pip install -r requirements.txt
pip install git+https://github.com/NandhaKishorM/laya.git
python export_onnx.py                    # 默认导出 multilingual,写到 ../models/
```

脚本会自动:
1. 从 HuggingFace Hub 拉取最新权重快照
2. 导出 ONNX 并修复 `Split` 算子兼容性问题
3. 用 PyTorch 结果做数值对齐校验(误差 ≥ 1e-3 会直接报错终止)
4. 保存 tokenizer + `rl_agent_config.json`

完成后,记得同步更新:
- 本文件(`models/VERSION.md`)的版本记录
- `com.laya4j.model.HuggingFaceFetcher.MODEL_VERSION` 常量

详细原理说明见 [`laya_fine_tune/README.md`](../laya_fine_tune/README.md)。

---

## 为什么文件名带版本号和 encoder 名称

- **版本号**(semver + commit SHA):不同 Laya Python 包版本导出的 ONNX 可能不兼容
- **encoder 名**(mmbert-base):明示用的是 mmBERT 而非 mBERT 或 BERT,避免误用同名的其他模型权重
- HuggingFace commit SHA 是不可变标识符,精确指向上游某次提交的权重

---

## 校验

```bash
python3 -c "
import onnxruntime as ort
sess = ort.InferenceSession('models/laya-decision-multilingual-mmbert-base-v0.3.5-1c5edc1.onnx')
print('inputs:', [i.name for i in sess.get_inputs()])
print('outputs:', [o.name for o in sess.get_outputs()])
"
```

应输出:
```
inputs: ['input_ids', 'attention_mask', 'marker_pos', 'marker_mask', 'qtype']
outputs: ['logits', 'act_logits']
```

---

## 参考

- mmBERT-base 论文: [Johns Hopkins mmBERT](https://huggingface.co/jhu-clsp/mmBERT-base)
- ModernBERT 论文: [Answer.AI ModernBERT](https://github.com/AnswerDotAI/ModernBERT)
- Laya 论文: [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)
- RLCD 训练方法: [README 里的 "Methodology" 节](https://github.com/NandhaKishorM/laya#methodology)
