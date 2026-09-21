# Model Version

## 当前打包的 ONNX 模型

**文件名**: `laya-decision-multilingual-mmbert-base-v0.3.4-1c5edc1.onnx`

### 来源

| 字段 | 值 |
|---|---|
| Laya Python 包版本 | **0.3.4** |
| HuggingFace commit SHA | `1c5edc17a7acd8701df6fc341c0d179f1c62c982`(短哈希 `1c5edc1`) |
| 模型子目录 | `multilingual` |
| 上游来源 | https://huggingface.co/convaiinnovations/laya/tree/multilingual |

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

**注意**:Laya multilingual 用的不是标准 BERT,是 Johns Hopkins 的 **mmBERT-base**(multilingual ModernBERT,byte-level BPE,RuSentNeMo-style tokenization)。特殊 token:
- `<bos>` = 2 (CLS 位置)
- `<eos>` = 1 (SEP 位置)
- `<mask>` = 4 (MASK 位置)
- `<pad>` = 0

构建 ONNX 时 Python 端用 `laya.load("convaiinnovations/laya", subfolder="multilingual")` 自动从 HF Hub 拉取 encoder + decision head + tokenizer。

### Decision Head (Laya 自定义)

| 字段 | 值 |
|---|---|
| `head_layers` | 2(2 层 Transformer encoder block) |
| `max_len` | 1024 |
| `head_max_len` | 256(留给 schema + options 的预算) |
| `temperature` | `[1.0, 1.0, 1.0]` (基座未校准) |
| 决策头结构 | `TransformerEncoder × 2 + type_emb(3×768) + scorer(LayerNorm→Linear→GELU→Linear→1) + act_head(Linear→256→GELU→Linear→2)` |

### 文件大小

| 文件 | 大小 | 说明 |
|---|---|---|
| `laya-decision-multilingual-mmbert-base-v0.3.4-1c5edc1.onnx` | **1.2 GB** | FP32 满血版(未量化) |
| `tokenizer/tokenizer.json` | 33 MB | mmBERT byte-level BPE vocab |
| `tokenizer/tokenizer_config.json` | 637 B | tokenizer 配置 |
| `rl_agent_config.json` | 472 B | decision head 配置 |

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

## 怎么重新生成

如果上游更新了:

```bash
# 1. Python 端导出
cd ../Laya
python export_onnx.py
# 生成 laya_onnx_export/laya_decision.onnx

# 2. 拷到项目,文件名带版本
NEW_VER="0.3.5-abc1234"   # 查 HF commit SHA 取短哈希
cp laya_onnx_export/laya_decision.onnx \
   ../Laya4j/models/laya-decision-multilingual-mmbert-base-v${NEW_VER}.onnx

# 3. 更新本文件 + HuggingFaceFetcher.MODEL_FILE
```

---

## 为什么文件名带版本号和 encoder 名称

- **版本号**(semver + commit SHA):不同 Laya Python 包版本导出的 ONNX 可能不兼容
- **encoder 名**(mmbert-base):明示用的是 mmBERT 而非 mBERT 或 BERT,避免误用同名的其他模型权重
- HuggingFace commit SHA 是不可变标识符,精确指向上游某次提交的权重

---

## 校验

```bash
python -c "
import onnxruntime as ort
sess = ort.InferenceSession('models/laya-decision-multilingual-mmbert-base-v0.3.4-1c5edc1.onnx')
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