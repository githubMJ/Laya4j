# train_ads — 广告出价决策微调

用 Laya 三种决策原语建模**广告计划出价调整**策略,微调 base 模型学会
"读中文投放指标 → 做出价动作"的业务策略。本目录自成一体的流水线:
数据构建 → 预处理 → 训练 → 评估 → 导出。

## 决策 schema

| 问题 | 类型 | 含义 |
|---|---|---|
| `bid_action` | choice | `raise_bid` / `keep` / `lower_bid` / `pause` / `add_budget` |
| `needs_adjustment` | noul | 当前是否需要动出价 |
| `adjust_magnitude` | score | 0=维持 1=微调(约5%) 2=中调(10%~20%) 3=大幅(>20%) |

状态输入为中文投放指标:计划名、时段、日预算、已消耗、展示/点击/CTR/CPM、
转化数、目标/实际转化成本。投放策略(指标→决策的规则)在 `gen_data.py` 的
`decide()` 中,**改策略后重新生成数据再微调即可**。

## 流水线

```bash
cd laya_fine_tune/train_ads

# 1. 构建数据集(规则策略 → 软目标 JSONL,默认 600 行)
python gen_data.py --n 600 --out data/finetune_ads.jsonl

# 2. 预处理为训练 items(通用脚本在上级目录)
python ../prepare_data.py --data data/finetune_ads.jsonl \
       --out items/train_items.pt --val-items 60

# 3. 微调(单卡;MPS fp32 / CUDA fp16 自适应)
python ../train.py --data items/train_items.pt --out out/ads \
       --epochs 3 --micro-batch 8 --grad-accum 2

# 4. 评估(手写留出用例,基线 vs 微调后同一路径)
python eval.py                        # base 0.3.21 基线
python eval.py --model-dir out/ads    # 微调后模型

# 5. 导出 ONNX → Java 端(需把 out/ads 推到 HF 或加本地导出支持)
python ../export_onnx.py --repo <hf-repo-of-checkpoint> --out-dir ../../models
```

## 基线(base 0.3.21,8 个手写用例)

```
bid_action       3/8 = 38%   ← 对业务策略零概念,一律猜 lower_bid
needs_adjustment 5/8 = 62%
```

微调后的目标:两项 ≥ 90%(训练数据即策略规则,理论上可近乎全对;
留出用例考察泛化而非记忆)。

## 文件

```
train_ads/
├── gen_data.py       # 数据构建:采样 7 种投放状态区域,decide() 规则产生软目标
├── data/             # 生成的 JSONL 数据集(入库,便于追溯策略版本)
├── items/            # 预处理产物(.pt,不入库)
├── eval.py           # 手写留出用例评估(不来自训练生成器)
├── out/              # 训练 checkpoint(不入库)
└── README.md
```
