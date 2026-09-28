"""
Ad-bidding decision evaluation (base vs fine-tuned checkpoint, same code path).

Hand-written held-out test cases (NOT from the training generator) measuring
how well a model internalizes the bid-adjustment policy:
  - bid_action (choice, 5 actions)
  - needs_adjustment (noul @0.5)

Usage:
    python eval_ads.py                                   # base 0.3.21 (HF download/cache)
    python eval_ads.py --model-dir out_zh_ads            # fine-tuned checkpoint dir
"""
import argparse
import json
import os

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from laya.common import build_model, build_sequence, QTYPES

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--model-dir", default=None,
                help="checkpoint dir (snapshot layout); omit = base 0.3.21 from HF")
args = ap.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[device] {device}")

model_dir = args.model_dir
if model_dir is None:
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download("convaiinnovations/laya-multilingual", allow_patterns=[
        "model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/*"])

with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
    cfg = json.load(f)
label = args.model_dir or "base-0.3.21"
print(f"[model] {label}")

tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"), pretrained=False)
model.load_state_dict({k: v.float() for k, v in
                       load_file(os.path.join(model_dir, "model.safetensors")).items()},
                      strict=False)
model = model.float().to(device).eval()

temps = cfg.get("temperature", [1.0, 1.0, 1.0])

QUESTIONS = {
    "bid_action": {"t": "choice",
                   "ins": "根据当前投放数据,下一步应该采取什么出价动作?",
                   "crit": {
                       "raise_bid": "成本达标但量不足,提高出价抢量",
                       "keep": "成本与量都在合理区间,维持当前出价",
                       "lower_bid": "实际成本超出目标且转化样本足够,降低出价控成本",
                       "pause": "持续无转化或严重超支,暂停计划止损",
                       "add_budget": "成本达标且预算提前耗尽,追加预算放量"}},
    "needs_adjustment": {"t": "noul",
                         "ins": "当前是否需要对出价进行调整(提价/降价/暂停)?",
                         "crit": {"false": "不需要调整", "true": "需要调整"}},
}
ACTION_LABELS = list(QUESTIONS["bid_action"]["crit"].keys())


def infer(state, question):
    ids, markers = build_sequence(tok, state, question,
                                  cfg["max_len"], cfg["head_max_len"])
    ids_t = torch.tensor([ids]).to(device)
    att = torch.ones_like(ids_t)
    mpos = torch.tensor([markers]).to(device)
    mmask = torch.ones_like(mpos, dtype=torch.bool).to(device)
    qtype = torch.tensor([QTYPES[question["t"]]]).to(device)
    with torch.no_grad():
        logits, _ = model(ids_t, att, mpos, mmask, qtype)
    row = logits[0].cpu().numpy()
    row = row / temps[QTYPES[question["t"]]]
    row = row - row.max()
    e = np.exp(row)
    return e / e.sum()


# (state, expected_action, expected_needs_adjustment)
TESTS = [
    ({"计划": "测试-A", "时段": "14:00", "日预算(元)": 1000, "已消耗(元)": 500, "预算消耗比例": "50%",
      "展示量": 20000, "点击量": 100, "CTR": "0.5%", "转化数": 0,
      "目标转化成本(元)": 50, "实际转化成本(元)": 180.0}, "pause", True),
    ({"计划": "测试-B", "时段": "15:00", "日预算(元)": 800, "已消耗(元)": 780, "预算消耗比例": "98%",
      "展示量": 80000, "点击量": 2400, "CTR": "3.0%", "转化数": 18,
      "目标转化成本(元)": 50, "实际转化成本(元)": 43.0}, "add_budget", False),
    ({"计划": "测试-C", "时段": "12:00", "日预算(元)": 1500, "已消耗(元)": 900, "预算消耗比例": "60%",
      "展示量": 40000, "点击量": 800, "CTR": "2.0%", "转化数": 8,
      "目标转化成本(元)": 50, "实际转化成本(元)": 112.5}, "lower_bid", True),
    ({"计划": "测试-D", "时段": "16:00", "日预算(元)": 1000, "已消耗(元)": 400, "预算消耗比例": "40%",
      "展示量": 30000, "点击量": 600, "CTR": "2.0%", "转化数": 4,
      "目标转化成本(元)": 50, "实际转化成本(元)": 100.0}, "lower_bid", True),
    ({"计划": "测试-E", "时段": "10:00", "日预算(元)": 1000, "已消耗(元)": 200, "预算消耗比例": "20%",
      "展示量": 10000, "点击量": 200, "CTR": "2.0%", "转化数": 1,
      "目标转化成本(元)": 50, "实际转化成本(元)": 100.0}, "keep", False),
    ({"计划": "测试-F", "时段": "20:00", "日预算(元)": 1000, "已消耗(元)": 150, "预算消耗比例": "15%",
      "展示量": 5000, "点击量": 20, "CTR": "0.4%", "转化数": 2,
      "目标转化成本(元)": 50, "实际转化成本(元)": 20.0}, "raise_bid", True),
    ({"计划": "测试-G", "时段": "19:00", "日预算(元)": 1000, "已消耗(元)": 500, "预算消耗比例": "50%",
      "展示量": 50000, "点击量": 1500, "CTR": "3.0%", "转化数": 10,
      "目标转化成本(元)": 50, "实际转化成本(元)": 50.0}, "keep", False),
    ({"计划": "测试-H", "时段": "21:00", "日预算(元)": 500, "已消耗(元)": 420, "预算消耗比例": "84%",
      "展示量": 30000, "点击量": 150, "CTR": "0.5%", "转化数": 0,
      "目标转化成本(元)": 40, "实际转化成本(元)": 95.0}, "pause", True),
]

n_action = n_adj = 0
print(f"\n{'=' * 72}\n广告出价决策评测 [{label}]\n{'=' * 72}")
for i, (state, exp_action, exp_adj) in enumerate(TESTS, 1):
    p_action = infer(state, QUESTIONS["bid_action"])
    pred_action = ACTION_LABELS[int(p_action.argmax())]
    p_adj = infer(state, QUESTIONS["needs_adjustment"])
    pred_adj = bool(p_adj[1] >= 0.5)
    ok_a, ok_n = pred_action == exp_action, pred_adj == exp_adj
    n_action += ok_a
    n_adj += ok_n
    print(f"{'✅' if ok_a else '❌'} #{i} action={pred_action:11s} exp={exp_action:11s}"
          f" (p={p_action.max():.2f}) | "
          f"{'✅' if ok_n else '❌'} adj={str(pred_adj):5s} exp={str(exp_adj):5s}"
          f" (P={p_adj[1]:.2f})")

print(f"\n结果: bid_action {n_action}/{len(TESTS)} = {n_action/len(TESTS):.0%} | "
      f"needs_adjustment {n_adj}/{len(TESTS)} = {n_adj/len(TESTS):.0%}")
