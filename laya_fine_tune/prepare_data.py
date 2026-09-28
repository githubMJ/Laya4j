"""
Prepare training items for Laya Chinese fine-tuning.

Reads a JSONL dataset in the upstream `LocalLLaMA/typed-decisions` format:
    {"state": {...}, "questions": {qid: {...}}, "gold": {qid: {"probabilities": {...}}}}

and tokenizes every (state, question) pair into the flat item format expected
by train.py:
    {"ids": [...], "markers": [...], "qtype": int, "target": [floats], "label": int}

Usage:
    python prepare_data.py --data data/finetune_zh.jsonl --out train_items.pt
    python prepare_data.py --data data/finetune_zh.jsonl --val-items 40   # hold out a val slice
"""
import argparse
import json
import os
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

from laya.common import build_sequence, render_options, QTYPES

DEFAULT_REPO = "convaiinnovations/laya-multilingual"


def build_training_item(tok, cfg, state, q, gold_q):
    """Tokenize one (state, question, gold) triple into a flat training item."""
    t = q["type"]
    crit = q.get("criteria", q.get("crit", {}))
    probs = gold_q["probabilities"]

    if t == "choice":
        keys = list(crit.keys())
        target = [probs.get(k, 0.0) for k in keys]
    elif t == "noul":
        target = [probs.get("false", 0.5), probs.get("true", 0.5)]
    elif t == "score":
        n_levels = len(crit) if isinstance(crit, list) else 4
        target = [probs.get(str(i), 0.0) for i in range(n_levels)]
    else:
        return None

    s = sum(target)
    target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
    label = target.index(max(target))

    expected_markers = len(render_options({"t": t, "crit": crit}))
    seq, markers = build_sequence(
        tok, state,
        {"t": t, "ins": q["instructions"], "crit": crit},
        cfg["max_len"], cfg["head_max_len"],
    )
    if len(markers) != expected_markers:
        return None
    return {"ids": seq, "markers": markers, "qtype": QTYPES[t],
            "target": target, "label": label}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="JSONL dataset in typed-decisions format")
    ap.add_argument("--out", default="train_items.pt", help="output .pt file")
    ap.add_argument("--model-dir", default=None,
                    help=f"checkpoint dir (snapshot layout); default: HF download of {DEFAULT_REPO}")
    ap.add_argument("--val-items", type=int, default=0,
                    help="hold out N items as validation (saved to <out>.val.pt)")
    ap.add_argument("--max-items", type=int, default=0, help="cap total items (0 = no cap)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model_dir = args.model_dir or snapshot_download(DEFAULT_REPO, allow_patterns=[
        "rl_agent_config.json", "encoder/config.json", "tokenizer/*"])
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)

    items, skipped = [], 0
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            state = row["state"] if isinstance(row["state"], dict) else json.loads(row["state"])
            questions = row["questions"] if isinstance(row["questions"], dict) else json.loads(row["questions"])
            gold = row["gold"] if isinstance(row["gold"], dict) else json.loads(row["gold"])
            for qid, q in questions.items():
                if qid not in gold:
                    continue
                it = build_training_item(tok, cfg, state, q, gold[qid])
                if it:
                    items.append(it)
                else:
                    skipped += 1
            if args.max_items and sum(1 for _ in items) >= args.max_items * 3:
                items = items[:args.max_items]
                break

    random.Random(args.seed).shuffle(items)
    if args.max_items:
        items = items[:args.max_items]

    val_items = []
    if args.val_items > 0:
        val_items, items = items[:args.val_items], items[args.val_items:]

    torch.save(items, args.out)
    print(f"Saved {len(items)} train items -> {args.out}"
          + (f" (+{len(val_items)} val -> {args.out}.val.pt)" if val_items else "")
          + (f" | skipped {skipped} malformed" if skipped else ""))

    by_type = {}
    for it in items:
        by_type[it["qtype"]] = by_type.get(it["qtype"], 0) + 1
    names = {v: k for k, v in QTYPES.items()}
    print("By question type:", {names[k]: v for k, v in sorted(by_type.items())})


if __name__ == "__main__":
    main()
