"""
Single-device RLCD fine-tuning for Laya (MPS / CUDA / CPU adaptive).

Extracted from the upstream Kaggle notebook (DDP variant) and adapted to run
on one device, so Chinese fine-tuning can be done locally on Apple Silicon.

Algorithm: RLCD-style policy gradient with proper-scoring-rule rewards
(log + spherical + ranked probability score) plus a soft cross-entropy
guidance term, followed by temperature calibration on a held-out slice.

Usage:
    python prepare_data.py --data data/finetune_zh.jsonl --out train_items.pt
    python train.py --data train_items.pt --out out_zh --epochs 3

After training, export with:
    python export_onnx.py --repo <hf-repo-of-the-checkpoint> --out-dir ../models
"""
import argparse
import json
import os
import random
import time

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from laya.common import build_model, proper_reward

CALIB_MAX = 400      # calibration slice size (never trained on)
LR_ENCODER = 2.5e-5
LR_HEAD = 1.0e-4
SIGMA_START = 0.4    # exploration noise at epoch 0
SIGMA_END = 0.1      # exploration noise at final epoch


def collate_train_batch(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids, "attention_mask": att,
        "marker_pos": mpos, "marker_mask": mmask, "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
    }


def fit_one_temp(sel):
    """Fit softmax temperature on held-out calibration logits (LBFGS)."""
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, :len(z)] = torch.tensor(z)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=None,
                    help="base checkpoint dir (snapshot layout); omit to download "
                         "convaiinnovations/laya-multilingual")
    ap.add_argument("--data", default="train_items.pt", help="preprocessed items .pt")
    ap.add_argument("--out", default="out_finetuned", help="output checkpoint dir")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device_s = args.device if args.device != "auto" else pick_device()
    device = torch.device(device_s)
    print(f"[device] {device_s}")

    # ---------- model ----------
    model_dir = args.model_dir
    if model_dir is None:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download("convaiinnovations/laya-multilingual", allow_patterns=[
            "model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/*"])

    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["gradient_checkpointing"] = True

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)

    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device)
    model.train()

    # ---------- data ----------
    all_items = torch.load(args.data, weights_only=False)
    order = list(range(len(all_items)))
    random.Random(20260922).shuffle(order)  # fixed split, independent of --seed
    n_calib = min(CALIB_MAX, max(1, len(all_items) // 10))
    calib_items = [all_items[i] for i in sorted(order[:n_calib])]
    train_items = [all_items[i] for i in sorted(order[n_calib:])]
    print(f"{len(train_items)} train items ({len(calib_items)} held out for calibration)")

    micro, accum = args.micro_batch, args.grad_accum

    enc_params = [p for n, p in model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": LR_ENCODER},
        {"params": head_params, "lr": LR_HEAD},
    ], weight_decay=0.01)

    total_updates = max(1, (len(train_items) // (micro * accum))) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=1e-6)

    # CUDA: fp16 autocast + GradScaler. MPS: fp32 (keeps RLCD policy gradients
    # well-behaved; autocast on MPS is unstable for this architecture).
    use_fp16 = device_s == "cuda"
    scaler = torch.amp.GradScaler(device_s, enabled=use_fp16)

    t0 = time.time()
    for epoch in range(args.epochs):
        random.shuffle(train_items)
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        accum_step = 0

        progress = epoch / max(1, args.epochs - 1)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * progress

        for b_idx in range(0, len(train_items), micro):
            chunk = train_items[b_idx:b_idx + micro]
            if not chunk:
                continue
            batch = collate_train_batch(chunk, tok.pad_token_id)
            dev_batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(device_s, dtype=torch.float16, enabled=use_fp16):
                logits, act = model(
                    dev_batch["input_ids"], dev_batch["attention_mask"],
                    dev_batch["marker_pos"], dev_batch["marker_mask"], dev_batch["qtype"],
                )

            logits = logits.float()
            mask = dev_batch["marker_mask"]
            k = mask.sum(-1, keepdim=True).float()
            target = dev_batch["target"]

            # 1. Sample G noisy logit distributions with zero-mean projection
            eps = torch.randn((4,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

            # 2. Proper scoring reward (w_sph=0.75 for soft targets)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), dev_batch["qtype"], mask,
                                  w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            # 3. Policy gradient loss + soft cross-entropy guidance
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + 1.0 * loss_ce) / accum + 0.0 * act.sum()

            scaler.scale(loss).backward()
            accum_step += 1

            if accum_step % accum == 0 or (b_idx + micro) >= len(train_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * accum
            n_batches += 1
            if n_batches % 20 == 0:
                print(f"  epoch {epoch+1}/{args.epochs} | step {n_batches} | "
                      f"loss {loss.item()*accum:.4f} | reward {r.mean().item():.3f} | "
                      f"lr {scheduler.get_last_lr()[0]:.2e}")

        print(f"=== epoch {epoch+1}/{args.epochs} done in {time.time()-t0:.1f}s | "
              f"avg loss {epoch_loss/max(1, n_batches):.4f} ===")

        # rolling checkpoint after each epoch
        ckpt_dir = os.path.join(args.out, "checkpoint_latest")
        os.makedirs(ckpt_dir, exist_ok=True)
        sd = {kk: v.half().contiguous().cpu() for kk, v in model.state_dict().items()}
        save_file(sd, os.path.join(ckpt_dir, "model.safetensors"))
        model.encoder.config.save_pretrained(os.path.join(ckpt_dir, "encoder"))
        tok.save_pretrained(os.path.join(ckpt_dir, "tokenizer"))

    # ---------- post-training temperature calibration ----------
    print("\nFitting post-training calibration temperatures...")
    del optimizer, scaler, scheduler
    if device_s == "cuda":
        torch.cuda.empty_cache()
    model.eval()
    calib_preds = []
    with torch.no_grad():
        for c_idx in range(0, len(calib_items), 16):
            c_chunk = calib_items[c_idx:c_idx + 16]
            cb = collate_train_batch(c_chunk, tok.pad_token_id)
            with torch.autocast(device_s, dtype=torch.float16, enabled=use_fp16):
                l_sub, _ = model(
                    cb["input_ids"].to(device), cb["attention_mask"].to(device),
                    cb["marker_pos"].to(device), cb["marker_mask"].to(device),
                    cb["qtype"].to(device),
                )
            l_np = l_sub.float().cpu().numpy()
            for r_i, it in enumerate(c_chunk):
                kk = len(it["markers"])
                calib_preds.append((it["qtype"], l_np[r_i, :kk], it["target"]))

    fitted_temps = [1.2, 1.2, 1.2]
    for qt in range(3):
        sel = [(z, t) for q_type, z, t in calib_preds if q_type == qt]
        if sel:
            fitted_temps[qt] = fit_one_temp(sel)
    print("Fitted temperatures (choice, score, noul):", [round(t, 3) for t in fitted_temps])

    os.makedirs(args.out, exist_ok=True)
    sd = {kk: v.half().contiguous().cpu() for kk, v in model.state_dict().items()}
    save_file(sd, os.path.join(args.out, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(args.out, "encoder"))
    tok.save_pretrained(os.path.join(args.out, "tokenizer"))

    cfg["fine_tuned"] = True
    cfg["temperature"] = fitted_temps
    cfg.pop("temperature_by_options", None)  # per-type fit; inherited overrides would hide it
    with open(os.path.join(args.out, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Fine-tuned model saved to {args.out}")


if __name__ == "__main__":
    main()
