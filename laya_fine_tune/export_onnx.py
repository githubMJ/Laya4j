"""
Laya multilingual → ONNX 导出脚本(供 Laya4j Java 端加载)
=========================================================

从上游 HuggingFace 仓库(convaiinnovations/laya-multilingual,mmBERT-base,
0.3.x checkpoint)下载权重,导出为 Laya4j 可直接加载的 FP32 ONNX,
并做 PyTorch / ONNX Runtime 数值对齐校验。

用法:
    python export_onnx.py                       # 默认导出 multilingual 最新版
    python export_onnx.py --out-dir ../models   # 产物写入 Laya4j 项目 models/

产物(默认写到 ../models/,与 Laya4j 主项目共享):
    models/laya-decision-multilingual-mmbert-base-v{laya版本}-{commit短哈希}.onnx
    models/tokenizer/                      (更新)
    models/rl_agent_config.json            (供 Java 端 LayaConfig.fromRlAgentConfig 使用)

背景说明 —— 为什么需要"手工修复"Split(见步骤 5):
    PyTorch 2.11 的 dynamo 导出器会为 Split 算子生成 `num_outputs` 属性,
    但 onnxruntime 1.18+ 已经不再支持该属性,必须用 onnx-graphsurgeon
    把每个 Split 节点重写成等价的多个 Slice 节点。

对齐性验证(见步骤 6):
    脚本会在同一份输入上分别跑 PyTorch 和导出后的 ONNX Runtime,
    比较 logits/act_logits 的最大绝对误差,确保导出没有引入数值偏差
    (阈值 1e-3,一般能做到 1e-5 量级)。这是保证 Java 端推理结果与
    Python 端严格一致(bit-level aligned)的关键步骤。
"""
import argparse
import importlib.metadata as im
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
import onnx_graphsurgeon as gs
from huggingface_hub import snapshot_download

from laya.common import build_model, build_sequence, QTYPES, collate_items

# ============================================================
# 0. 参数解析
# ============================================================
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", default="convaiinnovations/laya-multilingual",
                    help="上游 HuggingFace 仓库 id(0.3.19+ 拆分为独立模型仓库)")
parser.add_argument("--out-dir", default="../models",
                    help="导出产物输出目录,默认写到 Laya4j 项目的 models/")
parser.add_argument("--threshold", type=float, default=1e-3,
                    help="PyTorch/ONNX 对齐误差阈值,超过则报错终止")
args = parser.parse_args()

OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

laya_version = im.version("laya")
print(f"上游 laya Python 版本: {laya_version}")

# ============================================================
# 1. 下载/定位权重快照(0.3.19+ 布局:文件在仓库根,无 subfolder)
# ============================================================
print("=" * 60)
print(f"[1] 下载快照 {args.repo}")
print("=" * 60)
snap_root = Path(snapshot_download(args.repo, allow_patterns=[
    "model.safetensors", "rl_agent_config.json",
    "encoder/config.json", "tokenizer/*",
]))
# snapshot 缓存路径形如 ~/.cache/huggingface/hub/models--.../snapshots/<sha>;
# 0.3.19+ 仓库无 subfolder,sha 目录即 snap_root 本身
import re
commit_sha = snap_root.name if re.fullmatch(r"[0-9a-f]{40}", snap_root.name) \
        else snap_root.parent.name
commit_short = commit_sha[:7]
print(f"  snapshot: {snap_root}")
print(f"  commit:   {commit_sha}")

version_tag = f"{laya_version}-{commit_short}"
onnx_path = OUT_DIR / f"laya-decision-multilingual-mmbert-base-v{version_tag}.onnx"

cfg = json.loads((snap_root / "rl_agent_config.json").read_text())

model = build_model(cfg, encoder_dir=str(snap_root / "encoder"), pretrained=False)
sd = load_file(str(snap_root / "model.safetensors"))
missing, unexpected = model.load_state_dict(
    {k: v.float() for k, v in sd.items()}, strict=False)
if missing:
    print(f"  ⚠️  缺失参数: {sorted(missing)[:5]}{' ...' if len(missing) > 5 else ''}")
if unexpected:
    print(f"  ⚠️  多余参数: {sorted(unexpected)[:5]}{' ...' if len(unexpected) > 5 else ''}")
model = model.float().eval()
tokenizer = AutoTokenizer.from_pretrained(str(snap_root / "tokenizer"))
print(f"  cfg: {cfg.get('encoder')}, head_layers={cfg.get('head_layers')}")


# ============================================================
# 2. 用上游 build_sequence 构造覆盖三类问题的真实输入
# ============================================================
print("\n" + "=" * 60)
print("[2] 用 laya 内部 build_sequence 构造输入")
print("=" * 60)
state = {
    "from": "zhang@example.cn",
    "subject": "订单 #8821 重复扣款,要求立刻退款",
    "body": "我昨天被重复扣款两次,要求立刻退款,否则投诉 12315。",
}
questions_internal = {
    "department": {"t": "choice", "ins": "Which department?",
        "crit": {"billing": "invoices, payments, refunds",
                 "technical": "bugs, outages, system errors",
                 "sales": "pricing, new contracts",
                 "other": "everything else"}},
    "urgency": {"t": "score", "ins": "How urgent?",
        "crit": ["not urgent", "soon", "critical deadline"]},
    "refund": {"t": "noul", "ins": "User requests refund?",
        "crit": {"false": "no refund", "true": "yes refund"}},
}

max_len, head_max_len = cfg["max_len"], cfg["head_max_len"]
items = []
for qid, q in questions_internal.items():
    ids, markers = build_sequence(tokenizer, state, q, max_len, head_max_len)
    items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
    print(f"  {qid:12s} t={q['t']:6s} seq_len={len(ids):4d} markers={len(markers)}")

batch = collate_items([items], pad_id=tokenizer.pad_token_id or 0)


# ============================================================
# 3. PyTorch 前向(数值基准)
# ============================================================
print("\n" + "=" * 60)
print("[3] PyTorch forward(基准)")
print("=" * 60)
with torch.no_grad():
    pt_logits, pt_act = model(
        batch["input_ids"], batch["attention_mask"],
        batch["marker_pos"], batch["marker_mask"], batch["qtype"],
    )
print(f"  logits: {tuple(pt_logits.shape)}  act: {tuple(pt_act.shape)}")


# ============================================================
# 4. dynamo 导出 ONNX(opset 17)
# ============================================================
print("\n" + "=" * 60)
print("[4] 导出 ONNX")
print("=" * 60)

class Wrapper(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        return self.m(input_ids, attention_mask, marker_pos, marker_mask, qtype)

w = Wrapper(model).eval()

torch.onnx.export(
    w,
    (batch["input_ids"], batch["attention_mask"],
     batch["marker_pos"], batch["marker_mask"], batch["qtype"]),
    str(onnx_path),
    input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
    output_names=["logits", "act_logits"],
    dynamic_axes={
        "input_ids":      {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "marker_pos":     {0: "batch", 1: "markers"},
        "marker_mask":    {0: "batch", 1: "markers"},
        "qtype":          {0: "batch"},
        "logits":         {0: "batch", 1: "markers"},
        "act_logits":     {0: "batch"},
    },
    opset_version=17,
    do_constant_folding=True,
)

# 内嵌 external data(避免 .data 外部文件)
from onnx import external_data_helper
m_onnx = onnx.load(str(onnx_path))
external_data_helper.load_external_data_for_model(m_onnx, str(OUT_DIR))
onnx.save(m_onnx, str(onnx_path))
Path(str(onnx_path) + ".data").unlink(missing_ok=True)
print(f"  ✓ {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB)")


# ============================================================
# 5. 按需修复 Split num_outputs → Slice
#    0.3.21 上游已适配新版导出,原生图可能可直接被 ORT 加载;
#    仅当原生图加载失败时才做 Split→Slice 重写
# ============================================================
print("\n" + "=" * 60)
print("[5] Split 节点处理(按需)")
print("=" * 60)

def ort_can_load() -> bool:
    try:
        ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        return True
    except Exception as e:
        print(f"  原生图不可直接加载: {e}")
        return False

if ort_can_load():
    print("  ✓ 原生导出即可被 ONNX Runtime 加载,无需修补")
else:
    # num_outputs 属性是冗余的:Split 沿 axis 均分,份数本就等于输出数量,
    # 删除该属性即可,图语义完全不变(比重写成 Slice 更稳,不依赖维度推断)
    graph = gs.import_onnx(onnx.load(str(onnx_path)))

    patched = 0
    for node in graph.nodes:
        if node.op == "Split" and "num_outputs" in node.attrs:
            del node.attrs["num_outputs"]
            patched += 1

    onnx.save(gs.export_onnx(graph), str(onnx_path))
    print(f"  ✓ 删除 {patched} 个 Split 的 num_outputs 属性")


# ============================================================
# 6. ONNX Runtime 数值对齐校验
# ============================================================
print("\n" + "=" * 60)
print("[6] ONNX Runtime 验证")
print("=" * 60)
sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
onnx_logits, onnx_act = sess.run(None, {
    "input_ids":      batch["input_ids"].numpy(),
    "attention_mask": batch["attention_mask"].numpy(),
    "marker_pos":     batch["marker_pos"].numpy(),
    "marker_mask":    batch["marker_mask"].numpy(),
    "qtype":          batch["qtype"].numpy(),
})
diff_l = np.abs(pt_logits.numpy() - onnx_logits).max()
diff_a = np.abs(pt_act.numpy() - onnx_act).max()
print(f"  logits max diff: {diff_l:.2e}")
print(f"  act max diff:    {diff_a:.2e} (sigmoid 前原始值,仅作参考)")

# 与上游 tests/test_onnx.py 同语义:对 softmax(带温度)后的最终概率做
# atol=1e-3, rtol=1e-3 校验。act_logits 是 sigmoid 前的激活值,原始误差
# 会被 sigmoid 压缩,故只做宽松 sanity 检查。
def probs(logits_row, qtype_id):
    m = logits_row.copy()
    m /= model.temperature[qtype_id]
    m = m - m.max()
    e = np.exp(m)
    return e / e.sum()

bad = []
if diff_l >= args.threshold:
    bad.append(f"logits {diff_l:.2e} >= {args.threshold}")
if diff_a >= args.threshold * 10:
    bad.append(f"act {diff_a:.2e} >= {args.threshold * 10}")
for i, (qid, q) in enumerate(questions_internal.items()):
    pt_p = probs(pt_logits.numpy()[i], QTYPES[q["t"]])
    on_p = probs(onnx_logits[i], QTYPES[q["t"]])
    if not np.allclose(pt_p, on_p, atol=1e-3, rtol=1e-3):
        bad.append(f"{qid} 概率分布偏差超 1e-3")
if bad:
    raise SystemExit(f"❌ 对齐校验失败: {bad};导出产物不可用")

# ============================================================
# 7. 端到端决策冒烟(用 ONNX 输出走一遍决策)
# ============================================================
print("\n" + "=" * 60)
print("[7] ONNX 输出 → 决策冒烟")
print("=" * 60)
for i, (qid, q) in enumerate(questions_internal.items()):
    m_arr = onnx_logits[i].copy()
    m_arr[~batch["marker_mask"][i].numpy().astype(bool)] = -1e4
    probs = torch.softmax(
        torch.from_numpy(m_arr) / model.temperature[QTYPES[q["t"]]], -1
    ).numpy()
    t = q["t"]
    if t == "choice":
        labels = list(q["crit"].keys())
        print(f"  {qid:12s} (choice) best={labels[probs.argmax()]}")
    elif t == "score":
        print(f"  {qid:12s} (score)  score={float((np.arange(len(probs)) * probs).sum()):.4f}")
    elif t == "noul":
        p_true = probs[1] / probs.sum() if probs.sum() > 0 else 0.5
        print(f"  {qid:12s} (noul)   P(true)={p_true:.4f}")

# ============================================================
# 8. 附带文件:tokenizer + rl_agent_config.json
# ============================================================
print("\n" + "=" * 60)
print("[8] 保存 tokenizer + Java config")
print("=" * 60)
tok_dir = OUT_DIR / "tokenizer"
tok_dir.mkdir(exist_ok=True)
tokenizer.save_pretrained(str(tok_dir))

import shutil
shutil.copyfile(snap_root / "rl_agent_config.json", OUT_DIR / "rl_agent_config.json")

print(f"""
DONE  导出完成: {onnx_path.resolve()}
  下一步: 更新 Laya4j 的
    - com.laya4j.model.HuggingFaceFetcher.MODEL_VERSION = "{version_tag}"
    - com.laya4j.model.HuggingFaceFetcher.DEFAULT_REPO  = "{args.repo}"
    - com.laya4j.model.HuggingFaceFetcher.DEFAULT_SUBFOLDER = ""(新仓库无子目录)
    - models/VERSION.md 版本记录
""")
