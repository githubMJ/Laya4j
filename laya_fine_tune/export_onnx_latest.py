"""
Laya multilingual → ONNX 导出(上游 0.3.5 最新代码版)
基于 export_onnx.py,差异:
  1. laya 包来自 laya-upstream editable 安装(0.3.5 最新 main)
  2. 快照路径参数化,自动取 HF 缓存中的最新快照
  3. 产物带版本号命名,直接可放入 Laya4j/models/
"""
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
import laya
from laya.common import (
    DecisionModel, build_model, build_sequence, QTYPES, collate_items,
)

SNAP = Path(snapshot_download(
    "convaiinnovations/laya", allow_patterns=["multilingual/*"])) / "multilingual"
EXPORT_DIR = Path("laya_onnx_export")
EXPORT_DIR.mkdir(exist_ok=True)
import importlib.metadata as im
LAYA_VER = im.version("laya")
SNAP_ID = SNAP.parent.name[:7]
ONNX_NAME = f"laya-decision-multilingual-mmbert-base-v{LAYA_VER}-{SNAP_ID}.onnx"
print(f"[export] laya={LAYA_VER} snapshot={SNAP_ID}")

# ============ 1. 加载模型 + 权重 ============
print("=" * 60)
print("[1] 加载 DecisionModel + 真实权重")
print("=" * 60)
cfg = json.loads((SNAP / "rl_agent_config.json").read_text())
model = build_model(cfg)
sd = load_file(str(SNAP / "model.safetensors"))
model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
model = model.float().eval()
tokenizer = AutoTokenizer.from_pretrained(str(SNAP / "tokenizer"))
print(f"  cfg: {cfg.get('encoder')}, head_layers={cfg['head_layers']}, "
      f"temperature={model.temperature.tolist()}")

# ============ 2. 测试 schema ============
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

# ============ 3. PyTorch forward ============
print("\n" + "=" * 60)
print("[3] PyTorch forward")
print("=" * 60)
with torch.no_grad():
    pt_logits, pt_act = model(
        batch["input_ids"], batch["attention_mask"],
        batch["marker_pos"], batch["marker_mask"], batch["qtype"],
    )
print(f"  logits: {tuple(pt_logits.shape)}  act: {tuple(pt_act.shape)}")

# ============ 4. 导出 ONNX(dynamo) ============
print("\n" + "=" * 60)
print("[4] 导出 ONNX")
print("=" * 60)
onnx_path = EXPORT_DIR / ONNX_NAME

class W(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        return self.m(input_ids, attention_mask, marker_pos, marker_mask, qtype)

w = W(model).eval()

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

# 内嵌 external data
from onnx import external_data_helper
m_onnx = onnx.load(str(onnx_path))
external_data_helper.load_external_data_for_model(m_onnx, str(EXPORT_DIR))
onnx.save(m_onnx, str(onnx_path))
(EXPORT_DIR / (ONNX_NAME + ".data")).unlink(missing_ok=True)
print(f"  OK {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB)")

# ============ 5. 修复 Split num_outputs → Slice ============
print("\n" + "=" * 60)
print("[5] 修复 Split 节点(num_outputs → Slice)")
print("=" * 60)
graph = gs.import_onnx(onnx.load(str(onnx_path)))

patched = 0
nodes_to_remove = []
for node in graph.nodes:
    if node.op != "Split":
        continue
    num_outputs = node.attrs.get("num_outputs")
    if num_outputs is None:
        continue
    inp = node.inputs[0]
    matmul = next((p for p in inp.inputs if p.op == "MatMul"), None)
    last_dim = 0
    if matmul is not None:
        weight = matmul.inputs[1]
        if hasattr(weight, 'shape') and weight.shape and len(weight.shape) >= 2:
            last_dim = weight.shape[1]
    if last_dim == 0:
        last_dim = num_outputs * 1152
    split_size = last_dim // num_outputs

    nodes_to_remove.append(node)
    for i, out in enumerate(node.outputs):
        start = i * split_size
        end = (i + 1) * split_size
        slice_node = gs.Node(
            op="Slice",
            name=f"{node.name}_slice_{i}",
            inputs=[
                inp,
                gs.Constant(name=f"{node.name}_start_{i}", values=np.array([start], dtype=np.int64)),
                gs.Constant(name=f"{node.name}_end_{i}",   values=np.array([end],   dtype=np.int64)),
                gs.Constant(name=f"{node.name}_axes_{i}",  values=np.array([-1],    dtype=np.int64)),
                gs.Constant(name=f"{node.name}_step_{i}",  values=np.array([1],     dtype=np.int64)),
            ],
            outputs=[out],
        )
        graph.nodes.append(slice_node)
    patched += 1

for n in nodes_to_remove:
    n.inputs.clear()
    n.outputs.clear()
graph.cleanup()
onnx.save(gs.export_onnx(graph), str(onnx_path))
print(f"  OK 修复 {patched} 个 Split")

# ============ 6. ONNX Runtime 验证 ============
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
print(f"  act max diff:    {diff_a:.2e}")
print(f"  {'PASS 数值一致' if diff_l < 1e-3 else 'WARN 误差偏大'}")

# ============ 7. ONNX 输出 → 决策 ============
print("\n" + "=" * 60)
print("[7] ONNX 输出 → 决策")
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

# ============ 8. 附带文件 ============
print("\n" + "=" * 60)
print("[8] 保存 tokenizer + Java config")
print("=" * 60)
tok_dir = EXPORT_DIR / "tokenizer"
tok_dir.mkdir(exist_ok=True)
tokenizer.save_pretrained(str(tok_dir))

java_cfg = {
    "model_path":   ONNX_NAME,
    "tokenizer_dir": "tokenizer/",
    "max_len":      max_len,
    "head_max_len": head_max_len,
    "qtypes":       QTYPES,
    "temperature":  model.temperature.tolist(),
}
(EXPORT_DIR / "java_config.json").write_text(json.dumps(java_cfg, indent=2))

print(f"\nDONE 导出完成: {EXPORT_DIR.absolute()}")
print(f"   - {ONNX_NAME} ({onnx_path.stat().st_size/1e6:.1f} MB)")
print(f"   - tokenizer/")
print(f"   - java_config.json")
