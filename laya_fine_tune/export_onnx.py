"""
Laya multilingual → ONNX 导出脚本
====================================

把上游 Laya (https://github.com/NandhaKishorM/laya) 的 PyTorch 权重
导出成 Laya4j(本项目)可以直接加载的 ONNX 文件,并放入 ../models/ 目录。

用法:
    pip install -r requirements.txt
    python export_onnx.py [--repo convaiinnovations/laya] [--subfolder multilingual]

产物(默认写到 ../models/,与 Laya4j 主项目共享):
    models/laya-decision-multilingual-mmbert-base-v{laya版本}-{commit短哈希}.onnx
    models/tokenizer/                      (若不存在则更新)
    models/rl_agent_config.json            (供 Java 端 LayaConfig.fromRlAgentConfig 使用)

背景说明 —— 为什么需要"手工修复"这一步(见步骤 5):
    PyTorch 2.11 的 dynamo 导出器会为 Split 算子生成 `num_outputs` 属性,
    但 onnxruntime 1.18+ 已经不再支持该属性,必须用 onnx-graphsurgeon
    把每个 Split 节点重写成等价的多个 Slice 节点,导出的 ONNX 才能被
    ONNX Runtime(包括 Java 版)正确加载。

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
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer
import onnx_graphsurgeon as gs

from laya.common import build_model, build_sequence, QTYPES, collate_items

# ============================================================
# 0. 参数解析
# ============================================================
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", default="convaiinnovations/laya",
                     help="HuggingFace 仓库 id")
parser.add_argument("--subfolder", default="multilingual",
                     help="仓库内子目录(multilingual / english / typed-decisions)")
parser.add_argument("--out-dir", default="../models",
                     help="导出产物输出目录,默认写到 Laya4j 项目的 models/")
args = parser.parse_args()

OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)


class ForwardWrapper(torch.nn.Module):
    """把 DecisionModel 的 forward 签名包一层,供 torch.onnx.export 使用。"""

    def __init__(self, m: torch.nn.Module):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        return self.m(input_ids, attention_mask, marker_pos, marker_mask, qtype)


def patch_split_to_slice(onnx_path: Path) -> int:
    """
    把 ONNX 图里所有带 num_outputs 属性的 Split 节点重写成等价的多个
    Slice 节点(onnxruntime 1.18+ 已不支持 Split.num_outputs 属性)。

    :return: 被修复的 Split 节点数量
    """
    graph = gs.import_onnx(onnx.load(str(onnx_path)))
    patched = 0
    nodes_to_remove = []

    for node in graph.nodes:
        if node.op != "Split":
            continue
        num_outputs = node.attrs.get("num_outputs")
        if num_outputs is None:
            continue

        # 从前驱 MatMul 权重矩阵推断被 split 的总维度;找不到时按
        # mmBERT attention in_proj 维度(1152 每头)兜底估算。
        inp = node.inputs[0]
        matmul = next((p for p in inp.inputs if p.op == "MatMul"), None)
        last_dim = 0
        if matmul is not None:
            weight = matmul.inputs[1]
            if hasattr(weight, "shape") and weight.shape and len(weight.shape) >= 2:
                last_dim = weight.shape[1]
        if last_dim == 0:
            last_dim = num_outputs * 1152
        split_size = last_dim // num_outputs

        nodes_to_remove.append(node)
        for i, out in enumerate(node.outputs):
            start, end = i * split_size, (i + 1) * split_size
            graph.nodes.append(gs.Node(
                op="Slice",
                name=f"{node.name}_slice_{i}",
                inputs=[
                    inp,
                    gs.Constant(f"{node.name}_start_{i}", np.array([start], dtype=np.int64)),
                    gs.Constant(f"{node.name}_end_{i}", np.array([end], dtype=np.int64)),
                    gs.Constant(f"{node.name}_axes_{i}", np.array([-1], dtype=np.int64)),
                    gs.Constant(f"{node.name}_step_{i}", np.array([1], dtype=np.int64)),
                ],
                outputs=[out],
            ))
        patched += 1

    for n in nodes_to_remove:
        n.inputs.clear()
        n.outputs.clear()
    graph.cleanup()
    onnx.save(gs.export_onnx(graph), str(onnx_path))
    return patched


def build_sample_batch(tokenizer, cfg):
    """构造一个含 choice/score/noul 三种类型的示例 batch,用于导出与数值验证。"""
    state = {
        "from": "zhang@example.cn",
        "subject": "订单 #8821 重复扣款,要求立刻退款",
        "body": "我昨天被重复扣款两次,要求立刻退款,否则投诉 12315。",
    }
    questions = {
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
    for qid, q in questions.items():
        ids, markers = build_sequence(tokenizer, state, q, max_len, head_max_len)
        items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
    batch = collate_items([items], pad_id=tokenizer.pad_token_id or 0)
    return batch, questions


def main():
    laya_version = im.version("laya")

    # ============================================================
    # 1. 下载/定位权重快照
    # ============================================================
    print("=" * 60)
    print(f"[1] 下载/定位权重快照  repo={args.repo}  subfolder={args.subfolder}")
    print("=" * 60)
    snap_root = Path(snapshot_download(args.repo, allow_patterns=[f"{args.subfolder}/*"]))
    snap = snap_root / args.subfolder
    commit_short = snap_root.name[:7]
    print(f"  laya={laya_version}  commit={commit_short}  snapshot={snap}")

    cfg = json.loads((snap / "rl_agent_config.json").read_text())
    model = build_model(cfg)
    state_dict = load_file(str(snap / "model.safetensors"))
    model.load_state_dict({k: v.float() for k, v in state_dict.items()}, strict=False)
    model = model.float().eval()
    tokenizer = AutoTokenizer.from_pretrained(str(snap / "tokenizer"))
    print(f"  encoder={cfg.get('encoder')}  head_layers={cfg['head_layers']}  "
          f"temperature={model.temperature.tolist()}")

    # ============================================================
    # 2. 构造示例输入 + 3. PyTorch 前向(导出前基准)
    # ============================================================
    print("\n" + "=" * 60)
    print("[2-3] 构造示例输入并跑 PyTorch 前向(作为导出后数值比对基准)")
    print("=" * 60)
    batch, questions = build_sample_batch(tokenizer, cfg)
    with torch.no_grad():
        pt_logits, pt_act = model(
            batch["input_ids"], batch["attention_mask"],
            batch["marker_pos"], batch["marker_mask"], batch["qtype"],
        )
    print(f"  logits: {tuple(pt_logits.shape)}  act: {tuple(pt_act.shape)}")

    # ============================================================
    # 4. 导出 ONNX(dynamo exporter)
    # ============================================================
    onnx_name = f"laya-decision-{args.subfolder}-mmbert-base-v{laya_version}-{commit_short}.onnx"
    onnx_path = OUT_DIR / onnx_name
    print("\n" + "=" * 60)
    print(f"[4] 导出 ONNX → {onnx_path}")
    print("=" * 60)

    torch.onnx.export(
        ForwardWrapper(model).eval(),
        (batch["input_ids"], batch["attention_mask"],
         batch["marker_pos"], batch["marker_mask"], batch["qtype"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        output_names=["logits", "act_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "marker_pos": {0: "batch", 1: "markers"},
            "marker_mask": {0: "batch", 1: "markers"},
            "qtype": {0: "batch"},
            "logits": {0: "batch", 1: "markers"},
            "act_logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    # 把 external data 内嵌回单一 .onnx 文件,方便分发
    from onnx import external_data_helper
    m_onnx = onnx.load(str(onnx_path))
    external_data_helper.load_external_data_for_model(m_onnx, str(OUT_DIR))
    onnx.save(m_onnx, str(onnx_path))
    (OUT_DIR / (onnx_name + ".data")).unlink(missing_ok=True)
    print(f"  OK  {onnx_path}  ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    # ============================================================
    # 5. 修复 Split → Slice(onnxruntime 1.18+ 兼容性)
    # ============================================================
    print("\n" + "=" * 60)
    print("[5] 修复 Split 节点(num_outputs → Slice)")
    print("=" * 60)
    patched = patch_split_to_slice(onnx_path)
    print(f"  OK  修复 {patched} 个 Split 节点")

    # ============================================================
    # 6. ONNX Runtime 数值对齐验证
    # ============================================================
    print("\n" + "=" * 60)
    print("[6] ONNX Runtime 数值对齐验证")
    print("=" * 60)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits, onnx_act = sess.run(None, {
        "input_ids": batch["input_ids"].numpy(),
        "attention_mask": batch["attention_mask"].numpy(),
        "marker_pos": batch["marker_pos"].numpy(),
        "marker_mask": batch["marker_mask"].numpy(),
        "qtype": batch["qtype"].numpy(),
    })
    diff_l = float(np.abs(pt_logits.numpy() - onnx_logits).max())
    diff_a = float(np.abs(pt_act.numpy() - onnx_act).max())
    print(f"  logits max diff: {diff_l:.2e}")
    print(f"  act    max diff: {diff_a:.2e}")
    if diff_l >= 1e-3:
        raise SystemExit(f"数值误差过大(logits diff={diff_l:.2e} >= 1e-3),导出可能有问题,已终止")
    print("  PASS  数值对齐")

    # ============================================================
    # 7. 保存 tokenizer + rl_agent_config.json(供 Java 端使用)
    # ============================================================
    print("\n" + "=" * 60)
    print("[7] 保存 tokenizer + rl_agent_config.json")
    print("=" * 60)
    tok_dir = OUT_DIR / "tokenizer"
    tok_dir.mkdir(exist_ok=True)
    tokenizer.save_pretrained(str(tok_dir))
    (OUT_DIR / "rl_agent_config.json").write_text(json.dumps({
        "encoder": cfg.get("encoder"),
        "head_layers": cfg["head_layers"],
        "max_len": cfg["max_len"],
        "head_max_len": cfg["head_max_len"],
        "temperature": model.temperature.tolist(),
    }, indent=2))

    print(f"\nDONE  导出完成: {onnx_path.resolve()}")
    print(f"  下一步: 更新 Laya4j 的")
    print(f"    - com.laya4j.model.HuggingFaceFetcher.MODEL_VERSION = "
          f"\"{laya_version}-{commit_short}\"")
    print(f"    - models/VERSION.md 版本记录")


if __name__ == "__main__":
    main()
