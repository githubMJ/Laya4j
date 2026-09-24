"""
Laya 中文决策示例 — macOS (Apple Silicon, MPS 加速)

用法:
    python mac_mps.py

前置:
    pip install laya
    Apple Silicon (M1/M2/M3/M4);MPS 自动启用
"""
import time

import torch
from laya import Router

# ---------- 1. 中文状态:客服工单 ----------
STATE = {
    "from": "张先生 <zhang@example.cn>",
    "subject": "订单 #8821 重复扣款,要求立刻退款",
    "body": (
        "你好,我昨天用支付宝支付了一笔 1280 元的订单,"
        "今天又收到一条同样金额的扣款通知。请立刻核实并退款,"
        "否则我将向 12315 投诉并取消后续所有合作。"
    ),
}

# ---------- 2. 类型化问题:choice / score / noul ----------
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing":   "invoices, payments, refunds 发票、付款、退款",
            "technical": "bugs, outages, system errors 漏洞、宕机、系统错误",
            "sales":     "pricing, new contracts 报价、新合同",
            "other":     "everything else 其他",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent 不紧急", "soon 较快", "critical deadline 紧急截止"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave? 用户是否威胁取消或离开?",
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund? 用户是否明确要求退款?",
    },
}

THRESHOLD = 0.85


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    device = pick_device()
    print(f"[device] {device}")

    router = Router(preload=True, device=device)

    # 仅路由检查(无前向)
    r = router.route(STATE, QUESTIONS)
    print(f"\n【仅路由检查】 model={r.model}  reason={r.reason}")

    # 完整预测(带计时)
    t0 = time.perf_counter()
    res = router.predict(STATE, QUESTIONS)
    dt_ms = (time.perf_counter() - t0) * 1000
    print(f"\n推理耗时: {dt_ms:.1f} ms")

    print("\n【路由决策】")
    for k, v in res["routing"].items():
        if k == "detection" and isinstance(v, dict):
            print(f"  detection:")
            for dk, dv in v.items():
                print(f"    {dk}: {dv}")
        else:
            print(f"  {k}: {v}")

    print("\n【答案】")
    for qid, ans in res["answers"].items():
        t = ans.get("type", QUESTIONS[qid]["type"])
        if t == "choice":
            print(f"  {qid:18s} → {ans['choice']:10s}  conf={ans['confidence']:.3f}")
        elif t == "score":
            print(f"  {qid:18s} → score={ans['score']:.2f}   conf={ans['confidence']:.3f}")
        elif t == "noul":
            print(f"  {qid:18s} → P(true)={ans['noul']:.3f}   conf={ans['confidence']:.3f}")

    # 置信度门控演示
    dept = res["answers"]["department"]
    conf = dept["confidence"]
    print(f"\n【置信度门控:department, threshold={THRESHOLD}】")
    if conf >= THRESHOLD:
        print(f"  ✅ conf={conf:.3f} >= {THRESHOLD} → 自动路由到 [{dept['choice']}] 部门")
    else:
        print(f"  ⚠️  conf={conf:.3f} <  {THRESHOLD} → 转人工分诊,候选 [{dept['choice']}]")


if __name__ == "__main__":
    main()
