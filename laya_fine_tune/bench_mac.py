"""
Laya 压测 — macOS (MPS)
"""
import statistics
import time

import torch
from laya import Router

STATE = {
    "from": "张先生 <zhang@example.cn>",
    "subject": "订单 #8821 重复扣款,要求立刻退款",
    "body": (
        "你好,我昨天用支付宝支付了一笔 1280 元的订单,"
        "今天又收到一条同样金额的扣款通知。请立刻核实并退款,"
        "否则我将向 12315 投诉并取消后续所有合作。"
    ),
}

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds 发票、付款、退款",
            "technical": "bugs, outages, system errors 漏洞、宕机、系统错误",
            "sales": "pricing, new contracts 报价、新合同",
            "other": "everything else 其他",
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


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def bench(label: str, fn, n: int) -> dict:
    """预热 1 次,然后跑 n 次,统计延迟。"""
    fn()  # warmup
    samples_ms = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples_ms.append((time.perf_counter() - t0) * 1000)

    return {
        "label": label,
        "n": n,
        "p50": statistics.median(samples_ms),
        "p95": sorted(samples_ms)[int(0.95 * n) - 1],
        "min": min(samples_ms),
        "max": max(samples_ms),
        "mean": statistics.mean(samples_ms),
        "qps": 1000.0 / statistics.mean(samples_ms),
    }


def main() -> None:
    device = pick_device()
    print(f"[device] {device}")
    router = Router(preload=True, device=device)

    print("预热 + 路由检查...")
    _ = router.route(STATE, QUESTIONS)
    _ = router.predict(STATE, QUESTIONS)

    print("\n开始压测 (50 次/场景) ...")
    r1 = bench("仅路由(route)", lambda: router.route(STATE, QUESTIONS), n=50)
    r2 = bench("单次 predict(1 state, 4 questions)", lambda: router.predict(STATE, QUESTIONS), n=50)
    # 不同 state 复用同一组 questions
    def alt_predict():
        router.predict({"body": "服务挂了快两小时了,严重影响业务。"}, QUESTIONS)
    r3 = bench("predict(不同 state)", alt_predict, n=50)

    def row(r):
        return (
            f"  {r['label']:40s}  "
            f"p50={r['p50']:7.2f} ms  "
            f"p95={r['p95']:7.2f} ms  "
            f"min={r['min']:7.2f}  "
            f"max={r['max']:7.2f}  "
            f"mean={r['mean']:7.2f} ms  "
            f"qps={r['qps']:7.2f}/s"
        )

    print("\n" + "=" * 120)
    print(f"{'场景':40s}  {'p50':>10s}  {'p95':>10s}  {'min':>8s}  {'max':>8s}  {'mean':>10s}  {'QPS':>10s}")
    print("-" * 120)
    for r in (r1, r2, r3):
        print(row(r))
    print("=" * 120)


if __name__ == "__main__":
    main()
