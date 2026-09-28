"""
Generate the ad-bidding decision fine-tune dataset (data/finetune_ads.jsonl).

Scenario: 信息流/搜索广告计划的出价决策。模型读中文投放指标状态,输出:
  - bid_action       (choice): raise_bid / keep / lower_bid / pause / add_budget
  - needs_adjustment (noul):  当前是否需要动出价
  - adjust_magnitude (score): 0=维持 1=微调(约5%) 2=中调(10%~20%) 3=大幅(>20%)

Gold labels come from an explainable rule policy (`decide`) — edit it to
change the deployment policy the model should internalize.

Usage:
    python gen_data_ads.py --n 600 --out data/finetune_ads.jsonl
"""
import argparse
import json
import os
import random

random.seed(11)

QUESTIONS = {
    "bid_action": {
        "type": "choice",
        "instructions": "根据当前投放数据,下一步应该采取什么出价动作?",
        "criteria": {
            "raise_bid": "成本达标但量不足,提高出价抢量",
            "keep": "成本与量都在合理区间,维持当前出价",
            "lower_bid": "实际成本超出目标且转化样本足够,降低出价控成本",
            "pause": "持续无转化或严重超支,暂停计划止损",
            "add_budget": "成本达标且预算提前耗尽,追加预算放量",
        },
    },
    "needs_adjustment": {
        "type": "noul",
        "instructions": "当前是否需要对出价进行调整(提价/降价/暂停)?",
    },
    "adjust_magnitude": {
        "type": "score",
        "instructions": "若需要调整,调整幅度应为哪一档? 0=维持, 1=微调(约5%), 2=中调(10%~20%), 3=大幅(>20%)",
        "criteria": ["维持", "微调约5%", "中调10%~20%", "大幅>20%"],
    },
}


def decide(m):
    """指标 → (action, needs_adj, magnitude, reason)。投放策略改这里即可。"""
    target, actual, conv = m["target_cost"], m["actual_cost"], m["conversions"]
    spent_ratio = m["spent"] / m["budget"] if m["budget"] else 0
    ctr = m["clicks"] / m["impressions"] if m["impressions"] else 0

    if conv == 0 and spent_ratio >= 0.3:
        return "pause", True, 3, "超预算30%仍无转化,暂停止损"
    if actual <= target * 1.1 and spent_ratio >= 0.92 and m["hour"] < 22:
        return "add_budget", False, 0, "成本达标且预算提前耗尽,追加预算"
    if actual > target * 1.3 and conv >= 5:
        over = actual / target
        mag = 3 if over > 1.5 else 2
        return "lower_bid", True, mag, f"成本超目标{int((over-1)*100)}%,样本充足,降价控成本"
    if actual > target * 1.15 and conv >= 3:
        return "lower_bid", True, 2, "成本轻度超标,中幅下调"
    if actual > target * 1.1:
        return "keep", False, 0, "成本略超标但转化样本不足,维持观察"
    if actual < target * 0.75 and (ctr < 0.015 or m["clicks"] < 20):
        mag = 2 if actual < target * 0.6 else 1
        return "raise_bid", True, mag, "成本远低于目标且量不足,提价抢量"
    return "keep", False, 0, "各项指标健康,维持出价"


def sample_metrics(regime):
    target = random.choice([30, 40, 50, 60, 80, 100])
    budget = random.choice([500, 800, 1000, 1500, 2000])
    hour = random.randint(8, 23)
    if regime == "pause":
        spent = budget * random.uniform(0.35, 0.8)
        impressions = random.randint(8000, 30000)
        actual = round(random.uniform(target * 2, target * 4), 1)
        conv = 0
    elif regime == "add_budget":
        actual = round(target * random.uniform(0.7, 1.05), 1)
        spent = budget * random.uniform(0.93, 1.0)
        conv = random.randint(8, 30)
        impressions = random.randint(40000, 120000)
    elif regime == "lower_big":
        actual = round(target * random.uniform(1.3, 1.8), 1)
        spent = budget * random.uniform(0.3, 0.7)
        conv = random.randint(6, 20)
        impressions = random.randint(20000, 60000)
    elif regime == "lower_mid":
        actual = round(target * random.uniform(1.15, 1.3), 1)
        spent = budget * random.uniform(0.25, 0.6)
        conv = random.randint(3, 10)
        impressions = random.randint(15000, 50000)
    elif regime == "keep_watch":
        actual = round(target * random.uniform(1.1, 1.15), 1)
        spent = budget * random.uniform(0.2, 0.5)
        conv = random.randint(0, 2)
        impressions = random.randint(8000, 30000)
    elif regime == "raise":
        actual = round(target * random.uniform(0.4, 0.75), 1)
        spent = budget * random.uniform(0.15, 0.5)
        conv = random.randint(2, 12)
        impressions = random.randint(3000, 20000)
    else:  # keep
        actual = round(target * random.uniform(0.85, 1.1), 1)
        spent = budget * random.uniform(0.2, 0.85)
        conv = random.randint(3, 25)
        impressions = random.randint(15000, 80000)
    clicks = max(1, int(impressions * random.uniform(0.005, 0.035)))
    return {"budget": budget, "spent": round(spent, 1), "impressions": impressions,
            "clicks": clicks, "conversions": conv, "target_cost": target,
            "actual_cost": actual, "hour": hour}


def render_state(m, plan):
    ctr = round(m["clicks"] / m["impressions"] * 100, 2) if m["impressions"] else 0.0
    cpm = round(m["spent"] / m["impressions"] * 1000, 1) if m["impressions"] else 0.0
    return {"计划": plan, "时段": f"{m['hour']:02d}:00",
            "日预算(元)": m["budget"], "已消耗(元)": m["spent"],
            "预算消耗比例": f"{m['spent']/m['budget']*100:.0f}%",
            "展示量": m["impressions"], "点击量": m["clicks"],
            "CTR": f"{ctr}%", "CPM(元)": cpm, "转化数": m["conversions"],
            "目标转化成本(元)": m["target_cost"], "实际转化成本(元)": m["actual_cost"]}


PLAN_NAMES = ["穿搭女装-信息流-01", "3C数码-搜索-夜抛", "家居日用-信息流-A3",
              "美妆个护-千川-直投", "本地餐饮-同城推-02", "教育课程-搜索-品牌",
              "家电以旧换新-信息流-B1", "母婴用品-千川-极速推"]


def soft_action(action, confidence=0.9):
    labels = list(QUESTIONS["bid_action"]["criteria"].keys())
    spread = (1 - confidence) / (len(labels) - 1)
    return {k: (confidence if k == action else spread) for k in labels}


def soft_magnitude(mag):
    dist = [0.03, 0.03, 0.03, 0.03]
    dist[mag] = 0.88
    return {str(i): v for i, v in enumerate(dist)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out", default="data/finetune_ads.jsonl")
    args = ap.parse_args()

    regimes = ["pause", "add_budget", "lower_big", "lower_mid",
               "keep_watch", "raise", "keep"]
    os.makedirs("data", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.n):
            regime = regimes[i % len(regimes)]
            m = sample_metrics(regime)
            action, needs_adj, mag, reason = decide(m)
            state = render_state(m, PLAN_NAMES[i % len(PLAN_NAMES)])
            gold = {
                "bid_action": {"probabilities": soft_action(action)},
                "needs_adjustment": {"probabilities": {
                    "false": 0.0 if needs_adj else 1.0,
                    "true": 1.0 if needs_adj else 0.0}},
                "adjust_magnitude": {"probabilities": soft_magnitude(mag)},
            }
            f.write(json.dumps({
                "state": state, "questions": QUESTIONS, "gold": gold,
                "meta": {"regime": regime, "action": action, "reason": reason},
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {args.n} rows -> {args.out} (regimes: {regimes})")


if __name__ == "__main__":
    main()
