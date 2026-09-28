"""
Generate the Chinese seed fine-tune dataset (data/finetune_zh.jsonl).

Targets the three weak spots measured on the 0.3.21 base model:
  1. threat detection (noul)  — subtle Chinese threats all scored P<0.05
  2. difficulty grading (score, 4 levels) — everything collapsed to "moderate"
  3. intent classification (choice, 6 preset labels) — refund over-activated,
     "other"/"information" nearly never predicted

Seed data only: mix of hand-written cases and light template expansion.
Real deployments should grow this to 200-500 items per weak scenario,
LLM-generated + human-reviewed, before fine-tuning.

Usage:
    python gen_data_zh.py                 # writes data/finetune_zh.jsonl
"""
import json
import os
import random

random.seed(7)

# preset labels (must match laya.triage_questions())
INTENT_LABELS = ["refund", "technical_help", "billing_question",
                 "information", "cancellation", "other"]

rows = []


def add(state, questions, gold):
    rows.append({"state": state, "questions": questions, "gold": gold})


def noul_probs(flag):
    return {"false": 0.0 if flag else 1.0, "true": 1.0 if flag else 0.0}


def choice_probs(label, confidence=0.9):
    spread = (1.0 - confidence) / (len(INTENT_LABELS) - 1)
    return {k: (confidence if k == label else spread) for k in INTENT_LABELS}


# ============================================================
# 1. threat detection (noul) — the biggest weak spot
# ============================================================
THREAT_Q = {"threat": {"type": "noul",
                       "instructions": "Does this message contain a threat, intimidation, or a promise of retaliation?"}}

THREAT_TRUE = [
    "如果你们不退款,我会找人在你们公司楼下等着。",
    "再不解决问题我会让你在整个行业都接不到单。",
    "三天之内不给答复,后果自负,我有的是办法。",
    "我知道你家住哪个小区,最好给我处理清楚。",
    "别逼我把这些聊天记录发给你们所有的客户。",
    "你最好想清楚,得罪我对你没好处,我有的是时间陪你耗。",
    "我会一直盯着你,你跑到哪我都能找到你。",
    "叫我兄弟们晚上去你门店坐坐,看看你还敢不敢装死。",
    "这事你要是不给我办了,咱们就同归于尽。",
    "我已经记下你的工号了,你下班小心点。",
]
THREAT_FALSE = [
    "如果你们不退款,我会向消费者协会投诉。",
    "再不解决我就去平台申请仲裁,走正规流程。",
    "三天之内不给答复,我就只能退货了,望理解。",
    "希望贵司重视,我会保留追究法律责任的权利。",
    "再拖下去我只能给差评并申请平台介入了。",
    "请尽快处理,否则我会考虑取消后续所有合作。",
    "我已经咨询了律师,会通过法律途径解决。",
    "这是最后一次沟通了,不处理我就卸载你们的APP。",
    "我会把这次糟糕的体验分享到社交平台上。",
    "麻烦尽快退款,不然我只能拒收快递了。",
]

for msg in THREAT_TRUE + THREAT_FALSE:
    flag = msg in THREAT_TRUE
    add({"post": msg}, THREAT_Q, {"threat": {"probabilities": noul_probs(flag)}})

# 隐晦变体(模板扩展)
SUBTLE_TRUE_TPL = [
    "处理不好这事,你以后{}可就难了。",
    "我认识不少人,都能帮我在{}上说说话。",
    "别以为换个客服就没事了,我记性可{}。",
]
SUBTLE_FILL = ["在这行", "你们高层", "很好", "不差", "好得很"]
for tpl in SUBTLE_TRUE_TPL:
    for f in SUBTLE_FILL[:3]:
        add({"post": tpl.format(f)}, THREAT_Q,
            {"threat": {"probabilities": noul_probs(True)}})

# ============================================================
# 2. intent classification (choice, 6 preset labels)
# ============================================================
INTENT_Q = {"intent": {"type": "choice",
                       "instructions": "Which department or category does this customer message belong to?",
                       "criteria": {
                           "refund": "money returned or a duplicate charge reversed",
                           "technical_help": "a bug, outage or integration problem",
                           "billing_question": "a question about an invoice, plan or payment method",
                           "information": "general information, pricing or how-to",
                           "cancellation": "wants to cancel or downgrade",
                           "other": "none of the other options fits"}}}

INTENT_CASES = [
    ("我昨天被重复扣款两次,请把钱退给我。", "refund", 0.95),
    ("多收的运费什么时候退回我的账户?", "refund", 0.9),
    ("app 登录一直转圈,验证码也收不到。", "technical_help", 0.9),
    ("网页版导出报表功能报错 500。", "technical_help", 0.92),
    ("这个月的发票什么时候开?抬头要改一下。", "billing_question", 0.9),
    ("帮我看看这期账单里这笔 39 元是什么费用。", "billing_question", 0.92),
    ("企业版有什么功能?和专业版差在哪?", "information", 0.9),
    ("支持哪些支付方式?可以对公转账吗?", "information", 0.9),
    ("我要注销账号,把年费会员也退订了。", "cancellation", 0.93),
    ("帮我把套餐从专业版降级到基础版。", "cancellation", 0.9),
    ("你们公司注册地址是哪里?我想寄一份纸质函件。", "other", 0.9),
    ("物流显示揽收后 5 天没有更新了,帮我查查。", "other", 0.85),
    ("修改收货地址在哪里操作?", "information", 0.88),
    ("这个商品支持七天无理由退货吗?", "information", 0.85),
    ("我想投诉上次那个客服的服务态度。", "other", 0.9),
    ("能给我开发票吗?需要报销。", "billing_question", 0.9),
]
for msg, label, conf in INTENT_CASES:
    add({"body": msg}, INTENT_Q, {"intent": {"probabilities": choice_probs(label, conf)}})

# 边界:退款 vs 退换货 vs 注销
BOUNDARY_CASES = [
    ("商品有质量问题,我要退货退款。", "refund", 0.85),          # 退货退款 → refund 偏高
    ("我不要退款,帮我换一台新的。", "other", 0.75),              # 换货不是退款
    ("会员不想续费了,下个月别自动扣费。", "cancellation", 0.88),  # 取消续费
    ("扣款失败是不是我余额不足?", "billing_question", 0.88),
]
for msg, label, conf in BOUNDARY_CASES:
    add({"body": msg}, INTENT_Q, {"intent": {"probabilities": choice_probs(label, conf)}})

# ============================================================
# 3. difficulty grading (score, 4 levels, soft ordinal targets)
# ============================================================
DIFF_Q = {"difficulty": {"type": "score",
                         "instructions": "How hard is this request for an AI assistant? 0=trivial lookup, 1=easy, 2=moderate multi-step, 3=hard specialist reasoning.",
                         "criteria": ["trivial: a lookup or one-liner",
                                      "easy: short answer, no reasoning",
                                      "moderate: several steps",
                                      "hard: long multi-step reasoning or specialist knowledge"]}}

# (request, 分布) — 软序数目标,允许相邻档位分摊
DIFF_CASES = [
    ("'hello' 翻译成中文。", [0.75, 0.2, 0.05, 0.0]),
    ("今天北京的天气怎么样?", [0.7, 0.25, 0.05, 0.0]),
    ("把这句话改得正式一点:咱俩谁跟谁啊。", [0.2, 0.65, 0.15, 0.0]),
    ("写一首关于秋天的五言绝句。", [0.05, 0.6, 0.3, 0.05]),
    ("用 Python 统计一段文本的词频并排序输出。", [0.0, 0.2, 0.6, 0.2]),
    ("排查一个偶发的线上数据库死锁问题。", [0.0, 0.05, 0.35, 0.6]),
    ("从法律、财务、技术三个维度分析这份 SaaS 合同。", [0.0, 0.0, 0.2, 0.8]),
    ("设计一个支持百万级 QPS 的分布式限流方案并论证取舍。", [0.0, 0.0, 0.1, 0.9]),
    ("1 加 1 等于几?", [0.9, 0.1, 0.0, 0.0]),
    ("帮我写一封催款的商务邮件,语气要坚定但礼貌。", [0.0, 0.3, 0.55, 0.15]),
    ("分析这份用户行为日志,找出留存下降的原因。", [0.0, 0.05, 0.4, 0.55]),
    ("把这段中文摘要翻译成英文,保持专业术语。", [0.1, 0.5, 0.35, 0.05]),
]
for req, dist in DIFF_CASES:
    add({"request": req}, DIFF_Q,
        {"difficulty": {"probabilities": {str(i): v for i, v in enumerate(dist)}}})

# ============================================================
# 4. refund boundary (noul) — 开票/换货/议价不是退款
# ============================================================
REFUND_Q = {"refund_requested": {"type": "noul",
                                 "instructions": "Does the user explicitly request a refund?"}}

REFUND_CASES = [
    ("请把我多付的钱退回来。", True),
    ("订单重复扣款,要求退一笔。", True),
    ("商品有问题,我要求退货退款。", True),
    ("能给我开发票吗?需要报销。", False),
    ("这个价格还能再便宜点吗?", False),
    ("我不要钱,帮我换一台全新的。", False),
    ("会员到期了,不想续费了。", False),
    ("这笔运费我申请运费险赔付。", False),
    ("退款一般几个工作日到账?", True),
    ("发票明细能写成办公用品吗?", False),
]
for msg, flag in REFUND_CASES:
    add({"body": msg}, REFUND_Q, {"refund_requested": {"probabilities": noul_probs(flag)}})


# ============================================================
# write out
# ============================================================
os.makedirs("data", exist_ok=True)
out = "data/finetune_zh.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Wrote {len(rows)} rows -> {out}")
by_q = {}
for r in rows:
    for qid in r["gold"]:
        by_q[qid] = by_q.get(qid, 0) + 1
print("By question:", by_q)
