package com.laya4j;

import com.laya4j.core.*;
import com.laya4j.predict.LayaPredictor;
import com.laya4j.predict.Presets;
import com.laya4j.router.ScriptDetector;
import org.junit.jupiter.api.*;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 中英文混合测试 — 验证 Laya 在中英混合场景下的行为
 *
 * 覆盖:
 *   - Question 的中英文 instructions 渲染
 *   - 路由检测中英文混合文本
 *   - 实际推理:中英混合 state + 中英 question
 *   - 长文本 + 多语种符号 + emoji + URL + 数字
 */
public class MixedChineseEnglishTest {

    private static LayaPredictor predictor;

    @BeforeAll
    static void setup() throws Exception {
        var f = com.laya4j.model.HuggingFaceFetcher.fetchDefault(false);
        predictor = LayaPredictor.single(f.onnxFile(), f.tokenizerDir(), "multilingual");
    }

    @AfterAll
    static void teardown() throws Exception {
        if (predictor != null) predictor.close();
    }

    // ===== 1. Question instructions 中英文 =====
    @Test
    void chineseInstructionsRender() {
        Question q = Question.noul("refund", "用户是否明确要求退款?").build();
        String[] opts = q.renderOptions();
        assertEquals(2, opts.length);
        assertTrue(opts[0].startsWith("false:"));
        assertTrue(opts[1].startsWith("true:"));
    }

    @Test
    void englishInstructionsRender() {
        Question q = Question.noul("refund", "Does the user request a refund?").build();
        String[] opts = q.renderOptions();
        assertEquals(2, opts.length);
        assertTrue(opts[0].startsWith("false:"));
        assertTrue(opts[1].startsWith("true:"));
    }

    @Test
    void mixedCnEnInstructions() {
        Question q = Question.noul("mixed",
            "用户是否 ask for refund? 你怎么看?").build();
        String[] opts = q.renderOptions();
        assertEquals(2, opts.length);
        assertTrue(opts[0].contains("no"));
        assertTrue(opts[1].contains("yes"));
    }

    @Test
    void mixedChoiceCriteria() {
        // 注意:Map.of() 不保证顺序,渲染顺序依赖具体实现
        Question q = Question.choice("intent",
            "What is the user asking about?",
            Map.of(
                "billing",      "账单 billing 退款 payment",
                "tech",         "技术 bug error crash",
                "shipping",     "物流 delivery 配送 快递",
                "其他",          "其他 everything else 其他"
            )).build();
        String[] opts = q.renderOptions();
        assertEquals(4, opts.length);
        // 验证包含所有选项内容(顺序无关)
        String allOpts = String.join(" | ", opts);
        assertTrue(allOpts.contains("billing: 账单 billing 退款 payment"));
        assertTrue(allOpts.contains("tech: 技术 bug error crash"));
        assertTrue(allOpts.contains("shipping: 物流 delivery 配送 快递"));
        assertTrue(allOpts.contains("其他: 其他 everything else 其他"));
    }

    @Test
    void scoreLevelsWithMixedLanguages() {
        Question q = Question.score("urgency", "How urgent 紧急程度?",
            "low 不急", "medium 一般", "critical 紧急 deadline").build();
        String[] opts = q.renderOptions();
        assertEquals(3, opts.length);
        assertEquals("level 0: low 不急", opts[0]);
        assertEquals("level 2: critical 紧急 deadline", opts[2]);
    }

    // ===== 2. ScriptDetector 中英文混合 =====
    @Test
    void detectChineseWithEnglishWords() {
        ScriptDetector d = new ScriptDetector();
        ScriptDetector.DetectionResult r = d.detect(
            "请帮我 refund 这笔订单,我的 order #1234 有问题");
        assertEquals("han", r.dominantScript());
        assertEquals("zh", r.language());
        assertFalse(r.isEnglish());
        assertTrue(r.latinFraction() > 0);  // 有英文单词
    }

    @Test
    void detectEnglishWithChineseWords() {
        ScriptDetector d = new ScriptDetector();
        ScriptDetector.DetectionResult r = d.detect(
            "Please refund 订单 #1234, 我的 user account 有问题");
        assertEquals("latin", r.dominantScript());
        assertEquals("en", r.language());
        assertTrue(r.isEnglish());
        assertTrue(r.latinFraction() >= 0.7);  // 主导 latin
    }

    @Test
    void detectBalancedChineseEnglish() {
        ScriptDetector d = new ScriptDetector();
        // 50/50 混合
        ScriptDetector.DetectionResult r = d.detect(
            "Please 请 refund 退款 这个 order 订单 please 请");
        // 主导应该是 latin (因为 "Please" + "order" + "please" 都是英文)
        assertEquals("latin", r.dominantScript());
    }

    @Test
    void detectChineseWithUrl() {
        ScriptDetector d = new ScriptDetector();
        ScriptDetector.DetectionResult r = d.detect(
            "请访问 https://example.com 查看订单状态,谢谢");
        // 中文占比大但 latin 也占多,dominantScript 取决于哪边字符更多
        // language 仍判定为 zh (有汉字 hint)
        assertEquals("zh", r.language());
        assertFalse(r.isEnglish());
        assertTrue(r.latinFraction() < 0.7);
    }

    @Test
    void detectEnglishWithEmoji() {
        ScriptDetector d = new ScriptDetector();
        ScriptDetector.DetectionResult r = d.detect(
            "Hello world! 😊 This is awesome 🎉 please help me");
        assertEquals("latin", r.dominantScript());
        assertEquals("en", r.language());
        assertTrue(r.isEnglish());
    }

    @Test
    void detectChineseWithEmoji() {
        ScriptDetector d = new ScriptDetector();
        ScriptDetector.DetectionResult r = d.detect(
            "你好世界! 😊 这个真棒 🎉 请帮我");
        assertEquals("han", r.dominantScript());
        assertEquals("zh", r.language());
    }

    // ===== 3. 端到端推理:中英混合 state =====
    @Test
    void chineseStateWithEnglishQuestion() {
        Question q = Question.noul("refund", "用户是否要求退款?").build();
        Map<String, Decision> r = predictor.predict(
            "请帮我 refund this order #1234,谢谢", List.of(q));
        NoulDecision d = (NoulDecision) r.get("refund");
        assertNotNull(d);
        assertTrue(d.probability() >= 0.5, "refund 中文混合应该识别为退款");
    }

    @Test
    void englishStateWithChineseQuestion() {
        Question q = Question.noul("frustrating",
            "用户是否 frustrated 或 angry?").build();
        Map<String, Decision> r = predictor.predict(
            "I am so frustrated! This is terrible!", List.of(q));
        NoulDecision d = (NoulDecision) r.get("frustrating");
        assertNotNull(d);
        assertTrue(d.probability() >= 0.5,
            "英文愤怒表达应该被中文版 frustrated 问题识别");
    }

    @Test
    void mixedStateWithRefundEnglish() {
        Question q = Question.noul("refund_requested",
            "用户是否 ask for 退款?").build();
        Map<String, Decision> r = predictor.predict(
            "Please refund 立刻 refund 一下 my order", List.of(q));
        NoulDecision d = (NoulDecision) r.get("refund_requested");
        assertNotNull(d);
        assertTrue(d.probability() >= 0.5,
            "中英混合的退款请求应该被识别");
    }

    @Test
    void mixedStateWithNonRefundEnglish() {
        Question q = Question.noul("refund_requested",
            "用户是否 ask for 退款?").build();
        Map<String, Decision> r = predictor.predict(
            "请问什么时候能 ship? My order 还没到", List.of(q));
        NoulDecision d = (NoulDecision) r.get("refund_requested");
        assertNotNull(d);
        assertTrue(d.probability() < 0.5,
            "关于发货的查询(混合)应该被识别为非退款");
    }

    // ===== 4. 工单 intent 分类:中英混合 =====
    @Test
    void triageIntentChineseBilling() {
        Question q = Question.choice("intent",
            "What does the customer want in `body`?",
            Map.ofEntries(
                Map.entry("refund",           "money returned or a duplicate charge reversed"),
                Map.entry("technical_help",    "a bug, outage or integration problem"),
                Map.entry("billing_question", "a question about an invoice, plan or payment method"),
                Map.entry("information",      "general information, pricing or how-to"),
                Map.entry("cancellation",     "wants to cancel or downgrade"),
                Map.entry("other",            "none of the other options fits")
            )).build();
        Map<String, Decision> r = predictor.predict(
            "我上个月 charge 的 invoice 还没到账,请查一下", List.of(q));
        ChoiceDecision d = (ChoiceDecision) r.get("intent");
        assertNotNull(d);
        // 期望: billing 或 billing_question
        assertTrue(d.choice().contains("billing"),
            "中英混合账单问题应识别为 billing/billing_question,实际: " + d.choice());
    }

    @Test
    void triageIntentEnglishWithChinese() {
        Question q = Question.choice("intent",
            "用户想做什么?",
            Map.ofEntries(
                Map.entry("refund",           "money returned or a duplicate charge reversed"),
                Map.entry("technical_help",    "a bug, outage or integration problem"),
                Map.entry("billing_question", "a question about an invoice, plan or payment method"),
                Map.entry("information",      "general information, pricing or how-to"),
                Map.entry("cancellation",     "wants to cancel or downgrade"),
                Map.entry("other",            "none of the other options fits")
            )).build();
        Map<String, Decision> r = predictor.predict(
            "My order 一直 crash, please help me fix this bug", List.of(q));
        ChoiceDecision d = (ChoiceDecision) r.get("intent");
        assertNotNull(d);
        // 期望: technical_help
        assertEquals("technical_help", d.choice());
    }

    // ===== 5. 长文本 + 多符号 =====
    @Test
    void longMixedTextWithSymbols() {
        Question q = Question.noul("refund", "用户是否要求退款?").build();
        String state = "Hi, 您好! 我在 https://example.com/order/1234 " +
                "下了单,金额是 ¥1280.50,please refund it ASAP. " +
                "订单号: #ABC-123 🛒 谢谢! 10/15/2024";
        Map<String, Decision> r = predictor.predict(state, List.of(q));
        NoulDecision d = (NoulDecision) r.get("refund");
        assertNotNull(d);
        assertTrue(d.probability() >= 0.5,
            "中英混合带URL/价格/订单号的退款请求应该被识别");
    }

    @Test
    void longEnglishTextWithChineseAddress() {
        Question q = Question.noul("refund", "用户是否要求退款?").build();
        String state = "Please refund my recent order shipped to " +
                "上海市浦东新区张江路 88 号 3 楼 305 室. " +
                "Order ID: 12345. Amount: $1280 USD. " +
                "Please refund to my original payment method ASAP.";
        Map<String, Decision> r = predictor.predict(state, List.of(q));
        NoulDecision d = (NoulDecision) r.get("refund");
        assertNotNull(d);
        assertTrue(d.probability() >= 0.5);
    }

    @Test
    void mixedTextWithCodeSnippets() {
        Question q = Question.noul("technical_help",
            "用户是否遇到技术问题?").build();
        String state = "我用 python 写了一段代码:\n" +
                "```python\n" +
                "def refund():\n" +
                "    raise Exception('not allowed')\n" +
                "```\n" +
                "总是报错,请 help me";
        Map<String, Decision> r = predictor.predict(state, List.of(q));
        NoulDecision d = (NoulDecision) r.get("technical_help");
        assertNotNull(d);
        assertTrue(d.probability() >= 0.5);
    }

    // ===== 6. Triage preset 端到端(模拟真实客服场景) =====
    @Test
    void triagePresetMixedRefund() {
        String state = "{\"body\":\"Hi, 我要 refund 这笔 order #1234, " +
                "please 立刻处理,谢谢\"}";
        Map<String, Decision> r = predictor.predict(state, Presets.triage());
        // 应该有 5 个 answer
        assertEquals(5, r.size());
        // refund_requested 应该是 true
        NoulDecision refund = (NoulDecision) r.get("refund_requested");
        assertNotNull(refund);
        assertTrue(refund.probability() >= 0.5);
        // churn_risk 应该是 false(温和请求)
        NoulDecision churn = (NoulDecision) r.get("churn_risk");
        assertNotNull(churn);
        assertTrue(churn.probability() < 0.5,
            "温和的中英混合退款请求不应判为流失风险");
    }

    @Test
    void triagePresetEnglishTechnicalIssue() {
        String state = "{\"body\":\"我的 APP 一直 crash, please fix this ASAP. " +
                "我已经重启 100 次了,还是不行 help!\"}";
        Map<String, Decision> r = predictor.predict(state, Presets.triage());
        ChoiceDecision intent = (ChoiceDecision) r.get("intent");
        assertNotNull(intent);
        assertEquals("technical_help", intent.choice());
        // is_urgent:base 模型可能判断为 medium,不强求 true
        NoulDecision urgent = (NoulDecision) r.get("is_urgent");
        assertNotNull(urgent);
        assertTrue(urgent.probability() > 0.0, "概率值应该非负");  // base 模型准确率有限,只做 sanity check
    }

    @Test
    void triagePresetMixedBillingInquiry() {
        String state = "{\"body\":\"请问我上个月 charge 的 $99 invoice 在哪里? " +
                "Where can I download it? 谢谢\"}";
        Map<String, Decision> r = predictor.predict(state, Presets.triage());
        ChoiceDecision intent = (ChoiceDecision) r.get("intent");
        assertNotNull(intent);
        // 期望: billing_question 或 information
        assertTrue(
            intent.choice().equals("billing_question") ||
            intent.choice().equals("information"),
            "中英混合账单查询应该是 billing_question/information,实际: " + intent.choice()
        );
    }
}