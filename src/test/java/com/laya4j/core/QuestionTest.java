package com.laya4j.core;

import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class QuestionTest {

    @Test
    void choiceRequiresChoices() {
        assertThrows(IllegalStateException.class, () ->
                Question.choice("intent", "Which?", null).build());
        assertThrows(IllegalStateException.class, () ->
                Question.choice("intent", "Which?", Map.of()).build());
    }

    @Test
    void scoreRequiresLevels() {
        assertThrows(IllegalStateException.class, () ->
                Question.score("urgency", "How?", (String[]) null).build());
        assertThrows(IllegalStateException.class, () ->
                Question.score("urgency", "How?", new String[0]).build());
    }

    @Test
    void noulBuildsWithDefaultCriteria() {
        Question q = Question.noul("refund", "User asks for refund?").build();
        String[] opts = q.renderOptions();
        assertEquals(2, opts.length);
        assertEquals("false: no, the statement does not hold", opts[0]);
        assertEquals("true:  yes, the statement holds", opts[1]);
    }

    @Test
    void noulWithCustomCriteria() {
        Question q = Question.noul("refund", "User asks?", "no refund requested", "yes refund requested").build();
        String[] opts = q.renderOptions();
        assertEquals("false: no refund requested", opts[0]);
        assertEquals("true:  yes refund requested", opts[1]);
    }

    @Test
    void choiceRenderOptionsWithDescription() {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("billing", "invoices, payments");
        m.put("tech", "bugs");
        Question q = Question.choice("intent", "Which?", m).build();
        String[] opts = q.renderOptions();
        assertEquals("billing: invoices, payments", opts[0]);
        assertEquals("tech: bugs", opts[1]);
    }

    @Test
    void choiceRenderOptionsWithoutDescription() {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("a", null);
        m.put("b", "");
        Question q = Question.choice("which", "Which?", m).build();
        String[] opts = q.renderOptions();
        assertEquals("a", opts[0]);
        assertEquals("b", opts[1]);
    }

    @Test
    void scoreRenderOptionsWithLevels() {
        Question q = Question.score("urgency", "How urgent?",
                "low", "medium", "high").build();
        String[] opts = q.renderOptions();
        assertEquals("level 0: low", opts[0]);
        assertEquals("level 1: medium", opts[1]);
        assertEquals("level 2: high", opts[2]);
    }

    @Test
    void decisionTypeEnum() {
        assertEquals(0, DecisionType.CHOICE.id());
        assertEquals(1, DecisionType.SCORE.id());
        assertEquals(2, DecisionType.NOUL.id());
        assertEquals(DecisionType.CHOICE, DecisionType.fromId(0));
        assertEquals(DecisionType.SCORE, DecisionType.fromId(1));
        assertEquals(DecisionType.NOUL, DecisionType.fromId(2));
        assertThrows(IllegalArgumentException.class, () -> DecisionType.fromId(99));
    }

    @Test
    void layaConfigDefaults() {
        LayaConfig c = LayaConfig.defaults();
        assertEquals(1024, c.maxLen());
        assertEquals(256, c.headMaxLen());
        assertEquals(6, c.maxPrefixes());
        assertArrayEquals(new double[]{1.0, 1.0, 1.0}, c.temperature());
    }
}