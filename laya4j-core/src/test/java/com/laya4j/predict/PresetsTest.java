package com.laya4j.predict;

import com.laya4j.core.Question;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class PresetsTest {

    @Test
    void triageReturnsExpectedQuestions() {
        List<Question> qs = Presets.triage();
        assertNotNull(qs);
        assertTrue(qs.size() >= 3);
        assertTrue(qs.stream().anyMatch(q -> q.name().equals("intent")));
        assertTrue(qs.stream().anyMatch(q -> q.name().equals("is_urgent")));
        assertTrue(qs.stream().anyMatch(q -> q.name().equals("churn_risk")));
    }

    @Test
    void guardReturnsExpectedQuestions() {
        List<Question> qs = Presets.guard();
        assertTrue(qs.stream().anyMatch(q -> q.name().equals("prompt_injection")));
        assertTrue(qs.stream().anyMatch(q -> q.name().equals("jailbreak")));
    }

    @Test
    void moderationReturnsExpectedQuestions() {
        List<Question> qs = Presets.moderation();
        assertTrue(qs.stream().anyMatch(q -> q.name().equals("toxic")));
        assertTrue(qs.stream().anyMatch(q -> q.name().equals("threat")));
    }

    @Test
    void modelRouterReturnsExpectedQuestions() {
        List<Question> qs = Presets.modelRouter();
        assertTrue(qs.stream().anyMatch(q -> q.name().equals("difficulty")));
        assertTrue(qs.stream().anyMatch(q -> q.name().equals("domain")));
        assertTrue(qs.stream().anyMatch(q -> q.name().equals("needs_tools")));
    }
}