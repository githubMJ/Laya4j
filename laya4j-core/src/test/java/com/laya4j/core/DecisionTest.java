package com.laya4j.core;

import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class DecisionTest {

    @Test
    void choiceDecisionExposesProbabilities() {
        Map<String, Double> probs = new LinkedHashMap<>();
        probs.put("a", 0.7);
        probs.put("b", 0.3);
        ChoiceDecision c = new ChoiceDecision("a", 0.7, probs);
        assertEquals("a", c.choice());
        assertEquals(0.7, c.confidence());
        assertEquals(2, c.probabilities().size());
        assertEquals(DecisionType.CHOICE, c.type());
        // probabilities 是副本,不可变
        c.probabilities().clear();
        assertEquals(0.7, c.probabilities().get("a"));
    }

    @Test
    void scoreDecisionNearestLevel() {
        Map<Integer, Double> dist = new LinkedHashMap<>();
        dist.put(0, 0.1);
        dist.put(1, 0.7);
        dist.put(2, 0.2);
        ScoreDecision s = new ScoreDecision(1.1, 0.5, dist, new String[]{"low", "medium", "high"});
        assertEquals("medium", s.nearestLevel());
        assertEquals(DecisionType.SCORE, s.type());
    }

    @Test
    void scoreDecisionClampsNearestLevel() {
        Map<Integer, Double> dist = new LinkedHashMap<>();
        dist.put(0, 1.0);
        ScoreDecision s = new ScoreDecision(-1.0, 1.0, dist, new String[]{"a"});
        assertEquals("a", s.nearestLevel());
    }

    @Test
    void noulDecisionBoolean() {
        NoulDecision yes = new NoulDecision(0.9, 0.9);
        NoulDecision no = new NoulDecision(0.1, 0.9);
        assertTrue(yes.asBoolean(0.5));
        assertFalse(no.asBoolean(0.5));
        assertEquals(0.9, yes.confidence());
        assertEquals(DecisionType.NOUL, yes.type());
    }
}