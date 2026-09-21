package com.laya4j.core;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * CHOICE 类型结果:从候选标签中选一个,带每选项的概率分布
 */
public final class ChoiceDecision implements Decision {

    private final String choice;
    private final double confidence;
    private final Map<String, Double> probabilities;

    public ChoiceDecision(String choice, double confidence,
                          Map<String, Double> probabilities) {
        this.choice = choice;
        this.confidence = confidence;
        this.probabilities = probabilities;
    }

    public String choice() { return choice; }

    @Override
    public double confidence() { return confidence; }

    @Override
    public DecisionType type() { return DecisionType.CHOICE; }

    /** 每选项的概率(label -> P,加和 = 1) */
    public Map<String, Double> probabilities() {
        return new LinkedHashMap<>(probabilities);
    }

    @Override
    public String toString() {
        return "ChoiceDecision{choice=" + choice + ", confidence=" +
                String.format("%.4f", confidence) + "}";
    }
}