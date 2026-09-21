package com.laya4j.core;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * SCORE 类型结果:序数分级
 *   score = Σ(i * P(level_i)),i ∈ [0, N)
 *   distribution = 每 level 的概率
 */
public final class ScoreDecision implements Decision {

    private final double score;
    private final double confidence;
    private final Map<Integer, Double> distribution;
    private final String[] levels;

    public ScoreDecision(double score, double confidence,
                         Map<Integer, Double> distribution, String[] levels) {
        this.score = score;
        this.confidence = confidence;
        this.distribution = distribution;
        this.levels = levels;
    }

    public double score() { return score; }
    public Map<Integer, Double> distribution() { return new LinkedHashMap<>(distribution); }
    public String[] levels() { return levels; }

    @Override
    public double confidence() { return confidence; }

    @Override
    public DecisionType type() { return DecisionType.SCORE; }

    /** 把 score 映射回最近的 level 字符串(用于直接给人类看) */
    public String nearestLevel() {
        if (levels == null || levels.length == 0) return null;
        int idx = Math.round((float) score);
        if (idx < 0) idx = 0;
        if (idx >= levels.length) idx = levels.length - 1;
        return levels[idx];
    }

    @Override
    public String toString() {
        return "ScoreDecision{score=" + String.format("%.4f", score) +
                ", level=" + nearestLevel() + ", confidence=" +
                String.format("%.4f", confidence) + "}";
    }
}