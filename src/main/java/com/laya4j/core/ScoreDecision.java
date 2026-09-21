package com.laya4j.core;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * SCORE result: ordinal scale
 *
 * Returned by LayaOnnxModel.predict() when qtype=SCORE.
 *
 * Outputs:
 *   score = Σ(i × P(level_i)), i ∈ [0, N)
 *   distribution = per-level probabilities
 */
public final class ScoreDecision implements Decision {

    private final double score;
    private final double confidence;
    private final Map<Integer, Double> distribution;
    private final String[] levels;

    /**
     * @param score        weighted expected value ∈ [0, N-1]
     * @param confidence   normalized Shannon entropy (1 = fully certain)
     * @param distribution per-level probabilities (level → P, sums to 1.0)
     * @param levels       level string array (used by nearestLevel)
     */
    public ScoreDecision(double score, double confidence,
                         Map<Integer, Double> distribution, String[] levels) {
        this.score = score;
        this.confidence = confidence;
        this.distribution = distribution;
        this.levels = levels;
    }

    /** @return weighted expected score = Σ(i × P_i) */
    public double score() { return score; }

    /** @return per-level probabilities (level → P); defensive copy */
    public Map<Integer, Double> distribution() {
        return new LinkedHashMap<>(distribution);
    }

    /** @return level string array (nullable) */
    public String[] levels() { return levels; }

    /** {@inheritDoc} */
    @Override
    public double confidence() { return confidence; }

    /** {@inheritDoc} */
    @Override
    public DecisionType type() { return DecisionType.SCORE; }

    /**
     * Map score to nearest level string (round to nearest int, clamp to array bounds)
     * @return nearest level string, or null if levels is empty
     */
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