package com.laya4j.core;

/**
 * NOUL 类型结果:校准的 P(true) ∈ [0, 1]
 */
public final class NoulDecision implements Decision {

    private final double probability;     // P(true)
    private final double confidence;      // max(p, 1-p)

    public NoulDecision(double probability, double confidence) {
        this.probability = probability;
        this.confidence = confidence;
    }

    public double probability() { return probability; }

    /** 阈值判断:true / false */
    public boolean asBoolean(double threshold) {
        return probability >= threshold;
    }

    @Override
    public double confidence() { return confidence; }

    @Override
    public DecisionType type() { return DecisionType.NOUL; }

    @Override
    public String toString() {
        return "NoulDecision{P(true)=" + String.format("%.4f", probability) +
                ", confidence=" + String.format("%.4f", confidence) + "}";
    }
}