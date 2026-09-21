package com.laya4j.core;

/**
 * NOUL result: calibrated P(true) ∈ [0, 1]
 *
 * Returned by LayaOnnxModel.predict() when qtype=NOUL.
 *
 * markers[0]=false (default 'no, the statement does not hold')
 * markers[1]=true (default 'yes, the statement holds')
 */
public final class NoulDecision implements Decision {

    private final double probability;
    private final double confidence;

    /**
     * @param probability P(true) ∈ [0, 1] (softmax-normalized)
     * @param confidence  max(P(true), P(false))
     */
    public NoulDecision(double probability, double confidence) {
        this.probability = probability;
        this.confidence = confidence;
    }

    /** @return P(true) ∈ [0, 1] */
    public double probability() { return probability; }

    /**
     * Threshold to boolean
     * @param threshold classification threshold, typically 0.5
     * @return probability >= threshold
     */
    public boolean asBoolean(double threshold) {
        return probability >= threshold;
    }

    /** {@inheritDoc} */
    @Override
    public double confidence() { return confidence; }

    /** {@inheritDoc} */
    @Override
    public DecisionType type() { return DecisionType.NOUL; }

    @Override
    public String toString() {
        return "NoulDecision{P(true)=" + String.format("%.4f", probability) +
                ", confidence=" + String.format("%.4f", confidence) + "}";
    }
}