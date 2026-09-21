package com.laya4j.core;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * CHOICE result: pick one from candidate labels with full probability distribution
 *
 * Returned by LayaOnnxModel.predict() when qtype=CHOICE.
 */
public final class ChoiceDecision implements Decision {

    private final String choice;
    private final double confidence;
    private final Map<String, Double> probabilities;

    /**
     * @param choice        top-probability label
     * @param confidence   max(P(choice_i)) ∈ [0.5, 1.0]
     * @param probabilities per-option P(label → P, sums to 1.0, immutable copy)
     */
    public ChoiceDecision(String choice, double confidence,
                          Map<String, Double> probabilities) {
        this.choice = choice;
        this.confidence = confidence;
        this.probabilities = probabilities;
    }

    /** @return top-probability label */
    public String choice() { return choice; }

    /** {@inheritDoc} */
    @Override
    public double confidence() { return confidence; }

    /** {@inheritDoc} */
    @Override
    public DecisionType type() { return DecisionType.CHOICE; }

    /**
     * @return per-option probabilities (label → P, sums to 1.0); defensive copy
     */
    public Map<String, Double> probabilities() {
        return new LinkedHashMap<>(probabilities);
    }

    @Override
    public String toString() {
        return "ChoiceDecision{choice=" + choice + ", confidence=" +
                String.format("%.4f", confidence) + "}";
    }
}