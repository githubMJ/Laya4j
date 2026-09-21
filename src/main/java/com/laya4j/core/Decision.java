package com.laya4j.core;

/**
 * Sealed result types: ChoiceDecision / ScoreDecision / NoulDecision
 *
 * Mirrors Python: res['answers'][qid] = {choice: ..., confidence: ...}
 *
 * Usage:
 * <pre>{@code
 * Decision d = predictor.predict(state, questions).get("intent");
 * if (d instanceof ChoiceDecision cd) {
 *     System.out.println("Choice: " + cd.choice());
 * }
 * }</pre>
 */
public sealed interface Decision
        permits ChoiceDecision, ScoreDecision, NoulDecision {

    /**
     * Calibrated confidence (P(top answer)) ∈ [0.5, 1.0]
     *
     * Semantics:
     *   - CHOICE: max(P(choice_i))
     *   - SCORE:  1 - H(probabilities) / log(N); 1 = fully certain
     *   - NOUL:   max(P(true), P(false))
     *
     * @return confidence ∈ [0.0, 1.0]
     */
    double confidence();

    /**
     * @return the decision primitive type
     */
    DecisionType type();
}