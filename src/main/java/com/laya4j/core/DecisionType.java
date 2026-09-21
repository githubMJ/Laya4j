package com.laya4j.core;

/**
 * Laya decision primitive types
 *
 * Reference: laya.common.QTYPES
 *   choice=0, score=1, noul=2
 *
 * The three types correspond to three different output semantics:
 *   - CHOICE:  multi-class, pick one from candidate labels
 *   - SCORE:   ordinal scale, output expected value in 0..N-1
 *   - NOUL:    yes/no probability, P(true) ∈ [0, 1]
 */
public enum DecisionType {
    /** Multi-class: pick one from candidate labels (argmax of probabilities) */
    CHOICE(0),
    /** Ordinal scale: output expected value (float) in 0..N-1 */
    SCORE(1),
    /** Yes/no probability: output P(true) ∈ [0, 1] */
    NOUL(2);

    private final int id;

    DecisionType(int id) {
        this.id = id;
    }

    /**
     * @return Integer ID matching laya.common.QTYPES (ONNX qtype input)
     */
    public int id() {
        return id;
    }

    /**
     * Reverse-lookup enum from ONNX qtype integer ID.
     *
     * @param id ONNX qtype input value (0=choice, 1=score, 2=noul)
     * @return the corresponding DecisionType
     * @throws IllegalArgumentException if id is not 0/1/2
     */
    public static DecisionType fromId(int id) {
        for (DecisionType t : values()) {
            if (t.id == id) return t;
        }
        throw new IllegalArgumentException("Unknown qtype id: " + id);
    }
}