package com.laya4j.core;

/**
 * Laya 决策原语类型
 *
 * 参考 laya.common.QTYPES:
 *   choice=0, score=1, noul=2
 *
 * 这三种类型对应三种不同的输出语义:
 *   - CHOICE:  多分类,从候选标签里选一个
 *   - SCORE:   序数分级,输出 0~N-1 的浮点期望
 *   - NOUL:    是/否概率,P(true) ∈ [0, 1]
 */
public enum DecisionType {
    CHOICE(0),
    SCORE(1),
    NOUL(2);

    private final int id;

    DecisionType(int id) {
        this.id = id;
    }

    public int id() {
        return id;
    }

    public static DecisionType fromId(int id) {
        for (DecisionType t : values()) {
            if (t.id == id) return t;
        }
        throw new IllegalArgumentException("Unknown qtype id: " + id);
    }
}