package com.laya4j.core;

/**
 * 决策结果的 sealed 类型:ChoiceDecision / ScoreDecision / NoulDecision
 *
 * 对应 laya Python 端的 res["answers"][qid] = {"choice": ..., "confidence": ...} 等
 */
public sealed interface Decision
        permits ChoiceDecision, ScoreDecision, NoulDecision {

    /** 该决策的置信度(已校准的 P(top answer),∈ [0.5, 1.0]) */
    double confidence();

    DecisionType type();
}