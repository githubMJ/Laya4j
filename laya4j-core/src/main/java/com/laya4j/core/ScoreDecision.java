package com.laya4j.core;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * {@code SCORE} 原语的决策结果:在一个有序量表(ordinal scale)上给出
 * 加权期望值,而非单一离散标签。
 *
 * <p>由 {@code LayaModel.predict()} 在 {@link Question} 的类型为
 * {@link DecisionType#SCORE} 时返回。
 *
 * <p>输出语义:
 * <pre>
 *   score        = Σ(i × P(level_i)),  i ∈ [0, N)  —— 各等级概率的加权期望值
 *   distribution = 每个等级的独立概率(等级序号 → 概率,理论上求和为 1.0)
 * </pre>
 *
 * <p>相比直接取 argmax 得到的离散等级,加权期望值能反映模型在相邻等级间的
 * "犹豫程度"(例如 score=1.8 意味着模型认为更接近等级 2 但对等级 1 也有一定把握),
 * 在需要平滑排序或与其他连续型信号做加权融合的场景下更有价值。
 *
 * <p>不可变(immutable):构造时对 {@code distribution}/{@code levels}
 * 做防御性拷贝,访问器同样返回拷贝。
 */
public final class ScoreDecision implements Decision {

    private final double score;
    private final double confidence;
    private final Map<Integer, Double> distribution;
    private final String[] levels;

    /**
     * @param score        加权期望值,理论范围 [0, N-1](N 为等级数)
     * @param confidence   基于归一化香农熵的置信度,1.0 表示分布完全集中(最确定)
     * @param distribution 每个等级的独立概率(等级序号 → 概率);
     *                     允许为 {@code null},此时视为空分布
     * @param levels       等级的可读名称数组,供 {@link #nearestLevel()} 使用;
     *                     允许为 {@code null}(此时 {@link #nearestLevel()} 返回 null)
     */
    public ScoreDecision(double score, double confidence, Map<Integer, Double> distribution, String[] levels) {
        this.score = score;
        this.confidence = confidence;
        this.distribution = distribution == null ? Map.of() : new LinkedHashMap<>(distribution);
        this.levels = levels == null ? null : levels.clone();
    }

    /** @return 加权期望值 = Σ(i × P_i),理论范围 [0, N-1] */
    public double score() {
        return score;
    }

    /**
     * 返回每个等级的独立概率分布。
     *
     * @return 等级序号 → 概率的映射(理论上求和为 1.0);每次调用返回防御性拷贝
     */
    public Map<Integer, Double> distribution() {
        return new LinkedHashMap<>(distribution);
    }

    /** @return 等级可读名称数组的防御性拷贝(可能为 {@code null}) */
    public String[] levels() {
        return levels == null ? null : levels.clone();
    }

    /** {@inheritDoc} */
    @Override
    public double confidence() {
        return confidence;
    }

    /** {@inheritDoc} */
    @Override
    public DecisionType type() {
        return DecisionType.SCORE;
    }

    /**
     * 把连续的 {@link #score()} 映射到最接近的等级名称。
     *
     * <p>算法:四舍五入到最近的整数索引,再夹紧(clamp)到
     * {@code levels} 数组的合法下标范围内,避免越界。
     *
     * @return 最接近的等级名称;若 {@code levels} 为 {@code null} 或空数组则返回 {@code null}
     */
    public String nearestLevel() {
        if (levels == null || levels.length == 0) {
            return null;
        }
        int idx = Math.round((float) score);
        idx = Math.max(0, Math.min(levels.length - 1, idx));
        return levels[idx];
    }

    @Override
    public String toString() {
        return "ScoreDecision{score=" + String.format("%.4f", score)
                + ", level=" + nearestLevel()
                + ", confidence=" + String.format("%.4f", confidence) + "}";
    }
}
