package com.laya4j.core;

/**
 * {@code NOUL} 原语的决策结果:经过 RLCD 校准的是非概率。
 *
 * <p>"Noul" 一词源自 Laya Python 端的术语,是 "no/yes" 与 "null"(空判断原语)
 * 的合成造词,代表一个不依赖固定阈值、天然带有不确定性表达的是非判断原语——
 * 相比传统二分类器只给出 0/1 标签,NOUL 输出的是可直接用于风险加权、
 * 排序、多级阈值分流等场景的连续概率值。
 *
 * <p>由 {@code LayaModel.predict()} 在 {@link Question} 的类型为
 * {@link DecisionType#NOUL} 时返回。其内部对应模型的两个 marker 位置:
 * {@code markers[0]} = false(默认描述 "no, the statement does not hold"),
 * {@code markers[1]} = true(默认描述 "yes, the statement holds")。
 *
 * <p>不可变(immutable),线程安全。
 */
public final class NoulDecision implements Decision {

    private final double probability;
    private final double confidence;

    /**
     * @param probability P(true),取值范围 [0.0, 1.0](softmax 归一化后的结果)
     * @param confidence  {@code max(P(true), P(false))},取值范围 [0.5, 1.0]
     * @throws LayaException 当 probability 或 confidence 不在 [0.0, 1.0] 区间时抛出
     */
    public NoulDecision(double probability, double confidence) {
        if (probability < 0.0 || probability > 1.0 || Double.isNaN(probability)) {
            throw new LayaException("probability must be in [0,1], got " + probability);
        }
        if (confidence < 0.0 || confidence > 1.0 || Double.isNaN(confidence)) {
            throw new LayaException("confidence must be in [0,1], got " + confidence);
        }
        this.probability = probability;
        this.confidence = confidence;
    }

    /** @return P(true),取值范围 [0.0, 1.0] */
    public double probability() {
        return probability;
    }

    /**
     * 按给定阈值把连续概率转换为布尔判断。
     *
     * <p>由于概率已经过校准,大多数场景下 {@code threshold=0.5} 即可得到
     * 合理的判断;对精确率/召回率有特殊要求的场景(如内容安全审核),
     * 可以传入更高或更低的阈值做业务侧调优,而无需重新训练模型。
     *
     * @param threshold 分类阈值,常用 0.5
     * @return {@code probability() >= threshold}
     */
    public boolean asBoolean(double threshold) {
        return probability >= threshold;
    }

    /** {@inheritDoc} */
    @Override
    public double confidence() {
        return confidence;
    }

    /** {@inheritDoc} */
    @Override
    public DecisionType type() {
        return DecisionType.NOUL;
    }

    @Override
    public String toString() {
        return "NoulDecision{P(true)=" + String.format("%.4f", probability)
                + ", confidence=" + String.format("%.4f", confidence) + "}";
    }
}
