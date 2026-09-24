package com.laya4j.core;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * {@code CHOICE} 原语的决策结果:从候选标签中选出置信度最高的一个,
 * 并附带完整的概率分布。
 *
 * <p>由 {@code LayaModel.predict()} 在 {@link Question} 的类型为
 * {@link DecisionType#CHOICE} 时返回。
 *
 * <p>不可变(immutable):构造时对传入的 {@code probabilities} 做防御性拷贝,
 * {@link #probabilities()} 每次调用也返回新的拷贝,因此外部无法通过持有的
 * 引用修改内部状态,可安全地在多线程间共享。
 */
public final class ChoiceDecision implements Decision {

    private final String choice;
    private final double confidence;
    private final Map<String, Double> probabilities;

    /**
     * @param choice        置信度最高的候选标签(argmax 结果)
     * @param confidence    {@code max(P(choice_i))},取值范围 [0.0, 1.0]
     * @param probabilities 各候选标签的概率分布(标签 → 概率,理论上求和为 1.0);
     *                      构造时会被防御性拷贝,调用方后续修改原 Map 不影响本对象
     * @throws LayaException 当 choice 为空或 probabilities 为 null 时抛出
     */
    public ChoiceDecision(String choice, double confidence, Map<String, Double> probabilities) {
        if (choice == null || choice.isEmpty()) {
            throw new LayaException("ChoiceDecision.choice must not be null/empty");
        }
        Objects.requireNonNull(probabilities, "probabilities must not be null");
        this.choice = choice;
        this.confidence = confidence;
        this.probabilities = new LinkedHashMap<>(probabilities);
    }

    /** @return 置信度最高的候选标签 */
    public String choice() {
        return choice;
    }

    /** {@inheritDoc} */
    @Override
    public double confidence() {
        return confidence;
    }

    /** {@inheritDoc} */
    @Override
    public DecisionType type() {
        return DecisionType.CHOICE;
    }

    /**
     * 返回各候选标签的概率分布。
     *
     * @return 标签 → 概率的映射(理论上求和为 1.0);每次调用返回独立的防御性拷贝,
     *         保持插入顺序(与 {@link Question} 中 choices 的声明顺序一致)
     */
    public Map<String, Double> probabilities() {
        return new LinkedHashMap<>(probabilities);
    }

    @Override
    public String toString() {
        return "ChoiceDecision{choice=" + choice
                + ", confidence=" + String.format("%.4f", confidence) + "}";
    }
}
