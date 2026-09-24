package com.laya4j.core;

/**
 * Laya 决策结果的封闭接口(sealed interface)。
 *
 * <p>Laya 的每次推理输出都会归入以下三种原语之一:
 * <ul>
 *   <li>{@link ChoiceDecision} —— 多分类:从候选标签里选一个</li>
 *   <li>{@link ScoreDecision}  —— 序数打分:输出 0..N-1 区间的期望值</li>
 *   <li>{@link NoulDecision}   —— 是非概率:输出校准过的 P(true) ∈ [0,1]</li>
 * </ul>
 *
 * <p>使用 {@code sealed} 声明的好处是:调用方可以用 {@code switch} 表达式
 * 或 {@code instanceof} 模式匹配穷尽处理所有分支,编译器会在缺漏分支时报错,
 * 无需再依赖运行时的 {@code instanceof} 链式判断。
 *
 * <p>对应 Python 端 {@code laya} 库的返回结构:
 * {@code res['answers'][question_id] = {"choice": ..., "confidence": ...}}
 *
 * <p>典型用法:
 * <pre>{@code
 * Decision d = predictor.predict(state, questions).get("intent");
 * switch (d) {
 *     case ChoiceDecision cd -> System.out.println("选择: " + cd.choice());
 *     case ScoreDecision sd  -> System.out.println("打分: " + sd.score());
 *     case NoulDecision nd   -> System.out.println("P(true)=" + nd.probability());
 * }
 * }</pre>
 *
 * @see DecisionType 三种决策原语对应的类型枚举
 */
public sealed interface Decision permits ChoiceDecision, ScoreDecision, NoulDecision {

    /**
     * 返回该决策的校准置信度(calibrated confidence),取值范围 [0.0, 1.0]。
     *
     * <p>置信度含义随决策类型不同而不同:
     * <ul>
     *   <li>{@code CHOICE}: {@code max(P(choice_i))},即最优候选的概率</li>
     *   <li>{@code SCORE}:  {@code 1 - H(probabilities) / log(N)},
     *       基于归一化香农熵,值越接近 1 表示分布越集中(越确定)</li>
     *   <li>{@code NOUL}:   {@code max(P(true), P(false))}</li>
     * </ul>
     *
     * <p>由于模型使用 RLCD(Reinforcement Learning with Calibrated Decisions)
     * 训练,该置信度经过校准,可以直接作为业务决策阈值使用(例如
     * "confidence &lt; 0.6 时转人工复核"),而不仅仅是 softmax 输出的原始最大值。
     *
     * @return 置信度,恒为 [0.0, 1.0] 区间内的值
     */
    double confidence();

    /**
     * 返回该决策所属的原语类型,用于在不方便使用 {@code switch}
     * 模式匹配的场景下做类型分派(例如反射式的批量结果收集)。
     *
     * @return 决策原语类型:{@link DecisionType#CHOICE}、
     *         {@link DecisionType#SCORE} 或 {@link DecisionType#NOUL}
     */
    DecisionType type();
}
