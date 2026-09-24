package com.laya4j.core;

/**
 * Laya 决策原语类型枚举。
 *
 * <p>对应 Python 端 {@code laya.common.QTYPES} 常量表:
 * {@code {"choice": 0, "score": 1, "noul": 2}}。该整数 ID 会作为
 * ONNX 模型的 {@code qtype} 输入张量的值,决定推理时走哪一种解码路径,
 * 因此 Java 侧的枚举顺序 / id 必须与 Python 端严格保持一致,不可随意调整。
 *
 * <p>三种类型对应不同的输出语义:
 * <ul>
 *   <li>{@link #CHOICE} —— 多分类,从候选标签中挑一个(argmax)</li>
 *   <li>{@link #SCORE}  —— 序数量表,输出 0..N-1 区间的期望值(浮点数)</li>
 *   <li>{@link #NOUL}   —— 是非概率,输出校准过的 P(true) ∈ [0, 1]</li>
 * </ul>
 *
 * @see Question 定义某个具体问题使用哪种决策类型
 * @see Decision 决策类型对应的结果密封接口
 */
public enum DecisionType {

    /** 多分类:从候选标签中选择置信度最高的一个(对结果取 argmax)。 */
    CHOICE(0),

    /** 序数量表:输出 0..N-1 区间的加权期望值(浮点数,而非离散标签)。 */
    SCORE(1),

    /** 是非概率:输出经过 RLCD 校准的 P(true) ∈ [0, 1]。 */
    NOUL(2);

    private final int id;

    DecisionType(int id) {
        this.id = id;
    }

    /**
     * 返回与 {@code laya.common.QTYPES} 一致的整数 ID。
     *
     * <p>该值会被直接写入 ONNX 推理的 {@code qtype} 输入张量,
     * 是 Java 与 Python 两端协议对齐的关键常量。
     *
     * @return ONNX {@code qtype} 输入值:0=choice、1=score、2=noul
     */
    public int id() {
        return id;
    }

    /**
     * 根据 ONNX {@code qtype} 整数值反查对应的枚举常量。
     *
     * <p>典型用途:从模型输出或序列化数据中恢复出的原始 int 类型码,
     * 转换回类型安全的 {@link DecisionType}。
     *
     * @param id ONNX qtype 输入值(必须是 0、1 或 2)
     * @return 对应的 {@link DecisionType}
     * @throws IllegalArgumentException 当 id 不是 0/1/2 时抛出
     */
    public static DecisionType fromId(int id) {
        for (DecisionType t : values()) {
            if (t.id == id) {
                return t;
            }
        }
        throw new IllegalArgumentException("Unknown qtype id: " + id);
    }
}
