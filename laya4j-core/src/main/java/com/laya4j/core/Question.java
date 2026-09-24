package com.laya4j.core;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 单个 Laya 问题(question)定义:描述"要模型回答什么、以什么形式回答"。
 *
 * <p>对应 Laya Python 端的三种 schema:
 * <pre>
 *   {"type": "choice", "instructions": "...", "criteria": {"billing": "...", ...}}
 *   {"type": "score",  "instructions": "...", "criteria": ["low", "medium", ...]}
 *   {"type": "noul",   "instructions": "..."}
 * </pre>
 *
 * <p>本类使用 Builder 模式构造,通过静态工厂方法
 * {@link #choice(String, String, Map)}、{@link #score(String, String, String...)}、
 * {@link #noul(String, String)} 按类型创建;字段一旦构造完成即不可变,
 * 所有返回集合/数组类型的访问器都提供防御性拷贝。
 *
 * <p>典型用法:
 * <pre>{@code
 * Question refund = Question.noul("refund", "用户是否要求退款?").build();
 * Question urgency = Question.score("urgency", "紧急程度?", "low", "medium", "critical").build();
 * Map<String, Decision> r = predictor.predict(state, List.of(refund, urgency));
 * }</pre>
 */
public final class Question {

    private final String name;
    private final DecisionType type;
    private final String instructions;
    private final Map<String, String> choices;
    private final String[] levels;
    private final String falseDesc;
    private final String trueDesc;

    private Question(Builder b) {
        this.name = b.name;
        this.type = b.type;
        this.instructions = b.instructions;
        this.choices = b.choices == null ? null : new LinkedHashMap<>(b.choices);
        this.levels = b.levels == null ? null : b.levels.clone();
        this.falseDesc = b.falseDesc;
        this.trueDesc = b.trueDesc;
    }

    /** @return 问题的唯一标识,同时也是返回结果 {@code Map<String, Decision>} 的 key */
    public String name() {
        return name;
    }

    /** @return 该问题使用的决策原语类型 */
    public DecisionType type() {
        return type;
    }

    /** @return 提供给模型的问题说明文本(自然语言) */
    public String instructions() {
        return instructions;
    }

    /**
     * @return {@code CHOICE} 类型的候选标签 → 描述文本映射;
     *         其他类型下为 {@code null}。每次调用返回防御性拷贝。
     */
    public Map<String, String> choices() {
        return choices == null ? null : new LinkedHashMap<>(choices);
    }

    /**
     * @return {@code SCORE} 类型的等级名称数组(按等级 0..N-1 排列);
     *         其他类型下为 {@code null}。每次调用返回防御性拷贝。
     */
    public String[] levels() {
        return levels == null ? null : levels.clone();
    }

    /** @return {@code NOUL} 类型中 "false" 分支的描述文本;其他类型下为 {@code null} */
    public String falseDesc() {
        return falseDesc;
    }

    /** @return {@code NOUL} 类型中 "true" 分支的描述文本;其他类型下为 {@code null} */
    public String trueDesc() {
        return trueDesc;
    }

    /**
     * 把本问题的候选项渲染成模型可读的文本数组,对应 Python 端
     * {@code laya.common.render_options}。返回数组的每个元素会与模型输出的
     * 一个 marker 位置一一对应,顺序至关重要,不可随意调整。
     *
     * <p>各类型的渲染规则:
     * <ul>
     *   <li>{@code CHOICE}: 每个候选项渲染为 {@code "标签: 描述"}
     *       (描述为空时仅保留标签本身)</li>
     *   <li>{@code SCORE}: 每个等级渲染为 {@code "level {i}: 名称"}</li>
     *   <li>{@code NOUL}: 固定两项,{@code "false: 描述"} 与 {@code "true:  描述"}
     *       (为与 Python 端 tokenizer 对齐,{@code true} 分支前缀刻意保留两个空格)</li>
     * </ul>
     *
     * @return 渲染后的候选项文本数组,与 marker 顺序严格对应
     */
    public String[] renderOptions() {
        return switch (type) {
            case CHOICE -> renderChoiceOptions();
            case SCORE -> renderScoreOptions();
            case NOUL -> renderNoulOptions();
        };
    }

    private String[] renderChoiceOptions() {
        String[] arr = new String[choices.size()];
        int i = 0;
        for (Map.Entry<String, String> e : choices.entrySet()) {
            String desc = e.getValue();
            arr[i++] = (desc == null || desc.isEmpty()) ? e.getKey() : e.getKey() + ": " + desc;
        }
        return arr;
    }

    private String[] renderScoreOptions() {
        String[] r = new String[levels.length];
        for (int i = 0; i < levels.length; i++) {
            r[i] = "level " + i + ": " + (levels[i] == null ? "" : levels[i]);
        }
        return r;
    }

    private String[] renderNoulOptions() {
        return new String[]{
                "false: " + (falseDesc == null || falseDesc.isEmpty()
                        ? "no, the statement does not hold" : falseDesc),
                // 注意:"true:" 后保留两个空格,是与 Python tokenizer 对齐的必要格式
                "true:  " + (trueDesc == null || trueDesc.isEmpty()
                        ? "yes, the statement holds" : trueDesc)
        };
    }

    /**
     * 创建一个 {@code CHOICE} 类型问题的 builder。
     *
     * @param name         问题唯一标识
     * @param instructions 问题说明文本
     * @param choices      候选标签 → 描述文本映射(至少一项,建议用
     *                     {@link LinkedHashMap} 以保证候选项顺序稳定)
     * @return 待 {@link Builder#build()} 完成构造的 builder
     */
    public static Builder choice(String name, String instructions, Map<String, String> choices) {
        return new Builder().name(name).type(DecisionType.CHOICE)
                .instructions(instructions).choices(choices);
    }

    /**
     * 创建一个 {@code SCORE} 类型问题的 builder。
     *
     * @param name         问题唯一标识
     * @param instructions 问题说明文本
     * @param levels       等级名称数组,按 0..N-1 顺序排列(至少一项)
     * @return 待 {@link Builder#build()} 完成构造的 builder
     */
    public static Builder score(String name, String instructions, String... levels) {
        return new Builder().name(name).type(DecisionType.SCORE)
                .instructions(instructions).levels(levels);
    }

    /**
     * 创建一个 {@code NOUL} 类型问题的 builder,使用默认的 true/false 描述文本。
     *
     * @param name         问题唯一标识
     * @param instructions 问题说明文本
     * @return 待 {@link Builder#build()} 完成构造的 builder
     */
    public static Builder noul(String name, String instructions) {
        return new Builder().name(name).type(DecisionType.NOUL)
                .instructions(instructions)
                .falseDesc("no, the statement does not hold")
                .trueDesc("yes, the statement holds");
    }

    /**
     * 创建一个 {@code NOUL} 类型问题的 builder,使用自定义的 true/false 描述文本。
     *
     * <p>自定义描述可以让模型更准确地理解业务语境,例如把默认的
     * "yes/no, the statement holds" 换成更贴合场景的
     * "yes, this is a refund request" / "no, this is not a refund request"。
     *
     * @param name         问题唯一标识
     * @param instructions 问题说明文本
     * @param falseDesc    "false" 分支的描述文本
     * @param trueDesc     "true" 分支的描述文本
     * @return 待 {@link Builder#build()} 完成构造的 builder
     */
    public static Builder noul(String name, String instructions, String falseDesc, String trueDesc) {
        return new Builder().name(name).type(DecisionType.NOUL)
                .instructions(instructions)
                .falseDesc(falseDesc).trueDesc(trueDesc);
    }

    /**
     * {@link Question} 的构造器,提供流式(fluent)API。
     *
     * <p>推荐通过 {@link Question#choice}、{@link Question#score}、
     * {@link Question#noul} 等静态工厂方法获取预填充好类型相关字段的实例,
     * 而不是直接 {@code new Builder()}。
     */
    public static final class Builder {
        private String name;
        private DecisionType type;
        private String instructions;
        private Map<String, String> choices;
        private String[] levels;
        private String falseDesc;
        private String trueDesc;

        /** 设置问题唯一标识。 */
        public Builder name(String name) {
            this.name = name;
            return this;
        }

        /** 设置决策原语类型。 */
        public Builder type(DecisionType type) {
            this.type = type;
            return this;
        }

        /** 设置提供给模型的问题说明文本。 */
        public Builder instructions(String instructions) {
            this.instructions = instructions;
            return this;
        }

        /** 设置 {@code CHOICE} 类型的候选标签 → 描述文本映射。 */
        public Builder choices(Map<String, String> choices) {
            this.choices = choices;
            return this;
        }

        /** 设置 {@code SCORE} 类型的等级名称数组。 */
        public Builder levels(String[] levels) {
            this.levels = levels;
            return this;
        }

        /** 设置 {@code NOUL} 类型 "false" 分支的描述文本。 */
        public Builder falseDesc(String falseDesc) {
            this.falseDesc = falseDesc;
            return this;
        }

        /** 设置 {@code NOUL} 类型 "true" 分支的描述文本。 */
        public Builder trueDesc(String trueDesc) {
            this.trueDesc = trueDesc;
            return this;
        }

        /**
         * 校验必填字段并构造不可变的 {@link Question} 实例。
         *
         * @return 构造完成的 {@link Question}
         * @throws IllegalStateException 当 name/type/instructions 缺失,
         *         或 CHOICE 类型缺少 choices,或 SCORE 类型缺少 levels 时抛出
         */
        public Question build() {
            if (name == null || type == null || instructions == null) {
                throw new IllegalStateException("name/type/instructions required");
            }
            if (type == DecisionType.CHOICE && (choices == null || choices.isEmpty())) {
                throw new IllegalStateException("CHOICE 需要 choices");
            }
            if (type == DecisionType.SCORE && (levels == null || levels.length == 0)) {
                throw new IllegalStateException("SCORE 需要 levels");
            }
            return new Question(this);
        }
    }
}
