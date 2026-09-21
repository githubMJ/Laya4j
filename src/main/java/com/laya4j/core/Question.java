package com.laya4j.core;

import java.util.Map;

/**
 * 单个 Laya 问题定义
 *
 * 对应 laya Python 端:
 *   {"type": "choice", "instructions": "...", "criteria": {"billing": "...", ...}}
 *   {"type": "score",  "instructions": "...", "criteria": ["low", "medium", ...]}
 *   {"type": "noul",   "instructions": "..."}
 *
 * 这里用 builder 模式构造,字段语义不变
 */
public final class Question {

    private final String name;             // question id
    private final DecisionType type;
    private final String instructions;
    private final Map<String, String> choices;  // CHOICE: label -> description
    private final String[] levels;              // SCORE:  levels
    private final String falseDesc;             // NOUL:   false 的描述
    private final String trueDesc;              // NOUL:   true 的描述

    private Question(Builder b) {
        this.name = b.name;
        this.type = b.type;
        this.instructions = b.instructions;
        this.choices = b.choices;
        this.levels = b.levels;
        this.falseDesc = b.falseDesc;
        this.trueDesc = b.trueDesc;
    }

    public String name() { return name; }
    public DecisionType type() { return type; }
    public String instructions() { return instructions; }
    public Map<String, String> choices() { return choices; }
    public String[] levels() { return levels; }
    public String falseDesc() { return falseDesc; }
    public String trueDesc() { return trueDesc; }

    /** 渲染 options 文本(对应 laya.common.render_options) */
    public String[] renderOptions() {
        switch (type) {
            case CHOICE:
                String[] arr = new String[choices.size()];
                int i = 0;
                for (Map.Entry<String, String> e : choices.entrySet()) {
                    String desc = e.getValue();
                    arr[i++] = (desc == null || desc.isEmpty())
                        ? e.getKey()
                        : e.getKey() + ": " + desc;
                }
                return arr;
            case SCORE:
                String[] r = new String[levels.length];
                for (int j = 0; j < levels.length; j++) {
                    r[j] = "level " + j + ": " + (levels[j] == null ? "" : levels[j]);
                }
                return r;
            case NOUL:
                return new String[]{
                    "false: " + (falseDesc == null || falseDesc.isEmpty()
                        ? "no, the statement does not hold" : falseDesc),
                    "true:  " + (trueDesc == null || trueDesc.isEmpty()
                        ? "yes, the statement holds" : trueDesc)
                };
            default:
                throw new IllegalStateException();
        }
    }

    public static Builder choice(String name, String instructions, Map<String, String> choices) {
        return new Builder().setName(name).setType(DecisionType.CHOICE)
                .setInstructions(instructions).setChoices(choices);
    }

    public static Builder score(String name, String instructions, String... levels) {
        return new Builder().setName(name).setType(DecisionType.SCORE)
                .setInstructions(instructions).setLevels(levels);
    }

    public static Builder noul(String name, String instructions) {
        return new Builder().setName(name).setType(DecisionType.NOUL)
                .setInstructions(instructions)
                .setFalseDesc("no, the statement does not hold")
                .setTrueDesc("yes, the statement holds");
    }

    public static Builder noul(String name, String instructions, String falseDesc, String trueDesc) {
        return new Builder().setName(name).setType(DecisionType.NOUL)
                .setInstructions(instructions)
                .setFalseDesc(falseDesc).setTrueDesc(trueDesc);
    }

    public static final class Builder {
        private String name;
        private DecisionType type;
        private String instructions;
        private Map<String, String> choices;
        private String[] levels;
        private String falseDesc;
        private String trueDesc;

        public Builder setName(String name) { this.name = name; return this; }
        public Builder setType(DecisionType t) { this.type = t; return this; }
        public Builder setInstructions(String s) { this.instructions = s; return this; }
        public Builder setChoices(Map<String, String> m) { this.choices = m; return this; }
        public Builder setLevels(String[] a) { this.levels = a; return this; }
        public Builder setFalseDesc(String s) { this.falseDesc = s; return this; }
        public Builder setTrueDesc(String s) { this.trueDesc = s; return this; }

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