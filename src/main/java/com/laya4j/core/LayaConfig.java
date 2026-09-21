package com.laya4j.core;

/**
 * Laya 模型全局配置
 *
 * 对应 rl_agent_config.json 的字段(基座默认值)
 */
public record LayaConfig(
        int maxLen,        // 最大序列长度(multilingual=1024, typed-decisions=1024)
        int headMaxLen,    // schema 部分的最大长度(默认 256)
        int maxPrefixes,   // schema prefix 最大个数
        double[] temperature  // [choice, score, noul] 各自的温度
) {
    public static LayaConfig defaults() {
        return new LayaConfig(1024, 256, 6, new double[]{1.0, 1.0, 1.0});
    }
}