package com.laya4j.core;

/**
 * Laya model global configuration
 *
 * Mirrors rl_agent_config.json (base defaults)
 *
 * @param maxLen      max sequence length (multilingual=1024, typed-decisions=1024)
 * @param headMaxLen  schema portion max length (default 256)
 * @param maxPrefixes max schema prefixes
 * @param temperature [choice, score, noul] temperatures (untuned [1.0, 1.0, 1.0])
 */
public record LayaConfig(
        int maxLen,
        int headMaxLen,
        int maxPrefixes,
        double[] temperature
) {
    /**
     * Base default config (multilingual):
     *   maxLen=1024, headMaxLen=256, maxPrefixes=6, temperature=[1, 1, 1]
     *
     * @return default config instance
     */
    public static LayaConfig defaults() {
        return new LayaConfig(1024, 256, 6, new double[]{1.0, 1.0, 1.0});
    }
}