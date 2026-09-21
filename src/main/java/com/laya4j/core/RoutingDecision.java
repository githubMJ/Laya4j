package com.laya4j.core;

/**
 * 路由决策元数据
 *
 * 对应 laya Python 端 res["routing"] = {"model": ..., "repo": ..., "reason": ...}
 *
 * model 取值:
 *   "english"         → 用 laya (ModernBERT-large, 英文)
 *   "multilingual"    → 用 laya-multilingual (mmBERT-base, 100+ 语言)
 *   "typed-decisions" → 用 laya-typed-decisions (ModernBERT-large)
 */
public record RoutingDecision(
        String model,
        String repo,
        String reason,
        DetectionProfile profile
) {
    /** 路由检测到的语言画像 */
    public record DetectionProfile(
            String dominantScript,
            double latinFraction,
            double nonLatinFraction,
            String language,
            boolean isEnglish
    ) {}

    @Override
    public String toString() {
        return String.format("RoutingDecision{model=%s, lang=%s, reason=%s}",
                model, profile != null ? profile.language() : "?", reason);
    }
}