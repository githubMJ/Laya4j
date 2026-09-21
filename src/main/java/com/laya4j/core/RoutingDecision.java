package com.laya4j.core;

/**
 * Routing decision metadata
 *
 * Mirrors Python: res['routing'] = {model: ..., repo: ..., reason: ...}
 *
 * model values:
 *   "english"         → use laya (ModernBERT-large, English)
 *   "multilingual"    → use laya-multilingual (mmBERT-base, 100+ langs)
 *   "typed-decisions" → use laya-typed-decisions (ModernBERT-large)
 */
public record RoutingDecision(
        String model,
        String repo,
        String reason,
        DetectionProfile profile
) {
    /**
     * Detected language profile
     *
     * @param dominantScript dominant Unicode script (han / kana / hangul / latin / cyrillic / arabic)
     * @param latinFraction  Latin character fraction ∈ [0, 1]
     * @param nonLatinFraction non-Latin character fraction ∈ [0, 1]
     * @param language       detected language code (zh / en / ja / ko / ru / ar / de / es / ...)
     * @param isEnglish      whether treated as English (affects english checkpoint selection)
     */
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