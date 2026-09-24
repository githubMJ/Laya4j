package com.laya4j.core;

/**
 * 路由决策的元数据:记录本次推理最终选用了哪个模型 checkpoint,以及判定依据。
 *
 * <p>由 {@code LayaRouter} 在完成脚本/语言检测并选定 checkpoint 后产出,
 * 对应 Python 端 {@code res['routing'] = {"model": ..., "repo": ..., "reason": ...}}
 * 的返回结构。即便不需要实际推理,也可以单独调用路由检查
 * (例如 {@code LayaPredictor.route(state)})来预览"这段文本会被路由到哪个模型"。
 *
 * @param model   选中的模型标识,例如:
 *                <ul>
 *                  <li>{@code "english"}         —— 英文专用 checkpoint</li>
 *                  <li>{@code "multilingual"}    —— 多语种 checkpoint(100+ 语言)</li>
 *                  <li>{@code "typed-decisions"} —— 面向结构化决策微调的 checkpoint</li>
 *                </ul>
 * @param repo    该模型对应的 HuggingFace 仓库路径,便于溯源/下载
 * @param reason  路由判定的可读原因说明(例如"检测到非拉丁文字,选择多语种模型")
 * @param profile 触发本次路由判定的语言/文字检测结果详情
 */
public record RoutingDecision(String model, String repo, String reason, DetectionProfile profile) {

    /**
     * 输入文本的语言/文字体系检测结果。
     *
     * @param dominantScript   占比最高的 Unicode 文字体系,取值集合:
     *                         {@code han}(汉字)/ {@code kana}(假名)/
     *                         {@code hangul}(谚文)/ {@code latin}(拉丁字母)/
     *                         {@code cyrillic}(西里尔字母)/ {@code arabic}(阿拉伯字母)
     * @param latinFraction    拉丁字母字符占比,取值范围 [0.0, 1.0]
     * @param nonLatinFraction 非拉丁字母字符占比,取值范围 [0.0, 1.0]
     * @param language         检测出的语言代码,例如 {@code zh}/{@code en}/{@code ja}/
     *                         {@code ko}/{@code ru}/{@code ar}/{@code de}/{@code es} 等
     * @param isEnglish        是否被判定为英文(直接影响是否路由到 english checkpoint)
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
