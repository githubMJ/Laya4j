package com.laya4j.router;

import com.laya4j.core.Decision;
import com.laya4j.core.LayaException;
import com.laya4j.core.Question;
import com.laya4j.core.RoutingDecision;
import com.laya4j.model.LayaOnnxModel;

import java.util.List;
import java.util.Map;

/**
 * Laya 多 checkpoint 路由器
 *
 * 复刻 laya.Router:
 *   - 检测输入脚本/语言
 *   - 选对应 checkpoint
 *     * 非拉丁/英文外 → multilingual
 *     * 拉丁 + 英文 → english
 *     * 拉丁 + 非英文(德/法/西/葡等)→ multilingual
 *   - 返回 Decision map + RoutingDecision
 *
 * 单个 model 也可当作 router(永远用同一个 checkpoint)
 *
 * 用法:
 *   LayaRouter router = new LayaRouter.Builder()
 *       .multilingual(multilingualModel)
 *       .english(englishModel)
 *       .build();
 *   Result r = router.predict(state, questions);
 *   r.answers.forEach((k, v) -> System.out.println(k + ": " + v));
 */
public class LayaRouter {

    private final LayaOnnxModel defaultModel;
    private final LayaOnnxModel englishModel;
    private final LayaOnnxModel multilingualModel;
    private final LayaOnnxModel typedDecisionsModel;
    private final ScriptDetector detector;

    private LayaRouter(Builder b) {
        this.defaultModel = b.defaultModel;
        this.englishModel = b.englishModel;
        this.multilingualModel = b.multilingualModel;
        this.typedDecisionsModel = b.typedDecisionsModel;
        this.detector = b.detector != null ? b.detector : new ScriptDetector();
        if (defaultModel == null && englishModel == null && multilingualModel == null
                && typedDecisionsModel == null) {
            throw new LayaException("At least one model required");
        }
    }

    /** 单条推理入口 */
    public Result predict(String state, List<Question> questions) {
        if (state == null) state = "";
        ScriptDetector.DetectionResult det = detector.detect(state);
        LayaOnnxModel model = selectModel(det);
        String reason = buildReason(det, model.modelId());
        Map<String, Decision> answers = model.predict(state, questions);

        RoutingDecision routing = new RoutingDecision(
                model.modelId(),
                "convaiinnovations/laya/" + model.modelId(),
                reason,
                new RoutingDecision.DetectionProfile(
                        det.dominantScript(),
                        det.latinFraction(),
                        det.nonLatinFraction(),
                        det.language(),
                        det.isEnglish()
                )
        );
        return new Result(answers, routing);
    }

    /** 仅路由检查(不跑 forward) */
    public RoutingDecision route(String state) {
        ScriptDetector.DetectionResult det = detector.detect(state);
        LayaOnnxModel model = selectModel(det);
        return new RoutingDecision(
                model.modelId(),
                "convaiinnovations/laya/" + model.modelId(),
                buildReason(det, model.modelId()),
                new RoutingDecision.DetectionProfile(
                        det.dominantScript(), det.latinFraction(),
                        det.nonLatinFraction(), det.language(), det.isEnglish()
                )
        );
    }

    private LayaOnnxModel selectModel(ScriptDetector.DetectionResult det) {
        // 路由策略(对照 laya.Router):
        //   1. 显式 typed-decisions 优先? 我们保留简单策略,默认 multilingual
        //   2. 非 Latin → multilingual
        //   3. Latin + English → english
        //   4. Latin + 非英语 → multilingual
        if (!det.dominantScript().equals("latin") && multilingualModel != null) {
            return multilingualModel;
        }
        if (det.isEnglish() && englishModel != null) {
            return englishModel;
        }
        if (multilingualModel != null) {
            return multilingualModel;
        }
        return defaultModel != null ? defaultModel
                : englishModel != null ? englishModel : typedDecisionsModel;
    }

    private String buildReason(ScriptDetector.DetectionResult det, String model) {
        if (!det.dominantScript().equals("latin")) {
            return "non-Latin script (" + det.dominantScript() + ", " +
                    String.format("%.0f%%", det.nonLatinFraction() * 100) +
                    " of letters); the English checkpoint cannot read it";
        }
        if (det.isEnglish()) {
            return "English Latin text";
        }
        return "Latin script but language looks like '" + det.language() + "', not English";
    }

    /** 路由 + 决策的完整结果 */
    public record Result(Map<String, Decision> answers, RoutingDecision routing) {}

    public static class Builder {
        private LayaOnnxModel defaultModel;
        private LayaOnnxModel englishModel;
        private LayaOnnxModel multilingualModel;
        private LayaOnnxModel typedDecisionsModel;
        private ScriptDetector detector;

        public Builder defaultModel(LayaOnnxModel m) { this.defaultModel = m; return this; }
        public Builder english(LayaOnnxModel m) { this.englishModel = m; return this; }
        public Builder multilingual(LayaOnnxModel m) { this.multilingualModel = m; return this; }
        public Builder typedDecisions(LayaOnnxModel m) { this.typedDecisionsModel = m; return this; }
        public Builder detector(ScriptDetector d) { this.detector = d; return this; }

        public LayaRouter build() {
            return new LayaRouter(this);
        }
    }

    public void close() throws Exception {
        if (englishModel != null) englishModel.close();
        if (multilingualModel != null) multilingualModel.close();
        if (typedDecisionsModel != null) typedDecisionsModel.close();
    }
}