package com.laya4j.router;

import com.laya4j.core.Decision;
import com.laya4j.core.LayaException;
import com.laya4j.core.Question;
import com.laya4j.core.RoutingDecision;
import com.laya4j.model.LayaModel;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Laya multi-checkpoint router.
 *
 * <p>Ports {@code laya.Router}:
 * <ul>
 *   <li>detect input script/language via {@link ScriptDetector}</li>
 *   <li>pick the matching checkpoint:
 *     <ul>
 *       <li>non-Latin / non-English → multilingual</li>
 *       <li>Latin + English → english</li>
 *       <li>Latin + non-English (de/fr/es/pt/...) → multilingual</li>
 *     </ul>
 *   </li>
 *   <li>return Decision map + {@link RoutingDecision}</li>
 * </ul>
 *
 * <p>Works with any {@link LayaModel} implementation; a single model can also be
 * used as a router (always routing to the same checkpoint via {@code defaultModel}).
 *
 * <p>Usage:
 * <pre>{@code
 * LayaRouter router = LayaRouter.builder()
 *     .multilingual(multilingualModel)
 *     .english(englishModel)
 *     .build();
 * Result r = router.predict(state, questions);
 * r.answers().forEach((k, v) -> System.out.println(k + ": " + v));
 * }</pre>
 */
public class LayaRouter {

    /** Default HuggingFace repo used for the informational {@code repo} field. */
    public static final String DEFAULT_REPO_BASE = "convaiinnovations/laya";

    private final LayaModel defaultModel;
    private final LayaModel englishModel;
    private final LayaModel multilingualModel;
    private final LayaModel typedDecisionsModel;
    private final ScriptDetector detector;
    private final String repoBase;

    private LayaRouter(Builder b) {
        this.defaultModel = b.defaultModel;
        this.englishModel = b.englishModel;
        this.multilingualModel = b.multilingualModel;
        this.typedDecisionsModel = b.typedDecisionsModel;
        this.detector = b.detector != null ? b.detector : new ScriptDetector();
        this.repoBase = b.repoBase != null ? b.repoBase : DEFAULT_REPO_BASE;
        if (defaultModel == null && englishModel == null && multilingualModel == null
                && typedDecisionsModel == null) {
            throw new LayaException("At least one model required");
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Single-prompt inference entry: route + forward in one call. */
    public Result predict(String state, List<Question> questions) {
        if (state == null) state = "";
        ScriptDetector.DetectionResult det = detector.detect(state);
        LayaModel model = selectModel(det);
        Map<String, Decision> answers = model.predict(state, questions);
        return new Result(answers, routingFor(det, model));
    }

    /** Routing-only check (no forward pass). */
    public RoutingDecision route(String state) {
        ScriptDetector.DetectionResult det = detector.detect(state);
        return routingFor(det, selectModel(det));
    }

    private RoutingDecision routingFor(ScriptDetector.DetectionResult det, LayaModel model) {
        return new RoutingDecision(
                model.modelId(),
                repoBase + "/" + model.modelId(),
                buildReason(det),
                new RoutingDecision.DetectionProfile(
                        det.dominantScript(),
                        det.latinFraction(),
                        det.nonLatinFraction(),
                        det.language(),
                        det.isEnglish()
                )
        );
    }

    private LayaModel selectModel(ScriptDetector.DetectionResult det) {
        // Routing strategy (mirrors laya.Router):
        //   1. non-Latin → multilingual
        //   2. Latin + English → english
        //   3. Latin + non-English → multilingual
        //   4. fallback chain: default → english → typed-decisions
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

    private String buildReason(ScriptDetector.DetectionResult det) {
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

    /** Combined routing + decision result. */
    public record Result(Map<String, Decision> answers, RoutingDecision routing) {}

    public static final class Builder {
        private LayaModel defaultModel;
        private LayaModel englishModel;
        private LayaModel multilingualModel;
        private LayaModel typedDecisionsModel;
        private ScriptDetector detector;
        private String repoBase;

        public Builder defaultModel(LayaModel m) { this.defaultModel = m; return this; }
        public Builder english(LayaModel m) { this.englishModel = m; return this; }
        public Builder multilingual(LayaModel m) { this.multilingualModel = m; return this; }
        public Builder typedDecisions(LayaModel m) { this.typedDecisionsModel = m; return this; }
        public Builder detector(ScriptDetector d) { this.detector = d; return this; }

        /**
         * Override the HuggingFace repo prefix used in {@link RoutingDecision#repo()}
         * (default {@code convaiinnovations/laya}).
         */
        public Builder repoBase(String repoBase) { this.repoBase = repoBase; return this; }

        public LayaRouter build() {
            return new LayaRouter(this);
        }
    }

    /**
     * Closes all registered models exactly once. The same instance registered in
     * multiple slots (or as default) is closed a single time.
     */
    public void close() throws Exception {
        Set<LayaModel> closed = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
        LayaModel[] all = {defaultModel, englishModel, multilingualModel, typedDecisionsModel};
        for (LayaModel m : all) {
            if (m != null && closed.add(m)) {
                m.close();
            }
        }
    }
}
