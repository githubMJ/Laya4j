package com.laya4j.predict;

import com.laya4j.core.Decision;
import com.laya4j.core.LayaConfig;
import com.laya4j.core.LayaException;
import com.laya4j.core.Question;
import com.laya4j.core.RoutingDecision;
import com.laya4j.model.HuggingFaceFetcher;
import com.laya4j.model.LayaModel;
import com.laya4j.model.LayaOnnxModel;
import com.laya4j.router.LayaRouter;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Laya top-level entry point - wraps {@link LayaRouter}.
 *
 * <p>Thread-safe: a single instance can be shared across threads.
 * Must be {@link #close() closed} to release native ONNX resources
 * (recommended: try-with-resources).
 *
 * <p>Usage:
 * <pre>{@code
 * // local files
 * try (LayaPredictor p = LayaPredictor.single(
 *         Path.of("models/laya-decision-...onnx"),
 *         Path.of("models/tokenizer"), "multilingual")) {
 *     Map<String, Decision> out = p.predict(state, Presets.triage());
 * }
 *
 * // auto-resolve from project models/ dir, ~/.cache/laya, or the HuggingFace Hub
 * try (LayaPredictor p = LayaPredictor.autoLoad()) { ... }
 * }</pre>
 */
public class LayaPredictor implements AutoCloseable {

    private final LayaRouter router;
    private final LayaModel[] allModels;

    private LayaPredictor(LayaRouter router, LayaModel[] allModels) {
        this.router = router;
        this.allModels = allModels;
    }

    /** Convenience: decision-only prediction (routing metadata discarded). */
    public Map<String, Decision> predict(String state, List<Question> questions) {
        return router.predict(state, questions).answers();
    }

    /** Full result: decisions + routing metadata (which checkpoint was used and why). */
    public LayaRouter.Result predictWithRouting(String state, List<Question> questions) {
        return router.predict(state, questions);
    }

    /** Routing-only check (no forward pass). */
    public RoutingDecision route(String state) {
        return router.route(state);
    }

    @Override
    public void close() throws Exception {
        router.close();
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Single-checkpoint predictor backed by explicit local files. */
    public static LayaPredictor single(Path onnxPath, Path tokenizerDir, String modelId) {
        LayaOnnxModel m = new LayaOnnxModel(onnxPath, tokenizerDir, modelId, LayaConfig.defaults());
        LayaRouter r = LayaRouter.builder().defaultModel(m).build();
        return new LayaPredictor(r, new LayaModel[]{m});
    }

    /**
     * Convenience factory: resolve the default multilingual checkpoint
     * (project {@code models/} dir → {@code ~/.cache/laya/} → HuggingFace Hub)
     * and load the given ONNX file.
     *
     * @param onnxPath     ONNX file to load (usually {@code fetched.onnxFile()})
     * @param refreshCache force re-download of tokenizer/config from the Hub
     * @deprecated use {@link #autoLoad()} or {@link #autoLoad(boolean)} instead
     */
    @Deprecated(since = "0.2.0")
    public static LayaPredictor autoLoad(Path onnxPath, boolean refreshCache) throws Exception {
        var fetched = HuggingFaceFetcher.fetchDefault(refreshCache);
        return single(onnxPath, fetched.tokenizerDir(), "multilingual");
    }

    /**
     * Resolve + load the default multilingual checkpoint automatically
     * (project {@code models/} dir → {@code ~/.cache/laya/} → HuggingFace Hub).
     */
    public static LayaPredictor autoLoad() throws Exception {
        return autoLoad(false);
    }

    /** As {@link #autoLoad()} with an optional forced re-download from the Hub. */
    public static LayaPredictor autoLoad(boolean refreshCache) throws Exception {
        var fetched = HuggingFaceFetcher.fetchDefault(refreshCache);
        return single(fetched.onnxFile(), fetched.tokenizerDir(), "multilingual");
    }

    public static final class Builder {
        private Path englishOnnx;
        private Path englishTokenizer;
        private Path multilingualOnnx;
        private Path multilingualTokenizer;
        private Path typedOnnx;
        private Path typedTokenizer;
        private LayaConfig config = LayaConfig.defaults();
        private LayaModel englishModel;
        private LayaModel multilingualModel;
        private LayaModel typedModel;

        public Builder englishOnnx(Path p) { this.englishOnnx = p; return this; }
        public Builder englishTokenizer(Path p) { this.englishTokenizer = p; return this; }
        public Builder multilingualOnnx(Path p) { this.multilingualOnnx = p; return this; }
        public Builder multilingualTokenizer(Path p) { this.multilingualTokenizer = p; return this; }
        public Builder typedDecisionsOnnx(Path p) { this.typedOnnx = p; return this; }
        public Builder typedDecisionsTokenizer(Path p) { this.typedTokenizer = p; return this; }
        public Builder config(LayaConfig c) { this.config = c; return this; }

        /** Register a pre-built model for the english slot (any {@link LayaModel} implementation). */
        public Builder english(LayaModel m) { this.englishModel = m; return this; }

        /** Register a pre-built model for the multilingual slot (any {@link LayaModel} implementation). */
        public Builder multilingual(LayaModel m) { this.multilingualModel = m; return this; }

        /** Register a pre-built model for the typed-decisions slot (any {@link LayaModel} implementation). */
        public Builder typedDecisions(LayaModel m) { this.typedModel = m; return this; }

        public LayaPredictor build() {
            LayaModel en = englishModel != null ? englishModel
                    : englishOnnx != null
                    ? new LayaOnnxModel(englishOnnx, englishTokenizer, "english", config) : null;
            LayaModel ml = multilingualModel != null ? multilingualModel
                    : multilingualOnnx != null
                    ? new LayaOnnxModel(multilingualOnnx, multilingualTokenizer, "multilingual", config) : null;
            LayaModel td = typedModel != null ? typedModel
                    : typedOnnx != null
                    ? new LayaOnnxModel(typedOnnx, typedTokenizer, "typed-decisions", config) : null;

            List<LayaModel> models = new ArrayList<>();
            if (en != null) models.add(en);
            if (ml != null) models.add(ml);
            if (td != null) models.add(td);
            if (models.isEmpty()) {
                throw new LayaException("At least one model required");
            }
            LayaRouter router = LayaRouter.builder()
                    .english(en).multilingual(ml).typedDecisions(td)
                    .defaultModel(models.get(0))
                    .build();
            return new LayaPredictor(router, models.toArray(new LayaModel[0]));
        }
    }
}
