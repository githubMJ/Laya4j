package com.laya4j.predict;

import com.laya4j.core.Decision;
import com.laya4j.core.LayaConfig;
import com.laya4j.core.LayaException;
import com.laya4j.core.Question;
import com.laya4j.core.RoutingDecision;
import com.laya4j.model.HuggingFaceFetcher;
import com.laya4j.model.LayaOnnxModel;
import com.laya4j.router.LayaRouter;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Laya top-level entry - wraps LayaRouter
 */
public class LayaPredictor implements AutoCloseable {

    private final LayaRouter router;
    private final LayaOnnxModel[] allModels;

    private LayaPredictor(LayaRouter router, LayaOnnxModel[] allModels) {
        this.router = router;
        this.allModels = allModels;
    }

    public Map<String, Decision> predict(String state, List<Question> questions) {
        return router.predict(state, questions).answers();
    }

    public LayaRouter.Result predictWithRouting(String state, List<Question> questions) {
        return router.predict(state, questions);
    }

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

    public static LayaPredictor single(Path onnxPath, Path tokenizerDir, String modelId) {
        LayaOnnxModel m = new LayaOnnxModel(onnxPath, tokenizerDir, modelId, LayaConfig.defaults());
        LayaRouter r = new LayaRouter.Builder().defaultModel(m).build();
        return new LayaPredictor(r, new LayaOnnxModel[]{m});
    }

    public static LayaPredictor autoLoad(Path onnxPath, boolean refreshCache) throws Exception {
        var fetched = HuggingFaceFetcher.fetchDefault(refreshCache);
        return single(onnxPath, fetched.tokenizerDir(), "multilingual");
    }

    public static final class Builder {
        private Path englishOnnx;
        private Path englishTokenizer;
        private Path multilingualOnnx;
        private Path multilingualTokenizer;
        private Path typedOnnx;
        private Path typedTokenizer;
        private LayaConfig config = LayaConfig.defaults();

        public Builder englishOnnx(Path p) { this.englishOnnx = p; return this; }
        public Builder englishTokenizer(Path p) { this.englishTokenizer = p; return this; }
        public Builder multilingualOnnx(Path p) { this.multilingualOnnx = p; return this; }
        public Builder multilingualTokenizer(Path p) { this.multilingualTokenizer = p; return this; }
        public Builder typedDecisionsOnnx(Path p) { this.typedOnnx = p; return this; }
        public Builder typedDecisionsTokenizer(Path p) { this.typedTokenizer = p; return this; }
        public Builder config(LayaConfig c) { this.config = c; return this; }

        public LayaPredictor build() throws Exception {
            List<LayaOnnxModel> models = new ArrayList<>();
            LayaOnnxModel en = englishOnnx != null
                    ? new LayaOnnxModel(englishOnnx, englishTokenizer, "english", config) : null;
            LayaOnnxModel ml = multilingualOnnx != null
                    ? new LayaOnnxModel(multilingualOnnx, multilingualTokenizer, "multilingual", config) : null;
            LayaOnnxModel td = typedOnnx != null
                    ? new LayaOnnxModel(typedOnnx, typedTokenizer, "typed-decisions", config) : null;
            if (en != null) models.add(en);
            if (ml != null) models.add(ml);
            if (td != null) models.add(td);
            if (models.isEmpty()) {
                throw new LayaException("At least one model required");
            }
            LayaRouter router = new LayaRouter.Builder()
                    .english(en).multilingual(ml).typedDecisions(td)
                    .defaultModel(models.get(0))
                    .build();
            return new LayaPredictor(router, models.toArray(new LayaOnnxModel[0]));
        }
    }
}