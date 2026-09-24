package com.laya4j.model;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.laya4j.core.ChoiceDecision;
import com.laya4j.core.Decision;
import com.laya4j.core.DecisionType;
import com.laya4j.core.LayaConfig;
import com.laya4j.core.InferenceException;
import com.laya4j.core.ModelLoadException;
import com.laya4j.core.NoulDecision;
import com.laya4j.core.Question;
import com.laya4j.core.ScoreDecision;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.LongBuffer;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Laya multilingual DecisionModel - ONNX Runtime Java inference
 *
 * Wraps a single checkpoint (no router). For multi-checkpoint auto-routing, use LayaRouter.
 *
 * Usage:
 *   LayaOnnxModel model = new LayaOnnxModel(onnxPath, tokenizerDir);
 *   Map{@code <String, Decision>} results = model.predict(state, questions);
 *
 * Inputs:
 *   - state: any String / serialized JSON / Map (eventually serialized to a String)
 *   - questions: List{@code <Question>} (all questions share the same state in one inference)
 *
 * Outputs:
 *   - Map{@code <questionName, Decision>} (Decision is ChoiceDecision / ScoreDecision / NoulDecision)
 */
public class LayaOnnxModel implements LayaModel {

    private final String modelId;        // "english" / "multilingual" / "typed-decisions"
    private final OrtEnvironment env;
    private final OrtSession session;
    private final HuggingFaceTokenizer tokenizer;
    private final SequenceBuilder sequenceBuilder;
    private final double[] temperature;   // [choice, score, noul]
    private final int maxLen;
    private final int headMaxLen;

    /**
     * @param onnxPath      ONNX model path (required)
     * @param tokenizerDir  tokenizer dir (containing tokenizer.json + tokenizer_config.json)
     * @param modelId       model identifier ("multilingual", etc.), appears in RoutingDecision
     * @param config        LayaConfig (defaults() is sufficient)
     */
    public LayaOnnxModel(Path onnxPath, Path tokenizerDir, String modelId, LayaConfig config) {
        if (onnxPath == null || !onnxPath.toFile().exists()) {
            throw new ModelLoadException("ONNX model not found: " + onnxPath);
        }
        if (tokenizerDir == null || !tokenizerDir.toFile().isDirectory()) {
            throw new ModelLoadException("Tokenizer dir not found: " + tokenizerDir);
        }
        this.modelId = modelId;
        this.maxLen = config.maxLen();
        this.headMaxLen = config.headMaxLen();
        this.temperature = config.temperature().clone();
        try {
            this.env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            opts.setIntraOpNumThreads(Math.max(1, Runtime.getRuntime().availableProcessors() / 2));
            opts.setMemoryPatternOptimization(true);
            this.session = env.createSession(onnxPath.toString(), opts);
            // DJL 需要 file:// URL
            this.tokenizer = HuggingFaceTokenizer.builder()
                    .optTokenizerPath(java.nio.file.Paths.get(
                            tokenizerDir.resolve("tokenizer.json").toUri()))
                    .optAddSpecialTokens(false)
                    .build();
            this.sequenceBuilder = new SequenceBuilder(tokenizer, maxLen, headMaxLen);
        } catch (OrtException | IOException e) {
            throw new ModelLoadException("Failed to load Laya model: " + modelId, e);
        }
    }

    /**
     * Single inference: input state + questions, return Decision per question
     *
     * @param state     any string (JSON supported)
     * @param questions all questions of one inference (share one ONNX forward)
     */
    public Map<String, Decision> predict(String state, List<Question> questions) {
        if (questions == null || questions.isEmpty()) {
            return Map.of();
        }
        try {
            // 1. 构造 batch
            List<SequenceBuilder.BuiltSequence> built = new java.util.ArrayList<>();
            int[] qtypes = new int[questions.size()];
            int maxMarkers = 1, maxSeqLen = 1;
            for (int i = 0; i < questions.size(); i++) {
                Question q = questions.get(i);
                SequenceBuilder.BuiltSequence bs = sequenceBuilder.build(state, q);
                built.add(bs);
                qtypes[i] = q.type().id();
                if (bs.markers().length > maxMarkers) maxMarkers = bs.markers().length;
                if (bs.ids().length > maxSeqLen) maxSeqLen = bs.ids().length;
            }
            int batch = questions.size();

            // 2. 构造 ONNX tensor(用 ByteBuffer/LongBuffer)
            ByteOrder bo = ByteOrder.nativeOrder();
            LongBuffer idsBuf = ByteBuffer.allocateDirect(batch * maxSeqLen * 8).order(bo).asLongBuffer();
            LongBuffer maskBuf = ByteBuffer.allocateDirect(batch * maxSeqLen * 8).order(bo).asLongBuffer();
            LongBuffer mposBuf = ByteBuffer.allocateDirect(batch * maxMarkers * 8).order(bo).asLongBuffer();
            ByteBuffer mmaskBuf = ByteBuffer.allocateDirect(batch * maxMarkers).order(bo);
            long[] tmp = new long[Math.max(maxSeqLen, maxMarkers)];
            for (int i = 0; i < batch; i++) {
                SequenceBuilder.BuiltSequence bs = built.get(i);
                // ids + attention mask
                for (int j = 0; j < bs.ids().length; j++) tmp[j] = bs.ids()[j];
                for (int j = bs.ids().length; j < maxSeqLen; j++) tmp[j] = 0;
                idsBuf.put(copyPrefix(tmp, maxSeqLen));
                for (int j = 0; j < bs.ids().length; j++) tmp[j] = 1;
                for (int j = bs.ids().length; j < maxSeqLen; j++) tmp[j] = 0;
                maskBuf.put(copyPrefix(tmp, maxSeqLen));
                // marker_pos + marker_mask
                for (int j = 0; j < bs.markers().length; j++) tmp[j] = bs.markers()[j];
                for (int j = bs.markers().length; j < maxMarkers; j++) tmp[j] = 0;
                mposBuf.put(copyPrefix(tmp, maxMarkers));
                for (int j = 0; j < bs.markers().length; j++) mmaskBuf.put((byte) 1);
                for (int j = bs.markers().length; j < maxMarkers; j++) mmaskBuf.put((byte) 0);
            }
            idsBuf.rewind(); maskBuf.rewind(); mposBuf.rewind(); mmaskBuf.rewind();
            LongBuffer qtypeBuf = ByteBuffer.allocateDirect(batch * 8).order(bo).asLongBuffer();
            for (int v : qtypes) qtypeBuf.put(v);
            qtypeBuf.rewind();

            // 3. ONNX forward
            try (OnnxTensor tIds = OnnxTensor.createTensor(env, idsBuf, new long[]{batch, maxSeqLen});
                 OnnxTensor tAtt = OnnxTensor.createTensor(env, maskBuf, new long[]{batch, maxSeqLen});
                 OnnxTensor tMpos = OnnxTensor.createTensor(env, mposBuf, new long[]{batch, maxMarkers});
                 OnnxTensor tMmask = OnnxTensor.createTensor(env, mmaskBuf, new long[]{batch, maxMarkers}, OnnxJavaType.BOOL);
                 OnnxTensor tQt = OnnxTensor.createTensor(env, qtypeBuf, new long[]{batch})) {

                Map<String, OnnxTensor> inputs = new java.util.HashMap<>();
                inputs.put("input_ids", tIds);
                inputs.put("attention_mask", tAtt);
                inputs.put("marker_pos", tMpos);
                inputs.put("marker_mask", tMmask);
                inputs.put("qtype", tQt);

                OrtSession.Result result = session.run(inputs);
                float[][] logits = (float[][]) result.get(0).getValue();
                // act_logits 当前不暴露 API,留给 LayaRouter 用
                result.close();
                inputs.clear();

                // 4. 后处理
                Map<String, Decision> out = new LinkedHashMap<>();
                for (int i = 0; i < batch; i++) {
                    Question q = questions.get(i);
                    float[] li = logits[i].clone();
                    int valid = built.get(i).markers().length;
                    for (int j = valid; j < li.length; j++) li[j] = -1e4f;
                    double[] probs = softmax(li, temperature[q.type().id()]);
                    out.put(q.name(), toDecision(q, probs));
                }
                return out;
            }
        } catch (OrtException e) {
            throw new InferenceException("ONNX inference failed for model " + modelId, e);
        }
    }

    private Decision toDecision(Question q, double[] probs) {
        return switch (q.type()) {
            case CHOICE -> {
                int best = argmax(probs);
                Map<String, Double> map = new LinkedHashMap<>();
                String[] labels = q.choices().keySet().toArray(new String[0]);
                for (int i = 0; i < labels.length && i < probs.length; i++) {
                    map.put(labels[i], probs[i]);
                }
                yield new ChoiceDecision(labels[best], probs[best], map);
            }
            case SCORE -> {
                double expected = 0;
                for (int i = 0; i < probs.length; i++) expected += i * probs[i];
                Map<Integer, Double> dist = new LinkedHashMap<>();
                for (int i = 0; i < probs.length; i++) dist.put(i, probs[i]);
                double[] confs = new double[probs.length];
                for (int i = 0; i < probs.length; i++) confs[i] = probs[i];
                yield new ScoreDecision(expected, confidenceFromDistribution(probs), dist, q.levels());
            }
            case NOUL -> {
                // markers[0]=false, markers[1]=true
                double pFalse = probs.length > 0 ? probs[0] : 0.5;
                double pTrue  = probs.length > 1 ? probs[1] : 0.5;
                double sum = pFalse + pTrue;
                if (sum > 0) {
                    pFalse /= sum;
                    pTrue /= sum;
                }
                yield new NoulDecision(pTrue, Math.max(pTrue, pFalse));
            }
        };
    }

    private static double[] softmax(float[] logits, double temp) {
        double[] s = new double[logits.length];
        double max = Double.NEGATIVE_INFINITY;
        for (float l : logits) if (l > max) max = l;
        double sum = 0;
        for (int i = 0; i < logits.length; i++) { s[i] = Math.exp((logits[i] - max) / temp); sum += s[i]; }
        for (int i = 0; i < s.length; i++) s[i] /= sum;
        return s;
    }

    private static int argmax(double[] x) {
        int best = 0;
        for (int i = 1; i < x.length; i++) if (x[i] > x[best]) best = i;
        return best;
    }

    private static double confidenceFromDistribution(double[] probs) {
        if (probs.length < 2) return 1.0;
        double h = 0;
        for (double p : probs) {
            if (p > 1e-12) h -= p * Math.log(p);
        }
        return Math.max(0, Math.min(1, 1.0 - h / Math.log(probs.length)));
    }

    /** 辅助:LongBuffer.put 需要 long[],但我们要 copy 一段 */
    private static long[] copyPrefix(long[] src, int len) {
        long[] r = new long[len];
        System.arraycopy(src, 0, r, 0, Math.min(src.length, len));
        return r;
    }

    @Override
    public String modelId() { return modelId; }

    /**
     * Releases the native OrtSession.
     *
     * <p>Note: the shared {@link OrtEnvironment} is a process-wide singleton and
     * is intentionally NOT closed here — closing it would break any other
     * LayaOnnxModel (or ONNX Runtime user) running in the same JVM.
     */
    @Override
    public void close() throws OrtException {
        session.close();
    }
}