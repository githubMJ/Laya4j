package com.laya4j.model;

import com.laya4j.core.Decision;
import com.laya4j.core.Question;

import java.util.List;
import java.util.Map;

/**
 * Abstraction over a single Laya decision checkpoint.
 *
 * <p>The default implementation is {@link LayaOnnxModel} (ONNX Runtime).
 * Implement this interface to plug in a custom inference backend
 * (remote service, quantized runtime, test double, ...).
 *
 * <p>Implementations must be safe to use from multiple threads while
 * {@link #predict} is in flight, and must remain usable after {@link #close()}
 * has NOT been called.
 */
public interface LayaModel extends AutoCloseable {

    /**
     * @return model identifier (e.g. {@code "multilingual"}, {@code "english"}),
     *         surfaced in {@link com.laya4j.core.RoutingDecision#model()}
     */
    String modelId();

    /**
     * Run one batched inference: all questions share a single forward pass
     * over the same {@code state}.
     *
     * @param state     any string (plain text or serialized JSON)
     * @param questions questions to answer (must not be empty for real work)
     * @return decision per question name, in question order
     */
    Map<String, Decision> predict(String state, List<Question> questions);

    /** Releases native resources. Idempotent-friendly: closing twice must not throw. */
    @Override
    void close() throws Exception;
}
