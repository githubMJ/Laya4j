package com.laya4j.examples;

import com.laya4j.core.Decision;
import com.laya4j.core.Question;
import com.laya4j.predict.LayaPredictor;

import java.util.List;
import java.util.Map;

/**
 * Laya performance benchmark (test/example utility, not part of the published API).
 *
 * <p>Usage:
 * <pre>{@code
 * Benchmark.run(predictor, questions, state, n=100);
 * }</pre>
 *
 * <p>Outputs: p50 / p95 / p99 / mean / QPS / min / max
 */
public final class Benchmark {

    private Benchmark() {}

    public record Result(
            int n,
            long minNs,
            long p50Ns,
            long p95Ns,
            long p99Ns,
            long maxNs,
            double meanNs,
            double qps
    ) {
        @Override
        public String toString() {
            return String.format(
                    "Benchmark[n=%d] p50=%.1fms p95=%.1fms p99=%.1fms mean=%.1fms qps=%.1f",
                    n, p50Ns / 1e6, p95Ns / 1e6, p99Ns / 1e6, meanNs / 1e6, qps);
        }
    }

    /** Stress test: same questions run n times. */
    public static Result run(LayaPredictor p, List<Question> questions, String state, int n) {
        // warmup
        for (int i = 0; i < 3; i++) p.predict(state, questions);

        long[] samples = new long[n];
        long total = 0;
        long min = Long.MAX_VALUE, max = Long.MIN_VALUE;
        for (int i = 0; i < n; i++) {
            long t0 = System.nanoTime();
            Map<String, Decision> r = p.predict(state, questions);
            long dt = System.nanoTime() - t0;
            samples[i] = dt;
            total += dt;
            if (dt < min) min = dt;
            if (dt > max) max = dt;
        }
        java.util.Arrays.sort(samples);
        long p50 = samples[n / 2];
        long p95 = samples[Math.min(n - 1, (int) (n * 0.95))];
        long p99 = samples[Math.min(n - 1, (int) (n * 0.99))];
        double mean = (double) total / n;
        double qps = 1e9 / mean;
        return new Result(n, min, p50, p95, p99, max, mean, qps);
    }

    /** Compare latency across state lengths. */
    public static java.util.Map<String, Result> varyingLength(LayaPredictor p, List<Question> questions,
                                                              java.util.Map<String, String> states, int perStateN) {
        java.util.Map<String, Result> out = new java.util.LinkedHashMap<>();
        for (var e : states.entrySet()) {
            out.put(e.getKey(), run(p, questions, e.getValue(), perStateN));
        }
        return out;
    }

    /** Compare latency across question-set sizes. */
    public static java.util.Map<Integer, Result> varyingQuestionCount(
            LayaPredictor p, String state,
            java.util.List<java.util.List<Question>> questionSets, int perSetN) {
        java.util.Map<Integer, Result> out = new java.util.LinkedHashMap<>();
        for (var qs : questionSets) {
            out.put(qs.size(), run(p, qs, state, perSetN));
        }
        return out;
    }
}
