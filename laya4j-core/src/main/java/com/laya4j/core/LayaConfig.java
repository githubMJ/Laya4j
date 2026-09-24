package com.laya4j.core;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Laya model global configuration.
 *
 * <p>Mirrors {@code rl_agent_config.json} shipped with each checkpoint.
 * Use {@link #fromRlAgentConfig(Path)} to load the config bundled with a model,
 * or {@link #builder()} for custom values.
 *
 * @param maxLen      max total sequence length (multilingual=1024)
 * @param headMaxLen  question/options portion budget (default 256)
 * @param temperature per-type softmax temperatures [choice, score, noul]
 */
public record LayaConfig(int maxLen, int headMaxLen, double[] temperature) {

    public LayaConfig {
        if (maxLen <= 0) {
            throw new LayaException("maxLen must be positive, got " + maxLen);
        }
        if (headMaxLen <= 0 || headMaxLen > maxLen) {
            throw new LayaException(
                    "headMaxLen must be in (0, maxLen], got " + headMaxLen + " (maxLen=" + maxLen + ")");
        }
        if (temperature == null || temperature.length != DecisionType.values().length) {
            throw new LayaException("temperature must have exactly "
                    + DecisionType.values().length + " entries [choice, score, noul]");
        }
        for (double t : temperature) {
            if (t <= 0 || Double.isNaN(t)) {
                throw new LayaException("temperature values must be positive, got "
                        + Arrays.toString(temperature));
            }
        }
        temperature = temperature.clone();
    }

    /**
     * Base default config (multilingual):
     * maxLen=1024, headMaxLen=256, temperature=[1, 1, 1].
     */
    public static LayaConfig defaults() {
        return new LayaConfig(1024, 256, new double[]{1.0, 1.0, 1.0});
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Load config from a checkpoint's {@code rl_agent_config.json}
     * (keys: {@code max_len}, {@code head_max_len}, {@code temperature}).
     * Falls back to {@link #defaults()} if the file is missing or malformed.
     */
    public static LayaConfig fromRlAgentConfig(Path file) {
        if (file == null || !Files.isRegularFile(file)) {
            return defaults();
        }
        try {
            String json = Files.readString(file);
            int maxLen = intField(json, "max_len", 1024);
            int headMaxLen = intField(json, "head_max_len", 256);
            double[] temperature = temperatureField(json);
            return new LayaConfig(maxLen, headMaxLen, temperature);
        } catch (Exception e) {
            return defaults();
        }
    }

    /** @return softmax temperature for the given decision type */
    public double temperatureFor(DecisionType type) {
        return temperature[type.id()];
    }

    private static int intField(String json, String key, int fallback) {
        Matcher m = Pattern.compile("\"" + key + "\"\\s*:\\s*(-?\\d+)").matcher(json);
        return m.find() ? Integer.parseInt(m.group(1)) : fallback;
    }

    private static double[] temperatureField(String json) {
        Matcher m = Pattern.compile("\"temperature\"\\s*:\\s*\\[([^\\]]*)\\]").matcher(json);
        if (!m.find()) {
            return new double[]{1.0, 1.0, 1.0};
        }
        String[] parts = m.group(1).split(",");
        double[] out = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            out[i] = Double.parseDouble(parts[i].trim());
        }
        return out;
    }

    public static final class Builder {
        private int maxLen = 1024;
        private int headMaxLen = 256;
        private double[] temperature = {1.0, 1.0, 1.0};

        public Builder maxLen(int maxLen) { this.maxLen = maxLen; return this; }

        public Builder headMaxLen(int headMaxLen) { this.headMaxLen = headMaxLen; return this; }

        /** Per-type softmax temperatures [choice, score, noul] (copied). */
        public Builder temperature(double... temperature) { this.temperature = temperature; return this; }

        public LayaConfig build() {
            return new LayaConfig(maxLen, headMaxLen, temperature);
        }
    }
}
