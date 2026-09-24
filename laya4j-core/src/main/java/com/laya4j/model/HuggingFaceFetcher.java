package com.laya4j.model;

import com.laya4j.core.ModelFetchException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.util.Set;

/**
 * Resolves and downloads Laya model artifacts from the HuggingFace Hub.
 *
 * <p>Lookup order (highest priority first):
 * <ol>
 *   <li>project-local {@code models/} directory (offline, packaged ONNX)</li>
 *   <li>{@code ~/.cache/laya/} local cache (auto or manually placed)</li>
 *   <li>HuggingFace Hub (network download; public model, no token required)</li>
 * </ol>
 *
 * <p>The project-local directory can be overridden with the system property
 * {@code laya4j.models.dir} — recommended when running inside another application
 * where the process working directory is not the project root.
 *
 * <p>Usage:
 * <pre>{@code
 * var f = HuggingFaceFetcher.fetchDefault(false);
 * // f.tokenizerDir() : tokenizer directory
 * // f.onnxFile()     : ONNX model file
 * }</pre>
 *
 * <p>Model version is aligned with the Laya Python side (see models/VERSION.md).
 */
public class HuggingFaceFetcher {

    private static final Logger log = LoggerFactory.getLogger(HuggingFaceFetcher.class);

    /** System property to override the project-local models directory. */
    public static final String MODELS_DIR_PROPERTY = "laya4j.models.dir";

    /**
     * Model version (must match models/VERSION.md):
     * laya-decision-multilingual-mmbert-base-v{version}.onnx
     *
     * <p>Naming: task=decision, language=multilingual, encoder=mmbert-base
     * (JHU mmBERT-base, 322M), version=semver + HF commit short hash.
     */
    public static final String MODEL_VERSION = "0.3.5-1c5edc1";
    public static final String MODEL_ENCODER = "mmbert-base";
    public static final String MODEL_FILE = "laya-decision-multilingual-" + MODEL_ENCODER + "-v" + MODEL_VERSION + ".onnx";

    public static final String DEFAULT_REPO = "convaiinnovations/laya";
    public static final String DEFAULT_SUBFOLDER = "multilingual";

    /** Resolved artifact locations. */
    public record Fetched(Path tokenizerDir, Path modelDir, Path configFile, Path onnxFile) {}

    private static final Set<String> REQUIRED_FILES = Set.of(
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "encoder/config.json",
            "rl_agent_config.json"
    );

    /** Resolve the default multilingual checkpoint. */
    public static Fetched fetchDefault(boolean forceRefresh) throws Exception {
        return fetch(DEFAULT_REPO, DEFAULT_SUBFOLDER, defaultCacheDir(), forceRefresh);
    }

    /** Resolve a specific repo/subfolder, using {@link #projectModelsDir()} for local lookup. */
    public static Fetched fetch(String repo, String subfolder, Path cacheRoot, boolean forceRefresh) throws Exception {
        return fetch(repo, subfolder, projectModelsDir(), cacheRoot, forceRefresh);
    }

    /**
     * Full-resolution entry point.
     *
     * @param repo           HuggingFace repo id, e.g. {@code "convaiinnovations/laya"}
     * @param subfolder      subfolder inside the repo, e.g. {@code "multilingual"}
     * @param localModelsDir project-local models directory probed first (may not exist)
     * @param cacheRoot      local cache root (e.g. {@code ~/.cache/laya})
     * @param forceRefresh   re-download tokenizer/config even if cached
     */
    public static Fetched fetch(String repo, String subfolder, Path localModelsDir,
                                Path cacheRoot, boolean forceRefresh) throws Exception {
        Path projectTokDir = localModelsDir.resolve("tokenizer");
        Path projectConfigFile = localModelsDir.resolve("rl_agent_config.json");

        Path cacheModels = cacheRoot.resolve(subfolder);
        Path cacheTokDir = cacheModels.resolve("tokenizer");
        Path cacheConfigFile = cacheModels.resolve("rl_agent_config.json");

        Path tokenizerDir = null;
        Path configFile = null;

        // ===== 1. project-local models/ =====
        if (Files.isDirectory(projectTokDir) && Files.exists(projectConfigFile)) {
            tokenizerDir = projectTokDir;
            configFile = projectConfigFile;
            log.info("[model] tokenizer (project-local): {}", projectTokDir);
            log.info("[model] config   (project-local): {}", projectConfigFile);
        }

        // ===== 2. local cache =====
        if (tokenizerDir == null && Files.isDirectory(cacheTokDir) && Files.exists(cacheConfigFile)) {
            tokenizerDir = cacheTokDir;
            configFile = cacheConfigFile;
            log.info("[model] tokenizer (local cache): {}", cacheTokDir);
        }

        // ===== 3. HuggingFace Hub download =====
        if (tokenizerDir == null || forceRefresh) {
            Path target = cacheModels;
            Files.createDirectories(target.resolve("tokenizer"));
            Files.createDirectories(target.resolve("model"));
            HttpClient http = newHttpClient();
            String token = System.getenv("HF_TOKEN");
            String base = "https://huggingface.co/" + repo + "/resolve/main/" + subfolder;
            log.info("[hf] repo={} subfolder={}", repo, subfolder);
            log.info("[hf] cache={}", target.toAbsolutePath());
            for (String rel : REQUIRED_FILES) {
                Path outFile = target.resolve(rel);
                downloadSmallFile(http, base + "/" + rel, outFile, token, forceRefresh);
            }
            tokenizerDir = cacheTokDir;
            configFile = cacheConfigFile;
        }

        Path onnxFile = locateOnnx(localModelsDir, cacheRoot, repo, forceRefresh);

        if (tokenizerDir == null) {
            throw new ModelFetchException("Failed to obtain tokenizer files");
        }
        return new Fetched(tokenizerDir, localModelsDir.resolve("model"), configFile, onnxFile);
    }

    private static Path locateOnnx(Path localModelsDir, Path cacheRoot, String repo,
                                   boolean forceRefresh) throws Exception {
        // 1) versioned file inside local models/
        Path versioned = localModelsDir.resolve(MODEL_FILE);
        if (Files.exists(versioned)) {
            log.info("[model] ONNX (project-local): {} ({} MB, v{})", versioned,
                    Files.size(versioned) / 1024 / 1024, MODEL_VERSION);
            return versioned;
        }
        // 2) any .onnx inside local models/
        if (Files.isDirectory(localModelsDir)) {
            try (var stream = Files.list(localModelsDir)) {
                Path any = stream.filter(p -> p.getFileName().toString().endsWith(".onnx"))
                        .findFirst().orElse(null);
                if (any != null) {
                    log.info("[model] ONNX (project-local): {}", any);
                    log.warn("[model] file name has no version; consider renaming to: {}", MODEL_FILE);
                    return any;
                }
            } catch (IOException e) {
                // ignore and continue with cache
            }
        }
        // 3) local cache
        Path cached = cacheRoot.resolve(MODEL_FILE);
        if (Files.exists(cached)) {
            log.info("[model] ONNX (local cache): {}", cached);
            return cached;
        }
        // 4) HuggingFace Hub (slow, ~1.2 GB)
        log.info("[model] ONNX not found locally; downloading from HF Hub (~1.2 GB). "
                + "Tip: place the ONNX file in the local models/ dir, version-aligned to {}", MODEL_VERSION);
        Path target = cacheRoot.resolve(MODEL_FILE);
        String url = "https://huggingface.co/" + repo + "/resolve/main/" + MODEL_FILE;
        downloadLfsFile(newHttpClient(), url, target, System.getenv("HF_TOKEN"), forceRefresh);
        return target;
    }

    private static HttpClient newHttpClient() {
        return HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.ALWAYS)
                .connectTimeout(Duration.ofSeconds(30))
                .build();
    }

    private static void downloadSmallFile(HttpClient http, String url, Path outFile,
                                          String token, boolean forceRefresh) throws Exception {
        if (!forceRefresh && Files.exists(outFile) && Files.size(outFile) > 100) {
            log.info("[hf] cached {} ({} bytes)", outFile.getFileName(), Files.size(outFile));
            return;
        }
        Files.createDirectories(outFile.getParent());
        HttpRequest.Builder b = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .timeout(Duration.ofSeconds(120))
                .GET();
        if (token != null && !token.isEmpty()) {
            b.header("Authorization", "Bearer " + token);
        }
        log.info("[hf] fetch {} ...", outFile.getFileName());
        HttpResponse<byte[]> resp = http.send(b.build(), HttpResponse.BodyHandlers.ofByteArray());
        if (resp.statusCode() != 200) {
            throw new ModelFetchException("HTTP " + resp.statusCode() + " for " + url);
        }
        Files.write(outFile, resp.body());
        log.info("[hf] OK {} ({} bytes)", outFile.getFileName(), Files.size(outFile));
    }

    private static void downloadLfsFile(HttpClient http, String url, Path outFile,
                                        String token, boolean forceRefresh) throws Exception {
        if (!forceRefresh && Files.exists(outFile) && Files.size(outFile) > 1024 * 1024) {
            log.info("[hf] cached {}", outFile.getFileName());
            return;
        }
        Files.createDirectories(outFile.getParent());
        long existingSize = forceRefresh ? 0 : (Files.exists(outFile) ? Files.size(outFile) : 0);
        HttpRequest.Builder b = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .timeout(Duration.ofMinutes(30))
                .GET();
        if (existingSize > 0) {
            b.header("Range", "bytes=" + existingSize + "-");
        }
        if (token != null && !token.isEmpty()) {
            b.header("Authorization", "Bearer " + token);
        }
        log.info("[hf] downloading {}{}", outFile.getFileName(),
                existingSize > 0 ? " (resume from " + existingSize + " bytes)" : "");
        long t0 = System.currentTimeMillis();
        HttpResponse<java.io.InputStream> resp = http.send(b.build(),
                HttpResponse.BodyHandlers.ofInputStream());
        if (resp.statusCode() != 200 && resp.statusCode() != 206) {
            throw new ModelFetchException("HTTP " + resp.statusCode() + " for " + url);
        }
        try (var in = resp.body();
             var out = Files.newOutputStream(outFile,
                     existingSize > 0 ? StandardOpenOption.APPEND : StandardOpenOption.CREATE)) {
            byte[] buf = new byte[256 * 1024];
            long total = existingSize;
            int n;
            int lastReport = 0;
            while ((n = in.read(buf)) > 0) {
                out.write(buf, 0, n);
                total += n;
                long elapsed = System.currentTimeMillis() - t0;
                if (elapsed - lastReport > 2000) {
                    double mbps = total / 1024.0 / 1024.0 / Math.max(1, elapsed / 1000.0);
                    log.info("[hf]   {} MB @ {} MB/s", total / 1024 / 1024,
                            String.format("%.1f", mbps));
                    lastReport = (int) elapsed;
                }
            }
        } catch (IOException e) {
            throw new ModelFetchException("Download failed: " + url, e);
        }
        double secs = (System.currentTimeMillis() - t0) / 1000.0;
        double mbps = Files.size(outFile) / 1024.0 / 1024.0 / Math.max(1, secs);
        log.info("[hf] OK {} MB in {}s ({} MB/s)", Files.size(outFile) / 1024 / 1024,
                String.format("%.1f", secs), String.format("%.1f", mbps));
    }

    /**
     * Project-local models directory.
     *
     * <p>Honors the {@value #MODELS_DIR_PROPERTY} system property; otherwise falls
     * back to {@code ./models} relative to the current working directory
     * (development convenience).
     */
    public static Path projectModelsDir() {
        String prop = System.getProperty(MODELS_DIR_PROPERTY);
        if (prop != null && !prop.isBlank()) {
            return Paths.get(prop).toAbsolutePath();
        }
        return Paths.get("").toAbsolutePath().resolve("models");
    }

    /** Local cache directory: {@code ~/.cache/laya}. */
    public static Path defaultCacheDir() {
        return Paths.get(System.getProperty("user.home"), ".cache", "laya");
    }
}
