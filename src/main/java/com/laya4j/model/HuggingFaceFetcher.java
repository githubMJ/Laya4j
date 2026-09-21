package com.laya4j.model;

import com.laya4j.core.LayaException;

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
 * Laya 权重获取器
 *
 * 查找顺序(优先级从高到低):
 *   1. 项目内 models/ 目录(打包好的 ONNX,带版本号)
 *   2. ~/.cache/laya/ 本地缓存(自动或手动放置)
 *   3. HuggingFace Hub 拉取(网络下载,公开模型无需 token)
 *
 * 用法:
 *   var f = HuggingFaceFetcher.fetchDefault(false);
 *   -> f.tokenizerDir() : tokenizer 文件目录
 *   -> f.onnxFile()     : ONNX 模型文件
 *
 * 模型版本与 Laya Python 端的版本严格对齐(见 models/VERSION.md)
 */
public class HuggingFaceFetcher {

    /**
     * 模型版本(必须跟 models/VERSION.md 一致):
     *   laya-decision-multilingual-mmbert-base-v{version}.onnx
     *
     * 命名规范:
     *   - 任务:decision (System 1 decision model)
     *   - 语言:multilingual (100+ 语言)
     *   - encoder:mmbert-base (Johns Hopkins mmBERT-base,322M)
     *   - 版本:semver + HF commit 短哈希
     */
    public static final String MODEL_VERSION = "0.3.4-1c5edc1";
    public static final String MODEL_ENCODER = "mmbert-base";
    public static final String MODEL_FILE = "laya-decision-multilingual-" + MODEL_ENCODER + "-v" + MODEL_VERSION + ".onnx";

    public static final String DEFAULT_REPO = "convaiinnovations/laya";
    public static final String DEFAULT_SUBFOLDER = "multilingual";

    public record Fetched(Path tokenizerDir, Path modelDir, Path configFile, Path onnxFile) {}

    private static final Set<String> REQUIRED_FILES = Set.of(
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "encoder/config.json",
            "rl_agent_config.json"
    );

    public static Fetched fetchDefault(boolean forceRefresh) throws Exception {
        return fetch(DEFAULT_REPO, DEFAULT_SUBFOLDER, defaultCacheDir(), forceRefresh);
    }

    public static Fetched fetch(String repo, String subfolder, Path cacheRoot, boolean forceRefresh) throws Exception {
        Path projectModels = projectModelsDir();
        Path projectTokDir = projectModels.resolve("tokenizer");
        Path projectConfigFile = projectModels.resolve("rl_agent_config.json");
        Path projectOnnxFile = projectModels.resolve(MODEL_FILE);

        Path cacheModels = cacheRoot.resolve(subfolder);
        Path cacheTokDir = cacheModels.resolve("tokenizer");
        Path cacheConfigFile = cacheModels.resolve("rl_agent_config.json");

        Path tokenizerDir = null;
        Path configFile = null;

        // ========== 1. 项目内 models/ 优先 ==========
        if (Files.isDirectory(projectTokDir) && Files.exists(projectConfigFile)) {
            tokenizerDir = projectTokDir;
            configFile = projectConfigFile;
            System.out.println("[model] tokenizer (项目内): " + projectTokDir);
            System.out.println("[model] config   (项目内): " + projectConfigFile);
        }

        // ========== 2. 本地缓存 ~/.cache/laya/ ==========
        if (tokenizerDir == null && Files.isDirectory(cacheTokDir) && Files.exists(cacheConfigFile)) {
            tokenizerDir = cacheTokDir;
            configFile = cacheConfigFile;
            System.out.println("[model] tokenizer (本地缓存): " + cacheTokDir);
        }

        // ========== 3. HuggingFace Hub 下载 ==========
        if (tokenizerDir == null || forceRefresh) {
            Path target = cacheModels;
            Files.createDirectories(target.resolve("tokenizer"));
            Files.createDirectories(target.resolve("model"));
            HttpClient http = HttpClient.newBuilder()
                    .followRedirects(HttpClient.Redirect.ALWAYS)
                    .connectTimeout(Duration.ofSeconds(30))
                    .build();
            String token = System.getenv("HF_TOKEN");
            String base = "https://huggingface.co/" + repo + "/resolve/main/" + subfolder;
            System.out.println("[hf]   repo=" + repo + " subfolder=" + subfolder);
            System.out.println("[hf]   cache=" + target.toAbsolutePath());
            for (String rel : REQUIRED_FILES) {
                Path outFile = target.resolve(rel);
                downloadSmallFile(http, base + "/" + rel, outFile, token, forceRefresh);
            }
            tokenizerDir = cacheTokDir;
            configFile = cacheConfigFile;
        }

        // ========== 找 ONNX 模型文件 ==========
        Path onnxFile = null;

        // 1) 项目内 models/ 下的带版本号文件
        if (Files.exists(projectOnnxFile)) {
            onnxFile = projectOnnxFile;
            System.out.println("[model] ONNX (项目内): " + projectOnnxFile + " (" +
                    Files.size(onnxFile) / 1024 / 1024 + " MB, v" + MODEL_VERSION + ")");
        }
        // 2) 项目内 models/ 下任何 .onnx 文件
        if (onnxFile == null && Files.isDirectory(projectModels)) {
            try (var stream = Files.list(projectModels)) {
                onnxFile = stream.filter(p -> p.getFileName().toString().endsWith(".onnx"))
                        .findFirst().orElse(null);
                if (onnxFile != null) {
                    System.out.println("[model] ONNX (项目内): " + onnxFile);
                    System.out.println("[warn] 文件名不带版本号,建议重命名为: " + MODEL_FILE);
                }
            } catch (IOException e) {
                // ignore
            }
        }
        // 3) 本地缓存
        if (onnxFile == null) {
            Path cached = cacheRoot.resolve(MODEL_FILE);
            if (Files.exists(cached)) {
                onnxFile = cached;
                System.out.println("[model] ONNX (本地缓存): " + cached);
            }
        }
        // 4) HuggingFace Hub 下载(慢)
        if (onnxFile == null) {
            System.out.println("[model] 本地没有 ONNX,从 HF Hub 下载(1.2 GB)...\n" +
                    "[model] 推荐: 把 ONNX 放到项目 models/ 目录,版本对齐 " + MODEL_VERSION);
            Path target = cacheRoot.resolve(MODEL_FILE);
            String url = "https://huggingface.co/" + repo + "/resolve/main/" + MODEL_FILE;
            HttpClient http = HttpClient.newBuilder()
                    .followRedirects(HttpClient.Redirect.ALWAYS)
                    .connectTimeout(Duration.ofSeconds(30))
                    .build();
            String token = System.getenv("HF_TOKEN");
            downloadLfsFile(http, url, target, token, forceRefresh);
            onnxFile = target;
        }

        if (tokenizerDir == null) {
            throw new LayaException("Failed to obtain tokenizer files");
        }
        return new Fetched(tokenizerDir, projectModels.resolve("model"), configFile, onnxFile);
    }

    private static void downloadSmallFile(HttpClient http, String url, Path outFile,
                                          String token, boolean forceRefresh) throws Exception {
        if (!forceRefresh && Files.exists(outFile) && Files.size(outFile) > 100) {
            System.out.println("[hf]   cached " + outFile.getFileName() + " (" + Files.size(outFile) + " bytes)");
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
        System.out.print("[hf]   fetch " + outFile.getFileName() + " ... ");
        HttpResponse<byte[]> resp = http.send(b.build(), HttpResponse.BodyHandlers.ofByteArray());
        if (resp.statusCode() != 200) {
            System.out.println("FAIL " + resp.statusCode());
            throw new LayaException("HTTP " + resp.statusCode() + " for " + url);
        }
        Files.write(outFile, resp.body());
        System.out.println("OK " + Files.size(outFile) + " bytes");
    }

    private static void downloadLfsFile(HttpClient http, String url, Path outFile,
                                        String token, boolean forceRefresh) throws Exception {
        if (!forceRefresh && Files.exists(outFile) && Files.size(outFile) > 1024 * 1024) {
            System.out.println("[hf]   cached " + outFile.getFileName());
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
        System.out.println("[hf]   downloading " + outFile.getFileName() +
                (existingSize > 0 ? " (resume from " + existingSize + " bytes)" : ""));
        long t0 = System.currentTimeMillis();
        HttpResponse<java.io.InputStream> resp = http.send(b.build(),
                HttpResponse.BodyHandlers.ofInputStream());
        if (resp.statusCode() != 200 && resp.statusCode() != 206) {
            throw new LayaException("HTTP " + resp.statusCode() + " for " + url);
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
                    System.out.printf("[hf]     %d MB @ %.1f MB/s%n", total / 1024 / 1024, mbps);
                    lastReport = (int) elapsed;
                }
            }
        }
        double secs = (System.currentTimeMillis() - t0) / 1000.0;
        double mbps = Files.size(outFile) / 1024.0 / 1024.0 / Math.max(1, secs);
        System.out.printf("[hf]   OK %d MB in %.1fs (%.1f MB/s)%n",
                Files.size(outFile) / 1024 / 1024, secs, mbps);
    }

    /** 项目内 models/ 目录 */
    public static Path projectModelsDir() {
        // cwd 是项目根(有 pom.xml)
        return Paths.get("").toAbsolutePath().resolve("models");
    }

    /** 本地缓存目录 */
    public static Path defaultCacheDir() {
        return Paths.get(System.getProperty("user.home"), ".cache", "laya");
    }
}