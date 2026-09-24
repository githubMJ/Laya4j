package com.laya4j.snake.config;

import com.laya4j.core.LayaConfig;
import com.laya4j.model.HuggingFaceFetcher;
import com.laya4j.model.LayaModel;
import com.laya4j.model.LayaOnnxModel;
import com.laya4j.snake.laya.HeuristicSnakeModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * {@link LayaModel} 装配:{@code mock}(启发式,默认,秒级启动)或
 * {@code onnx}(真实 Laya 权重,自动从项目 {@code models/} 目录或
 * HuggingFace Hub 解析)。
 *
 * <p>切换方式:{@code -Dlaya.snake.model=onnx}
 * (可配合 {@code -Dlaya4j.models.dir=/path/to/Laya4j/models} 指向仓库根的
 * 权重目录,详见 {@code com.laya4j.model.HuggingFaceFetcher})。
 */
@Configuration
public class ModelConfig {

    private static final Logger log = LoggerFactory.getLogger(ModelConfig.class);

    @Bean(destroyMethod = "close")
    public LayaModel layaModel(@Value("${laya.snake.model:mock}") String mode) throws Exception {
        if ("onnx".equalsIgnoreCase(mode)) {
            log.info("Loading real Laya ONNX model for Snake decisions ...");
            var fetched = HuggingFaceFetcher.fetchDefault(false);
            return new LayaOnnxModel(fetched.onnxFile(), fetched.tokenizerDir(),
                    "multilingual", LayaConfig.defaults());
        }
        log.info("Using heuristic mock LayaModel for Snake (set laya.snake.model=onnx for real weights)");
        return new HeuristicSnakeModel();
    }
}
