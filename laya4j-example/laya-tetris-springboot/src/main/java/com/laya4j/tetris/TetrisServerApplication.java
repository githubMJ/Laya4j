package com.laya4j.tetris;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Laya Tetris —— 用 Laya 的 choice/score/noul 三种决策原语驱动
 * 俄罗斯方块落子决策的 Spring Boot 示例服务。
 *
 * <p>运行:
 * <pre>{@code
 * # 1. 先在 Laya4j 项目根目录安装库到本地仓库
 * cd ../../.. && mvn install -DskipTests
 *
 * # 2. 启动本服务(默认启发式 mock 模型,秒级启动)
 * cd examples/laya-tetris-springboot && mvn spring-boot:run
 * #    打开 http://localhost:8082
 *
 * # 3. 切换真实 Laya 权重(自动从项目 models/ 或 HuggingFace Hub 解析)
 * mvn spring-boot:run \
 *     -Dspring-boot.run.jvmArguments="-Dlaya.tetris.model=onnx -Dlaya4j.models.dir=$(pwd)/../../models"
 * }</pre>
 */
@SpringBootApplication
public class TetrisServerApplication {

    public static void main(String[] args) {
        SpringApplication.run(TetrisServerApplication.class, args);
    }
}
