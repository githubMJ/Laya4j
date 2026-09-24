package com.laya4j.snake;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * 贪吃蛇 Web 示例入口:启动后访问 http://localhost:8083 观看
 * Laya 驱动贪吃蛇的实时决策动画。
 */
@SpringBootApplication
public class SnakeServerApplication {

    public static void main(String[] args) {
        SpringApplication.run(SnakeServerApplication.class, args);
    }
}
