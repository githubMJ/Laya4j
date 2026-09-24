package com.laya4j.snake.web;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * REST API:前端页面通过这三个接口驱动贪吃蛇决策动画。
 */
@RestController
public class SnakeController {

    private final SnakeGameService service;

    public SnakeController(SnakeGameService service) {
        this.service = service;
    }

    /** 查询当前局面(不推进游戏)。 */
    @GetMapping("/api/state")
    public SnakeGameService.Snapshot state() {
        return service.state();
    }

    /** 开一局新游戏。 */
    @PostMapping("/api/new")
    public SnakeGameService.Snapshot reset(@RequestParam(required = false) Long seed) {
        return service.reset(seed);
    }

    /** 对当前局面推进一次完整的 Laya 决策 + 移动。 */
    @PostMapping("/api/step")
    public SnakeGameService.Snapshot step() {
        return service.step();
    }
}
