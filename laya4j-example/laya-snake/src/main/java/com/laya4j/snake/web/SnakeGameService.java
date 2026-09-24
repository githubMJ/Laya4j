package com.laya4j.snake.web;

import com.laya4j.snake.engine.SnakeGame;
import com.laya4j.snake.laya.DecisionTrace;
import com.laya4j.snake.laya.SnakeDecisionService;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * 贪吃蛇游戏会话服务:持有引擎状态,每次 {@link #step()} 触发一次完整的
 * "渲染 state → 问 Laya(1 choice + 4 noul)→ 消费决策 → 走一步"流程,
 * 并输出可直接序列化给前端渲染的 {@link Snapshot}。
 *
 * <p>与 tetris 示例的 {@code TetrisGameService} 保持一致的分层:
 * Controller 只做参数透传,游戏状态与决策编排都在本类完成,
 * 真正的"问 Laya、消费决策"逻辑委托给 {@link SnakeDecisionService}。
 */
@Service
public class SnakeGameService {

    /** 棋盘 + 分数 + 最近一次决策轨迹的完整快照,供 REST API 直接返回。 */
    public record Snapshot(
            int boardSize,
            List<int[]> bodyCells,
            int[] food,
            String heading,
            int score, int steps, boolean gameOver,
            DecisionTrace lastDecision
    ) {}

    private final SnakeDecisionService decisionService;
    private SnakeGame game;

    public SnakeGameService(com.laya4j.model.LayaModel model) {
        this.decisionService = new SnakeDecisionService(model);
        reset(42L);
    }

    /** 开一局新游戏。 */
    public synchronized Snapshot reset(Long seed) {
        this.game = new SnakeGame(12, seed != null ? seed : 42L);
        return snapshot(null);
    }

    /** 只读取当前状态,不推进游戏。 */
    public synchronized Snapshot state() {
        return snapshot(null);
    }

    /**
     * 推进一步:对当前局面做一次完整的 Laya 决策 + 移动。
     * 若游戏已结束,直接返回当前快照(不再推理)。
     */
    public synchronized Snapshot step() {
        if (!game.alive()) {
            return snapshot(null);
        }
        DecisionTrace trace = decisionService.decideAndApply(game);
        return snapshot(trace);
    }

    private Snapshot snapshot(DecisionTrace lastDecision) {
        return new Snapshot(
                game.size(), game.bodyCells(),
                game.food() == null ? null : game.food().toArray(),
                game.heading().name(),
                game.score(), game.steps(), !game.alive(),
                lastDecision
        );
    }
}
