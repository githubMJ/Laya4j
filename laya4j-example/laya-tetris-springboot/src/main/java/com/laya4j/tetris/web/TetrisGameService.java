package com.laya4j.tetris.web;

import com.laya4j.model.LayaModel;
import com.laya4j.tetris.engine.Board;
import com.laya4j.tetris.engine.TetrisEngine;
import com.laya4j.tetris.laya.DecisionTrace;
import com.laya4j.tetris.laya.TetrisDecisionService;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * Tetris 游戏会话服务:持有引擎状态,每次 {@link #step()} 触发一次完整的
 * "枚举候选 → 问 Laya → 消费决策 → 落子"流程,并输出可直接序列化给
 * 前端渲染的 {@link Snapshot}。
 *
 * <p>与贪吃蛇示例的 {@code SnakeService} 保持一致的分层:
 * Controller 只做参数透传,游戏状态与决策编排都在本类完成,
 * 真正的"问 Laya、消费决策"逻辑委托给 {@link TetrisDecisionService}。
 */
@Service
public class TetrisGameService {

    /** 棋盘 + 分数 + 最近一次决策轨迹的完整快照,供 REST API 直接返回。 */
    public record Snapshot(
            int boardWidth, int boardHeight,
            List<int[]> filledCells,
            String currentPiece, String nextPiece,
            long score, int linesCleared, int turn, boolean gameOver,
            DecisionTrace lastDecision
    ) {}

    private final TetrisDecisionService decisionService;
    private TetrisEngine engine;

    public TetrisGameService(LayaModel model) {
        this.decisionService = new TetrisDecisionService(model);
        reset(42L);
    }

    /** 开一局新游戏。 */
    public synchronized Snapshot reset(Long seed) {
        this.engine = new TetrisEngine(seed != null ? seed : 42L);
        return snapshot(null);
    }

    /** 只读取当前状态,不推进游戏。 */
    public synchronized Snapshot state() {
        return snapshot(null);
    }

    /**
     * 推进一步:对当前方块做一次完整的 Laya 决策 + 落子。
     * 若游戏已结束,直接返回当前快照(不再推理)。
     */
    public synchronized Snapshot step() {
        if (engine.isGameOver()) {
            return snapshot(null);
        }
        DecisionTrace trace = decisionService.decideAndApply(engine);
        return snapshot(trace);
    }

    private Snapshot snapshot(DecisionTrace lastDecision) {
        Board board = engine.board();
        return new Snapshot(
                board.width(), board.height(), board.filledCells(),
                engine.current().name(), engine.next().name(),
                engine.score(), engine.linesCleared(), engine.turn(), engine.isGameOver(),
                lastDecision
        );
    }
}
