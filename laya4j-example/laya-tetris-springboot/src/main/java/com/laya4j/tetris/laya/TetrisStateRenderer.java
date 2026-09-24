package com.laya4j.tetris.laya;

import com.laya4j.tetris.engine.Board;
import com.laya4j.tetris.engine.Placement;
import com.laya4j.tetris.engine.TetrisEngine;

import java.util.List;

/**
 * 把 Tetris 引擎的当前局面渲染成 Laya 模型的 {@code state} 输入文本。
 *
 * <p>Laya 的推理接口是 {@code predict(String state, List<Question> questions)}——
 * {@code state} 是模型能看到的唯一"世界观测",因此这里把决策所需的全部
 * 几何事实(当前盘面高度/孔洞/凹凸度、候选简选列表各自的落子后指标)
 * 显式编码进一段紧凑 JSON 文本,而不是让模型自己"脑补"棋盘几何——
 * 这是从贪吃蛇示例中沿用下来的经验:把可计算的传感器事实写进 state,
 * 比单纯丢一个原始网格给通用 LLM 决策模型更有效。
 *
 * <p>产出的 JSON 同时也是 {@link com.laya4j.tetris.laya.HeuristicLayaModel}
 * (启发式 mock 实现)解析状态、复原候选列表统计量的数据源,保证
 * "真实模型看到的输入" 与 "mock 模型消费的输入" 完全一致。
 */
public final class TetrisStateRenderer {

    private TetrisStateRenderer() {}

    /**
     * 渲染当前局面 + 候选简选列表为 state 文本。
     *
     * @param engine    当前游戏引擎(读取棋盘/方块信息,不做任何修改)
     * @param shortlist 已排序的候选简选列表(参见
     *                  {@code TetrisDecisionService} 的简选逻辑),
     *                  下标即为 {@link TetrisQuestions} 中 {@code c0, c1, ...}
     *                  candidate 标签对应的索引
     * @return 紧凑单行 JSON 字符串
     */
    public static String render(TetrisEngine engine, List<Placement> shortlist) {
        Board board = engine.board();
        StringBuilder sb = new StringBuilder(512);
        sb.append("{\"board_width\":").append(board.width())
          .append(",\"board_height\":").append(board.height())
          .append(",\"current_piece\":\"").append(engine.current()).append('"')
          .append(",\"next_piece\":\"").append(engine.next()).append('"')
          .append(",\"turn\":").append(engine.turn())
          .append(",\"lines_cleared\":").append(engine.linesCleared())
          .append(",\"current_max_height\":").append(board.maxHeight())
          .append(",\"current_holes\":").append(board.holeCount())
          .append(",\"current_bumpiness\":").append(board.bumpiness())
          .append(",\"column_heights\":").append(java.util.Arrays.toString(board.columnHeights()))
          .append(",\"candidates\":[");
        for (int i = 0; i < shortlist.size(); i++) {
            if (i > 0) sb.append(',');
            Placement p = shortlist.get(i);
            sb.append("{\"id\":\"c").append(i).append('"')
              .append(",\"rotation\":").append(p.rotation())
              .append(",\"column\":").append(p.column())
              .append(",\"resulting_height\":").append(p.resultingHeight())
              .append(",\"resulting_max_height\":").append(p.resultingMaxHeight())
              .append(",\"resulting_holes\":").append(p.resultingHoles())
              .append(",\"resulting_bumpiness\":").append(p.resultingBumpiness())
              .append(",\"lines_cleared\":").append(p.linesCleared())
              .append('}');
        }
        sb.append("]}");
        return sb.toString();
    }
}
