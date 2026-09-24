package com.laya4j.tetris.laya;

import com.laya4j.core.Question;
import com.laya4j.tetris.engine.Placement;
import com.laya4j.tetris.engine.Tetromino;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 构造每一步落子决策需要问 Laya 的问题集合,三种决策原语一次用全:
 *
 * <ul>
 *   <li>{@code CHOICE placement}   —— 从候选简选列表(shortlist)里选一个落子方案</li>
 *   <li>{@code NOUL risk_c{i}}     —— 对简选列表中每个候选,判断"落在这里是否
 *       会让棋盘堆叠过高、面临随时game over的风险"(逐候选判断,与
 *       {@code CHOICE} 的 label 一一对应,呼应贪吃蛇示例里
 *       "每个方向一个 collision 判断" 的设计)</li>
 *   <li>{@code SCORE board_health} —— 对"落子前"的当前盘面整体健康度打分
 *       (纯展示用途,不参与门控决策,用于在 Web 界面里展示模型对局势的
 *       整体判断)</li>
 * </ul>
 *
 * <p>所有问题共享同一次 forward(同一个 {@code state} 输入),
 * 因此问题数量随 shortlist 大小线性增长:{@code 1 + N + 1} 个问题。
 */
public final class TetrisQuestions {

    /** {@code board_health} SCORE 问题的等级名称,从差到好排列。 */
    public static final String[] HEALTH_LEVELS = {"critical", "risky", "stable", "strong"};

    private TetrisQuestions() {}

    /**
     * @param piece     当前正在决策的方块(用于计算每个候选的包围盒宽度,
     *                  让 {@code describe} 文本里的 "columns X-Y" 准确)
     * @param shortlist 候选简选列表,下标对应 {@code choice} 问题里的
     *                  {@code c0, c1, ...} 标签,以及各自的 {@code risk_c{i}} 问题
     * @return 待送入 {@code LayaModel.predict} 的问题列表
     */
    public static List<Question> build(Tetromino piece, List<Placement> shortlist) {
        List<Question> qs = new ArrayList<>();

        Map<String, String> options = new LinkedHashMap<>();
        for (int i = 0; i < shortlist.size(); i++) {
            Placement p = shortlist.get(i);
            options.put("c" + i, p.describe(piece.width(p.rotation())));
        }
        qs.add(Question.choice("placement",
                "You are placing a falling Tetris piece. Each candidate below describes the "
                        + "resulting board after a hard drop at a specific rotation/column: "
                        + "lower total height is better, fewer holes is much better (holes cannot "
                        + "be filled later), lower bumpiness (smoother surface) is better, and "
                        + "clearing more lines is strongly rewarded. Pick the single best candidate.",
                options).build());

        for (int i = 0; i < shortlist.size(); i++) {
            Placement p = shortlist.get(i);
            qs.add(Question.noul("risk_c" + i,
                    "Candidate c" + i + " (" + p.describe(piece.width(p.rotation())) + "). "
                            + "Would applying this candidate push the stack dangerously close to "
                            + "the top of the board, risking an imminent game over?",
                    "no, this candidate keeps the stack at a safe height",
                    "yes, this candidate is dangerously high").build());
        }

        qs.add(Question.score("board_health",
                "Rate the overall health of the CURRENT board (before this move), considering "
                        + "stack height, hole count, and surface smoothness.",
                HEALTH_LEVELS).build());

        return qs;
    }
}
