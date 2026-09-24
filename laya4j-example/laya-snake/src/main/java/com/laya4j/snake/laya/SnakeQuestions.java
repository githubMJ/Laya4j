package com.laya4j.snake.laya;

import com.laya4j.core.Question;
import com.laya4j.snake.engine.SnakeGame;
import com.laya4j.snake.engine.SnakeGame.Direction;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 每一步向 Laya 提出的决策问题:
 * <ul>
 *   <li>{@code next_direction}(CHOICE):四个方向及其落点,模型选一个</li>
 *   <li>{@code fatal_up/down/left/right}(NOUL):往该方向走是否立即死亡</li>
 * </ul>
 */
public final class SnakeQuestions {

    public static final String CHOICE_NAME = "next_direction";

    private SnakeQuestions() {
    }

    public static List<Question> build(SnakeGame game) {
        Map<String, String> dirChoices = new LinkedHashMap<>();
        for (Direction d : Direction.values()) {
            dirChoices.put(d.lower(), describeDir(d, game));
        }
        List<Question> questions = new ArrayList<>();
        questions.add(Question.choice(CHOICE_NAME,
                "Which direction should the snake move this step? Avoid walls and its own body.",
                dirChoices).build());
        for (Direction d : Direction.values()) {
            questions.add(Question.noul("fatal_" + d.lower(),
                    "If the snake moves " + d.lower() + " this step, will it immediately die?",
                    "no, the cell is safe", "yes, wall or body there").build());
        }
        return questions;
    }

    static String describeDir(Direction d, SnakeGame game) {
        SnakeGame.Cell next = game.head().plus(d);
        return "move %s to cell (%d,%d)".formatted(d.lower(), next.x(), next.y());
    }
}
