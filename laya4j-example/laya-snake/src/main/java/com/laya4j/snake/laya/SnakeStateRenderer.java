package com.laya4j.snake.laya;

import com.laya4j.snake.engine.SnakeGame;
import com.laya4j.snake.engine.SnakeGame.Direction;

/**
 * 把当前局面渲染成给 Laya 的 state 输入(JSON 文本)。
 * 字段刻意保持扁平、可被 {@link HeuristicSnakeModel} 用正则抽取,
 * 同时对真实 Laya 模型也可读。
 */
public final class SnakeStateRenderer {

    private SnakeStateRenderer() {
    }

    public static String render(SnakeGame game) {
        SnakeGame.Cell head = game.head();
        SnakeGame.Cell food = game.food();
        StringBuilder sb = new StringBuilder();
        sb.append("{\"game\":\"snake\"");
        sb.append(",\"board_size\":").append(game.size());
        sb.append(",\"heading\":\"").append(game.heading().name()).append('"');
        sb.append(",\"head_x\":").append(head.x());
        sb.append(",\"head_y\":").append(head.y());
        if (food != null) {
            sb.append(",\"food_x\":").append(food.x());
            sb.append(",\"food_y\":").append(food.y());
        }
        sb.append(",\"body_length\":").append(game.body().size());
        for (Direction d : Direction.values()) {
            sb.append(",\"fatal_").append(d.lower()).append("\":")
              .append(game.isFatal(d) ? 1 : 0);
        }
        // 蛇身坐标,头在前,供模型/调试阅读
        sb.append(",\"body\":[");
        boolean first = true;
        for (int[] c : game.bodyCells()) {
            if (!first) sb.append(',');
            sb.append("[").append(c[0]).append(',').append(c[1]).append(']');
            first = false;
        }
        sb.append("]}");
        return sb.toString();
    }
}
