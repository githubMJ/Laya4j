package com.laya4j.snake.laya;

import com.laya4j.core.ChoiceDecision;
import com.laya4j.core.Decision;
import com.laya4j.core.NoulDecision;
import com.laya4j.core.Question;
import com.laya4j.model.LayaModel;
import com.laya4j.snake.engine.SnakeGame.Direction;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 启发式 {@link LayaModel} 实现,作为贪吃蛇示例的 mock 后端:
 * 经典贪心策略——候选方向按"靠近食物"排序,过滤致命与掉头。
 *
 * <p>与 tetris 示例的 {@code HeuristicLayaModel} 同样的设计原则:
 * 只通过 {@link #predict(String, List)} 接口约定的 {@code state} 文本
 * 获取信息(用正则从 {@link SnakeStateRenderer} 产出的 JSON 里抽取字段),
 * 不"偷看"游戏引擎内部状态——因此它可作为 {@code LayaModel} 的合法
 * 可插拔后端,在没有 1.2GB ONNX 权重时验证完整决策链路。
 */
public final class HeuristicSnakeModel implements LayaModel {

    @Override
    public String modelId() {
        return "snake-heuristic";
    }

    @Override
    public Map<String, Decision> predict(String state, List<Question> questions) {
        StateView view = StateView.parse(state);
        Map<String, Decision> out = new LinkedHashMap<>();
        for (Question q : questions) {
            if (q.type() == com.laya4j.core.DecisionType.CHOICE) {
                out.put(q.name(), choiceDirection(view));
            } else {
                out.put(q.name(), riskDecision(view, q.name()));
            }
        }
        return out;
    }

    private Decision choiceDirection(StateView view) {
        Map<String, Double> raw = new LinkedHashMap<>();
        String best = null;
        double bestScore = -Double.MAX_VALUE;
        for (String id : view.dirIds()) {
            Direction d = Direction.valueOf(id.toUpperCase());
            if (d == view.heading().opposite() || view.fatal(d)) {
                continue; // 启发式只考虑合法方向
            }
            int dist = Math.abs(headPlus(view, d)[0] - view.foodX())
                    + Math.abs(headPlus(view, d)[1] - view.foodY());
            raw.put(id, -(double) dist);
            if (-dist > bestScore) {
                bestScore = -dist;
                best = id;
            }
        }
        if (best == null) {
            // 无合法方向:均匀概率,选 heading,必死,交给上层兜底逻辑
            best = view.heading().lower();
            for (String id : view.dirIds()) {
                raw.put(id, 0.0);
            }
        }
        Map<String, Double> probs = softmax(raw, 0.5);
        return new ChoiceDecision(best, probs.getOrDefault(best, 0.0), probs);
    }

    private Decision riskDecision(StateView view, String questionName) {
        String dirId = questionName.substring("fatal_".length());
        boolean fatal = view.fatal(Direction.valueOf(dirId.toUpperCase()));
        return new NoulDecision(fatal ? 0.95 : 0.05, 0.9);
    }

    private static int[] headPlus(StateView view, Direction d) {
        int x = view.headX(), y = view.headY();
        return switch (d) {
            case UP -> new int[]{x, y - 1};
            case DOWN -> new int[]{x, y + 1};
            case LEFT -> new int[]{x - 1, y};
            case RIGHT -> new int[]{x + 1, y};
        };
    }

    private static Map<String, Double> softmax(Map<String, Double> raw, double temperature) {
        double max = raw.values().stream().mapToDouble(Double::doubleValue).max().orElse(0);
        double sum = 0;
        Map<String, Double> exp = new LinkedHashMap<>();
        for (var e : raw.entrySet()) {
            double v = Math.exp((e.getValue() - max) / temperature);
            exp.put(e.getKey(), v);
            sum += v;
        }
        Map<String, Double> probs = new LinkedHashMap<>();
        for (var e : exp.entrySet()) {
            probs.put(e.getKey(), e.getValue() / sum);
        }
        return probs;
    }

    @Override
    public void close() {
        // 无原生资源,空实现
    }

    /** 从 {@link SnakeStateRenderer} 产出的 JSON 文本里抽取决策所需字段的极简解析器。 */
    private record StateView(
            int headX, int headY, int foodX, int foodY,
            Direction heading, List<String> dirIds,
            Map<String, Integer> fatalFlags
    ) {
        static StateView parse(String json) {
            List<String> dirIds = List.of("up", "down", "left", "right");
            Map<String, Integer> fatal = new LinkedHashMap<>();
            for (String id : dirIds) {
                fatal.put(id, intField(json, "fatal_" + id, 0));
            }
            return new StateView(
                    intField(json, "head_x", 0),
                    intField(json, "head_y", 0),
                    intField(json, "food_x", -1),
                    intField(json, "food_y", -1),
                    Direction.valueOf(strField(json, "heading", "RIGHT")),
                    dirIds, fatal);
        }

        boolean fatal(Direction d) {
            return fatalFlags.getOrDefault(d.lower(), 0) == 1;
        }

        private static int intField(String json, String key, int fallback) {
            Matcher m = Pattern.compile("\"" + key + "\":(-?\\d+)").matcher(json);
            return m.find() ? Integer.parseInt(m.group(1)) : fallback;
        }

        private static String strField(String json, String key, String fallback) {
            Matcher m = Pattern.compile("\"" + key + "\":\"([A-Z]+)\"").matcher(json);
            return m.find() ? m.group(1) : fallback;
        }
    }
}
