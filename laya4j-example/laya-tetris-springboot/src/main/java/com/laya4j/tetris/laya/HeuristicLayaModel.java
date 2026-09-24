package com.laya4j.tetris.laya;

import com.laya4j.core.ChoiceDecision;
import com.laya4j.core.Decision;
import com.laya4j.core.NoulDecision;
import com.laya4j.core.Question;
import com.laya4j.core.ScoreDecision;
import com.laya4j.model.LayaModel;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 启发式 {@link LayaModel} 实现,用经典 Tetris AI 权重(高度/消行/孔洞/凹凸度)
 * 计算候选落点得分,输出与真实 Laya 模型同构的 {@link Decision}。
 *
 * <p>与贪吃蛇示例中的 {@code HeuristicSnakeModel} 同样的设计原则:
 * 只通过 {@link #predict(String, List)} 接口约定的 {@code state} 文本
 * 获取信息(用正则从 {@link TetrisStateRenderer} 产出的 JSON 里抽取字段),
 * 不"偷看"游戏引擎内部状态——这样它才能作为
 * {@code LayaRouter}/{@code LayaPredictor} 的合法可插拔后端,
 * 用于在没有 1.2GB ONNX 权重时快速验证决策链路是否打通。
 */
public final class HeuristicLayaModel implements LayaModel {

    private static final double W_HEIGHT = -0.510066;
    private static final double W_LINES = 0.760666;
    private static final double W_HOLES = -0.35663;
    private static final double W_BUMPINESS = -0.184483;

    /** 最高列高度达到棋盘高度的这个比例以上,视为"危险,可能马上 game over"。 */
    private static final double DANGER_HEIGHT_RATIO = 0.85;

    @Override
    public String modelId() {
        return "tetris-heuristic";
    }

    @Override
    public Map<String, Decision> predict(String state, List<Question> questions) {
        StateView view = StateView.parse(state);
        Map<String, Decision> out = new LinkedHashMap<>();
        for (Question q : questions) {
            switch (q.type()) {
                case CHOICE -> out.put(q.name(), choicePlacement(view));
                case NOUL -> out.put(q.name(), riskDecision(view, q.name()));
                case SCORE -> out.put(q.name(), boardHealth(view));
            }
        }
        return out;
    }

    private Decision choicePlacement(StateView view) {
        Map<String, Double> raw = new LinkedHashMap<>();
        String best = null;
        double bestScore = -Double.MAX_VALUE;
        for (StateView.Candidate c : view.candidates) {
            double score = W_HEIGHT * c.resultingHeight + W_LINES * c.linesCleared
                    + W_HOLES * c.resultingHoles + W_BUMPINESS * c.resultingBumpiness;
            raw.put(c.id, score);
            if (score > bestScore) {
                bestScore = score;
                best = c.id;
            }
        }
        Map<String, Double> probs = softmax(raw, 4.0);
        double bestP = best != null ? probs.getOrDefault(best, 0.0) : 0.0;
        return new ChoiceDecision(best != null ? best : "c0", bestP, probs);
    }

    private Decision riskDecision(StateView view, String questionName) {
        String candidateId = questionName.substring("risk_".length());
        StateView.Candidate c = view.candidate(candidateId);
        int dangerHeight = (int) Math.round(view.boardHeight * DANGER_HEIGHT_RATIO);
        boolean risky = c != null && c.resultingMaxHeight >= dangerHeight;
        return new NoulDecision(risky ? 0.95 : 0.05, 0.9);
    }

    private Decision boardHealth(StateView view) {
        // 用当前盘面(落子前)的高度/孔洞/凹凸度映射到 4 档等级,
        // 与 TetrisQuestions.HEALTH_LEVELS 顺序一致(critical..strong)
        double ratio = view.boardHeight == 0 ? 0 : (double) view.currentMaxHeight / view.boardHeight;
        int level;
        if (ratio >= 0.8 || view.currentHoles >= 6) {
            level = 0; // critical
        } else if (ratio >= 0.55 || view.currentHoles >= 3) {
            level = 1; // risky
        } else if (ratio >= 0.3 || view.currentBumpiness >= 6) {
            level = 2; // stable
        } else {
            level = 3; // strong
        }
        Map<Integer, Double> dist = new LinkedHashMap<>();
        for (int i = 0; i < TetrisQuestions.HEALTH_LEVELS.length; i++) {
            dist.put(i, i == level ? 0.85 : 0.05);
        }
        return new ScoreDecision(level, 0.85, dist, TetrisQuestions.HEALTH_LEVELS);
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

    /** 从 {@link TetrisStateRenderer} 产出的 JSON 文本里抽取决策所需字段的极简解析器。 */
    private record StateView(
            int boardWidth, int boardHeight,
            int currentMaxHeight, int currentHoles, int currentBumpiness,
            List<Candidate> candidates
    ) {
        record Candidate(String id, int resultingHeight, int resultingMaxHeight,
                         int resultingHoles, int resultingBumpiness, int linesCleared) {}

        Candidate candidate(String id) {
            return candidates.stream().filter(c -> c.id.equals(id)).findFirst().orElse(null);
        }

        static StateView parse(String json) {
            int boardWidth = intField(json, "board_width", 10);
            int boardHeight = intField(json, "board_height", 20);
            int currentMaxHeight = intField(json, "current_max_height", 0);
            int currentHoles = intField(json, "current_holes", 0);
            int currentBumpiness = intField(json, "current_bumpiness", 0);

            List<Candidate> candidates = new ArrayList<>();
            Matcher m = Pattern.compile(
                    "\\{\"id\":\"(c\\d+)\",\"rotation\":\\d+,\"column\":-?\\d+,"
                            + "\"resulting_height\":(\\d+),\"resulting_max_height\":(\\d+),"
                            + "\"resulting_holes\":(\\d+),\"resulting_bumpiness\":(\\d+),"
                            + "\"lines_cleared\":(\\d+)\\}"
            ).matcher(json);
            while (m.find()) {
                candidates.add(new Candidate(
                        m.group(1),
                        Integer.parseInt(m.group(2)),
                        Integer.parseInt(m.group(3)),
                        Integer.parseInt(m.group(4)),
                        Integer.parseInt(m.group(5)),
                        Integer.parseInt(m.group(6))
                ));
            }
            return new StateView(boardWidth, boardHeight, currentMaxHeight,
                    currentHoles, currentBumpiness, candidates);
        }

        private static int intField(String json, String key, int fallback) {
            Matcher m = Pattern.compile("\"" + key + "\":(-?\\d+)").matcher(json);
            return m.find() ? Integer.parseInt(m.group(1)) : fallback;
        }
    }
}
