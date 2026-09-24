package com.laya4j.snake.laya;

import com.laya4j.core.ChoiceDecision;
import com.laya4j.core.Decision;
import com.laya4j.core.NoulDecision;
import com.laya4j.model.LayaModel;
import com.laya4j.snake.engine.SnakeGame;
import com.laya4j.snake.engine.SnakeGame.Direction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 编排"一次贪吃蛇移动决策"的完整流程:
 * 渲染 state → 问 Laya(1 choice + 4 noul)→ 消费决策 → 走一步。
 *
 * <p>决策消费策略 —— "模型选择优先,安全规则兜底":
 * <ol>
 *   <li>优先采纳 {@code next_direction}(CHOICE)选出的方向</li>
 *   <li>若该方向对应的 {@code fatal_x}(NOUL)P(true) ≥ 0.5(模型或规则
 *       判断为致命),则改用规则策略挑选安全方向兜底</li>
 *   <li>若无任何安全方向,仍按规则策略的"最不坏"方向走,蛇撞死,游戏结束</li>
 * </ol>
 */
public final class SnakeDecisionService {

    private static final Logger log = LoggerFactory.getLogger(SnakeDecisionService.class);

    private static final List<String> DIR_IDS =
            List.of("up", "down", "left", "right");

    private final LayaModel model;

    public SnakeDecisionService(LayaModel model) {
        this.model = model;
    }

    /**
     * 对当前局面执行一次"决策 + 走一步"。
     *
     * @param game 待决策的游戏引擎(会被原地修改:蛇移动、吃食物、可能死亡)
     * @return 本次决策的完整轨迹,可直接序列化返回给前端
     */
    public DecisionTrace decideAndApply(SnakeGame game) {
        String state = SnakeStateRenderer.render(game);
        List<com.laya4j.core.Question> questions = SnakeQuestions.build(game);

        long t0 = System.currentTimeMillis();
        Map<String, Decision> answers = model.predict(state, questions);
        long latency = System.currentTimeMillis() - t0;

        Map<String, String> dirDescriptions = new LinkedHashMap<>();
        for (Direction d : Direction.values()) {
            dirDescriptions.put(d.lower(), SnakeQuestions.describeDir(d, game));
        }

        Map<String, Double> choiceProbs = Map.of();
        String choicePick = null;
        if (answers.get(SnakeQuestions.CHOICE_NAME) instanceof ChoiceDecision cd) {
            choiceProbs = cd.probabilities();
            choicePick = cd.choice();
        }

        Map<String, Double> risk = new LinkedHashMap<>();
        for (Direction d : Direction.values()) {
            if (answers.get("fatal_" + d.lower()) instanceof NoulDecision nd) {
                risk.put(d.lower(), nd.probability());
            }
        }

        // ---- 决策消费:模型选择优先,安全兜底纠错 ----
        String verdictDir = choicePick;
        String verdictMode = "model";
        String verdictReason;
        boolean modelPickFatal = choicePick == null
                || risk.getOrDefault(choicePick, 0.0) >= 0.5;
        if (modelPickFatal) {
            verdictDir = safePick(game, risk);
            verdictMode = "fallback";
            verdictReason = choicePick == null
                    ? "模型未给出选择, 启用规则安全策略"
                    : "模型选 %s 的致命风险 %.2f ≥ 0.5, 改用规则安全方向".formatted(
                            choicePick, risk.getOrDefault(choicePick, 0.0));
        } else {
            verdictReason = "模型选 %s 的致命风险 %.2f < 0.5, 直接采纳".formatted(
                    choicePick, risk.getOrDefault(choicePick, 0.0));
        }

        boolean ate = game.step(Direction.valueOf(verdictDir.toUpperCase()));

        log.info("决策路径 耗时={}ms choice={}({}) risk={} 裁决={}({}) ate={} len={}",
                latency, choicePick,
                String.format("%.3f", choiceProbs.getOrDefault(choicePick, 0.0)),
                risk, verdictDir, verdictMode, ate, game.body().size());

        return new DecisionTrace(model.modelId(), latency, state,
                questionSpecs(questions), DIR_IDS, dirDescriptions,
                choiceProbs, choicePick, risk, verdictDir, verdictMode, verdictReason,
                ate, game.body().size(), !game.alive());
    }

    /** 把决策链上每个问题的参数(名称/类型/指令/选项)转成可序列化快照。 */
    private static List<DecisionTrace.QuestionSpec> questionSpecs(List<com.laya4j.core.Question> questions) {
        return questions.stream().map(q -> {
            Map<String, String> options = switch (q.type()) {
                case CHOICE -> q.choices();
                case SCORE -> {
                    Map<String, String> levels = new LinkedHashMap<>();
                    String[] arr = q.levels();
                    for (int i = 0; i < arr.length; i++) {
                        levels.put(String.valueOf(i), arr[i]);
                    }
                    yield levels;
                }
                case NOUL -> Map.of("false", q.falseDesc(), "true", q.trueDesc());
            };
            return new DecisionTrace.QuestionSpec(
                    q.name(), q.type().name(), q.instructions(), options);
        }).toList();
    }

    /**
     * 规则安全策略:候选方向按"靠近食物"排序,过滤致命与掉头,取最优;
     * 实在没有则取"最不坏"方向(撞死)。
     */
    static String safePick(SnakeGame game, Map<String, Double> risk) {
        var head = game.head();
        var food = game.food();
        List<Direction> ranked = new ArrayList<>(List.of(Direction.values()));
        ranked.sort((a, b) -> {
            int da = food == null ? 0 : dist(head.plus(a), food);
            int db = food == null ? 0 : dist(head.plus(b), food);
            return Integer.compare(da, db);
        });
        Direction best = null;
        for (Direction d : ranked) {
            if (d == game.heading().opposite() || game.isFatal(d)) {
                continue;
            }
            if (best == null) {
                best = d;
            }
        }
        if (best == null) {
            best = ranked.stream()
                    .filter(d -> d != game.heading().opposite())
                    .findFirst().orElse(game.heading());
        }
        return best.lower();
    }

    private static int dist(SnakeGame.Cell a, SnakeGame.Cell b) {
        return Math.abs(a.x() - b.x()) + Math.abs(a.y() - b.y());
    }
}
