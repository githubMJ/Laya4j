package com.laya4j.tetris.laya;

import com.laya4j.core.ChoiceDecision;
import com.laya4j.core.Decision;
import com.laya4j.core.NoulDecision;
import com.laya4j.core.Question;
import com.laya4j.core.ScoreDecision;
import com.laya4j.model.LayaModel;
import com.laya4j.tetris.engine.Placement;
import com.laya4j.tetris.engine.Tetromino;
import com.laya4j.tetris.engine.TetrisEngine;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 编排"一次 Tetris 落子决策"的完整流程:
 * 枚举候选 → 简选 shortlist → 渲染 state → 问 Laya → 消费决策 → 应用落子。
 *
 * <p>决策消费策略与贪吃蛇示例一致 ——"模型选择优先,安全规则兜底纠错":
 * <ol>
 *   <li>优先采纳 {@code CHOICE placement} 问题选出的候选</li>
 *   <li>若该候选对应的 {@code NOUL risk_c{i}} 判断为"危险"(P(true) ≥ 0.5),
 *       则改用 shortlist 中风险最低、启发式得分最高的候选兜底,
 *       避免模型的单次误判直接导致游戏结束</li>
 *   <li>若 shortlist 为空(方块出生即无合法落点),游戏结束</li>
 * </ol>
 */
public final class TetrisDecisionService {

    private static final Logger log = LoggerFactory.getLogger(TetrisDecisionService.class);

    /**
     * 最近 1 秒内完成的决策时间戳,用来算出"服务端真实每秒决策数"——
     * 与前端"节奏"滑条设定值是两件事:滑条只控制轮询间隔,
     * 这里记录的才是 forward 推理 + 落子真正跑完所需的吞吐上限。
     */
    private static final java.util.Deque<Long> recentDecisionTimestamps = new java.util.concurrent.ConcurrentLinkedDeque<>();

    /** 候选简选列表的最大大小:过大会让单次 forward 的 marker 数量膨胀。 */
    public static final int SHORTLIST_SIZE = 5;

    private final LayaModel model;

    public TetrisDecisionService(LayaModel model) {
        this.model = model;
    }

    /**
     * 对当前方块执行一次完整的"决策 + 落子"。
     *
     * @param engine 待决策的游戏引擎(会被原地修改:方块落子、消行、切换下一方块)
     * @return 本次决策的完整轨迹,可直接序列化返回给前端
     */
    public DecisionTrace decideAndApply(TetrisEngine engine) {
        // 在方块被固化(lock)之前,先捕获"下落前"的方块类型、出生形态与盘面快照,
        // 供前端播放"出生显形 → 决策过渡 → 硬降下落"的完整三段式动画。
        Tetromino piece = engine.current();
        int spawnRotation = 0;
        int spawnColumn = TetrisEngine.spawnColumn(piece);
        List<int[]> boardBeforeCells = engine.board().filledCells();

        List<Placement> all = engine.candidatePlacements();
        if (all.isEmpty()) {
            return new DecisionTrace(model.modelId(), 0, "{}", List.of(), Map.of(), Map.of(),
                    null, Map.of(), null, null, "gameover",
                    piece.name(), -1, -1, -1, -1, -1, boardBeforeCells);
        }
        List<Placement> shortlist = all.subList(0, Math.min(SHORTLIST_SIZE, all.size()));

        String state = TetrisStateRenderer.render(engine, shortlist);
        List<Question> questions = TetrisQuestions.build(engine.current(), shortlist);

        long t0 = System.currentTimeMillis();
        Map<String, Decision> answers = model.predict(state, questions);
        long latency = System.currentTimeMillis() - t0;

        List<String> shortlistIds = idsInOrder(shortlist.size());
        Map<String, String> shortlistDescriptions = describeShortlist(piece, shortlist, shortlistIds);

        Map<String, Double> choiceProbs = Map.of();
        String choicePick = null;
        if (answers.get("placement") instanceof ChoiceDecision cd) {
            choiceProbs = cd.probabilities();
            choicePick = cd.choice();
        }

        Map<String, Double> risk = new LinkedHashMap<>();
        for (int i = 0; i < shortlist.size(); i++) {
            String qName = "risk_c" + i;
            if (answers.get(qName) instanceof NoulDecision nd) {
                risk.put("c" + i, nd.probability());
            }
        }

        String healthLevel = null;
        if (answers.get("board_health") instanceof ScoreDecision sd) {
            healthLevel = sd.nearestLevel();
        }

        // ---- 决策消费:模型选择优先,安全兜底纠错 ----
        String verdictId = choicePick;
        String verdictMode = "model";
        boolean picksExist = choicePick != null && indexOf(choicePick) < shortlist.size();
        boolean risky = picksExist && risk.getOrDefault(choicePick, 0.0) >= 0.5;
        if (!picksExist || risky) {
            // 兜底:shortlist 已按启发式得分降序排列,优先选风险最低的;
            // 若全部风险判断都为空(理论上不会),退化为 shortlist 第一名
            String safest = shortlistIds.stream()
                    .filter(id -> risk.getOrDefault(id, 0.0) < 0.5)
                    .findFirst()
                    .orElse(shortlistIds.get(0));
            verdictId = safest;
            verdictMode = "fallback";
        }

        Placement chosen = shortlist.get(indexOf(verdictId));
        engine.applyPlacement(chosen);

        long throughput = recordAndCountThroughput();
        log.info("决策路径 piece={} 耗时={}ms 吞吐={}/s 出生(rot={},col={}) -> 决策(rot={},col={}) -> 落地row={} 裁决={}({})",
                piece.name(), latency, throughput,
                spawnRotation, spawnColumn, chosen.rotation(), chosen.column(), chosen.landingRow(),
                verdictId, verdictMode);

        return new DecisionTrace(model.modelId(), latency, state, shortlistIds,
                shortlistDescriptions, choiceProbs, choicePick, risk, healthLevel, verdictId, verdictMode,
                piece.name(), spawnRotation, spawnColumn,
                chosen.rotation(), chosen.column(), chosen.landingRow(), boardBeforeCells);
    }

    /**
     * 把每个候选 id({@code c0, c1, ...})翻译成人类可读的中文描述,
     * 例如"旋转1档 · 第3列 · 消1行 · 无孔洞 · 高度+2",
     * 让前端不用只看一个编号,能直接理解"c0 具体是什么摆法"。
     */
    private static Map<String, String> describeShortlist(Tetromino piece, List<Placement> shortlist, List<String> ids) {
        Map<String, String> out = new LinkedHashMap<>();
        for (int i = 0; i < shortlist.size(); i++) {
            Placement p = shortlist.get(i);
            int pieceWidth = piece.width(p.rotation());
            int colEnd = p.column() + pieceWidth - 1;
            String colDesc = pieceWidth <= 1 ? ("第" + p.column() + "列") : ("第" + p.column() + "-" + colEnd + "列");
            StringBuilder sb = new StringBuilder();
            sb.append("旋转").append(p.rotation()).append("档 · ").append(colDesc);
            if (p.linesCleared() > 0) sb.append(" · 消").append(p.linesCleared()).append("行");
            sb.append(" · 孔洞").append(p.resultingHoles());
            sb.append(" · 高度").append(p.resultingMaxHeight());
            out.put(ids.get(i), sb.toString());
        }
        return out;
    }

    /**
     * 记录本次决策完成的时间戳,并清理 1 秒前的旧记录,
     * 返回清理后窗口内的决策次数(即"过去 1 秒服务端真实决策数/秒")。
     */
    private static long recordAndCountThroughput() {
        long now = System.currentTimeMillis();
        recentDecisionTimestamps.addLast(now);
        long cutoff = now - 1000;
        while (true) {
            Long head = recentDecisionTimestamps.peekFirst();
            if (head == null || head >= cutoff) break;
            recentDecisionTimestamps.pollFirst();
        }
        return recentDecisionTimestamps.size();
    }

    private static List<String> idsInOrder(int n) {
        return java.util.stream.IntStream.range(0, n).mapToObj(i -> "c" + i).toList();
    }

    private static int indexOf(String candidateId) {
        return Integer.parseInt(candidateId.substring(1));
    }
}
