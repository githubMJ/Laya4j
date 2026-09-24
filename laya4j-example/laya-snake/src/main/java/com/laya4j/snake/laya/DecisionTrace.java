package com.laya4j.snake.laya;

import java.util.List;
import java.util.Map;

/**
 * 一次移动决策的完整决策链轨迹,供 REST API / Web 页面实时序列化展示:
 * 向 Laya 提出的每个问题(参数)、模型各答案的概率分布、NOUL 风险、
 * 以及最终裁决路径——让"Laya 是如何决定这一步往哪走"全程可视化。
 *
 * @param model            产出本次决策的模型标识({@code LayaModel.modelId()})
 * @param latencyMs        本次 forward 推理耗时(毫秒)
 * @param state            本次推理使用的 state 输入文本(JSON)
 * @param questions        本次向模型提出的全部问题(名称/类型/指令/选项),
 *                         即决策链的完整输入参数
 * @param dirOptions       四个方向 id(up/down/left/right,固定顺序)
 * @param dirDescriptions  方向 id → 人类可读描述(落点坐标)
 * @param choiceProbs      CHOICE 问题(next_direction)的概率分布,key 为方向 id
 * @param choicePick       CHOICE 选出的方向(模型原始选择,未经安全兜底)
 * @param risk             方向 id → NOUL "是否致命" 的 P(true)
 * @param verdictDir       最终采纳的方向(可能因安全兜底而不同于 choicePick)
 * @param verdictMode      裁决方式:{@code "model"}(直接采纳模型选择)、
 *                         {@code "fallback"}(模型选择被判致命,改用规则安全方向)
 * @param verdictReason    裁决理由:为什么采纳/否决模型选择
 *                         (如 "risk(up)=0.05 < 0.5" / "risk(left)=0.95 ≥ 0.5, fallback")
 * @param ateFood          本步是否吃到食物(蛇变长)
 * @param bodyLength       本步走完后的蛇身长度
 * @param gameOver         本步走完后游戏是否结束(撞死)
 */
public record DecisionTrace(
        String model,
        long latencyMs,
        String state,
        List<QuestionSpec> questions,
        List<String> dirOptions,
        Map<String, String> dirDescriptions,
        Map<String, Double> choiceProbs,
        String choicePick,
        Map<String, Double> risk,
        String verdictDir,
        String verdictMode,
        String verdictReason,
        boolean ateFood,
        int bodyLength,
        boolean gameOver
) {
    /**
     * 决策链中单个问题的参数快照。
     *
     * @param name         问题名(如 {@code next_direction} / {@code fatal_up})
     * @param type         决策类型(CHOICE / SCORE / NOUL)
     * @param instructions 问题指令文本
     * @param options      选项 → 描述(NOUL 恒为 false/true 两项)
     */
    public record QuestionSpec(
            String name, String type, String instructions, Map<String, String> options
    ) {}
}
