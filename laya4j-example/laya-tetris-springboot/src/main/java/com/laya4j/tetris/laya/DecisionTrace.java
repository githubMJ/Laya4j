package com.laya4j.tetris.laya;

import java.util.List;
import java.util.Map;

/**
 * 一次完整落子决策的可观测轨迹:state 输入、CHOICE 概率分布、
 * 各候选的 NOUL 风险判断、当前盘面健康度打分,以及最终裁决。
 *
 * <p>同时携带完整的"方块生命周期"信息,供前端播放三段式动画:
 * <ol>
 *   <li><b>出生显形</b>:方块以旋转状态 0、居中列({@code spawnRotation}/
 *       {@code spawnColumn})出现在棋盘顶部——这是 Laya 开始决策<b>之前</b>
 *       玩家/观众看到的原始形状,对应真实俄罗斯方块"方块刚出现"的画面</li>
 *   <li><b>决策过渡</b>:方块从出生形态旋转/横移到 Laya 最终选定的
 *       {@code rotation}/{@code column}</li>
 *   <li><b>硬降下落</b>:方块从当前位置下落到 {@code landingRow} 并固化</li>
 * </ol>
 * 与 {@code Snapshot.filledCells}(落子并消行<b>后</b>的最终盘面)配合:
 * 先渲染 {@code boardBeforeCells} 并播放上述动画,动画结束后再切换为
 * 渲染最终盘面。
 *
 * <p>供 REST API / Web 页面直接序列化展示,让"Laya 是如何做出这次落子决策的"
 * 全程可视化,而不只是看到最终结果。
 *
 * @param model              产出本次决策的模型标识({@code LayaModel.modelId()})
 * @param latencyMs          本次 forward 推理耗时(毫秒)
 * @param state              本次推理使用的 state 输入文本(JSON)
 * @param shortlistIds       候选简选列表的 id 顺序({@code c0, c1, ...})
 * @param shortlistDescriptions 各候选 id → 人类可读的中文描述(旋转/落点列/
 *                              高度/孔洞/消行数),供前端在 c0/c1 等编号旁
 *                              直接展示"这个候选具体是什么样的摆法",
 *                              避免只看到一个不知所云的编号
 * @param choiceProbs        CHOICE 问题(placement)的概率分布,key 为候选 id
 * @param choicePick         CHOICE 问题选出的候选 id(模型原始选择,未经安全兜底)
 * @param risk               各候选 id → NOUL "是否危险" 的概率(P(true))
 * @param boardHealthLevel   SCORE 问题(board_health)选出的等级名称
 * @param verdictCandidateId 最终采纳的候选 id(可能因安全兜底而不同于 choicePick)
 * @param verdictMode        裁决方式:{@code "model"}(直接采纳模型选择)、
 *                           {@code "fallback"}(模型选择过于危险,改用启发式最优候选)、
 *                           {@code "gameover"}(无合法候选,游戏结束)
 * @param piece              本次下落的方块类型名称(对应
 *                           {@code Tetromino.name()},如 {@code "I"}/{@code "T"})
 * @param spawnRotation      方块出生时的旋转状态,恒为 0;{@code gameover} 时为 -1
 * @param spawnColumn        方块出生时居中摆放的包围盒左上角列;{@code gameover} 时为 -1
 * @param rotation           最终采纳候选的旋转状态;{@code gameover} 时为 -1(无有效落子)
 * @param column             最终采纳候选的包围盒左上角列;{@code gameover} 时为 -1
 * @param landingRow         最终采纳候选硬降后的包围盒左上角行;{@code gameover} 时为 -1
 * @param boardBeforeCells   本次方块开始下落<b>之前</b>的盘面已占据格子列表
 *                           (与 {@code Snapshot.filledCells} 的坐标格式一致)
 */
public record DecisionTrace(
        String model,
        long latencyMs,
        String state,
        List<String> shortlistIds,
        Map<String, String> shortlistDescriptions,
        Map<String, Double> choiceProbs,
        String choicePick,
        Map<String, Double> risk,
        String boardHealthLevel,
        String verdictCandidateId,
        String verdictMode,
        String piece,
        int spawnRotation,
        int spawnColumn,
        int rotation,
        int column,
        int landingRow,
        List<int[]> boardBeforeCells
) {}
