package com.laya4j.tetris.engine;

/**
 * 一个候选落子摆放方案及其落子后的盘面度量指标。
 *
 * <p>由 {@link TetrisEngine#candidatePlacements()} 枚举生成:对当前方块的
 * 每个去重旋转状态、每个合法列位置做一次硬降模拟,记录落子后(含消行)的
 * 盘面统计数据。这些统计数据既用于经典启发式排序(生成候选简选列表),
 * 也会被渲染成自然语言描述交给 Laya 做 {@code choice} 决策。
 *
 * @param rotation        旋转状态(0..3,已按 {@link Tetromino#cells} 语义)
 * @param column          方块包围盒左上角所在列(originCol)
 * @param landingRow       硬降后方块包围盒左上角所在行(originRow)
 * @param linesCleared    该摆放会消除的行数
 * @param resultingHoles   落子并消行后的孔洞总数
 * @param resultingHeight  落子并消行后的总高度(aggregate height)
 * @param resultingMaxHeight 落子并消行后的最高列高度
 * @param resultingBumpiness 落子并消行后的凹凸度
 * @param heuristicScore   经典启发式加权得分(仅用于候选简选排序,
 *                         不代表 Laya 的最终决策依据)
 */
public record Placement(
        int rotation,
        int column,
        int landingRow,
        int linesCleared,
        int resultingHoles,
        int resultingHeight,
        int resultingMaxHeight,
        int resultingBumpiness,
        double heuristicScore
) {

    /**
     * 面向模型/人类可读的候选摆放文本描述,格式统一,供 {@code choice}
     * 问题的候选项描述使用。
     *
     * @return 例如 {@code "rotation=1, columns 3-4 → +1 height, +0 holes, bumpiness=3, clears 1 line(s)"}
     */
    public String describe(int pieceWidth) {
        int colEnd = column + pieceWidth - 1;
        String cols = pieceWidth <= 1 ? ("column " + column) : ("columns " + column + "-" + colEnd);
        return String.format(
                "rotation=%d, %s -> total height=%d (max column height=%d), holes=%d, bumpiness=%d, clears %d line(s)",
                rotation, cols, resultingHeight, resultingMaxHeight, resultingHoles, resultingBumpiness, linesCleared);
    }
}
