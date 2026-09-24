package com.laya4j.tetris.engine;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * 俄罗斯方块游戏引擎:管理棋盘、当前方块/下一方块、分数与行数,
 * 并提供"枚举当前方块所有合法候选落点"与"应用某个候选落点"两个核心能力。
 *
 * <p>本引擎只负责"整块硬降"式的决策粒度(不模拟逐帧左右移动/软降),
 * 这正是 Laya 决策集成所需要的抽象层级:每次决策只需要在有限个
 * (旋转, 列) 候选摆放里选一个,而不需要处理连续控制信号。
 *
 * <p>经典的启发式权重(用于 {@link #heuristicScore}) 来自公开的
 * Tetris AI 研究(如 Pierre Dellacherie / El-Tetris 的加权公式):
 * 高度惩罚、消行奖励、孔洞惩罚、凹凸度惩罚。这套权重仅用于
 * <b>候选简选(shortlist)排序</b>与"安全兜底"逻辑,真正的落子选择
 * 交给 Laya 的 {@code choice} 决策(参见 {@code TetrisDecisionService})。
 */
public final class TetrisEngine {

    /** 经典启发式权重:aggregate height、lines cleared、holes、bumpiness。 */
    private static final double W_HEIGHT = -0.510066;
    private static final double W_LINES = 0.760666;
    private static final double W_HOLES = -0.35663;
    private static final double W_BUMPINESS = -0.184483;

    private final Random random;
    private final List<Tetromino> bag = new ArrayList<>();
    private int bagIndex = 0;

    private Board board;
    private Tetromino current;
    private Tetromino next;
    private long score;
    private int linesCleared;
    private int turn;
    private boolean gameOver;

    public TetrisEngine(long seed) {
        this.random = new Random(seed);
        this.board = new Board();
        refillBagIfNeeded();
        this.current = drawNext();
        this.next = drawNext();
    }

    /** 7-bag 随机发牌:每 7 个方块保证 7 种各出现一次,避免长时间不出某个方块。 */
    private void refillBagIfNeeded() {
        if (bagIndex >= bag.size()) {
            bag.clear();
            Collections.addAll(bag, Tetromino.values());
            Collections.shuffle(bag, random);
            bagIndex = 0;
        }
    }

    private Tetromino drawNext() {
        refillBagIfNeeded();
        return bag.get(bagIndex++);
    }

    /**
     * 枚举当前方块在棋盘上所有合法的(旋转, 列)候选落点,并计算每个候选
     * 落子(含消行)后的盘面度量指标。
     *
     * @return 候选落点列表,按 {@link Placement#heuristicScore()} 降序排列
     *         (启发式最优的排在最前面,便于取 shortlist / 兜底选择)
     */
    public List<Placement> candidatePlacements() {
        List<Placement> out = new ArrayList<>();
        int distinctRotations = current.distinctRotationCount();
        for (int rotation = 0; rotation < distinctRotations; rotation++) {
            int[][] cells = current.cells(rotation);
            int minCol = Integer.MAX_VALUE, maxCol = Integer.MIN_VALUE;
            for (int[] c : cells) {
                minCol = Math.min(minCol, c[1]);
                maxCol = Math.max(maxCol, c[1]);
            }
            int pieceWidth = maxCol - minCol + 1;
            for (int col = -minCol; col <= board.width() - 1 - maxCol; col++) {
                if (board.collides(current, rotation, 0, col)) {
                    continue; // 出生位置就冲突,该列不可用
                }
                int landingRow = board.hardDropRow(current, rotation, col, 0);

                Board simulated = board.copy();
                simulated.lock(current, rotation, landingRow, col);
                int lines = simulated.clearFullLines();

                out.add(new Placement(
                        rotation, col, landingRow, lines,
                        simulated.holeCount(), simulated.aggregateHeight(),
                        simulated.maxHeight(), simulated.bumpiness(),
                        heuristicScore(simulated.aggregateHeight(), lines,
                                simulated.holeCount(), simulated.bumpiness())
                ));
            }
        }
        out.sort((a, b) -> Double.compare(b.heuristicScore(), a.heuristicScore()));
        return out;
    }

    private static double heuristicScore(int aggHeight, int lines, int holes, int bumpiness) {
        return W_HEIGHT * aggHeight + W_LINES * lines + W_HOLES * holes + W_BUMPINESS * bumpiness;
    }

    /**
     * 应用一个候选落点:固化方块、消行、计分、切换到下一方块,
     * 并检测新方块是否一出生就与棋盘冲突(游戏结束条件)。
     *
     * @param placement 待应用的候选落点(通常来自 {@link #candidatePlacements()})
     */
    public void applyPlacement(Placement placement) {
        if (gameOver) {
            return;
        }
        board.lock(current, placement.rotation(), placement.landingRow(), placement.column());
        int lines = board.clearFullLines();
        linesCleared += lines;
        score += linesScoreDelta(lines);
        turn++;

        current = next;
        next = drawNext();
        if (board.collides(current, 0, 0, spawnColumn(current))) {
            gameOver = true;
        }
    }

    /**
     * 计算某个方块以旋转状态 0 "居中出生"时的包围盒左上角列(originCol)。
     *
     * <p>公开该方法是为了让决策层(参见 {@code TetrisDecisionService})
     * 能够还原"方块最初出现在棋盘上方时是什么形状、在什么位置"——
     * 即 Laya 真正开始决策之前的初始画面,用于前端播放
     * "先显示出生形态,再动画过渡到模型选定的旋转/列,最后下落"的完整过程。
     *
     * @param piece 待计算出生位置的方块
     * @return 旋转状态 0 下,居中摆放的包围盒左上角列
     */
    public static int spawnColumn(Tetromino piece) {
        int[][] cells = piece.cells(0);
        int minCol = Integer.MAX_VALUE, maxCol = Integer.MIN_VALUE;
        for (int[] c : cells) {
            minCol = Math.min(minCol, c[1]);
            maxCol = Math.max(maxCol, c[1]);
        }
        int width = maxCol - minCol + 1;
        return (Board.WIDTH - width) / 2 - minCol;
    }

    /** 经典计分规则:同时消除的行数越多,单次得分呈非线性增长。 */
    private static int linesScoreDelta(int lines) {
        return switch (lines) {
            case 1 -> 100;
            case 2 -> 300;
            case 3 -> 500;
            case 4 -> 800;
            default -> 0;
        };
    }

    public Board board() { return board; }
    public Tetromino current() { return current; }
    public Tetromino next() { return next; }
    public long score() { return score; }
    public int linesCleared() { return linesCleared; }
    public int turn() { return turn; }
    public boolean isGameOver() { return gameOver; }
}
