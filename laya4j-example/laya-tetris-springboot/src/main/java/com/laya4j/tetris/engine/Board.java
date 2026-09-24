package com.laya4j.tetris.engine;

import java.util.ArrayList;
import java.util.List;

/**
 * 俄罗斯方块的棋盘:固定宽高的格子网格,负责碰撞检测、落子固化、
 * 消行,以及供决策使用的一组盘面度量指标(高度、孔洞、凹凸度)。
 *
 * <p>坐标体系:{@code grid[row][col]},row=0 是最顶行,row 越大越靠下;
 * col=0 是最左列。{@code true} 表示该格子已被占据。
 */
public final class Board {

    /** 标准俄罗斯方块棋盘宽度。 */
    public static final int WIDTH = 10;
    /** 标准俄罗斯方块棋盘高度。 */
    public static final int HEIGHT = 20;

    private final boolean[][] grid;

    public Board() {
        this.grid = new boolean[HEIGHT][WIDTH];
    }

    /** 深拷贝构造,用于"模拟落子后的盘面"而不影响真实盘面。 */
    private Board(boolean[][] grid) {
        this.grid = grid;
    }

    /** @return 当前盘面的深拷贝 */
    public Board copy() {
        boolean[][] g = new boolean[HEIGHT][WIDTH];
        for (int r = 0; r < HEIGHT; r++) {
            g[r] = grid[r].clone();
        }
        return new Board(g);
    }

    /** @return 指定格子是否已被占据(越界视为占据,便于碰撞检测统一处理) */
    public boolean occupied(int row, int col) {
        if (row < 0 || row >= HEIGHT || col < 0 || col >= WIDTH) {
            return true;
        }
        return grid[row][col];
    }

    /**
     * 判断某个方块在给定 (originRow, originCol, rotation) 下是否与已有格子
     * 重叠或超出边界。
     *
     * @return {@code true} 表示该摆放位置不合法(碰撞)
     */
    public boolean collides(Tetromino piece, int rotation, int originRow, int originCol) {
        for (int[] c : piece.cells(rotation)) {
            int r = originRow + c[0];
            int col = originCol + c[1];
            if (col < 0 || col >= WIDTH || r >= HEIGHT) {
                return true;
            }
            if (r >= 0 && grid[r][col]) {
                return true;
            }
        }
        return false;
    }

    /**
     * 从给定的 (originRow, originCol, rotation) 位置开始,持续下移直到即将碰撞,
     * 模拟"硬降"(hard drop)后方块最终静止的位置。
     *
     * @return 硬降后方块最终的 originRow
     */
    public int hardDropRow(Tetromino piece, int rotation, int originCol, int startRow) {
        int row = startRow;
        while (!collides(piece, rotation, row + 1, originCol)) {
            row++;
        }
        return row;
    }

    /**
     * 把方块固化到棋盘上(原地修改),不做消行。
     *
     * @throws IllegalStateException 若目标位置与现有格子冲突(调用前应先用
     *         {@link #collides} 校验)
     */
    public void lock(Tetromino piece, int rotation, int originRow, int originCol) {
        for (int[] c : piece.cells(rotation)) {
            int r = originRow + c[0];
            int col = originCol + c[1];
            if (r < 0) {
                continue; // 方块顶部溢出棋盘外,视为该格子不落在可见区域
            }
            if (grid[r][col]) {
                throw new IllegalStateException("lock() 目标格子已被占据: (" + r + "," + col + ")");
            }
            grid[r][col] = true;
        }
    }

    /**
     * 消除所有已被完全填满的行,上方行整体下移填补空缺(原地修改)。
     *
     * @return 本次消除的行数
     */
    public int clearFullLines() {
        int cleared = 0;
        for (int r = HEIGHT - 1; r >= 0; r--) {
            if (isRowFull(r)) {
                removeRow(r);
                cleared++;
                r++; // 上方行下移后,同一 index 需要再检查一次
            }
        }
        return cleared;
    }

    private boolean isRowFull(int row) {
        for (int c = 0; c < WIDTH; c++) {
            if (!grid[row][c]) {
                return false;
            }
        }
        return true;
    }

    private void removeRow(int row) {
        for (int r = row; r > 0; r--) {
            grid[r] = grid[r - 1].clone();
        }
        grid[0] = new boolean[WIDTH];
    }

    /** @return 每列最上方被占据格子的高度(从底部数,空列为 0) */
    public int[] columnHeights() {
        int[] heights = new int[WIDTH];
        for (int c = 0; c < WIDTH; c++) {
            int h = 0;
            for (int r = 0; r < HEIGHT; r++) {
                if (grid[r][c]) {
                    h = HEIGHT - r;
                    break;
                }
            }
            heights[c] = h;
        }
        return heights;
    }

    /** @return 所有列高度之和,反映堆叠的整体高度 */
    public int aggregateHeight() {
        int sum = 0;
        for (int h : columnHeights()) {
            sum += h;
        }
        return sum;
    }

    /** @return 最高列的高度(用于判断是否接近棋盘顶部) */
    public int maxHeight() {
        int max = 0;
        for (int h : columnHeights()) {
            max = Math.max(max, h);
        }
        return max;
    }

    /**
     * @return 孔洞数:某列某格上方有方块占据、但该格自身为空的格子总数
     *         (这类空格无法通过正常下落再填补,是评估盘面质量的核心负向指标)
     */
    public int holeCount() {
        int holes = 0;
        for (int c = 0; c < WIDTH; c++) {
            boolean seenBlock = false;
            for (int r = 0; r < HEIGHT; r++) {
                if (grid[r][c]) {
                    seenBlock = true;
                } else if (seenBlock) {
                    holes++;
                }
            }
        }
        return holes;
    }

    /** @return 凹凸度(bumpiness):相邻列高度差绝对值之和,越小代表表面越平整 */
    public int bumpiness() {
        int[] heights = columnHeights();
        int sum = 0;
        for (int c = 0; c < WIDTH - 1; c++) {
            sum += Math.abs(heights[c] - heights[c + 1]);
        }
        return sum;
    }

    /** @return 棋盘宽度(列数) */
    public int width() {
        return WIDTH;
    }

    /** @return 棋盘高度(行数) */
    public int height() {
        return HEIGHT;
    }

    /** @return 指定格子是否被占据(不做越界兜底,供渲染遍历使用) */
    public boolean cellAt(int row, int col) {
        return grid[row][col];
    }

    /**
     * 返回所有已被占据格子的 (row, col) 坐标列表,按行优先顺序排列。
     *
     * <p>供 REST API 序列化(渲染棋盘)以及"下落动画"场景使用——
     * 动画需要"落子前"的静态盘面快照,这里返回的是调用时刻的即时状态,
     * 调用方应在方块固化(lock)之前调用以捕获"落子前"的盘面。
     *
     * @return 已占据格子坐标列表(每个元素是 {@code {row, col}})
     */
    public List<int[]> filledCells() {
        List<int[]> cells = new ArrayList<>();
        for (int r = 0; r < HEIGHT; r++) {
            for (int c = 0; c < WIDTH; c++) {
                if (grid[r][c]) {
                    cells.add(new int[]{r, c});
                }
            }
        }
        return cells;
    }
}
