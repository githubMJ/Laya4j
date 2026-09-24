package com.laya4j.tetris.engine;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * 俄罗斯方块的 7 种标准方块(Tetromino)及其旋转形态。
 *
 * <p>每种方块用 4 个旋转状态描述,每个状态是一组 (row, col) 相对坐标,
 * 限定在 4×4 的包围盒内(经典 SRS 旋转系统的简化版:本示例只关心
 * "落子决策"本身,因此不实现踢墙(wall kick)等精细规则,仅保留
 * 每种方块在原地旋转后的形状集合,足以支撑"选择旋转 + 列"的决策枚举)。
 *
 * <p>坐标体系:row 越大越靠下,col 越大越靠右,与 {@link Board} 的
 * 网格坐标系一致。
 */
public enum Tetromino {

    /** I 形:四格连成一条直线。 */
    I(new int[][][]{
            {{1, 0}, {1, 1}, {1, 2}, {1, 3}},
            {{0, 2}, {1, 2}, {2, 2}, {3, 2}},
            {{2, 0}, {2, 1}, {2, 2}, {2, 3}},
            {{0, 1}, {1, 1}, {2, 1}, {3, 1}},
    }),

    /** O 形:2×2 正方形,旋转不变。 */
    O(new int[][][]{
            {{0, 0}, {0, 1}, {1, 0}, {1, 1}},
            {{0, 0}, {0, 1}, {1, 0}, {1, 1}},
            {{0, 0}, {0, 1}, {1, 0}, {1, 1}},
            {{0, 0}, {0, 1}, {1, 0}, {1, 1}},
    }),

    /** T 形。 */
    T(new int[][][]{
            {{0, 1}, {1, 0}, {1, 1}, {1, 2}},
            {{0, 1}, {1, 1}, {1, 2}, {2, 1}},
            {{1, 0}, {1, 1}, {1, 2}, {2, 1}},
            {{0, 1}, {1, 0}, {1, 1}, {2, 1}},
    }),

    /** S 形。 */
    S(new int[][][]{
            {{0, 1}, {0, 2}, {1, 0}, {1, 1}},
            {{0, 1}, {1, 1}, {1, 2}, {2, 2}},
            {{0, 1}, {0, 2}, {1, 0}, {1, 1}},
            {{0, 1}, {1, 1}, {1, 2}, {2, 2}},
    }),

    /** Z 形。 */
    Z(new int[][][]{
            {{0, 0}, {0, 1}, {1, 1}, {1, 2}},
            {{0, 2}, {1, 1}, {1, 2}, {2, 1}},
            {{0, 0}, {0, 1}, {1, 1}, {1, 2}},
            {{0, 2}, {1, 1}, {1, 2}, {2, 1}},
    }),

    /** J 形。 */
    J(new int[][][]{
            {{0, 0}, {1, 0}, {1, 1}, {1, 2}},
            {{0, 1}, {0, 2}, {1, 1}, {2, 1}},
            {{1, 0}, {1, 1}, {1, 2}, {2, 2}},
            {{0, 1}, {1, 1}, {2, 0}, {2, 1}},
    }),

    /** L 形。 */
    L(new int[][][]{
            {{0, 2}, {1, 0}, {1, 1}, {1, 2}},
            {{0, 1}, {1, 1}, {2, 1}, {2, 2}},
            {{1, 0}, {1, 1}, {1, 2}, {2, 0}},
            {{0, 0}, {0, 1}, {1, 1}, {2, 1}},
    });

    /** 4 个旋转状态,每个状态是 4 个 {row, col} 坐标对。 */
    private final int[][][] rotations;

    Tetromino(int[][][] rotations) {
        this.rotations = rotations;
    }

    /**
     * 返回指定旋转状态(0..3,自动取模)的相对坐标集合。
     *
     * @param rotation 旋转状态索引,允许任意整数(内部取模到 0..3)
     * @return 该旋转状态下 4 个格子的 {row, col} 坐标数组
     */
    public int[][] cells(int rotation) {
        return rotations[Math.floorMod(rotation, rotations.length)];
    }

    /**
     * 返回指定旋转状态下,方块包围盒的宽度(列数)。
     *
     * <p>用于把候选落点渲染为"columns X-Y"这样的可读文本
     * (参见 {@link Placement#describe(int)})。
     *
     * @param rotation 旋转状态索引
     * @return 包围盒宽度(≥1)
     */
    public int width(int rotation) {
        int minCol = Integer.MAX_VALUE, maxCol = Integer.MIN_VALUE;
        for (int[] c : cells(rotation)) {
            minCol = Math.min(minCol, c[1]);
            maxCol = Math.max(maxCol, c[1]);
        }
        return maxCol - minCol + 1;
    }

    /**
     * 返回该方块在形状上互不相同的旋转状态数量。
     *
     * <p>例如 O 形任意旋转都一样,只有 1 种去重后的形态;S/Z 只有 2 种;
     * I/T/J/L 有 4 种。枚举候选落点时只需遍历去重后的旋转状态,
     * 避免生成大量重复的候选摆放。
     *
     * @return 去重后的旋转状态数量(1、2 或 4)
     */
    public int distinctRotationCount() {
        Set<String> seen = new LinkedHashSet<>();
        for (int[][] shape : rotations) {
            seen.add(shapeKey(shape));
        }
        return seen.size();
    }

    private static String shapeKey(int[][] shape) {
        StringBuilder sb = new StringBuilder();
        for (int[] c : shape) {
            sb.append(c[0]).append(',').append(c[1]).append(';');
        }
        return sb.toString();
    }
}
