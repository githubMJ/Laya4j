package com.laya4j.snake.engine;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * 贪吃蛇棋盘与规则:固定 size×size 网格,蛇撞墙或撞自己即死亡。
 * 纯游戏逻辑,不依赖 Laya4j,便于单独测试。
 */
public final class SnakeGame {

    /** 移动方向。 */
    public enum Direction {
        UP, DOWN, LEFT, RIGHT;

        /** 对头方向(蛇不能直接掉头)。 */
        public Direction opposite() {
            return switch (this) {
                case UP -> DOWN;
                case DOWN -> UP;
                case LEFT -> RIGHT;
                case RIGHT -> LEFT;
            };
        }

        public String lower() {
            return name().toLowerCase();
        }
    }

    /** 棋盘坐标,x 向右、y 向下。 */
    public record Cell(int x, int y) {
        public Cell plus(Direction d) {
            return switch (d) {
                case UP -> new Cell(x, y - 1);
                case DOWN -> new Cell(x, y + 1);
                case LEFT -> new Cell(x - 1, y);
                case RIGHT -> new Cell(x + 1, y);
            };
        }

        /** 供前端 JSON 序列化的 [x, y]。 */
        public int[] toArray() {
            return new int[]{x, y};
        }
    }

    private final int size;
    private final RandomGenerator rnd;

    private final Deque<Cell> body = new ArrayDeque<>();
    private Direction heading = Direction.RIGHT;
    private Cell food;
    private boolean alive = true;
    private int steps;

    public SnakeGame(int size, long seed) {
        if (size < 5) {
            throw new IllegalArgumentException("size must be >= 5");
        }
        this.size = size;
        this.rnd = new java.util.Random(seed);
        int mid = size / 2;
        // 初始 3 节蛇,位于中部,向右
        body.addFirst(new Cell(mid, mid));
        body.addLast(new Cell(mid - 1, mid));
        body.addLast(new Cell(mid - 2, mid));
        placeFood();
    }

    private void placeFood() {
        List<Cell> free = new ArrayList<>();
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                Cell c = new Cell(x, y);
                if (!body.contains(c)) {
                    free.add(c);
                }
            }
        }
        food = free.isEmpty() ? null : free.get(rnd.nextInt(free.size()));
    }

    /** 该方向是否致命(出界或撞身体;不吃食物时尾巴即将移走,不算碰撞)。 */
    public boolean isFatal(Direction d) {
        Cell head = body.peekFirst();
        Cell next = head.plus(d);
        if (next.x() < 0 || next.y() < 0 || next.x() >= size || next.y() >= size) {
            return true;
        }
        Deque<Cell> checked = new ArrayDeque<>(body);
        if (!next.equals(food)) {
            checked.pollLast();
        }
        return checked.contains(next);
    }

    /**
     * 朝指定方向走一步。返回 true 表示吃到食物。
     *
     * @throws IllegalStateException 蛇已死亡
     */
    public boolean step(Direction d) {
        if (!alive) {
            throw new IllegalStateException("game over");
        }
        if (d == heading.opposite()) {
            throw new IllegalArgumentException("cannot reverse: " + d);
        }
        if (isFatal(d)) {
            alive = false;
            return false;
        }
        heading = d;
        Cell head = body.peekFirst().plus(d);
        body.addFirst(head);
        steps++;
        if (head.equals(food)) {
            placeFood();
            return true;
        }
        body.pollLast();
        return false;
    }

    /** 蛇身坐标,头在前,供前端 JSON 序列化。 */
    public List<int[]> bodyCells() {
        return body.stream().map(Cell::toArray).toList();
    }

    public int size() { return size; }
    public Deque<Cell> body() { return new ArrayDeque<>(body); }
    public Cell head() { return body.peekFirst(); }
    public Cell food() { return food; }
    public Direction heading() { return heading; }
    public boolean alive() { return alive; }
    public int steps() { return steps; }
    public int score() { return body.size() - 3; }
}
