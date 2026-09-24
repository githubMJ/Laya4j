# laya-snake

用 Laya(`com.laya4j:laya4j-core`)驱动贪吃蛇的 **Web 动画示例**:
每步 1 个 `choice`(下一步方向)+ 4 个 `noul`(该方向是否致命),
演示 **"模型决策优先,规则兜底保安全"** 的混合决策模式,并在浏览器里
实时可视化每一步的决策轨迹(方向概率条、致命风险、裁决方式)。

## 运行

```bash
# 默认启发式 mock 模型(无需 1.2GB 权重,秒级启动)
mvn -pl laya4j-example/laya-snake spring-boot:run

# 真实 Laya 权重(从项目 models/ 目录或 HF Hub 自动解析)
mvn -pl laya4j-example/laya-snake spring-boot:run -Dspring-boot.run.jvmArguments="-Dlaya.snake.model=onnx"
```

启动后访问 **http://localhost:8083**:
- **棋盘动画**:蛇头高亮、蛇身渐变、食物脉冲;模型选择方向蓝色描边、
  致命方向红色虚线框实时叠加在棋盘上
- **决策面板**:四个方向的概率条(CHOICE)、安全/致命芯片(NOUL P(true))、
  裁决徽章(模型决策 / 规则兜底 / 撞死)与推理耗时
- **控制**:新开一局(可指定 seed)、单步、自动播放 + 节奏滑条

## 每步的决策请求

- `choice next_direction`:四个方向及其落点描述,让模型选方向
- `noul fatal_up / fatal_down / fatal_left / fatal_right`:问模型
  "往该方向走是否立即死亡"(撞墙或撞身体)

## 决策消费策略

1. 优先采纳 CHOICE 选出的方向
2. 若该方向 NOUL P(true) ≥ 0.5(致命),改用规则策略挑选安全方向兜底
3. 若无任何安全方向,按"最不坏"方向走,蛇撞死,游戏结束

## 代码分层(与 laya-tetris-springboot 同构)

```
com.laya4j.snake
├── engine/    SnakeGame            # 纯游戏逻辑,不依赖 Laya4j
├── laya/      SnakeStateRenderer   # 局面 → state JSON
│              SnakeQuestions       # 1 choice + 4 noul
│              HeuristicSnakeModel  # mock 后端(只读 state,不偷看引擎)
│              SnakeDecisionService # 编排:问 Laya → 消费决策 → 走一步
│              DecisionTrace        # 决策轨迹(供前端序列化)
├── config/    ModelConfig          # mock / onnx 切换
├── web/       SnakeController / SnakeGameService  # REST API
└── resources/static/index.html                      # 动画页面
```
