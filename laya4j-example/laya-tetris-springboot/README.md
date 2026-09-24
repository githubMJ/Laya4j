# laya-tetris-springboot

用 Laya(`com.laya4j:laya4j-core`)的 `choice` / `score` / `noul` 三种决策原语
驱动俄罗斯方块落子决策的 Spring Boot 示例服务,带实时 Web 决策可视化。
本模块**不属于** Laya4j 发布库,是一个独立可运行的集成示例工程。

## 决策设计

每个下落方块只做**一次落子决策**(硬降到底,不模拟逐帧移动),流程:

1. `TetrisEngine.candidatePlacements()` 枚举当前方块所有合法的
   (旋转, 列)组合,计算每个候选落子(含消行)后的高度/孔洞/凹凸度指标
2. 按经典启发式权重(高度惩罚、消行奖励、孔洞惩罚、凹凸度惩罚)取
   **前 5 名**作为简选列表(shortlist),避免单次 forward 的候选数量爆炸
3. 构造 **1 次 Laya forward** 涵盖的问题集合:
   - `CHOICE placement`:从 shortlist 里选一个最佳候选
   - `NOUL risk_c{i}`(每个候选一个):该候选是否会让堆叠"接近顶部、
     随时可能 game over"
   - `SCORE board_health`:给"落子前"的当前盘面整体健康度打分
     (`critical`/`risky`/`stable`/`strong`,纯展示用途)
4. **决策消费**(与贪吃蛇示例一致的"模型选择优先,安全兜底纠错"模式):
   - 模型选中的候选若被对应的 `risk_c{i}` 判定为危险,则改用
     shortlist 中风险最低、启发式得分最高的候选兜底
   - 兜底逻辑保证了"即使模型判断失误,也不会因为单次误判直接崩盘"

## 运行

```bash
# 1. 先在 Laya4j 项目根目录安装库到本地仓库
cd ../.. && mvn install -DskipTests

# 2. 启动本服务(默认启发式 mock 模型,秒级启动,无需 1.2GB 权重)
cd examples/laya-tetris-springboot
mvn spring-boot:run
# 打开 http://localhost:8082

# 3. 切换真实 Laya 权重(自动从项目 models/ 或 HuggingFace Hub 解析)
mvn spring-boot:run \
    -Dspring-boot.run.jvmArguments="-Dlaya.tetris.model=onnx -Dlaya4j.models.dir=$(pwd)/../../models"
```

## REST API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/state` | 查询当前局面(不推进游戏) |
| POST | `/api/new?seed=42` | 开一局新游戏 |
| POST | `/api/step` | 对当前方块推进一次完整的 Laya 决策 + 落子 |

## 目录结构

```
src/main/java/com/laya4j/tetris/
├── TetrisServerApplication.java
├── engine/            # 纯游戏逻辑,不依赖 Laya/Spring
│   ├── Tetromino.java     # 7 种方块及旋转形态
│   ├── Board.java         # 棋盘:碰撞/固化/消行/盘面指标
│   ├── Placement.java     # 一个候选落点 + 落子后指标
│   └── TetrisEngine.java  # 游戏状态 + 候选枚举 + 应用落子
├── laya/              # Laya 决策集成,只通过 LayaModel 接口交互
│   ├── TetrisStateRenderer.java   # 局面 → state 文本
│   ├── TetrisQuestions.java       # 构造 choice/noul/score 问题
│   ├── HeuristicLayaModel.java    # mock LayaModel(启发式,测试用)
│   ├── TetrisDecisionService.java # 编排:候选→简选→问 Laya→消费→落子
│   └── DecisionTrace.java         # 决策轨迹(供 API/UI 展示)
├── config/ModelConfig.java        # LayaModel Bean 装配(mock/onnx)
└── web/                           # REST + 会话状态
    ├── TetrisGameService.java
    └── TetrisController.java
```
