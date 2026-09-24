# laya4j-example

Laya4j 示例工程父模块(聚合器)。各子项目为**独立可运行**的示例,
不属于 Laya4j 发布库,不参与主库构建/发布。

## 子项目

| 子项目 | 说明 | 运行方式 |
|---|---|---|
| [`laya-tetris-springboot`](laya-tetris-springboot/) | Spring Boot 服务:用 Laya 的 `choice`/`score`/`noul` 三种决策原语驱动俄罗斯方块落子,REST + Web 页面实时展示决策轨迹 | `mvn -pl laya4j-example/laya-tetris-springboot spring-boot:run` |
| [`laya-snake`](laya-snake/) | 命令行贪吃蛇:每步 1 个 `choice`(下一步方向)+ 4 个 `noul`(该方向是否致命),模型决策优先、规则兜底 | `mvn -pl laya4j-example/laya-snake exec:java -Dexec.args="--mock"` |

## 构建

在仓库根目录执行 `mvn install` 即可随聚合构建一起编译;
单独构建某个子项目需先在根目录 `mvn install` 安装 `com.laya4j:laya4j-core`。

> 除 `laya-snake --mock` 外,其余示例运行时需要 Laya 模型权重
> (从项目 `models/` 目录或 HuggingFace Hub 自动解析,详见 `models/VERSION.md`)。

## 新增示例

在 `laya4j-example/` 下新建子目录 + `pom.xml`,并在本模块 `pom.xml` 的
`<modules>` 中登记即可。注意:子项目若使用自己的 parent(如
`spring-boot-starter-parent`),属性不会从本模块继承,版本需自行维护。
