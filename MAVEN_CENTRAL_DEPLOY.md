# 部署 Laya4j 到 Maven Central 指南

把 `com.laya4j:laya4j` 发布到 **Maven Central**,让全世界通过 `pom.xml` 直接依赖。

> **重要**:Laya4j 的 **ONNX 模型(1.2 GB)不会发布到 Maven Central**(Maven Central 单文件限制 100 MB,而且 jar 包超过几 MB 就极不友好)。
> 模型仍通过 HuggingFace Hub 或项目内 `models/` 目录分发,详见 `HuggingFaceFetcher` 的加载优先级。

---

## 前置条件(一次性)

### 1. 注册 Sonatype OSSRH 账号

1. 访问 https://issues.sonatype.org/ 用 GitHub 账号登录
2. 创建 issue:标题 `com.laya4j:parent` 申请 group id `com.laya4j`
3. 等 Sonatype 审核(一般几分钟到几小时)

### 2. 生成 GPG 签名 key

```bash
# 如果没装 gpg:brew install gnupg(macOS) / apt install gnupg(Linux)
gpg --version

# 生成 key(用真实邮箱替换)
gpg --gen-key
# 提示输入:
#   Real name: Your Name
#   Email: your.email@example.com
#   Passphrase: <一个强密码>

# 列出 public key
gpg --list-keys

# 输出 public key 到 Sonatype OSSRH:
# https://issues.sonatype.org/secure/ViewProfile.jspa -> "OSSRH" -> "PGP Public Keys"
# 上传 .asc 文件
```

### 3. 配置 `~/.m2/settings.xml`(认证信息)

```xml
<settings>
  <servers>
    <server>
      <id>central</id>
      <username>your_sonatype_username</username>
      <password>your_sonatype_password</password>
    </server>
  </servers>

  <profiles>
    <profile>
      <id>gpg</id>
      <properties>
        <gpg.keyname>your.email@example.com</gpg.keyname>
        <!-- passphrase 可选,本地 gpg-agent 会缓存 -->
        <gpg.passphrase>your_gpg_passphrase</gpg.passphrase>
      </properties>
    </profile>
  </profiles>
</settings>
```

---

## 发布流程(每次发布)

### 1. 更新版本号 + CHANGELOG

```bash
# 1. 修改 pom.xml 的 <version>0.1.0</version> → <version>0.2.0</version>
# 2. 在 CHANGELOG.md 写新版本的改动
# 3. git commit + tag
git add pom.xml CHANGELOG.md
git commit -m "release: v0.2.0"
git tag v0.2.0
```

### 2. 发布到 Maven Central

```bash
# 切换到 release profile(启用 source/javadoc/gpg 签名 + central-publishing 插件)
mvn clean deploy -P release
```

**执行步骤**:
1. `clean` - 清理 target/
2. `compile` + `test` - 跑全部测试(57/57 必须全过)
3. `package` - 打 jar + source jar + javadoc jar
4. `verify` - GPG 签名所有 jar
5. `deploy` - 上传到 Sonatype OSSRH staging repository
6. `central-publishing-maven-plugin` 自动 publish 到 Maven Central

### 3. 验证

等待 10-30 分钟(首次发布可能要 24h 等待审核),然后:

```bash
# 检查 Maven Central
curl -s "https://repo1.maven.org/maven2/com/laya4j/laya4j/maven-metadata.xml"
# 应返回最新的 <latest>0.2.0</latest>

# 在新项目里测试依赖
mvn dependency:get -Dartifact=com.laya4j:laya4j:0.2.0
```

---

## 版本号约定(语义化版本)

- **MAJOR**: 破坏 API 兼容(必须改用户代码)
- **MINOR**: 新功能(向后兼容)
- **PATCH**: bug fix

历史:
- 0.1.0 - 初始发布

---

## CI/CD 自动化(可选)

在 GitHub Actions 配置自动发布:

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags: ['v*']

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up JDK 17
        uses: actions/setup-java@v4
        with:
          distribution: 'temurin'
          java-version: '17'

      - name: Import GPG key
        uses: crazy-max/ghaction-import-gpg@v6
        with:
          gpg_private_key: ${{ secrets.GPG_PRIVATE_KEY }}
          passphrase: ${{ secrets.GPG_PASSPHRASE }}

      - name: Publish to Maven Central
        env:
          SONATYPE_USERNAME: ${{ secrets.SONATYPE_USERNAME }}
          SONATYPE_PASSWORD: ${{ secrets.SONATYPE_PASSWORD }}
        run: mvn clean deploy -P release
```

需要的 Secrets:
- `GPG_PRIVATE_KEY`:`gpg --export-secret-keys --armor your.email@example.com` 的输出
- `GPG_PASSPHRASE`:GPG key 的密码
- `SONATYPE_USERNAME` / `SONATYPE_PASSWORD`:Sonatype OSSRH 账号

---

## 发布后

发布成功后,用户可以在自己的项目里这样用:

```xml
<dependency>
    <groupId>com.laya4j</groupId>
    <artifactId>laya4j</artifactId>
    <version>0.2.0</version>
</dependency>
```

ONNX 模型仍需要单独获取(从 HuggingFace 或 GitHub Release),Java 代码会通过 `HuggingFaceFetcher` 自动加载。

---

## 故障排查

### "401 Unauthorized"
- 检查 `~/.m2/settings.xml` 的用户名密码
- Sonatype 的 token 跟普通账号密码不一致,推荐用 `slt-token` 机制(参见 Sonatype 文档)

### "GPG signing failed"
- 确保 `gpg-agent` 在跑(`echo "test" | gpg --clearsign` 测试)
- `passphrase` 在 settings.xml 里配对

### "401 - Project not found"
- Sonatype 的 issue 还没审核通过,先等

### "Deploy to Central failed: 400 Bad Request"
- POM 里缺 `<name>`, `<url>`, `<licenses>`, `<developers>`, `<scm>`
- 已经全部加上,见 `pom.xml`

---

## 参考

- [Sonatype Publishing Guide](https://central.sonatype.com/publish)
- [Maven Central Requirements](https://central.sonatype.com/pages/requirements)
- [GPG 签名 for Maven](https://docs.sonatype.org/en/general/concepts/use-maven-gpg-signed-components.html)
- [central-publishing-maven-plugin 文档](https://central.sonatype.com/publish/publish-with-maven)