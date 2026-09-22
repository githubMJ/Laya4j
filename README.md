# Laya4j

Java SDK for [Laya](https://github.com/NandhaKishorM/laya) System 1 decision models —
fast, non-autoregressive, calibrated decisions in a single forward pass via ONNX Runtime.

```java
LayaPredictor p = LayaPredictor.single(
    Path.of("models/laya-decision-multilingual-mmbert-base-v0.3.4-1c5edc1.onnx"),
    Path.of("models/tokenizer"),
    "multilingual");

Map<String, Decision> r = p.predict(state, Presets.triage());
r.forEach((k, v) -> System.out.println(k + ": " + v));
```

---

## Install

```xml
<dependency>
    <groupId>com.laya4j</groupId>
    <artifactId>laya4j</artifactId>
    <version>0.2.0</version>
</dependency>
```

Requires **Java 17+**. Maven Central publishing guide: [MAVEN_CENTRAL_DEPLOY.md](MAVEN_CENTRAL_DEPLOY.md).

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/laya4j/Laya4j.git
cd Laya4j

# 2. Build & test
mvn test

# 3. Run end-to-end demo (5 demos, includes Python alignment check)
mvn -Pdemo test-compile exec:java

# 4. Snake game AI demo (see below)
mvn -Pdemo test-compile exec:java -Dexec.mainClass=com.laya4j.examples.SnakeDemo
```

The demo includes a Python-alignment check (4 fixed cases, expected diff = 0.0000).

---

## How It Works

```
Java app
   │
   ▼  LayaPredictor.predict(state, questions)
┌─────────────────────────────────────────┐
│ ScriptDetector  ─►  LayaRouter         │
│   (detect script)    (pick checkpoint) │
│                      english / multiling │
│                      typed-decisions      │
└─────────────────────────────────────────┘
   │
   ▼  LayaOnnxModel.predict
   build_sequence  (SequenceBuilder)
        │  same byte-ids as Python
        ▼
   ONNX Runtime Java  ──►  decision logits
        │
        ▼  softmax + temperature
   Decision (ChoiceDecision | ScoreDecision | NoulDecision)
```

**Three primitives**: `choice` (multi-class), `score` (ordinal), `noul` (yes/no probability).

---

## Usage

### Built-in presets (mirror Python `laya.presets`)

```java
p.predict(state, Presets.triage());        // 5 questions: intent, is_urgent, frustration, refund_requested, churn_risk
p.predict(state, Presets.guard());         // jailbreak, prompt_injection, sensitive_data, harm_severity, topic
p.predict(state, Presets.moderation());    // toxic, harassment, threat, spam, severity
p.predict(state, Presets.modelRouter());   // difficulty, domain, needs_tools, is_sensitive
```

### Custom questions

```java
Question refund = Question.noul("refund",
    "User asks for refund?", "no refund", "yes refund").build();

Question urgency = Question.score("urgency",
    "How urgent?", "low", "medium", "critical").build();

Map<String, Decision> r = p.predict(state, List.of(refund, urgency));
```

### Routing check (no inference)

```java
RoutingDecision rd = p.route(state);
System.out.println(rd.model() + " / " + rd.profile().language());
```

### Custom inference backend (`LayaModel` interface)

`LayaPredictor` / `LayaRouter` are programmed against the `LayaModel` interface
(default impl: `LayaOnnxModel`). Plug in a remote service, quantized runtime, or test stub:

```java
LayaPredictor p = LayaPredictor.builder().multilingual(myLayaModel).build();
```

### Snake game AI demo

`examples/SnakeDemo` drives a snake with one forward pass per step:
1 `choice` (next direction) + 4 `noul` (is each direction fatal?), consumed as
"model choice first, game-rule safety fallback second":

```bash
# real Laya model (auto-resolved from models/ or HF Hub)
mvn -Pdemo test-compile exec:java -Dexec.mainClass=com.laya4j.examples.SnakeDemo

# heuristic mock model (implements LayaModel, no 1.2 GB weights needed)
mvn -Pdemo test-compile exec:java -Dexec.mainClass=com.laya4j.examples.SnakeDemo -Dexec.args="--mock"
```

Flags: `--steps N` `--size N` `--seed S` `--delay MS`.

---

## Model

`models/laya-decision-multilingual-mmbert-base-v0.3.4-1c5edc1.onnx` (1.2 GB FP32).

| Field | Value |
|---|---|
| Laya Python version | **0.3.4** |
| HF commit | `1c5edc17...` |
| Encoder | mmBERT-base (322M, vocab 256k, RoPE 8192) |
| Format | FP32, full-precision |

Full info: [`models/VERSION.md`](models/VERSION.md).

**Loading priority** (handled by `HuggingFaceFetcher`):
1. `models/` in the project (zero network)
2. `~/.cache/laya/` (local cache)
3. HuggingFace Hub (auto-download, no token required)

For fine-tuning on your domain: see [Fine-Tuning](#fine-tuning) below.

---

## Performance (Mac M4 / CPU)

| Metric | Value |
|---|---|
| Model load | 2.7 s |
| Single predict (5 questions) p50 | 148 ms |
| Throughput | 6.3 QPS |

For higher throughput use NVIDIA GPU + `onnxruntime-gpu` or run multiple JVM instances.

---

## Tests

```bash
mvn test                                      # unit tests
mvn -Pdemo test-compile exec:java             # 5 end-to-end demos (includes Python alignment check)
```

**Python alignment** (Test 5 of demo, 4 fixed cases):
```
java=0.9915 py=0.9915 diff=0.0000  我要求立即退款
java=0.9142 py=0.9142 diff=0.0000  Please cancel my subscription
java=0.0582 py=0.0582 diff=0.0000  服务挂了
java=0.0158 py=0.0158 diff=0.0000  Where is my order #1234
→ 4/4 bit-equal with Python
```

## Fine-Tuning

For domain-specific deployments, fine-tune on your own labeled data using the [Laya Kaggle notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb) (RLCD algorithm, ~4-5h on 2xT4). Re-export the ONNX to `models/` and update `HuggingFaceFetcher.MODEL_VERSION`.

---

## Publish to Maven Central

```bash
mvn clean deploy -P release
```

```
Laya4j/
├── pom.xml                          # Java 17 + ONNX Runtime 1.30.0 + DJL 0.30 + JUnit 5
├── README.md
├── MAVEN_CENTRAL_DEPLOY.md
├── CHANGELOG.md
├── models/                          # ONNX weights (NOT in git; see MAVEN_CENTRAL_DEPLOY.md)
│   ├── laya-decision-multilingual-mmbert-base-v0.3.4-1c5edc1.onnx  (1.2 GB)
│   ├── tokenizer/                                                   (33 MB)
│   └── VERSION.md
└── src/
    ├── main/java/com/laya4j/
    │   ├── core/        Question, Decision, DecisionType, RoutingDecision, LayaConfig, exceptions
    │   ├── model/       LayaModel, LayaOnnxModel, SequenceBuilder, HuggingFaceFetcher
    │   ├── router/      ScriptDetector, LayaRouter
    │   ├── predict/     LayaPredictor, Presets
    │   └── benchmark/   Benchmark
    └── test/java/com/laya4j/        # 57 unit tests
    ```
│   └── VERSION.md
├── src/main/java/com/laya4j/        # published library
│   ├── core/        Question, Decision (sealed), DecisionType, RoutingDecision, LayaConfig, exception hierarchy
│   ├── model/       LayaModel (interface), LayaOnnxModel, SequenceBuilder, HuggingFaceFetcher
│   ├── router/      ScriptDetector, LayaRouter
│   └── predict/     LayaPredictor, Presets
└── src/test/java/com/laya4j/
    ├── (unit tests)
    └── examples/    QuickStartDemo, SnakeDemo, Benchmark   # not in the published jar
```

---

## Related

- **Python source**: [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)
- **Python ONNX model source**: [`Laya/export_onnx.py`](../Laya/export_onnx.py) (sibling project)
- **HuggingFace model**: [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)


