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
    <version>0.1.0</version>
</dependency>
```

Requires **Java 17+**. Maven Central publishing guide: [MAVEN_CENTRAL_DEPLOY.md](MAVEN_CENTRAL_DEPLOY.md).

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/your-org/Laya4j.git
cd Laya4j

# 2. Build & test (57 tests)
mvn test

# 3. Run demo (5 end-to-end tests)
mvn exec:java
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

For higher throughput use NVIDIA GPU + `onnxruntime-gpu` (5-10x faster), or run multiple JVM instances.

---

## Tests

```bash
mvn test                    # 57 unit tests
mvn exec:java               # 5 end-to-end demos (includes Python alignment check)
```

**Python alignment** (Test 5 of demo, 4 fixed cases):
```
java=0.9915 py=0.9915 diff=0.0000  我要求立即退款
java=0.9142 py=0.9142 diff=0.0000  Please cancel my subscription
java=0.0582 py=0.0582 diff=0.0000  服务挂了
java=0.0158 py=0.0158 diff=0.0000  Where is my order #1234
→ 4/4 bit-equal with Python
```

---

## Fine-Tuning

Base `laya-multilingual` is general; **production deployments should fine-tune** because:

| Task | Base accuracy |
|---|---|
| Refund detection | 100% (good) |
| Prompt injection | 100% (good) |
| Triage intent | 83% |
| Toxicity detection | 83% |
| Threat detection | **50%** (needs fine-tune) |
| Difficulty grading | **0%** (needs fine-tune) |
| Domain detection | **40%** (needs fine-tune) |

Fine-tune in Python using [Laya's Kaggle notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb) (RLCD algorithm, ~4-5h on 2xT4), then re-export and place the new ONNX in `models/`. Update `HuggingFaceFetcher.MODEL_VERSION` to point at it.

---

## Publish to Maven Central

```bash
mvn clean deploy -P release
```

First-time setup: register at [Sonatype OSSRH](https://issues.sonatype.org/), generate a GPG key, and configure `~/.m2/settings.xml`. Full guide: [MAVEN_CENTRAL_DEPLOY.md](MAVEN_CENTRAL_DEPLOY.md).

Note: the 1.2 GB ONNX model is **not** published to Maven Central (file size limit + bandwidth). Models are distributed separately via HuggingFace Hub or `models/` in the project.

---

## Project Structure

```
Laya4j/
├── pom.xml                          # Java 17 + ORT 1.30.0 + DJL 0.30 + JUnit 5
├── README.md
├── MAVEN_CENTRAL_DEPLOY.md
├── CHANGELOG.md
├── models/
│   ├── laya-decision-multilingual-mmbert-base-v0.3.4-1c5edc1.onnx  (1.2 GB)
│   ├── tokenizer/                                                   (33 MB)
│   └── VERSION.md
├── src/main/java/com/laya4j/
│   ├── core/        (9 files)   Question, Decision (sealed), DecisionType, ...
│   ├── model/       (3 files)   LayaOnnxModel, SequenceBuilder, HuggingFaceFetcher
│   ├── router/      (2 files)   ScriptDetector, LayaRouter
│   ├── predict/     (2 files)   LayaPredictor, Presets
│   └── benchmark/   (1 file)    Benchmark
└── src/test/java/com/laya4j/      (57 unit tests)
```

---

## Related

- **Python source**: [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)
- **Python ONNX model source**: [`Laya/export_onnx.py`](../Laya/export_onnx.py) (sibling project)
- **HuggingFace model**: [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)

---

## License

Apache 2.0 (matches upstream Laya).
