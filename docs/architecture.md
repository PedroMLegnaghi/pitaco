# Architecture

## Pipeline

Every box is one CLI command (`pitaco <stage>`), chained by the Makefile.

```mermaid
flowchart TD
    A["Data prep<br/>pitaco data<br/><i>clean · scrub PII · dedup</i>"] --> B["Gold set · 300<br/>pitaco annotate<br/><i>hand-labeled, blind</i>"]
    A --> C["Silver set · 4,000<br/>pitaco label<br/><i>teacher LLM via OpenRouter</i>"]
    C --> D["Training data<br/>pitaco build-training-data<br/><i>valid + grounded only</i>"]
    D --> E["Baseline A<br/>pitaco train-bert<br/><i>BERTimbau, laptop GPU</i>"]
    D --> F["Student<br/>pitaco train-student<br/><i>Qwen3.5-4B + LoRA, Kaggle T4</i>"]
    B -->|test set only| G["Evaluation<br/>pitaco evaluate · report"]
    C -->|teacher as Baseline B| G
    E --> G
    F --> G
    G --> H["Publish<br/><i>HF model · dataset · demo</i>"]
```

## Splits and the leakage guard

```mermaid
flowchart TD
    U["Unique reviews · 32,490<br/><i>cleaned, PII-free, deduped</i>"] --> G["Gold · 300<br/><i>10+50 per star rating</i>"]
    U --> S["Silver pool<br/><i>36 near-copies of gold removed</i>"]
    G --> DEV["Dev · 50<br/>prompt + guideline tuning"]
    G --> TEST["Test · 250<br/>final evaluation only, logged"]
    S --> TR["Train · 3,600<br/>fit weights"]
    S --> VAL["Val · 400<br/>early stopping, thresholds"]
```

Enforced in code: `assert_no_leakage` (no shared id, no identical normalized text, no fuzzy ratio ≥ 90
between gold and silver), few-shot examples must come from silver (`check_fewshot_not_in_gold`), and
every evaluation on gold test is appended to `results/test_access.log`.

## Labeling loop (teacher)

```mermaid
flowchart LR
    R[review] --> P["prompt<br/>instructions + 12 examples"] --> K{"cache hit?"}
    K -->|yes| O[cached response]
    K -->|no| B{"spent + reserved +<br/>worst case ≤ cap?"}
    B -->|no| X[BudgetExceeded]
    B -->|yes| API["OpenRouter<br/>json_schema · reasoning off"] --> L[append to cache = spend ledger] --> O
    O --> V["parse + validate schema<br/>+ ground evidence"] --> W[PredictionRecord]
```

## One prediction format

Teacher, Claude, BERTimbau and student all write `results/predictions/<system>/<split>.jsonl`
(`PredictionRecord`: parsed analysis or label-only mentions, validity, grounding, latency, cost),
so `pitaco evaluate` scores every system the same way and `pitaco report` builds one table.
