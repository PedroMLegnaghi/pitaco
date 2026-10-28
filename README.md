# Pitaco

**Small Model, Same Verdict? Distilling Aspect-Based Sentiment Analysis into a 4B LLM for Brazilian Portuguese Reviews**

A small open-weight LLM (Qwen3.5-4B) is fine-tuned on reviews labeled by a large open-weight teacher
to read a Portuguese customer review and return structured JSON (aspects, sentiment, verbatim
evidence, summary). It is compared with the teacher, a BERTimbau classifier and its own zero-shot
base on a hand-labeled gold set — on quality, cost and latency.

> Work in progress. Results table will appear here. Plan: [docs/PLAN.md](docs/PLAN.md) ·
> Architecture: [docs/architecture.md](docs/architecture.md) ·
> Annotation rules: [docs/annotation_guidelines.md](docs/annotation_guidelines.md)

## Status

| Module | State |
|---|---|
| M0 setup · M1 data · M2 annotation app | done |
| M3 teacher labeling · M4 evaluation | code + tests done; not run yet |
| M5 BERTimbau · M6 student (Unsloth) · Kaggle notebook | drafted; not run yet |

## Reproduce

```bash
uv sync                      # Python 3.12 + locked dependencies
cp .env.example .env         # add your keys
make check                   # lint + tests
make data                    # Olist → clean, deduped, frozen splits
make annotate-dev            # local blind labeling app
make estimate-dev            # teacher cost estimates (no API call)
uv run pitaco label --model deepseek-v4.1-flash --split gold_dev --execute   # spends money
uv run pitaco evaluate --predictions results/predictions/<system>/gold_dev.jsonl --split gold_dev
```

## Data and license

Source: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(CC BY-NC-SA 4.0); derived data inherits that license. Code: MIT.
