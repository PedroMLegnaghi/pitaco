# Pitaco

**Small Model, Same Verdict? Distilling Aspect-Based Sentiment Analysis into a 4B LLM for Brazilian Portuguese Reviews**

Work in progress. A small open-weight LLM is fine-tuned to read a Portuguese customer review and return structured JSON (aspects, sentiment, evidence, summary), then compared against a large teacher model on quality, cost and latency. See [docs/PLAN.md](docs/PLAN.md).

## Quickstart

```bash
uv sync            # Python 3.12 + locked dependencies
cp .env.example .env
make check         # lint + tests
make env           # validate config and credentials
```
