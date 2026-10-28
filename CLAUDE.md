# Pitaco — context for Claude

Portfolio project for Pedro's application to iFood GEN AI 2027 (AI Engineer track, AI/ML Engineering profile). Deadline 2026-11-17. Full plan and decisions: `docs/PLAN.md`.

## How to work here
- Build one module at a time (M0…M8). For each module, briefly explain the concepts so Pedro can defend every decision in an interview, and warn about common mistakes (leakage, overfitting, wrong teacher, OOM).
- Pedro is intermediate in Python and a beginner with LLM fine-tuning and MLOps.
- **Never assume he knows a term.** The first time a concept, tool or metric appears (e.g. NER,
  spaCy, F1, notebook, kernel, secret, LoRA), explain in plain words what it is, why it matters
  here, and what he has to do with it — step by step for anything he must do himself.
- Ask before any paid API call (state the cost) and before running any model. Budget: US$5 total.
- Code, docs and comments in English. Review text and labels are Brazilian Portuguese.
- Never send PII or real API keys anywhere. Keys live in `.env` (gitignored).
- API spend is capped at US$5 in total; every LLM call must go through the cached, budget-capped client.

## Conventions
- Python 3.12 (matches Kaggle and HF ZeroGPU), managed with `uv`; `uv.lock` pins everything.
- Every stage is a CLI command (`pitaco <stage> --config configs/...yaml`) chained by the `Makefile`. Notebooks only call package code.
- Configs are YAML validated by pydantic models with `extra="forbid"`.
- `src/pitaco/schema.py` is the single source of truth for the output JSON and the taxonomy.
- The gold test set is read only by the final evaluation. Never tune anything on it.
- Run `make check` (ruff + pytest) before declaring a change done.
