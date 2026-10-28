# Prompt changelog

Prompts are tuned on **gold dev (50)** only. Metrics below use the parser with same-sentiment duplicate merging (strict validity is reported separately). Record the dev scores of every version here; never tune on gold test.

| Version | Change | Dev aspect macro-F1 | Dev joint macro-F1 | Notes |
|---|---|---|---|---|
| teacher v1 | Derived from annotation guidelines v1.1; 12 few-shot examples from silver_train | Flash 84.3 · Pro 87.7 · Qwen 86.1 | Flash 83.4 · Pro 87.7 · Qwen 69.5 | Joint micro-F1 ≈ 0.92 for Flash and Pro (paired diff −0.002, CI ±0.06). Errors: aspect listed twice (6–8%); Qwen paraphrased 2 evidence spans with "...". Dev cost: Flash $0.007, Pro $0.024, Qwen $0.060 |
| teacher v2 | Rule 2: one entry per aspect even with several spans; rule 4: no "..." joins | Flash 90.3 | Flash 89.5 | Chosen. Joint micro-F1 0.921 (= v1); κ 0.884. Duplicates persist (8% strict-invalid) → deterministic same-sentiment merge added to the parser; usable 98% |
| student v1 | Short instruction used for both zero-shot and fine-tuning (must stay identical) | — | — | — |
