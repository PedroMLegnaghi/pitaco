Results on gold_dev (n = gold reviews; F1 in %, 95% bootstrap CI)

| System | Valid JSON | Aspect macro-F1 | Joint macro-F1 | Sentiment F1 (matched) | Evidence grounded | Latency p50 (s) | US$/1k reviews |
|---|---|---|---|---|---|---|---|
| bertimbau | 100.0 | 87.2 [71.4–93.4] | 85.1 | 64.5 | — | 0.05 | — |
| student-qwen3-4b-qlora-base-zeroshot | 2.0 | 0.0 [0.0–0.0] | 0.0 | 0.0 | — | 5.25 | 0.25 |
| student-qwen3-4b-qlora | 100.0 | 91.1 [73.3–97.3] | 90.2 | 88.4 | 100.0 | 3.74 | 0.14 |
| teacher-deepseek-v4-pro-fewshot-v1 | 92.0 | 87.7 [72.0–95.4] | 87.7 | 100.0 | 100.0 | 1.54 | 0.48 |
| teacher-deepseek-v4.1-flash-fewshot-v1 | 94.0 | 84.3 [65.4–95.2] | 83.4 | 88.2 | 100.0 | 1.06 | 0.15 |
| teacher-deepseek-v4.1-flash-fewshot-v2 | 92.0 | 90.3 [74.2–97.5] | 89.5 | 88.2 | 100.0 | 0.92 | 0.12 |
| teacher-qwen3.5-397b-fewshot-v1 | 100.0 | 86.1 [71.3–92.6] | 69.5 | 92.6 | 96.9 | 2.41 | 1.20 |
