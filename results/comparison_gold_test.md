Results on gold_test (n = gold reviews; F1 in %, 95% bootstrap CI)

| System | Valid JSON | Aspect macro-F1 | Joint macro-F1 | Sentiment F1 (matched) | Evidence grounded | Latency p50 (s) | US$/1k reviews |
|---|---|---|---|---|---|---|---|
| bertimbau | 100.0 | 73.5 [67.3–79.4] | 65.6 | 62.8 | — | 0.05 | — |
| student-qwen3-4b-qlora-base-zeroshot | 2.0 | 0.5 [0.0–1.1] | 0.5 | 100.0 | 100.0 | 6.17 | 0.25 |
| student-qwen3-4b-qlora | 100.0 | 74.3 [68.8–79.7] | 73.2 | 65.0 | 100.0 | 4.18 | 0.14 |
| teacher-deepseek-v4.1-flash-fewshot-v2 | 96.0 | 77.2 [72.5–81.7] | 75.6 | 82.2 | 100.0 | 1.00 | 0.08 |
| teacher-deepseek-v4.1-flash-zeroshot-v2 | 95.6 | 75.7 [70.3–80.5] | 73.9 | 78.7 | 100.0 | 1.03 | 0.07 |
