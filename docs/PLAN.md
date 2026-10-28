# Pitaco — project plan (approved 2026-10-28)

**Working title:** Small Model, Same Verdict? Distilling Aspect-Based Sentiment Analysis into a 4B LLM for Brazilian Portuguese Reviews

**Research question:** can a small fine-tuned LLM match a large teacher's quality on Portuguese ABSA at a fraction of the cost and latency?

**Hard constraints:** application deadline **2026-11-17** (submit on 11-16) · ~10 h/week · US$5 API budget · free compute only.

## Decisions

| Topic | Decision | Why |
|---|---|---|
| Student | `Qwen3.5-4B`, 16-bit LoRA (Unsloth) | Newest Apache-2.0 model in range; Unsloth advises against QLoRA on Qwen3.5; 4B fits a T4 in 16-bit (~10 GB) |
| Student fallback | `Qwen3-4B-Instruct-2507` + QLoRA | Plain transformer, mature tooling; switch if Qwen3.5 blocks > 1 h |
| Teacher | **DeepSeek V4.1 Flash** (`deepseek/deepseek-v4.1-flash-1017`, MIT), prompt v2 — chosen 2026-10-28 on dev-50: same agreement as V4 Pro (joint micro-F1 0.92, paired diff −0.002 ± 0.06) at ⅓ the cost; Qwen3.5-397B too expensive for silver (weak prefix caching) | Labels redistributable, reproducible, ≈ US$0.12 per 1k reviews |
| Closed reference (optional) | Claude Sonnet 5.5 on test-250 only | Frontier ceiling, ~US$1 |
| API access | One OpenRouter account, hard cap `budget.max_usd` | Single key, single top-up |
| Gold set | 300 = dev 50 (prompt/guideline tuning) + test 250 (final eval only), star-stratified, labeled blind | Prompt tuning on test would inflate results |
| Gold labels | Aspect + sentiment only; evidence scored automatically (verbatim substring) | Saves ~1 h of labeling |
| Silver set | 4,000 teacher-labeled → train 3,600 / val 400; gold near-duplicates removed | Leakage guard |
| Compute | Laptop (RTX 3050 6 GB): data, BERTimbau, 0.8B smoke tests, CPU/GGUF benchmarks. Kaggle T4 (not P100): 4B student, merge, GGUF export | Laptop has 14 GB RAM; merge needs ~30 GB |
| Tracking | Weights & Biases (public project) | Free, linkable from README |
| Licenses | Data: CC BY-NC-SA 4.0 (inherited from Olist). Code: MIT | ShareAlike + NonCommercial |
| Demo | HF account created 2026-10-28 → ZeroGPU hosting only after ~11-27. Before the deadline: Gradio share link + GIF in README | HF free-tier 30-day rule |

## Evaluation (same 250 gold reviews for every system)

Rows: teacher zero-shot · teacher few-shot (= teacher–human agreement) · Claude reference · BERTimbau · student pre-FT · student LoRA 16-bit · student GGUF Q4_K_M (CPU/GPU) · student NF4.

Columns: JSON validity · aspect-detection F1 (per aspect + macro) · joint aspect+sentiment F1 · sentiment macro-F1 on matched aspects · evidence grounding % · latency p50/p95 · throughput · US$/1k reviews · VRAM/size. 95% bootstrap CIs.

## Schedule

| Week | Dates | Sessions |
|---|---|---|
| 1 — data & labels | Oct 28 – Nov 3 | M0 setup (1 h) · M1 data (2 h) · M2 guidelines + dev-50 (2 h) · M3 teacher + 4k silver (2 h) · test-250 labeling (3 h) |
| 2 — models & eval | Nov 4 – Nov 10 | M4 eval harness + Baseline B (2 h) · M5 BERTimbau (1.5 h) · M6 student (3 h) · M7 efficiency (1.5 h) · M8 errors + publish + README (2 h) |
| 3 — polish | Nov 11 – Nov 16 | Kaggle notebook · data-size ablation (500/1,500/3,600) · article (~1,500 words) · intra-annotator κ · demo GIF · submit 11-16 |

## Diagrams (to add to `docs/architecture.md` as Mermaid)

Pipeline overview · data splits & leakage guard · labeling loop · LoRA anatomy + T4 memory budget · SFT example with loss mask · evaluation harness · serving path.
