# Knowledge base — everything new in this project, explained

A learning guide for Pedro. Each entry answers: **what is it**, **why we use it here**, and **how you
use it**. Read top to bottom once; later use it as a glossary (Ctrl+F). Every term the project
uses should be here — if one is missing, ask and it will be added.

---

## 1. The project in one paragraph

Customers write reviews like *"Chegou antes do prazo, mas veio com defeito"*. We want a model that
reads a review and returns structured JSON: which **aspects** are evaluated (entrega, produto,
embalagem, atendimento, preco, outro), the **sentiment** of each (positivo, neutro, negativo), an
**evidence** quote copied from the review, and a one-sentence **summary**. A large model (the
**teacher**) does this well but costs money per review; a small model (the **student**) is cheap to
run. Research question: **can a small fine-tuned model match the large one at a fraction of the cost
and latency?**

Pipeline: raw reviews → cleaned data → you hand-label 300 (**gold**) → the teacher labels 4,000
(**silver**) → we train the student and a BERT baseline on silver → everyone is scored on your gold
test set → we publish the results.

---

## 2. Core ML / LLM concepts

**Model / parameters.** A neural network is a huge function with adjustable numbers called
*parameters* (or *weights*). "4B" = 4 billion parameters. More parameters ≈ more capable, but more
memory and slower.

**LLM (large language model).** A model trained to predict the next piece of text. Given a prompt,
it generates an answer token by token. Examples here: DeepSeek, Qwen.

**Token / tokenizer.** Models read text as *tokens* — pieces of words (roughly 3–4 characters in
Portuguese). The *tokenizer* converts text ↔ token IDs. API prices and memory are counted in tokens.

**Prompt, system / user / assistant messages.** A chat request is a list of messages: the *system*
message holds instructions ("you are an annotator… respond only with JSON"), the *user* message
holds the input (the review), the *assistant* message is the model's answer.

**Zero-shot vs few-shot.** *Zero-shot*: only instructions. *Few-shot*: instructions plus a few solved
examples inside the prompt. Our result: few-shot did **not** help the teacher on test (0.878 vs 0.877).

**Structured output / JSON schema.** A JSON schema is a formal description of the allowed JSON shape
(keys, types, allowed values). With `response_format: json_schema` the API constrains the model to
produce JSON matching it. It cannot enforce "each aspect at most once", which is why we still saw
duplicate aspects.

**Temperature.** Randomness of generation. 0 = always pick the most likely token (as repeatable as
possible). We use 0 for labeling and evaluation.

**Reasoning / thinking tokens.** Some models "think" in hidden text before answering. Those tokens are
billed and add latency. We disable thinking (DeepSeek had it on by default at "high").

**Prompt caching (provider side).** If many requests start with the same long text (our instructions +
examples), the provider can reuse its computation and charge ~1–10% of the normal input price for it.
93% of our silver input was cached — that is why 4,000 labels cost only $0.32.

**Knowledge distillation / teacher–student.** Use a large model's outputs as training labels for a
small model. Ours is *sequence-level* (black-box) distillation: we only see the teacher's text
answers, not its internal probabilities.

**Gold vs silver labels.** *Gold*: made by a human (you), trusted, used for evaluation. *Silver*: made
by a model (the teacher), cheaper, a bit noisy, used for training.

**Fine-tuning / SFT.** Continue training a pretrained model on your own examples so it learns your
task and format. *SFT (supervised fine-tuning)* = training on (input → desired output) pairs.

**LoRA (Low-Rank Adaptation).** Instead of changing all 4 billion weights, freeze them and train two
small matrices *A* and *B* added to each layer (W + B·A). Only ~0.8% of parameters are trained
(33M of 4.06B in our run) → far less memory, fast, and the result is a small *adapter* file (~100 MB)
that sits on top of the base model. *Rank r* (we use 16) = size of the small matrices; *alpha*
(32) scales their effect.

**QLoRA.** LoRA where the frozen base model is stored in **4-bit** precision (NF4 format). The 4B
model then takes ~2.5 GB instead of ~8 GB, so it fits small GPUs. The trainable adapters stay in
16-bit. Trade-off: a little quality loss from quantization.

**Quantization.** Storing weights with fewer bits (32 → 16 → 8 → 4). Smaller and often faster, slightly
less accurate. Formats: *NF4* (bitsandbytes, used in QLoRA), *GGUF Q4_K_M* (llama.cpp, used to run
models on CPUs). Experiment 4 measures quality/speed before vs after 4-bit.

**Precision: float32 / float16 / bfloat16.** How many bytes per number. float32 = 4 bytes, precise,
big, slow. float16 = 2 bytes, fast, small range of values (can overflow). bfloat16 = 2 bytes with a
wide range. **The T4 GPU has no bfloat16** — that is why Qwen3.5 (which overflows in float16) fell
back to float32 on Kaggle and we switched to Qwen3.

**GPU / VRAM / OOM.** The GPU does the math; its memory is *VRAM* (T4: 15 GB; your RTX 3050: 6 GB).
Model + training data in flight must fit, otherwise *CUDA out of memory (OOM)*. Fixes: smaller batch,
shorter sequences, 4-bit base, gradient checkpointing.

**Training vocabulary.**
- *Step*: one weight update. *Batch size*: examples per step. *Gradient accumulation*: add up several
  small batches before one update (simulates a bigger batch without the memory).
- *Epoch*: one pass over the whole training set.
- *Loss*: number measuring how wrong the model is; training lowers it.
- *Learning rate*: how big each update is. *Warmup*: start small, then ramp up. *Scheduler (cosine)*:
  slowly lower it toward the end.
- *Gradient checkpointing*: recompute some values instead of storing them → less memory, a bit slower.

**Train / validation / test.** *Train*: the model learns from it. *Validation*: checked during
training to choose settings (epochs, thresholds) — never trained on. *Test*: touched once, at the end,
for the final score. Using test for choices = cheating yourself (**leakage**).

**Overfitting / early stopping.** Overfitting = memorizing training examples instead of learning
general rules (training loss keeps falling, validation loss rises). Early stopping = stop when
validation stops improving and keep the best checkpoint.

**Chat template.** Each model family expects messages wrapped in special tokens, e.g. Qwen:
`<|im_start|>user … <|im_end|><|im_start|>assistant …`. Training and inference must use the **same**
template, or the model sees an unfamiliar format.

**Loss mask ("train on responses only").** During SFT we compute the loss only on the assistant's JSON,
not on the instructions or the review, so the model learns to *answer*, not to write reviews. We
verified it: ~62 loss tokens per example vs ~200 total tokens.

**Inference / latency / throughput.** *Inference*: using a trained model to make predictions.
*Latency*: time for one review (we report the median = *p50*, and *p95* = slow cases). *Throughput*:
reviews per second when processing many at once (batching).

**BERT / BERTimbau / classification head.** BERT is an older kind of model (110M parameters) that reads
text and outputs numbers instead of generating text. *BERTimbau* is BERT pretrained on Brazilian
Portuguese. We add a *classification head* (a small final layer) with 18 outputs, one per
aspect+sentiment pair.

**Multi-label classification, sigmoid, BCE, threshold.** *Multi-label*: several labels can be true at
once (entrega:negativo AND produto:positivo). Each label gets an independent probability (*sigmoid*,
0–1) and its own loss (*binary cross-entropy, BCE*). A *threshold* (tuned on validation: 0.25) decides
when a probability counts as "present".

---

## 3. Data concepts

**Olist dataset.** ~100k real orders from a Brazilian marketplace (2016–2018) published on Kaggle;
we use the 40,977 reviews that have text. License **CC BY-NC-SA 4.0** (credit, non-commercial,
share-alike) — so our derived dataset must use the same license.

**LGPD / PII / scrubbing.** LGPD = Brazil's data-protection law. *PII* = personally identifiable
information (names, phones, emails, CPF). *Scrubbing* = replacing it with placeholders like
`[TELEFONE]`. *Data minimization* = keep only the columns you need (we dropped `order_id`).

**NER / spaCy.** *NER (named-entity recognition)* = a model that finds names of people, places,
organizations in text. *spaCy* = a popular NLP library that ships NER models. We tested spaCy's
Portuguese model: it flagged ~11% of reviews as containing a person, almost all wrong (capitalized
verbs like "Comprei"), so we used targeted rules (regular expressions) instead.

**Regular expression (regex).** A pattern language for matching text, e.g. phone numbers. Our PII
rules are regexes, each unit-tested.

**Deduplication / near-duplicates.** Many reviews repeat ("Muito bom"). We normalize text (lowercase,
no accents/punctuation) and drop exact copies; then a fuzzy match (*rapidfuzz* similarity ≥ 90%)
removes paraphrases of gold reviews from the training pool.

**Data leakage.** When information from the test set reaches training or tuning, so the score looks
better than reality. Guards here: dedup before splitting, near-duplicate filter, few-shot examples
only from silver, prompt tuned only on dev, a test-access log.

**Stratified sampling.** Sampling an equal number per group (here: per star rating). Without it, the
test set would be mostly 5-star "chegou rápido" reviews and rare aspects would barely appear.

**Annotation guidelines / blind labeling.** Written rules for labeling, so labels are consistent and
the teacher prompt can follow the same rules. *Blind*: the annotator never sees the model's answer or
the star rating first (that would bias you — *anchoring*).

**Agreement / Cohen's κ (kappa).** How often two annotators (or you and the teacher) agree, corrected
for agreement by chance. κ = 1 perfect, 0 = chance level; > 0.8 is usually read as "near-perfect",
0.6–0.8 "substantial". Teacher vs you: 0.88 on dev, 0.74 on test.

---

## 4. Metrics

**Precision, recall, F1.** For "did the model find aspect X?":
- *Precision* = of the aspects the model claimed, how many were right (penalizes inventing).
- *Recall* = of the true aspects, how many the model found (penalizes missing).
- *F1* = their harmonic mean (high only if both are high). Ranges 0–1.

**Aspect detection F1.** Is the aspect present at all (sentiment ignored)?

**Joint (aspect + sentiment) F1.** A hit needs both right. A wrong sentiment counts as one false
positive and one false negative. This is our **headline** metric.

**Micro vs macro F1.** *Micro*: pool all decisions, then compute F1 — dominated by common aspects,
stable. *Macro*: compute F1 per aspect, then average — every aspect weighs the same, so rare ones
(outro: 7 examples) make it jumpy. We report both; micro is the headline.

**Sentiment on matched aspects.** Accuracy/F1 of the polarity only where both gold and model found the
aspect — isolates "reads sentiment right" from "finds the aspect".

**JSON validity: strict / repaired / usable.** *Strict*: the raw output follows the schema. *Repaired*:
an aspect listed twice with the same sentiment, merged deterministically. *Usable*: strict or repaired.
The table shows strict, so repairs cannot hide format errors.

**Evidence grounding rate.** Share of evidence quotes that really appear in the review (exact, or after
fixing case/spacing). A paraphrased or invented quote is a hallucination.

**Bootstrap confidence interval (CI).** Resample the 250 test reviews with replacement 1,000 times,
recompute the metric each time; the middle 95% of values is the 95% CI. It tells how much the score
could move with a different sample of reviews.

**Paired bootstrap / significance.** Same resampling, but both systems are scored on the *same*
resampled reviews and we look at the *difference*. If the 95% CI of the difference excludes 0, the
difference is *statistically significant* (unlikely to be luck). Example: teacher − BERTimbau =
+0.056 [0.021, 0.091] → significant.

**Cost per 1,000 reviews.** API: measured cost × 1000. Self-hosted student: GPU price per hour ÷
reviews per hour (reference: T4 at $0.35/h).

---

## 5. Models used

| Model | Role | Why / notes |
|---|---|---|
| **DeepSeek V4.1 Flash** (`deepseek/deepseek-v4.1-flash-1017`, MIT) | **Teacher** | Same agreement with you as V4 Pro on dev, 1/3 the price |
| DeepSeek V4 Pro, Qwen3.5-397B | Teacher candidates | Tested on dev only; Qwen too expensive (weak caching) |
| **Qwen3-4B-Instruct-2507** (Apache-2.0) | **Student** | Trains in float16 on the free T4 with QLoRA; no thinking mode |
| Qwen3.5-4B | Planned student, dropped | Needs bfloat16; on a T4 it falls back to float32 (2× memory, ~8× slower) |
| Qwen3.5-0.8B | Smoke test only | Tiny, same code path, fits your laptop |
| **BERTimbau** (`neuralmind/bert-base-portuguese-cased`) | **Baseline A** | Classic fast classifier; no evidence/summary |
| Claude Sonnet 5.5 | Optional reference | Dropped to save budget |

---

## 6. Tools — what, why, how

### Python environment
- **uv** — installs Python and packages, fast. `uv sync` installs exactly what `uv.lock` lists;
  `uv run <cmd>` runs inside the project environment; `uv add pkg` adds a dependency.
- **pyproject.toml / uv.lock** — `pyproject.toml` declares what the project needs (ranges);
  `uv.lock` pins the *exact* versions so everyone gets the same environment (**reproducibility**).
  Dependency *groups*: `dev` (tests/lint), `app` (Gradio), `train` (GPU stack).
- **.env** — your secret keys (never committed). `.env.example` shows which keys exist.

### Code quality
- **git / GitHub** — version history and the public repo. Each module = one commit.
- **pytest** — runs the `tests/` folder (`make test`). Tests prove metrics, parsing, splits, PII rules work.
- **ruff** — linter + formatter (style and common bugs). `make lint`, `make format`.
- **GitHub Actions (CI)** — runs lint + tests automatically on every push (`.github/workflows/ci.yml`).
- **Makefile** — shortcuts: `make check`, `make data`, `make annotate-dev`, `make estimate-dev`.

### The project's own CLI (`pitaco ...`, built with **Typer**)
| Command | Does |
|---|---|
| `pitaco check-env` | Validates config and which keys are set |
| `pitaco data` | Download Olist → clean → scrub PII → dedup → frozen splits |
| `pitaco annotate --split dev` | Local labeling app (Gradio) |
| `pitaco label --model X --split Y` | Teacher labeling; dry-run cost estimate unless `--execute` |
| `pitaco evaluate --predictions F --split S` | Score predictions vs gold, with CIs |
| `pitaco compare --a A --b B --split S` | Paired bootstrap between two systems |
| `pitaco report --split S` | Results table (`results/comparison_*.md`) |
| `pitaco build-training-data` | Silver labels → training files (+ 500/1,500 subsets) |
| `pitaco train-bert` | Train BERTimbau, predict gold |
| `pitaco train-student --config C` | LoRA/QLoRA fine-tuning |
| `pitaco predict-student --config C [--adapter D]` | Student predictions + speed + cost |

### Configs
- **YAML + pydantic** — every stage reads a YAML file (`configs/`); *pydantic* validates it, so a
  typo stops the run instead of being silently ignored. `extends: base.yaml` inherits shared settings.

### Data libraries
- **pandas / pyarrow** — tables in Python; parquet = compact table file format.
- **rapidfuzz** — fast fuzzy string similarity (near-duplicates, evidence matching).
- **JSONL** — one JSON object per line; all our predictions/labels use it.

### LLM access
- **OpenRouter** — one API (and one $5 balance) for many models. We call it with the **openai** Python
  SDK (OpenRouter speaks the same protocol). Our client caches every answer (never pay twice), keeps
  a spending ledger, and refuses calls that could exceed the $4 cap.

### Kaggle
- **Kaggle** — data-science site with datasets, competitions and free GPU notebooks.
- **Notebook (Jupyter, `.ipynb`)** — document of text + code *cells*; run a cell with ▶, output below.
- **Session** — the temporary machine running your notebook; wiped at the end.
- **Accelerator** — attached hardware; we need **GPU T4** (P100 is retired).
- **Internet (session option)** — must be On to download code/libraries; requires a verified phone.
- **Input / dataset** — files attached to the notebook (we attach Olist); they appear under `/kaggle/input/`.
- **Output** — files saved in `/kaggle/working`, downloadable after the run (`pitaco_outputs.tgz`).
- **Version / "Save & Run All (Commit)"** — runs the whole notebook in the background and keeps its
  log and outputs; your laptop can be off.
- **Secrets** — private values (e.g. `WANDB_API_KEY`) available to the notebook without appearing in code.
- **Quota** — ~30 GPU-hours per week, free.
- **Kaggle CLI / API** — `kaggle kernels push -p kaggle/` uploads and starts a notebook; `kaggle
  kernels status <user>/<slug>` shows if it is running/complete/error; `kaggle kernels output` downloads
  its log and files. "Kernel" is Kaggle's old word for notebook.

### Hugging Face
- **Hub** — site hosting models and datasets (like GitHub for ML). We will publish the student model
  and the gold+silver dataset there, each with a *card* (README describing use, data, metrics, limits).
- **transformers** — the standard library to load and run models.
- **datasets** — library to load data for training (reads our JSONL).
- **Spaces / ZeroGPU** — free hosted demo apps; ZeroGPU = shared GPU, needs an account older than 30 days.
- **HF_HOME** — where downloaded models are cached; we point it to the project's `.cache/`.

### Training stack
- **PyTorch (torch)** — the deep-learning engine; **CUDA** = NVIDIA's GPU platform it uses.
- **PEFT** — Hugging Face library implementing LoRA.
- **TRL** — Hugging Face library with `SFTTrainer` (supervised fine-tuning loop).
- **Unsloth** — wraps transformers/PEFT/TRL with faster, memory-saving kernels; must be imported
  *first* (we learned this from a crash).
- **bitsandbytes** — 4-bit/8-bit quantization (NF4) used by QLoRA; also the 8-bit optimizer.
- **Triton** — compiler for fast GPU kernels that Unsloth uses.

### Tracking and apps
- **Weights & Biases (W&B)** — website that records training runs (loss curves, settings) so you
  can compare and share them. Our runs log to the `pitaco` project.
- **Gradio** — Python library for quick web UIs. Used for the labeling app and later the demo.

---

## 7. Lessons learned (interview material)

1. **Choose by data, then by cost.** V4 Pro and V4.1 Flash were statistically identical on dev → pick
   the cheaper one.
2. **Prompt caching decides API cost.** Stable prompt prefix → 93% cached input → $0.32 for 4,000 labels.
3. **Few-shot examples gave no gain on test** — they cost ~2,000 extra input tokens per review.
4. **Dev scores overestimate** (0.92 dev vs 0.88 test) because dev was used to choose — that is why a
   locked test set exists.
5. **BERTimbau gets ~94% of the teacher's detection quality at ~20× the speed**, but cannot explain
   itself (no evidence, no summary).
6. **Hardware constrains model choice.** Qwen3.5 needs bfloat16; free T4s lack it → plan B (Qwen3 + QLoRA).
7. **Smoke-test before expensive runs.** A 30-step local run catches template/loss-mask/library bugs
   in minutes instead of failing a 2-hour cloud run.
8. **Environment mismatches are real bugs.** Upgrading torch on Kaggle broke torchaudio; fixed by
   import order and uninstalling unused packages.
9. **Measure strict validity separately from repaired outputs**, so post-processing cannot hide errors.
10. **Report uncertainty.** With 250 test reviews and rare aspects, confidence intervals matter more
    than the third decimal.
