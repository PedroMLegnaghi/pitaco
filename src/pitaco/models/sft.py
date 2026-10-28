"""`pitaco train-student` / `pitaco predict-student`: LoRA fine-tuning of the small LLM (Unsloth).

STATUS: draft, not yet executed. First run = local smoke test with Qwen3.5-0.8B
(configs/student/smoke_qwen35_08b.yaml) before the Kaggle T4 run. Verify on that smoke run:
the chat template output, the response-only loss mask, and that thinking stays disabled.

Key choices (see docs/PLAN.md):
- 16-bit LoRA, not QLoRA, for Qwen3.5 (Unsloth reports larger quantization error with QLoRA on
  this architecture; a 4B model fits a T4 in 16-bit). `load_in_4bit: true` turns it into QLoRA.
- Loss only on the assistant JSON (`train_on_responses_only`): the model should learn to answer,
  not to reproduce reviews.
- The same short instruction (prompts/student_v1.md) at training and inference time.
"""

import json
import time
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import Field

from pitaco.config import BaseConfig, StrictModel
from pitaco.labeling.prompts import student_instruction
from pitaco.predictions import record_from_raw, write_predictions
from pitaco.utils.seed import set_seed


class LoraConfig(StrictModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class TrainConfig(StrictModel):
    train_set: str = "all"
    epochs: float = 2
    learning_rate: float = 2e-4
    batch_size: int = 8
    grad_accum: int = 2
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    eval_steps: int = 50
    logging_steps: int = 10
    max_steps: int = -1  # > 0 only for smoke tests


class GenerationConfig(StrictModel):
    max_new_tokens: int = 256
    throughput_batch_size: int = 16


class StudentConfig(BaseConfig):
    base_model: str
    loader: Literal["vision", "language"] = "vision"  # Qwen3.5 ships as a unified VLM
    load_in_4bit: bool = False
    max_seq_length: int = 512
    prompts_dir: Path = Path("prompts")
    student_prompt_version: str = "v1"
    training_dir: Path = Path("data/training")
    instruction_part: str = "<|im_start|>user\n"
    response_part: str = "<|im_start|>assistant\n"
    chat_template_kwargs: dict = Field(default_factory=lambda: {"enable_thinking": False})
    lora: LoraConfig = LoraConfig()
    train: TrainConfig = TrainConfig()
    generation: GenerationConfig = GenerationConfig()
    gpu_usd_per_hour: float = 0.35  # reference on-demand T4 price used for the cost column
    output_dir: Path = Path("outputs/student")
    system_name: str


def _loader(cfg: StudentConfig):
    if cfg.loader == "vision":
        from unsloth import FastVisionModel

        return FastVisionModel
    from unsloth import FastLanguageModel

    return FastLanguageModel


def _text_tokenizer(tokenizer):
    return getattr(tokenizer, "tokenizer", tokenizer)  # VLM processors wrap the text tokenizer


def _as_parts(messages: list[dict]) -> list[dict]:
    """VLM chat templates expect content as a list of typed parts."""
    return [
        {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages
    ]


def render(tokenizer, cfg: StudentConfig, messages: list[dict], add_generation_prompt: bool) -> str:
    msgs = _as_parts(messages) if cfg.loader == "vision" else messages
    return tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **cfg.chat_template_kwargs,
    )


def load(cfg: StudentConfig, adapter: Path | None = None):
    loader = _loader(cfg)
    model, tokenizer = loader.from_pretrained(
        str(adapter or cfg.base_model),
        load_in_4bit=cfg.load_in_4bit,
        max_seq_length=cfg.max_seq_length,
        use_gradient_checkpointing="unsloth",
    )
    return loader, model, tokenizer


def train(cfg: StudentConfig) -> dict:
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    set_seed(cfg.seed)
    loader, model, tokenizer = load(cfg)
    peft_kwargs = dict(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias="none",
        random_state=cfg.seed,
    )
    if cfg.loader == "vision":
        model = loader.get_peft_model(
            model,
            finetune_vision_layers=False,  # text-only task: leave the vision encoder frozen
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            **peft_kwargs,
        )
    else:
        model = loader.get_peft_model(model, target_modules=cfg.lora.target_modules, **peft_kwargs)

    files = {
        "train": str(cfg.training_dir / f"train_{cfg.train.train_set}" / "sft.jsonl"),
        "val": str(cfg.training_dir / "val" / "sft.jsonl"),
    }
    ds = load_dataset("json", data_files=files)
    ds = ds.map(
        lambda ex: {"text": render(tokenizer, cfg, ex["messages"], add_generation_prompt=False)}
    )

    t = cfg.train
    trainer = SFTTrainer(
        model=model,
        processing_class=_text_tokenizer(tokenizer),
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        args=SFTConfig(
            output_dir=str(cfg.output_dir),
            dataset_text_field="text",
            max_length=cfg.max_seq_length,
            packing=False,
            num_train_epochs=t.epochs,
            max_steps=t.max_steps,
            learning_rate=t.learning_rate,
            per_device_train_batch_size=t.batch_size,
            per_device_eval_batch_size=t.batch_size,
            gradient_accumulation_steps=t.grad_accum,
            warmup_steps=t.warmup_ratio,  # transformers v5: a float < 1 is a fraction of training
            weight_decay=t.weight_decay,
            lr_scheduler_type=t.scheduler,
            optim="adamw_8bit",
            eval_strategy="steps",
            eval_steps=t.eval_steps,
            save_strategy="steps",
            save_steps=t.eval_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            logging_steps=t.logging_steps,
            report_to=[] if cfg.wandb.mode == "disabled" else ["wandb"],
            run_name=cfg.system_name,
            seed=cfg.seed,
        ),
    )
    trainer = train_on_responses_only(
        trainer, instruction_part=cfg.instruction_part, response_part=cfg.response_part
    )
    stats = trainer.train()
    adapter_dir = cfg.output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    return {"adapter": str(adapter_dir), "train_runtime_s": stats.metrics.get("train_runtime")}


def _generate(model, tokenizer, cfg: StudentConfig, prompts: list[str]) -> list[str]:
    import torch

    tok = _text_tokenizer(tokenizer)
    tok.padding_side = (
        "left"  # decoder-only batching: pad on the left so generation continues the text
    )
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=cfg.generation.max_new_tokens, do_sample=False)
    new_tokens = out[:, enc["input_ids"].shape[1] :]
    return tok.batch_decode(new_tokens, skip_special_tokens=True)


def predict(
    cfg: StudentConfig, adapter: Path | None, system: str, limit: int | None = None
) -> dict:
    """Greedy predictions on gold dev/test (batch size 1 → latency) + batched throughput → cost."""
    loader, model, tokenizer = load(cfg, adapter)
    loader.for_inference(model)
    instruction = student_instruction(cfg.prompts_dir, cfg.student_prompt_version)

    def prompt(text: str) -> str:
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"Avaliação: {text}"},
        ]
        return render(tokenizer, cfg, messages, add_generation_prompt=True)

    val = pd.read_json(cfg.paths.data_dir / "processed" / "silver_val.jsonl", lines=True)
    if limit:
        val = val.head(limit * cfg.generation.throughput_batch_size)
    batch = cfg.generation.throughput_batch_size
    start = time.perf_counter()
    for i in range(0, len(val), batch):
        _generate(model, tokenizer, cfg, [prompt(t) for t in val.text[i : i + batch]])
    seconds_per_review = (time.perf_counter() - start) / len(val)
    usd_per_review = seconds_per_review * cfg.gpu_usd_per_hour / 3600

    summary = {
        "throughput_reviews_per_s": 1 / seconds_per_review,
        "usd_per_1k_reviews": 1000 * usd_per_review,
    }
    for split in ("gold_dev", "gold_test"):
        reviews = pd.read_json(cfg.paths.data_dir / "processed" / f"{split}.jsonl", lines=True)
        if limit:
            reviews = reviews.head(limit)
        records = []
        for row in reviews.itertuples():
            t0 = time.perf_counter()
            raw = _generate(model, tokenizer, cfg, [prompt(row.text)])[0]
            records.append(
                record_from_raw(
                    row.review_id,
                    system,
                    raw,
                    row.text,
                    latency_s=time.perf_counter() - t0,
                    cost_usd=usd_per_review,
                )
            )
        write_predictions(
            cfg.paths.results_dir / "predictions" / system / f"{split}.jsonl", records
        )
        summary[split] = len(records)
    out = cfg.paths.results_dir / "efficiency" / f"{system}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    return summary
