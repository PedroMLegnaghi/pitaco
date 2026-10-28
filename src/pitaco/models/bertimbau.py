"""`pitaco train-bert`: BERTimbau multi-label classifier (Baseline A).

STATUS: draft, not yet executed (needs the `train` dependency group and user approval to run).

Same filtered silver data as the student. Threshold tuned on silver val; then gold dev/test are
predicted in the common format with label-only mentions (BERT produces no evidence or summary —
a capability gap the comparison table shows explicitly).
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import Field

from pitaco.config import BaseConfig
from pitaco.models.bert_labels import LABELS, decode, tune_threshold
from pitaco.predictions import PredictionRecord, write_predictions
from pitaco.schema import GoldMention
from pitaco.utils.seed import set_seed


class BertConfig(BaseConfig):
    model_name: str = "neuralmind/bert-base-portuguese-cased"
    training_dir: Path = Path("data/training")
    train_set: str = "all"
    max_length: int = 128  # reviews are short (max 208 chars ≈ 60 tokens)
    learning_rate: float = 3e-5
    epochs: int = 6
    batch_size: int = 16
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    early_stopping_patience: int = Field(2, ge=1)
    output_dir: Path = Path("outputs/bertimbau")
    system_name: str = "bertimbau"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def run(cfg: BertConfig) -> dict:
    import torch
    from datasets import Features, Sequence, Value, load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    set_seed(cfg.seed)
    os.environ.setdefault("WANDB_PROJECT", cfg.wandb.project)  # W&B project for this run
    files = {
        "train": str(cfg.training_dir / f"train_{cfg.train_set}" / "bert.jsonl"),
        "val": str(cfg.training_dir / "val" / "bert.jsonl"),
    }
    # float32 labels: BCEWithLogitsLoss fails on the float64 that JSON loading infers by default.
    features = Features(
        {
            "review_id": Value("string"),
            "text": Value("string"),
            "labels": Sequence(Value("float32")),
        }
    )
    ds = load_dataset("json", data_files=files, features=features)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    ds = ds.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=cfg.max_length), batched=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",  # sigmoid + BCE per label
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    )
    args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=64,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_ratio,  # transformers v5: a float < 1 is a fraction of training
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,  # best epoch by val loss, not the last one (overfitting guard)
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to=[] if cfg.wandb.mode == "disabled" else ["wandb"],
        run_name=f"{cfg.system_name}-{cfg.train_set}",
        seed=cfg.seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"].remove_columns(["review_id", "text"]),
        eval_dataset=ds["val"].remove_columns(["review_id", "text"]),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=[EarlyStoppingCallback(cfg.early_stopping_patience)],
    )
    trainer.train()

    val_probs = _sigmoid(
        trainer.predict(ds["val"].remove_columns(["review_id", "text"])).predictions
    )
    val_gold = [decode(labels, 0.5) for labels in ds["val"]["labels"]]
    threshold, val_score = tune_threshold(val_probs, val_gold)

    model.eval()
    device = model.device
    out = {"threshold": threshold, "val_joint_macro_f1": val_score}
    for split in ("gold_dev", "gold_test"):
        reviews = pd.read_json(cfg.paths.data_dir / "processed" / f"{split}.jsonl", lines=True)
        records = []
        for row in reviews.itertuples():  # batch size 1: the latency a single request would see
            start = time.perf_counter()
            with torch.no_grad():
                enc = tokenizer(
                    row.text, truncation=True, max_length=cfg.max_length, return_tensors="pt"
                ).to(device)
                probs = torch.sigmoid(model(**enc).logits)[0].float().cpu().numpy()
            latency = time.perf_counter() - start
            mentions = [
                GoldMention(aspecto=a, sentimento=s) for a, s in sorted(decode(probs, threshold))
            ]
            records.append(
                PredictionRecord(
                    review_id=row.review_id,
                    system=cfg.system_name,
                    raw=None,
                    analysis=None,
                    mentions=mentions,
                    valid_json=True,  # not applicable: labels come from a classifier head
                    valid_schema=True,
                    latency_s=latency,
                )
            )
        write_predictions(
            cfg.paths.results_dir / "predictions" / cfg.system_name / f"{split}.jsonl", records
        )
        out[split] = len(records)
    trainer.save_model(str(cfg.output_dir / "best"))
    return out
