"""`pitaco build-training-data`: teacher silver predictions → training files for both models.

Only fully valid teacher outputs are kept: schema-valid AND every evidence span grounded in the
review. A label we cannot verify is noise the student would learn to imitate.

The same filtered silver data feeds BERTimbau and the student, so the comparison is about the
model, not the data. Ablation subsets are nested and star-stratified (500 ⊂ 1,500 ⊂ all).
"""

import json
from pathlib import Path

import pandas as pd

from pitaco.labeling.prompts import student_instruction
from pitaco.models.bert_labels import encode
from pitaco.predictions import PredictionRecord, read_predictions
from pitaco.schema import ReviewAnalysis

STARS = (1, 2, 3, 4, 5)


def keep(record: PredictionRecord) -> bool:
    return (
        record.valid_schema
        and record.analysis is not None
        and record.n_grounded == record.n_evidence
    )


def compact_json(analysis: ReviewAnalysis) -> str:
    """Target string the student learns to emit: compact JSON, accents unescaped (fewer tokens)."""
    return json.dumps(analysis.model_dump(), ensure_ascii=False, separators=(",", ":"))


def sft_example(instruction: str, text: str, analysis: ReviewAnalysis) -> dict:
    """Chat-format example. Inference must build exactly the same system + user turns."""
    return {
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"Avaliação: {text}"},
            {"role": "assistant", "content": compact_json(analysis)},
        ]
    }


def nested_subsets(df: pd.DataFrame, sizes: list[int], seed: int) -> dict[int, pd.DataFrame]:
    """Star-stratified subsets where each smaller subset is contained in the larger ones."""
    ranked = df.sample(frac=1.0, random_state=seed)
    ranked["rank"] = ranked.groupby("review_score").cumcount()
    out = {}
    for size in sizes:
        per_star = size // len(STARS)
        out[size] = ranked[ranked["rank"] < per_star].drop(columns="rank")
    return out


def build(
    predictions_dir: Path,
    processed_dir: Path,
    out_dir: Path,
    prompts_dir: Path,
    student_prompt_version: str,
    ablation_sizes: list[int],
    seed: int,
) -> dict:
    instruction = student_instruction(prompts_dir, student_prompt_version)
    report: dict = {"source": str(predictions_dir)}
    frames = {}
    for split in ("silver_train", "silver_val"):
        reviews = pd.read_json(processed_dir / f"{split}.jsonl", lines=True)
        preds = read_predictions(predictions_dir / f"{split}.jsonl")
        kept = {rid: r for rid, r in preds.items() if keep(r)}
        df = reviews[reviews.review_id.isin(kept)].copy()
        df["analysis"] = df.review_id.map(lambda rid, kept=kept: kept[rid].analysis)
        frames[split] = df
        report[split] = {"reviews": len(reviews), "predicted": len(preds), "kept": len(df)}

    train_sets = {"all": frames["silver_train"]}
    for size, subset in nested_subsets(frames["silver_train"], ablation_sizes, seed).items():
        train_sets[f"n{size}"] = subset

    for name, df in {**train_sets, "val": frames["silver_val"]}.items():
        target = out_dir / ("val" if name == "val" else f"train_{name}")
        target.mkdir(parents=True, exist_ok=True)
        with (
            (target / "sft.jsonl").open("w", encoding="utf-8") as sft,
            (target / "bert.jsonl").open("w", encoding="utf-8") as bert,
        ):
            for row in df.itertuples():
                example = sft_example(instruction, row.text, row.analysis)
                sft.write(json.dumps(example, ensure_ascii=False) + "\n")
                labels = encode(row.analysis.pairs())
                bert.write(
                    json.dumps({"review_id": row.review_id, "text": row.text, "labels": labels})
                    + "\n"
                )
        report.setdefault("files", {})[name] = len(df)
    (out_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
