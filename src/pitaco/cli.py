"""Command-line entry point: `pitaco <stage> --config configs/<stage>.yaml`.

Each module adds its stage here; the Makefile chains them into the full pipeline.
"""

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from dotenv import find_dotenv, load_dotenv

from pitaco import __version__
from pitaco.config import load_config

app = typer.Typer(no_args_is_help=True, add_completion=False)

REQUIRED_KEYS = ("OPENROUTER_API_KEY", "HF_TOKEN", "WANDB_API_KEY")


@app.callback()
def main() -> None:
    """Pitaco pipeline: data → labeling → training → evaluation → publishing."""
    load_dotenv(find_dotenv(usecwd=True))


def _gpu_summary() -> str:
    if shutil.which("nvidia-smi") is None:
        return "no NVIDIA driver found"
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "nvidia-smi returned nothing"


@app.command("check-env")
def check_env(
    config: Annotated[Path, typer.Option(help="Config file")] = Path("configs/base.yaml"),
) -> None:
    """Validate the config and report which credentials are set (values are never printed)."""
    cfg = load_config(config)
    typer.echo(f"pitaco {__version__} · Python {platform.python_version()}")
    typer.echo(f"config  {config} OK (seed={cfg.seed}, budget=US${cfg.budget.max_usd:.2f})")
    typer.echo(f"gpu     {_gpu_summary()}")

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    kaggle_ok = kaggle_json.exists() or bool(
        os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")
    )
    keys = {key: bool(os.getenv(key)) for key in REQUIRED_KEYS} | {"KAGGLE": kaggle_ok}
    for key, present in keys.items():
        typer.echo(f"{'set    ' if present else 'MISSING'} {key}")
    if not all(keys.values()):
        raise typer.Exit(code=1)


@app.command("data")
def data(
    config: Annotated[Path, typer.Option(help="Config file")] = Path("configs/data.yaml"),
) -> None:
    """Download, clean, scrub PII, deduplicate and freeze the gold/silver splits."""
    from pitaco.data.pipeline import DataConfig, run

    manifest = run(load_config(config, DataConfig))
    stats = manifest["stats"]
    typer.echo(
        f"raw {stats['raw_rows']} → with text {stats['with_text']} → unique {stats['unique_texts']}"
    )
    typer.echo(f"PII replaced in {stats['reviews_with_pii']} reviews: {stats['pii_replacements']}")
    removed = manifest["near_duplicates_removed_from_pool"]
    typer.echo(f"near-duplicates of gold removed from pool: {removed}")
    typer.echo(f"splits: {manifest['splits']}")


@app.command("annotate")
def annotate(
    split: Annotated[str, typer.Option(help="gold split to label: dev or test")] = "dev",
    annotator: Annotated[str, typer.Option(help="who is labeling")] = "pedro",
    port: Annotated[int, typer.Option(help="local port")] = 7860,
    config: Annotated[Path, typer.Option(help="Config file")] = Path("configs/base.yaml"),
) -> None:
    """Launch the local, blind labeling app (needs the `app` dependency group)."""
    from pitaco.labeling.annotate_app import build_app

    if split not in {"dev", "test"}:
        raise typer.BadParameter("split must be dev or test")
    cfg = load_config(config)
    reviews = cfg.paths.data_dir / "processed" / f"gold_{split}.jsonl"
    labels = cfg.paths.data_dir / "annotations" / f"gold_{split}.jsonl"
    typer.echo(f"labeling {reviews} → {labels}")
    build_app(reviews, labels, cfg.seed, annotator).launch(
        server_name="127.0.0.1", server_port=port, share=False, inbrowser=True
    )


@app.command("label")
def label(
    model: Annotated[str, typer.Option(help="model key from configs/teacher.yaml")],
    split: Annotated[str, typer.Option(help="gold_dev | gold_test | silver_train | silver_val")],
    mode: Annotated[str, typer.Option(help="fewshot | zeroshot")] = "fewshot",
    limit: Annotated[int | None, typer.Option(help="only the first N reviews")] = None,
    prompt_version: Annotated[str | None, typer.Option(help="override config prompt")] = None,
    execute: Annotated[bool, typer.Option("--execute", help="actually call the API")] = False,
    config: Annotated[Path, typer.Option(help="Config file")] = Path("configs/teacher.yaml"),
) -> None:
    """Label a split with a teacher LLM. Without --execute: cost estimate only, no API call."""
    import json

    from pitaco.labeling import teacher

    if mode not in {"fewshot", "zeroshot"}:
        raise typer.BadParameter("mode must be fewshot or zeroshot")
    cfg = load_config(config, teacher.TeacherConfig)
    if prompt_version:
        cfg = cfg.model_copy(update={"prompt_version": prompt_version})
    if model not in cfg.models:
        raise typer.BadParameter(f"unknown model; choose from {sorted(cfg.models)}")
    teacher.check_fewshot_not_in_gold(cfg)
    typer.echo(json.dumps(teacher.estimate(cfg, model, mode, split, limit), indent=2))
    if not execute:
        typer.echo("dry run — nothing was sent. Re-run with --execute to call the API.")
        return
    typer.echo(json.dumps(teacher.run(cfg, model, mode, split, limit), indent=2))


@app.command("reparse")
def reparse_cmd(
    system: Annotated[str, typer.Option(help="system folder under results/predictions")],
    split: Annotated[str, typer.Option(help="split file to re-parse")],
    config: Annotated[Path, typer.Option(help="Config file")] = Path("configs/base.yaml"),
) -> None:
    """Re-apply the current parser to stored raw outputs (no model call)."""
    import pandas as pd

    from pitaco.predictions import reparse, write_predictions

    cfg = load_config(config)
    reviews = pd.read_json(cfg.paths.data_dir / "processed" / f"{split}.jsonl", lines=True)
    path = cfg.paths.results_dir / "predictions" / system / f"{split}.jsonl"
    records = reparse(path, dict(zip(reviews.review_id, reviews.text, strict=True)))
    write_predictions(path, records)
    typer.echo(
        f"{system}/{split}: {len(records)} re-parsed, {sum(r.repaired for r in records)} repaired"
    )


@app.command("evaluate")
def evaluate_cmd(
    predictions: Annotated[Path, typer.Option(help="predictions JSONL")],
    split: Annotated[str, typer.Option(help="gold_dev | gold_test")],
    system: Annotated[str | None, typer.Option(help="name in the results table")] = None,
    n_boot: Annotated[int, typer.Option(help="bootstrap resamples")] = 1000,
    config: Annotated[Path, typer.Option(help="Config file")] = Path("configs/base.yaml"),
) -> None:
    """Score predictions against the gold labels (writes results/metrics/<system>__<split>.json)."""
    from pitaco.eval.evaluate import run_evaluation

    cfg = load_config(config)
    name = system or predictions.parent.name
    r = run_evaluation(
        predictions,
        cfg.paths.data_dir / "annotations",
        cfg.paths.results_dir,
        split,
        name,
        n_boot,
        cfg.seed,
    )
    ci = r["ci95"]["aspect_macro_f1"]
    typer.echo(
        f"{name} on {split}: valid {r['valid_schema_rate']:.1%} · aspect macro-F1 "
        f"{r['aspect_macro_f1']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}] · "
        f"joint macro-F1 {r['joint_macro_f1']:.3f} · kappa {r['kappa_mean']:.3f}"
    )


@app.command("compare")
def compare_cmd(
    a: Annotated[Path, typer.Option(help="predictions JSONL of system A")],
    b: Annotated[Path, typer.Option(help="predictions JSONL of system B")],
    split: Annotated[str, typer.Option(help="gold_dev | gold_test")],
    metric: Annotated[str, typer.Option(help="metric name")] = "joint_macro_f1",
    config: Annotated[Path, typer.Option(help="Config file")] = Path("configs/base.yaml"),
) -> None:
    """Paired bootstrap: is A really better than B on the same reviews?"""
    import json

    from pitaco.eval.evaluate import compare

    cfg = load_config(config)
    typer.echo(
        json.dumps(compare(a, b, cfg.paths.data_dir / "annotations", split, metric), indent=2)
    )


@app.command("report")
def report_cmd(
    split: Annotated[str, typer.Option(help="gold_dev | gold_test")] = "gold_test",
    config: Annotated[Path, typer.Option(help="Config file")] = Path("configs/base.yaml"),
) -> None:
    """Build results/comparison_<split>.md from all evaluated systems."""
    from pitaco.eval.evaluate import report

    typer.echo(report(load_config(config).paths.results_dir, split))


@app.command("build-training-data")
def build_training_data(
    config: Annotated[Path, typer.Option(help="Config file")] = Path("configs/training_data.yaml"),
) -> None:
    """Filter teacher silver labels and write SFT + BERT training files (incl. ablation subsets)."""
    import json

    from pitaco.config import BaseConfig
    from pitaco.models.training_data import build

    class TrainingDataConfig(BaseConfig):
        teacher_system: str
        prompts_dir: Path
        student_prompt_version: str
        ablation_sizes: list[int]
        out_dir: Path

    cfg = load_config(config, TrainingDataConfig)
    report = build(
        cfg.paths.results_dir / "predictions" / cfg.teacher_system,
        cfg.paths.data_dir / "processed",
        cfg.out_dir,
        cfg.prompts_dir,
        cfg.student_prompt_version,
        cfg.ablation_sizes,
        cfg.seed,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("train-bert")
def train_bert(
    config: Annotated[Path, typer.Option(help="Config file")] = Path("configs/bertimbau.yaml"),
) -> None:
    """Train the BERTimbau baseline and predict gold dev/test (needs the `train` group)."""
    from pitaco.models import bertimbau

    typer.echo(bertimbau.run(load_config(config, bertimbau.BertConfig)))


@app.command("train-student")
def train_student(
    config: Annotated[Path, typer.Option(help="Config file")],
) -> None:
    """LoRA fine-tuning of the student LLM (GPU; needs the `train` group)."""
    from pitaco.models import sft

    typer.echo(sft.train(load_config(config, sft.StudentConfig)))


@app.command("predict-student")
def predict_student(
    config: Annotated[Path, typer.Option(help="Config file")],
    adapter: Annotated[
        Path | None, typer.Option(help="LoRA adapter dir; omit = base model")
    ] = None,
    limit: Annotated[int | None, typer.Option(help="smoke test: only N reviews per split")] = None,
) -> None:
    """Predict gold dev/test with the student (base model zero-shot if no adapter)."""
    from pitaco.models import sft

    cfg = load_config(config, sft.StudentConfig)
    system = cfg.system_name if adapter else f"{cfg.system_name}-base-zeroshot"
    typer.echo(sft.predict(cfg, adapter, system, limit))


if __name__ == "__main__":
    app()
