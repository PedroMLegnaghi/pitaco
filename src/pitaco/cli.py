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
from dotenv import load_dotenv

from pitaco import __version__
from pitaco.config import load_config

app = typer.Typer(no_args_is_help=True, add_completion=False)

REQUIRED_KEYS = ("OPENROUTER_API_KEY", "HF_TOKEN", "WANDB_API_KEY")


@app.callback()
def main() -> None:
    """Pitaco pipeline: data → labeling → training → evaluation → publishing."""
    load_dotenv()


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


if __name__ == "__main__":
    app()
