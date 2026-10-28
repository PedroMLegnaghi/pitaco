"""YAML config loading with strict validation.

Every stage reads one YAML file. A stage file can start with `extends: base.yaml` to inherit the
shared settings; keys in the stage file override the inherited ones. `extra="forbid"` turns a typo
such as `sede: 42` into an immediate error instead of a silently ignored setting.
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Paths(StrictModel):
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")


class WandbConfig(StrictModel):
    project: str = "pitaco"
    entity: str | None = None
    mode: Literal["online", "offline", "disabled"] = "online"


class Budget(StrictModel):
    # The whole project has US$5; the config may never allow more than that.
    max_usd: float = Field(gt=0, le=5.0)


class BaseConfig(StrictModel):
    project: str
    seed: int = 42
    paths: Paths = Paths()
    wandb: WandbConfig = WandbConfig()
    budget: Budget


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base` without mutating either."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file, resolving a top-level `extends:` relative to the file's directory."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = data.pop("extends", None)
    if parent is None:
        return data
    return deep_merge(load_yaml(path.parent / parent), data)


def load_config[ConfigT: BaseModel](path: str | Path, model: type[ConfigT] = BaseConfig) -> ConfigT:
    return model.model_validate(load_yaml(path))
