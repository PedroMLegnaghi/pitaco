from pathlib import Path

import pytest
from pydantic import ValidationError

from pitaco.config import BaseConfig, deep_merge, load_config

BASE = Path(__file__).parents[1] / "configs" / "base.yaml"


def test_base_config_loads():
    cfg = load_config(BASE)
    assert isinstance(cfg, BaseConfig)
    assert cfg.seed == 42
    assert cfg.budget.max_usd <= 5.0


def test_typo_in_config_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("project: pitaco\nsede: 42\nbudget: {max_usd: 1.0}\n")
    with pytest.raises(ValidationError, match="sede"):
        load_config(bad)


def test_budget_above_project_total_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("project: pitaco\nbudget: {max_usd: 50}\n")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_extends_inherits_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text(BASE.read_text())
    stage = tmp_path / "stage.yaml"
    stage.write_text("extends: base.yaml\nseed: 7\nwandb: {mode: offline}\n")
    cfg = load_config(stage)
    assert cfg.seed == 7
    assert cfg.wandb.mode == "offline"
    assert cfg.wandb.project == "pitaco"  # inherited, not wiped by the partial override


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1, "c": 2}}
    merged = deep_merge(base, {"a": {"b": 9}})
    assert merged == {"a": {"b": 9, "c": 2}}
    assert base == {"a": {"b": 1, "c": 2}}
