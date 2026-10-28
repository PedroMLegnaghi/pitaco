.PHONY: setup lint format test check env data

setup:            ## install the locked environment
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest

check: lint test  ## run before every commit

env:              ## validate config and credentials
	uv run pitaco check-env

data:             ## M1: raw Olist → clean, deduped, frozen splits
	uv run pitaco data --config configs/data.yaml
