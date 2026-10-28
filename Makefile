.PHONY: setup lint format test check env data annotate-dev annotate-test estimate-dev

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

annotate-dev:     ## M2: label the 50 dev reviews (local app, blind)
	uv run --group app pitaco annotate --split dev

annotate-test:    ## label the 250 test reviews — only after guidelines v1.1
	uv run --group app pitaco annotate --split test

estimate-dev:     ## M3: teacher cost estimates on gold dev (dry run — no API call)
	for m in deepseek-v4.1-flash deepseek-v4-pro qwen3.5-397b; do uv run pitaco label --model $$m --split gold_dev; done
