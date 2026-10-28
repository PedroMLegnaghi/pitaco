.PHONY: setup lint format test check env data annotate-dev annotate-test

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
