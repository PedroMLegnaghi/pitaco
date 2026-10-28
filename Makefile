.PHONY: setup lint format test check env

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
