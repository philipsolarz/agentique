.PHONY: lint type test check console

lint:
	uv run ruff check .
	uv run ruff format --check .

type:
	uv run ty check

test:
	uv run pytest

check: lint type test

# Dev-only: the console REPL with full run capture (writes runs/<ts>-console/).
console:
	uv run python -m scenarios.console
