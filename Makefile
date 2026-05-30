.PHONY: lint type test check

lint:
	uv run ruff check .
	uv run ruff format --check .

type:
	uv run ty check

test:
	uv run pytest

check: lint type test
