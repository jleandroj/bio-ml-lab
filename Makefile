.PHONY: install lint format type test check clean docker-build docker-test

IMAGE ?= biomllab:dev

install:  ## Create .venv and install package + dev tools (editable)
	uv venv
	uv pip install -e ".[dev]"

lint:  ## Lint with ruff
	uv run ruff check .

format:  ## Auto-format with ruff
	uv run ruff format .

type:  ## Type-check with mypy (strict)
	uv run mypy

test:  ## Run the test suite
	uv run pytest

check: lint type test  ## Run lint + type + tests (what CI runs)

docker-build:  ## Build the reproducible Docker image
	docker build -t $(IMAGE) .

docker-test: docker-build  ## Run the test suite inside the container
	docker run --rm $(IMAGE)

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info src/*.egg-info
