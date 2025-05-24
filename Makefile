.PHONY: help install install-dev test test-fast test-slow test-cov test-watch lint format type-check security clean run run-api run-both docker-build docker-run

# Default target
help:
	@echo "Available targets:"
	@echo "  install       - Install production dependencies"
	@echo "  install-dev   - Install all dependencies including dev"
	@echo "  test          - Run all tests"
	@echo "  test-fast     - Run tests excluding slow ones"
	@echo "  test-slow     - Run only slow tests"
	@echo "  test-cov      - Run tests with coverage report"
	@echo "  test-watch    - Run tests in watch mode"
	@echo "  lint          - Run all linters"
	@echo "  format        - Format code with black and isort"
	@echo "  type-check    - Run mypy type checking"
	@echo "  security      - Run security checks"
	@echo "  clean         - Clean up generated files"
	@echo "  run           - Run the application (Gradio)"
	@echo "  run-api       - Run REST API only"
	@echo "  run-both      - Run both Gradio and API"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-run    - Run Docker container"

# Installation targets
install:
	uv sync --no-dev

install-dev:
	uv sync --all-extras

# Testing targets
test:
	uv run pytest

test-fast:
	uv run pytest -m "not slow"

test-slow:
	uv run pytest -m "slow"

test-cov:
	uv run pytest --cov=windsurf_dreambooth --cov-report=html --cov-report=term

test-watch:
	uv run ptw -- --testmon

test-parallel:
	uv run pytest -n auto

test-unit:
	uv run pytest -m "unit"

test-integration:
	uv run pytest -m "integration"

test-benchmark:
	uv run pytest --benchmark-only

test-profile:
	uv run pytest --profile

# Code quality targets
lint:
	uv run flake8 windsurf_dreambooth tests
	uv run pylint windsurf_dreambooth
	uv run mypy windsurf_dreambooth

lint-fix:
	uv run black windsurf_dreambooth tests
	uv run isort windsurf_dreambooth tests
	uv run flake8 windsurf_dreambooth tests --extend-ignore=E203,W503

format:
	uv run black windsurf_dreambooth tests
	uv run isort windsurf_dreambooth tests

type-check:
	uv run mypy windsurf_dreambooth --install-types --non-interactive

# Security targets
security:
	uv run bandit -r windsurf_dreambooth
	uv run safety check
	uv run pip-audit

# Development targets
run:
	uv run python main.py

run-api:
	uv run python main.py --mode api

run-both:
	uv run python main.py --mode both

run-dev:
	GRADIO_RELOAD=1 uv run python main.py

# Docker targets
docker-build:
	docker build -t windsurf-dreambooth:latest .

docker-run:
	docker run -it --rm \
		--gpus all \
		-p 8000:8000 \
		-v $(PWD)/uploads:/app/uploads \
		-v $(PWD)/outputs:/app/outputs \
		-v $(PWD)/finetuned_models:/app/finetuned_models \
		windsurf-dreambooth:latest

# Clean targets
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf htmlcov .coverage coverage.xml
	rm -rf build dist

# Documentation targets
docs:
	uv run sphinx-build -b html docs docs/_build/html

docs-serve:
	cd docs/_build/html && python -m http.server

# Pre-commit hooks
pre-commit:
	uv run pre-commit install

pre-commit-run:
	uv run pre-commit run --all-files

# Profiling targets
profile-memory:
	uv run mprof run python main.py
	uv run mprof plot

profile-cpu:
	uv run py-spy record -o profile.svg -- python main.py

# Database migrations (if needed in future)
db-upgrade:
	echo "No database migrations yet"

db-downgrade:
	echo "No database migrations yet"