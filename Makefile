# LangGraph Research Assistant - Makefile
# Common commands for development and deployment

.PHONY: help install run run-query test lint format docker-build docker-run docker-dev clean

# Default target
help:
	@echo "LangGraph Research Assistant - Available Commands"
	@echo "================================================"
	@echo ""
	@echo "Development:"
	@echo "  make install     - Install dependencies"
	@echo "  make run         - Run in interactive mode"
	@echo "  make run-query   - Run single query (use QUERY='...')"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run linting"
	@echo "  make format      - Format code"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run in Docker (interactive)"
	@echo "  make docker-dev   - Run in Docker (dev mode)"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean       - Clean temporary files"
	@echo ""

# Install dependencies
install:
	pip install -r requirements.txt

# Install dev dependencies
install-dev:
	pip install -r requirements.txt
	pip install black isort mypy pytest pytest-asyncio

# Run in interactive mode
run:
	python -m src.research_assistant.main

# Run with verbose logging
run-verbose:
	python -m src.research_assistant.main -v

# Run single query (usage: make run-query QUERY="Tell me about Apple")
run-query:
	python -m src.research_assistant.main -q "$(QUERY)"

# Run tests
test:
	pytest tests/ -v

# Run tests with coverage
test-coverage:
	pytest tests/ -v --cov=src/research_assistant --cov-report=html

# Lint code
lint:
	mypy src/research_assistant --ignore-missing-imports

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Build Docker image
docker-build:
	docker build -t research-assistant:latest .

# Run in Docker (interactive mode)
docker-run:
	docker-compose up research-assistant

# Run in Docker (dev mode with volume mounts)
docker-dev:
	docker-compose --profile dev up research-assistant-dev

# Stop Docker containers
docker-stop:
	docker-compose down

# Clean temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage 2>/dev/null || true

# Show project structure
tree:
	@find . -type f -name "*.py" | grep -v __pycache__ | grep -v .venv | sort
