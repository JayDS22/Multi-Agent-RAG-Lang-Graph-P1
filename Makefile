# Makefile for Multi-Agent RAG System
# Author: Jay Guwalani

.PHONY: help install install-dev test test-verbose clean lint format type-check run-examples docker-build docker-run docs

# Default target
help:
	@echo "Multi-Agent RAG System - Available Commands:"
	@echo ""
	@echo "Installation:"
	@echo "  install          Install package and dependencies"
	@echo "  install-dev      Install package with development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  test             Run all tests"
	@echo "  test-verbose     Run tests with verbose output"
	@echo "  lint             Run code linting (flake8)"
	@echo "  format           Format code (black)"
	@echo "  type-check       Run type checking (mypy)"
	@echo ""
	@echo "Examples:"
	@echo "  run-examples     Run all example scripts"
	@echo "  run-research     Run research example only"
	@echo "  run-document     Run document example only"
	@echo "  run-workflow     Run full workflow example"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run Docker container"
	@echo ""
	@echo "Utilities:"
	@echo "  clean            Clean temporary files and caches"
	@echo "  docs             Generate documentation"

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e .[dev]

# Testing
test:
	pytest tests/ -v

test-verbose:
	pytest tests/ -v -s --tb=long

test-coverage:
	pytest tests/ --cov=multi_agent_rag --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 multi_agent_rag.py examples/ tests/ --max-line-length=100 --ignore=E203,W503

format:
	black multi_agent_rag.py examples/ tests/ --line-length=100

type-check:
	mypy multi_agent_rag.py --ignore-missing-imports

# Examples
run-examples: run-research run-document run-workflow

run-research:
	@echo "Running research example..."
	python examples/research_example.py

run-document:
	@echo "Running document example..."
	python examples/document_example.py

run-workflow:
	@echo "Running full workflow example..."
	python examples/full_workflow_example.py

# Docker
docker-build:
	docker build -t multi-agent-rag:latest .

docker-run:
	docker run -e OPENAI_API_KEY=${OPENAI_API_KEY} -e TAVILY_API_KEY=${TAVILY_API_KEY} multi-agent-rag:latest

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

# Utilities
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/

# Environment setup
setup-env:
	@echo "Setting up development environment..."
	python -m venv venv
	@echo "Activate with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"
	@echo "Then run: make install-dev"

# Documentation
docs:
	@echo "Documentation available in docs/ directory"
	@echo "Architecture: docs/architecture.md"
	@echo "API Reference: docs/api_reference.md"
	@echo "Deployment: docs/deployment_guide.md"

# Quality checks (run all)
check: lint type-check test
	@echo "All quality checks passed!"

# Release preparation
prepare-release: clean format lint type-check test
	@echo "Ready for release!"
	@echo "Version: $(shell python setup.py --version)"
	@echo "To release: git tag v$(shell python setup.py --version) && git push --tags"

# Performance testing
benchmark:
	@echo "Running performance benchmarks..."
	python -m pytest tests/ -v -k "performance" --benchmark-only

# Security checks
security:
	@echo "Running security checks..."
	pip-audit
	bandit -r multi_agent_rag.py

# Development workflow
dev: install-dev format lint type-check test
	@echo "Development workflow complete!"

# Production deployment checks
deploy-check: clean test security
	@echo "Deployment checks complete!"

# GitHub Actions simulation
ci: install-dev lint type-check test
	@echo "CI pipeline simulation complete!"
