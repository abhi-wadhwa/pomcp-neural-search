.PHONY: install dev test lint format typecheck run-ui demo docker clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/

run-ui:
	streamlit run src/viz/app.py

demo:
	python examples/demo.py

docker:
	docker build -t pomcp-neural-search .

docker-run:
	docker run -p 8501:8501 pomcp-neural-search

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
