.PHONY: install install-dev test lint format clean docs

install:
	pip install -e .

install-dev:
	pip install -e .[dev]

test:
	pytest tests/ -v

lint:
	flake8 torchslim tests examples
	mypy torchslim

format:
	black torchslim tests examples

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/ dist/ *.egg-info/

docs:
	cd docs && make html

demo:
	python examples/basic_usage.py

custom-demo:
	python examples/custom_method_example.py
