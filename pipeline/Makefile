setup:
	python3 -m venv .venv && \
	source .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt

# Format code with black
format: black . --line-length 99

# Lint code with flake8
lint: flake8 .

# Sort imports with isort
sort: isort .

# Run tests with pytest
test: pytest

# Clean up the virtual environment
clean: rm -rf .venv