
setup:
	python3 -m venv .venv && \
	source .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt

format:
	black parser.py --line-length 99

sort: 
	isort .

test: 
	pytest

clean: 
	rm -rf .venv