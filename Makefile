
setup:
	python3 -m venv .venv && \
	source .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt

format:
	black . --line-length 99

sort: 
	isort .

clean: 
	rm -rf .venv