.PHONY: install train sample fid api test clean

install:
	pip install -r requirements.txt

train:
	python -m src.train --config configs/default.yaml

sample:
	python -m src.sample --n 64

fid:
	python -m src.fid --config configs/default.yaml

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

test:
	pytest -q tests/

clean:
	rm -rf artifacts/checkpoints artifacts/samples artifacts/*.gif
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
