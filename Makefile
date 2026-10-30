.PHONY: install data ingest run test lint eval-retrieval eval-generation docker-build docker-run

install:
	pip install -r requirements-dev.txt

data:
	python -m scripts.generate_catalog
	python -m scripts.build_eval_set

ingest:
	python -m scripts.ingest

run:
	uvicorn app.main:create_app --factory --reload --port 8000

test:
	pytest

lint:
	ruff check . && ruff format --check .

eval-retrieval:
	python -m eval.run_eval retrieval

eval-generation:
	python -m eval.run_eval generation --limit 40

docker-build:
	docker build -t retail-rag-agent:local .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env retail-rag-agent:local
