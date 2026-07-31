.PHONY: install test smoke docker-build

install:
	python -m pip install -e ".[analysis,test]"

test:
	pytest -q

smoke:
	python examples/rerank_candidates.py

docker-build:
	docker compose --env-file .env -f docker/compose.yaml build
