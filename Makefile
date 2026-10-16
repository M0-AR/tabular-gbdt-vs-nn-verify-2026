.PHONY: setup fast full test docker-fast docker-full
setup:
	pip install -r requirements.txt
fast:
	python3 src/experiments.py --fast
full:
	python3 src/experiments.py --full
test:
	python3 -m pytest tests/ -v
docker-fast:
	docker compose run --rm bench-fast
docker-full:
	docker compose run --rm bench
