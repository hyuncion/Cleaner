.PHONY: install dev test demo labels pipeline clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest -q

demo:
	python scripts/create_demo_photos.py --out demo_photos --count 120

labels:
	python scripts/create_demo_labels.py --photo-dir demo_photos --max 80

pipeline:
	python scripts/run_pipeline.py --photo-dir demo_photos --max-embeddings 120 --top-n 20

clean:
	rm -rf cleaner_data demo_photos .pytest_cache .ruff_cache src/*.egg-info
