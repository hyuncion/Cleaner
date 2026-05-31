.PHONY: install dev test demo run clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest -q

demo:
	python scripts/create_demo_photos.py --out demo_photos --count 120

run:
	streamlit run app.py

clean:
	rm -rf cleaner_data demo_photos .pytest_cache .ruff_cache src/*.egg-info
