.PHONY: install run test eval eval-llm docker data

install:
	pip install -r requirements-dev.txt

run:
	streamlit run app.py

test:
	python -m pytest

eval:
	python evals/run_evals.py

eval-llm:
	python evals/run_evals.py --engine llm

data:
	python scripts/generate_sample_data.py

docker:
	docker compose up --build
