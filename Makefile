PY ?= python
CONFIG ?= config.yaml

.PHONY: install dev run data prep forecast policy optimize sensitivity monitor test lint format typecheck security check app eda profile scale-test docker

install:            ## runtime dependencies
	$(PY) -m pip install -r requirements.txt
dev:                ## runtime + dev tools + git hooks
	$(PY) -m pip install -r requirements-dev.txt && pre-commit install

run:                ## full batch pipeline: data -> clean -> forecast -> policy -> optimise
	$(PY) run_pipeline.py --config $(CONFIG)
data:
	$(PY) data/generate.py --config $(CONFIG)
prep:
	$(PY) src/prep.py --config $(CONFIG)
forecast:
	$(PY) src/forecast.py --config $(CONFIG)
policy:
	$(PY) src/policy.py --config $(CONFIG)
optimize:
	$(PY) src/optimizer.py --config $(CONFIG)
sensitivity:        ## re-solve under alternative assumptions -> reports/sensitivity.csv
	$(PY) scripts/sensitivity.py --config $(CONFIG)
monitor:            ## make monitor ACTUALS=new_sales.csv
	$(PY) -m src.monitoring --actuals $(ACTUALS) --config $(CONFIG)

test:
	$(PY) -m pytest --cov --cov-report=term
lint:
	ruff check .
format:
	black . && ruff check . --fix
typecheck:
	$(PY) -m mypy src
security:
	bandit -q -c pyproject.toml -r src data app.py run_pipeline.py scripts && pip-audit -r requirements.txt
check: lint typecheck test security   ## everything CI runs

app:
	streamlit run app.py
eda:                ## re-execute the EDA notebook in place
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
profile:            ## time and peak memory per stage
	$(PY) scripts/profile_pipeline.py --config $(CONFIG) --out reports/profile.csv
scale-test:         ## 10x network (250 branches, 3,000 SKUs)
	$(PY) scripts/profile_pipeline.py --config configs/scale_10x.yaml --out reports/profile_10x.csv
docker:
	docker build -t stockflow .
