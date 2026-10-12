PY ?= python

.PHONY: install run data prep forecast policy optimize test app eda

install:
	$(PY) -m pip install -r requirements.txt

run:            ## full pipeline: data -> clean -> forecast -> policy -> optimise
	$(PY) run_pipeline.py

data:
	$(PY) data/generate.py
prep:
	$(PY) src/prep.py
forecast:
	$(PY) src/forecast.py
policy:
	$(PY) src/policy.py
optimize:
	$(PY) src/optimizer.py

test:
	$(PY) -m pytest -q

eda:            ## re-execute the EDA notebook in place
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

app:
	streamlit run app.py
