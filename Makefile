# Makefile for The Culinary Fingerprint
PYTHON ?= .venv/Scripts/python.exe

.PHONY: install preprocess d1 benchmark d2 d3 d4 poster test validate all clean

install:
	python -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

preprocess:
	$(PYTHON) src/preprocess.py
	$(PYTHON) src/split_data.py

d1: preprocess
	$(PYTHON) src/build_networks.py

benchmark: preprocess
	$(PYTHON) src/null_model.py --check --benchmark

d2: d1 benchmark
	$(PYTHON) src/backbone.py
	$(PYTHON) src/visualize.py

d3: d2
	$(PYTHON) src/analysis.py
	$(PYTHON) src/report.py

d4:
	$(PYTHON) evaluate.py

poster:
	$(PYTHON) src/poster.py

test:
	$(PYTHON) -m pytest -v

validate:
	$(PYTHON) run_all.py --validate-only

all:
	$(PYTHON) run_all.py
