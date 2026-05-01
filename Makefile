# PolyFuzz top-level Makefile.
#
# This is the artefact's reproduce-results entry point. Each target
# below corresponds to a claim in the report. After running, the
# console output and JSON files in results/ should match the numbers
# quoted in the writeup (modulo trial-level RNG variance).
#
# Targets:
#   make build       Compile the mock C target with gcov + UBSan.
#   make install     pip install the polyfuzz package in editable mode.
#   make test        Run the unit-test suite (pytest).
#   make demo        20-second fuzz against the mock target.
#   make eval        Multi-trial evaluation (3 trials × 25s × 3 variants).
#   make all         build + install + test + demo + eval.
#   make clean       Remove build artefacts and result directories.
#
# Variables:
#   TRIALS=N         Number of trials for `make eval`. Default 3.
#   BUDGET_SEC=N     Per-trial seconds for `make eval`. Default 25.
#   PYTHON=python    Python executable to use.

PYTHON      ?= python
TRIALS      ?= 3
BUDGET_SEC  ?= 25

.PHONY: all build install test demo eval clean

all: build install test demo eval

build:
	$(MAKE) -C target

install: build
	pip install --break-system-packages -e .

test:
	$(PYTHON) -m pytest tests/ -v

demo: build
	$(PYTHON) scripts/run_mock.py --budget-sec 20 \
		--output-dir results/demo

eval: build
	$(PYTHON) scripts/multi_trial.py \
		--trials $(TRIALS) \
		--budget-sec $(BUDGET_SEC) \
		--output-dir results/multi_trial

clean:
	$(MAKE) -C target clean
	rm -rf results/ build/ dist/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
