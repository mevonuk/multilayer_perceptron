VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

OPTIMIZER ?= gd
PROGRAM_MODE ?= all

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

activate:
	@echo "Run:"
	@echo "source $(VENV)/bin/activate"

help:
	@echo "Run:"
	@echo "optimizer choices: gd, nest, rms, adam"
	@echo "program_mode choices: all, pre_process, train, predict"
	@echo "make run PROGRAM_MODE=program_mode OPTIMIZER=optimizer"
	@echo "note that additional options are available from the command line"

plot:
	$(PYTHON) plotter.py

run:
	$(PYTHON) tester.py --program_mode $(PROGRAM_MODE) --optimizer $(OPTIMIZER)

preprocess:
	$(PYTHON) tester.py --program_mode pre_process --optimizer $(OPTIMIZER)

train:
	$(PYTHON) tester.py --program_mode train --optimizer $(OPTIMIZER)

predict:
	$(PYTHON) tester.py --program_mode predict --optimizer $(OPTIMIZER)

compare:
	$(PYTHON) tester.py --optimizer compare

clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pkl" -delete

deactivate:
	@echo "Run:"
	@echo "deactivate"

.PHONY: clean, activate, deactivate, run, preprocess, train, predict, plot, help
