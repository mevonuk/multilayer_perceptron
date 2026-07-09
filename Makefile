VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

OPTIMIZER ?= gd
PROGRAM_MODE ?= all
ACTIVATION ?= softmax

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

activate:
	@echo "Run:"
	@echo "source $(VENV)/bin/activate"

help:
	@echo "Running program using 'make run':"
	@echo "optimizer choices: gd, nest, rms, adam"
	@echo "program_mode choices: all, preprocess, train, predict"
	@echo "activation choces: sigmoid, softmax\n"
	@echo "make run PROGRAM_MODE=program_mode OPTIMIZER=optimizer ACTIVATION=activation"
	@echo "\nNote that additional options are available from the command line, see README."

plot:
	$(PYTHON) plotter.py

run:
	$(PYTHON) tester.py --program_mode $(PROGRAM_MODE) --optimizer $(OPTIMIZER) --activation $(ACTIVATION)

split:
	$(PYTHON) tester.py --program_mode split_preprocess --optimizer $(OPTIMIZER) --activation $(ACTIVATION)

preprocess:
	$(PYTHON) tester.py --program_mode preprocess --optimizer $(OPTIMIZER) --activation $(ACTIVATION)

train:
	$(PYTHON) tester.py --program_mode train --optimizer $(OPTIMIZER) --activation $(ACTIVATION)

predict:
	$(PYTHON) tester.py --program_mode predict --optimizer $(OPTIMIZER) --activation $(ACTIVATION)

compare:
	$(PYTHON) tester.py --optimizer compare --activation $(ACTIVATION)

deactivate:
	@echo "Run:"
	@echo "deactivate"

clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pkl" -delete
	find . -type f -name "*.pdf" -delete

.PHONY: clean, activate, deactivate, run, preprocess, train, predict, plot, help
