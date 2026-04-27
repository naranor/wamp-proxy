.PHONY: run test bench lint format clean help

PYTHONPATH := src:.

help:
	@echo "WAMP-proxy development commands:"
	@echo "  make run      - Run the proxy server"
	@echo "  make test     - Run unit and API tests"
	@echo "  make bench    - Run final tri-modal validation"
	@echo "  make train    - Train router and needle classifiers"
	@echo "  make calibrate - Run filter threshold calibration"
	@echo "  make lint     - Check code style with Ruff"
	@echo "  make format   - Format code with Ruff"
	@echo "  make clean    - Remove cache and temporary files"

run:
	PYTHONPATH=$(PYTHONPATH) python main.py

train:
	PYTHONPATH=$(PYTHONPATH) python tools/train_massive.py

test:
	PYTHONPATH=$(PYTHONPATH) pytest

bench:
	PYTHONPATH=$(PYTHONPATH) python benchmarks/final_validation_no_proxy.py

calibrate:
	PYTHONPATH=$(PYTHONPATH) python benchmarks/calibration.py

lint:
	ruff check .

format:
	ruff format .

clean:
	rm -rf `find . -type d -name __pycache__`
	rm -rf .pytest_cache
	rm -rf .ruff_cache
