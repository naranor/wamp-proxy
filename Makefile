.PHONY: run test bench lint format clean help

PYTHONPATH := src:.

help:
	@echo "WAMP-proxy development commands:"
	@echo "  make run      - Run the proxy server"
	@echo "  make test     - Run unit and API tests"
	@echo "  make bench    - Run all efficiency benchmarks"
	@echo "  make calibrate - Run filter threshold calibration"
	@echo "  make lint     - Check code style with Ruff"
	@echo "  make format   - Format code with Ruff"
	@echo "  make clean    - Remove cache and temporary files"

run:
	PYTHONPATH=$(PYTHONPATH) python main.py

test:
	PYTHONPATH=$(PYTHONPATH) pytest

bench:
	PYTHONPATH=$(PYTHONPATH) python benchmarks/needle_test.py
	PYTHONPATH=$(PYTHONPATH) python benchmarks/multi_doc_qa.py
	PYTHONPATH=$(PYTHONPATH) python benchmarks/coherence_test.py

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
