# Induction engine — one-command targets.
# The baseline is pure stdlib; `pytest` is the only dev dependency.

PY ?= python3
FLASK_DIR = data/corpus/flask
CLICK_DIR = data/corpus/click

.PHONY: run heldout test clean deps

## run: clone (if needed) + ingest + induce pallets/flask, then emit out/
run:
	@test -d $(FLASK_DIR)/.git || GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1000 https://github.com/pallets/flask $(FLASK_DIR)
	$(PY) ingest.py --repo-path $(FLASK_DIR) --slug pallets/flask
	$(PY) run.py --slug pallets/flask
	@echo "→ open out/inspector.html"

## heldout: same, for the unseen repo pallets/click (generalisation check)
heldout:
	@test -d $(CLICK_DIR)/.git || GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1000 https://github.com/pallets/click $(CLICK_DIR)
	$(PY) ingest.py --repo-path $(CLICK_DIR) --slug pallets/click
	$(PY) run.py --slug pallets/click

## test: run the suite (held-out tests activate once click is cached)
test:
	$(PY) -m pytest -q

## deps: install the only dev dependency
deps:
	$(PY) -m pip install pytest

## clean: remove generated output (keeps the cached corpus)
clean:
	rm -rf out
