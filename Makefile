.PHONY: help install notebooks test

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create the uv venv with dev deps (marimo, pytest) and editable tspjax
	uv sync --extra dev

notebooks: install  ## Open the notebooks/ folder in marimo (right venv)
	uv run --extra dev marimo edit --watch notebooks

test: install  ## Run the test suite
	uv run --extra dev pytest -q
