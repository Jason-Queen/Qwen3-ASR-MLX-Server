PYTHON_VERSION ?= 3.11
PORT ?= 8989
ROLE ?=

.PHONY: sync test run smoke

sync:
	uv sync --python $(PYTHON_VERSION)

test:
	uv run python -m unittest discover -s tests

run:
	env QWEN_MLX_PORT=$(PORT) uv run python qwen3_asr_mlx_server.py

smoke:
	env QWEN_MLX_SMOKE_ROLE=$(ROLE) ./scripts/smoke_local_api.sh $(PORT)
