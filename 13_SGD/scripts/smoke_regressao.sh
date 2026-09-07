#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/01_regressao_gd_manual.py --samples 30 --steps 12 --output-dir outputs/smoke/01
uv run python scripts/02_regressao_comparar_learning_rates.py --samples 30 --steps 12 --output-dir outputs/smoke/02
uv run python scripts/03_ruido_gradiente_por_batch_size.py --samples 30 --draws 40 --batch-sizes 1 8 30 --output-dir outputs/smoke/03
