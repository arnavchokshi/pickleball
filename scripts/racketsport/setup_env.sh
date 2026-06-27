#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-racketsport.txt

mkdir -p models/checkpoints runs/phase0 data/testclips
echo "local Phase 0 environment ready: $ROOT/.venv"

