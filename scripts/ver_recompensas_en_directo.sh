#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

export PYTHONPATH="${PWD}"
exec "${PWD}/.venv/bin/python" \
  "${PWD}/scripts/ver_recompensas_en_directo.py" "$@"
