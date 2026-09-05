#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PLAYGROUND_COMMIT="9c2dce4a3519cd4bb9d299bf28a6ef3f5086844b"
UV_VERSION="0.11.8"
# shellcheck source=platform.sh
source "${SCRIPT_DIR}/platform.sh"
export PATH="${HOME}/.local/bin:${PATH}"

failures=0
ok() { printf '[OK]   %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }
fail() { printf '[FAIL] %s\n' "$*"; failures=$((failures + 1)); }

printf '%s\n' 'TARANTULIN doctor - Windows/WSL'
printf '%s\n' '================================'
printf 'Workspace: %s\n' "${REPO_ROOT}"

if tarantulin_is_wsl; then ok "Ejecucion dentro de WSL"; else fail "No se detecta WSL"; fi
if tarantulin_path_is_windows_mount "$(realpath "${REPO_ROOT}")"; then
  fail "El runtime esta sobre un filesystem Windows; debe estar en ext4 de WSL"
else
  ok "Runtime en filesystem Linux/ext4"
fi
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  printf 'Sistema: %s\n' "${PRETTY_NAME:-desconocido}"
  [[ "${ID:-}" == ubuntu ]] && ok "Distribucion Ubuntu" || warn "Distribucion no validada para este repositorio"
fi

requested="$(tarantulin_requested_accelerator 2>/dev/null)" || requested="invalido"
resolved="$(tarantulin_resolve_accelerator 2>/dev/null)" || resolved="invalido"
printf 'Acelerador solicitado: %s\n' "${requested}"
printf 'Acelerador resuelto  : %s\n' "${resolved}"
if tarantulin_accelerator_preflight "${resolved}"; then ok "Preflight del acelerador"; else fail "Preflight del acelerador"; fi

for command in bash git curl python3 rsync; do
  if command -v "${command}" >/dev/null 2>&1; then ok "Comando ${command}"; else fail "Falta ${command}"; fi
done
if command -v uv >/dev/null 2>&1; then
  uv_actual="$(uv --version 2>/dev/null || true)"
  uv_actual_version="$(awk '{print $2}' <<< "${uv_actual}")"
  if [[ "${uv_actual_version}" == "${UV_VERSION}" ]]; then
    ok "${uv_actual} (version fijada)"
  else
    fail "Se esperaba uv ${UV_VERSION}; se detecto '${uv_actual}'"
  fi
else
  fail "Falta uv; ejecuta .\\install.ps1"
fi

if tarantulin_has_nvidia; then
  ok "NVIDIA visible desde WSL"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/       /'
else
  warn "nvidia-smi no expone GPU"
fi
if tarantulin_has_amd_rocm; then ok "AMD ROCm visible (experimental en WSL)"; else warn "rocminfo no expone GPU AMD"; fi

if [[ -f "${REPO_ROOT}/pyproject.toml" && -f "${REPO_ROOT}/uv.lock" ]] && grep -q "${PLAYGROUND_COMMIT}" "${REPO_ROOT}/uv.lock"; then
  ok "MuJoCo Playground fijado en uv.lock (${PLAYGROUND_COMMIT:0:12})"
else
  fail "pyproject.toml/uv.lock no fijan el commit esperado de MuJoCo Playground"
fi

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  ok "Entorno Python presente"
  export PYTHONPATH="${REPO_ROOT}"
  export TARANTULIN_RESOLVED_BACKEND_PROFILE="${resolved}"
  if [[ "${resolved}" == cpu ]]; then export JAX_PLATFORM_NAME=cpu; fi
  "${VENV_DIR}/bin/python" - <<'PY'
from __future__ import annotations

import sys

try:
  import jax
  import mujoco
  from mujoco import mjx
  from tarantulin.entorno_tarantulin_mjx import TarantulinIncorporarse
except Exception as exc:
  print(f"[FAIL] Imports Python/MJX: {type(exc).__name__}: {exc}")
  raise SystemExit(1)

print(f"[OK]   Python {sys.version.split()[0]}")
print(f"[OK]   JAX {jax.__version__}; backend={jax.default_backend()}; devices={jax.devices()}")
print(f"[OK]   MuJoCo {mujoco.__version__}; mjx={mjx.__name__}; env={TarantulinIncorporarse.__name__}")
expected = __import__("os").environ.get("TARANTULIN_RESOLVED_BACKEND_PROFILE", "cpu")
backend = jax.default_backend()
if expected in {"nvidia", "amd"} and backend != "gpu":
  print(f"[FAIL] Se esperaba backend GPU para {expected}, se obtuvo {backend}")
  raise SystemExit(2)
if expected == "cpu" and backend != "cpu":
  print(f"[FAIL] Se esperaba backend CPU, se obtuvo {backend}")
  raise SystemExit(2)
PY
  status=$?
  if (( status == 0 )); then ok "Prueba de imports y backend"; else fail "Prueba de imports y backend"; fi
else
  fail "Entorno Python ausente; ejecuta .\\install.ps1"
fi

printf '%s\n' '--------------------------------'
if (( failures > 0 )); then
  printf 'Diagnostico terminado con %d fallo(s).\n' "${failures}"
  exit 1
fi
printf '%s\n' 'Diagnostico correcto.'
