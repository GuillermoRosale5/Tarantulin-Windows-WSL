#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOGS_DIR="${REPO_ROOT}/logs_tarantulin_mjx"
LAST_RUN="${LOGS_DIR}/ultima_run.txt"
XML_DIR="${REPO_ROOT}/tarantulin/xmls"

cd "${REPO_ROOT}"

python_viewer_env() {
  export PYTHONPATH="${REPO_ROOT}"
  export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.10}"
  export MUJOCO_GL="${MUJOCO_VIEWER_GL:-glfw}"
}

current_run() {
  if [[ -f "${LAST_RUN}" ]]; then
    local saved
    saved="$(<"${LAST_RUN}")"
    if [[ -n "${saved}" && -d "${saved}" ]]; then
      printf '%s\n' "${saved}"
      return
    fi
  fi
  find "${LOGS_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null |
    sort -nr |
    awk 'NR == 1 {print $2}' || true
}

select_from_list() {
  local prompt="$1"
  local count="$2"
  local selected
  read -r -p "${prompt}" selected
  if [[ ! "${selected}" =~ ^[0-9]+$ ]] || (( selected < 1 || selected > count )); then
    echo "Opcion no valida: ${selected}. Elige un numero entre 1 y ${count}." >&2
    exit 1
  fi
  printf '%s\n' "${selected}"
}

visualizar_ultimo_checkpoint() {
  echo "Abriendo ultimo checkpoint..."
  exec ./scripts/tarantulin_wsl.sh view-results "$@"
}

visualizar_checkpoint_anterior() {
  local run_dir
  run_dir="$(current_run)"
  if [[ -z "${run_dir}" ]]; then
    echo "No encuentro ninguna run en ${LOGS_DIR}." >&2
    exit 1
  fi

  mapfile -t checkpoints < <(
    find "${run_dir}/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%P\n' 2>/dev/null |
      awk '/^[0-9]+$/' |
      sort -nr
  )

  if (( ${#checkpoints[@]} <= 1 )); then
    echo "No hay checkpoints anteriores en la run actual: ${run_dir}" >&2
    echo "Run encontrada, pero checkpoints disponibles: ${#checkpoints[@]}" >&2
    exit 1
  fi

  echo ""
  echo "Run actual:"
  echo "  ${run_dir}"
  echo ""
  echo "Checkpoints anteriores:"
  local i step ckpt_path
  for (( i = 1; i < ${#checkpoints[@]}; i++ )); do
    step="${checkpoints[$i]}"
    ckpt_path="${run_dir}/checkpoints/${step}"
    printf '  %2d) step %-12s  %s\n' "${i}" "${step}" "$(date -d "@$(stat -c %Y "${ckpt_path}")" '+%Y-%m-%d %H:%M:%S')"
  done

  local choice
  local max_choice="$(( ${#checkpoints[@]} - 1 ))"
  choice="$(select_from_list "Elige checkpoint anterior [1-${max_choice}]: " "${max_choice}")"
  ckpt_path="${run_dir}/checkpoints/${checkpoints[$choice]}"
  echo "Abriendo checkpoint: ${ckpt_path}"
  exec ./scripts/tarantulin_wsl.sh view-results --checkpoint-path "${ckpt_path}" "$@"
}

visualizar_xml() {
  mapfile -t xmls < <(find "${XML_DIR}" -maxdepth 1 -type f -name '*.xml' -printf '%f\n' | sort)
  if (( ${#xmls[@]} == 0 )); then
    echo "No encuentro XMLs en ${XML_DIR}." >&2
    exit 1
  fi

  echo ""
  echo "XMLs disponibles:"
  local i xml_path
  for (( i = 0; i < ${#xmls[@]}; i++ )); do
    xml_path="${XML_DIR}/${xmls[$i]}"
    printf '  %2d) %-36s  %s\n' "$(( i + 1 ))" "${xmls[$i]}" "$(date -d "@$(stat -c %Y "${xml_path}")" '+%Y-%m-%d %H:%M:%S')"
  done

  local choice xml
  choice="$(select_from_list "Elige XML [1-${#xmls[@]}]: " "${#xmls[@]}")"
  xml="${XML_DIR}/${xmls[$(( choice - 1 ))]}"

  python_viewer_env
  echo "Abriendo XML sin simular: ${xml}"
  "${REPO_ROOT}/.venv/bin/python" - "${xml}" <<'PY'
from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

xml = Path(sys.argv[1]).resolve()
model = mujoco.MjModel.from_xml_path(str(xml))
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

print(f"XML: {xml.name}")
print(f"qpos0 z cuerpo: {data.qpos[2]:.4f} m")
print("Vista estatica. Cierra la ventana para terminar.")

with mujoco.viewer.launch_passive(model, data) as viewer:
  while viewer.is_running():
    viewer.sync()
    time.sleep(0.02)
PY
}

main() {
  echo "=============================================="
  echo "Visualizador TARANTULIN"
  echo "=============================================="
  echo "  1) Visualizar ultimo checkpoint"
  echo "  2) Visualizar checkpoint anterior"
  echo "  3) Ver XML sin simular"
  echo ""

  local choice
  choice="$(select_from_list "Elige opcion [1-3]: " 3)"
  case "${choice}" in
    1) visualizar_ultimo_checkpoint "$@" ;;
    2) visualizar_checkpoint_anterior "$@" ;;
    3) visualizar_xml ;;
  esac
}

main "$@"
