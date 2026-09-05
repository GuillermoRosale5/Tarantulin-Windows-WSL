#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOGS_DIR="${REPO_ROOT}/logs_tarantulin_mjx"
LAST_RUN="${LOGS_DIR}/ultima_run.txt"
XML_DIR="${REPO_ROOT}/tarantulin/xmls"

cd "${REPO_ROOT}"

configurar_entorno_visualizador() {
  export PYTHONPATH="${REPO_ROOT}"
  export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.10}"
  export MUJOCO_GL="${MUJOCO_VIEWER_GL:-glfw}"
}

ejecucion_actual() {
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

elegir_de_lista() {
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
  echo "Abriendo ultimo checkpoint local (puede ser un entrenamiento parcial)..."
  echo "Esta opcion sigue logs_tarantulin_mjx/ultima_run.txt; no usa la red publicada."
  exec ./scripts/tarantulin.sh visualizar-resultados "$@"
}

visualizar_red_preentrenada() {
  echo "Abriendo la red preentrenada recomendada..."
  echo "Fase 2 | paso 45.932.544 | episodio 1500"
  echo "Esta opcion no consulta logs_tarantulin_mjx/ultima_run.txt."
  exec ./scripts/tarantulin.sh visualizar-red-preentrenada "$@"
}

visualizar_checkpoint_anterior() {
  local directorio_ejecucion
  directorio_ejecucion="$(ejecucion_actual)"
  if [[ -z "${directorio_ejecucion}" ]]; then
    echo "No encuentro ninguna ejecucion en ${LOGS_DIR}." >&2
    exit 1
  fi

  mapfile -t checkpoints < <(
    find "${directorio_ejecucion}/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%P\n' 2>/dev/null |
      awk '/^[0-9]+$/' |
      sort -nr
  )

  if (( ${#checkpoints[@]} <= 1 )); then
    echo "No hay checkpoints anteriores en la ejecucion actual: ${directorio_ejecucion}" >&2
    echo "Ejecucion encontrada, pero checkpoints disponibles: ${#checkpoints[@]}" >&2
    exit 1
  fi

  echo ""
  echo "Ejecucion actual:"
  echo "  ${directorio_ejecucion}"
  echo ""
  echo "Checkpoints anteriores:"
  local i paso ruta_checkpoint
  for (( i = 1; i < ${#checkpoints[@]}; i++ )); do
    paso="${checkpoints[$i]}"
    ruta_checkpoint="${directorio_ejecucion}/checkpoints/${paso}"
    printf '  %2d) paso %-12s  %s\n' "${i}" "${paso}" "$(date -d "@$(stat -c %Y "${ruta_checkpoint}")" '+%Y-%m-%d %H:%M:%S')"
  done

  local eleccion
  local maximo="$(( ${#checkpoints[@]} - 1 ))"
  eleccion="$(elegir_de_lista "Elige checkpoint anterior [1-${maximo}]: " "${maximo}")"
  ruta_checkpoint="${directorio_ejecucion}/checkpoints/${checkpoints[$eleccion]}"
  echo "Abriendo checkpoint: ${ruta_checkpoint}"
  exec ./scripts/tarantulin.sh visualizar-resultados --ruta-checkpoint "${ruta_checkpoint}" "$@"
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

  local eleccion xml
  eleccion="$(elegir_de_lista "Elige XML [1-${#xmls[@]}]: " "${#xmls[@]}")"
  xml="${XML_DIR}/${xmls[$(( eleccion - 1 ))]}"

  configurar_entorno_visualizador
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
  echo "  1) Visualizar red preentrenada recomendada (fase 2, paso 45.932.544)"
  echo "  2) Visualizar ultimo checkpoint local (puede ser parcial)"
  echo "  3) Visualizar checkpoint local anterior"
  echo "  4) Ver XML sin simular"
  echo ""

  local eleccion
  eleccion="$(elegir_de_lista "Elige opcion [1-4]: " 4)"
  case "${eleccion}" in
    1) visualizar_red_preentrenada "$@" ;;
    2) visualizar_ultimo_checkpoint "$@" ;;
    3) visualizar_checkpoint_anterior "$@" ;;
    4) visualizar_xml ;;
  esac
}

main "$@"
