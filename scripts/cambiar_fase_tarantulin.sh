#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOGS_DIR="${REPO_ROOT}/logs_tarantulin_mjx"
LAST_RUN="${LOGS_DIR}/ultima_run.txt"
PHASE="${1:-}"
STOP_CURRENT=1

usage() {
  cat <<'EOF'
Uso:
  scripts/cambiar_fase_tarantulin.sh [fase]

Fases:
  1  mantener_pose_xml
  2  llegar_desde_suelo
  3  recuperar_desde_caida

Opciones:
  --no-stop   solo escribe la solicitud; no para el entrenamiento actual

Este script esta pensado para usarse con curriculum_auto_tarantulin.sh.
Escribe fase_solicitada.txt en la ultima run y, por defecto, para el chunk
actual para que el supervisor relance inmediatamente con la fase elegida.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-stop) STOP_CURRENT=0; shift ;;
    --help|-h) usage; exit 0 ;;
    1|2|3) PHASE="$1"; shift ;;
    *) echo "Argumento no reconocido: $1" >&2; usage; exit 1 ;;
  esac
done

choose_phase() {
  cat >&2 <<'EOF'
Cambiar fase curricular:
  1) mantener_pose_xml
  2) llegar_desde_suelo
  3) recuperar_desde_caida
EOF
  local opcion
  read -r -p "Elige fase [1-3]: " opcion < /dev/tty
  case "${opcion}" in
    1|2|3) printf '%s\n' "${opcion}" ;;
    *) echo "Opcion no reconocida: ${opcion}. Usa 1, 2 o 3." >&2; exit 1 ;;
  esac
}

if [[ -z "${PHASE}" ]]; then
  if [[ -t 0 && -t 1 ]]; then
    PHASE="$(choose_phase)"
  else
    echo "Indica fase 1, 2 o 3." >&2
    exit 1
  fi
fi

if [[ ! -f "${LAST_RUN}" ]]; then
  echo "No encuentro ${LAST_RUN}; no se cual es la ultima run." >&2
  exit 1
fi

RUN_DIR="$(<"${LAST_RUN}")"
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "La ultima run no existe: ${RUN_DIR}" >&2
  exit 1
fi

printf '%s\n' "${PHASE}" > "${RUN_DIR}/fase_solicitada.txt"
python3 - "${RUN_DIR}/fase_solicitada.json" "${PHASE}" <<'PY'
import json
import sys
from datetime import datetime

path, phase = sys.argv[1:]
names = {
    "1": "mantener_pose_xml",
    "2": "llegar_desde_suelo",
    "3": "recuperar_desde_caida",
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "fase_solicitada": int(phase),
            "nombre": names[str(phase)],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        f,
        indent=2,
        sort_keys=True,
    )
PY

echo "Solicitud escrita: fase ${PHASE} en ${RUN_DIR}"

if (( STOP_CURRENT == 1 )); then
  echo "Parando entrenamiento actual para que el supervisor relance la fase..."
  "${SCRIPT_DIR}/tarantulin_wsl.sh" stop || true
else
  echo "No paro el entrenamiento actual; el cambio se aplicara cuando termine el chunk."
fi
