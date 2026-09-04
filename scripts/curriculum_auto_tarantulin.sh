#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOGS_DIR="${REPO_ROOT}/logs_tarantulin_mjx"

TOTAL_STEPS=200000000
CHUNK_STEPS=5000000
PERFIL_PPO="${TARANTULIN_PERFIL_PPO:-lite}"
RUN_NAME="TarantulinCurriculum-$(date +%Y%m%d-%H%M%S)"
PHASE="${TARANTULIN_FASE_RECOMPENSA:-1}"
SETUP_FIRST=1
RESET_FIRST=1
APPEND_CSV=1
STOP_ON_REQUEST=1

usage() {
  cat <<'EOF'
Uso:
  scripts/curriculum_auto_tarantulin.sh [opciones]

Opciones:
  --perfil-ppo debug|lite|lite_fast|full
  --run-name NOMBRE
  --start-phase 1|2|3
  --total-steps N       default 200000000
  --chunk-steps N       default 5000000
  --no-setup
  --no-reset-first
  --no-stop-on-request  no para el chunk actual al recibir fase_solicitada.txt

Control manual:
  scripts/cambiar_fase_tarantulin.sh

El supervisor entrena por chunks. Al terminar cada chunk evalua metricas recientes
y sube de fase si el rendimiento es suficiente. Si pides una fase manual, para
el chunk actual y relanza restaurando el ultimo checkpoint.
EOF
}

validate_phase() {
  case "${1:-}" in
    1|2|3) return 0 ;;
    *) echo "Fase invalida: ${1:-<vacia>}. Usa 1, 2 o 3." >&2; exit 1 ;;
  esac
}

validate_profile() {
  case "${1:-}" in
    debug|lite|lite_fast|full) return 0 ;;
    *) echo "Perfil PPO invalido: ${1:-<vacio>}." >&2; exit 1 ;;
  esac
}

validate_run_name() {
  local value="${1:-}"
  if [[ ! "${value}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || [[ "${value}" == "." || "${value}" == ".." ]]; then
    echo "Nombre de run no valido: ${value:-<vacio>}" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --perfil-ppo|--perfil_ppo|--ppo-profile) PERFIL_PPO="$2"; shift 2 ;;
    --perfil-ppo=*|--perfil_ppo=*|--ppo-profile=*) PERFIL_PPO="${1#*=}"; shift ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --run-name=*) RUN_NAME="${1#*=}"; shift ;;
    --start-phase|--fase-recompensa|--fase_recompensa) PHASE="$2"; shift 2 ;;
    --start-phase=*|--fase-recompensa=*|--fase_recompensa=*) PHASE="${1#*=}"; shift ;;
    --total-steps) TOTAL_STEPS="$2"; shift 2 ;;
    --total-steps=*) TOTAL_STEPS="${1#*=}"; shift ;;
    --chunk-steps) CHUNK_STEPS="$2"; shift 2 ;;
    --chunk-steps=*) CHUNK_STEPS="${1#*=}"; shift ;;
    --no-setup) SETUP_FIRST=0; shift ;;
    --no-reset-first) RESET_FIRST=0; shift ;;
    --no-stop-on-request) STOP_ON_REQUEST=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Argumento no reconocido: $1" >&2; usage; exit 1 ;;
  esac
done

validate_profile "${PERFIL_PPO}"
validate_phase "${PHASE}"
validate_run_name "${RUN_NAME}"

RUN_DIR="${LOGS_DIR}/${RUN_NAME}"
repo_real="$(realpath -e "${REPO_ROOT}")"
logs_real="$(realpath -m "${LOGS_DIR}")"
run_real="$(realpath -m "${RUN_DIR}")"
if [[ -L "${LOGS_DIR}" || "${logs_real}" != "${repo_real}/logs_tarantulin_mjx" || \
      -L "${RUN_DIR}" || "$(dirname -- "${run_real}")" != "${logs_real}" ]]; then
  echo "Ruta de curriculum insegura o enlazada fuera del runtime: ${RUN_DIR}" >&2
  exit 1
fi
REQUEST_FILE="${RUN_DIR}/fase_solicitada.txt"
STATE_FILE="${RUN_DIR}/curriculum_auto_estado.json"
mkdir -p "${RUN_DIR}"

phase_name() {
  case "$1" in
    1) printf '%s' "mantener_pose_xml" ;;
    2) printf '%s' "llegar_desde_suelo" ;;
    3) printf '%s' "recuperar_desde_caida" ;;
  esac
}

write_state() {
  local status="$1"
  python3 - "$STATE_FILE" "$status" "$RUN_NAME" "$RUN_DIR" "$PHASE" "$TOTAL_STEPS" "$CHUNK_STEPS" "$COMPLETED_STEPS" "$PERFIL_PPO" <<'PY'
import json
import sys
from datetime import datetime

path, status, run_name, run_dir, phase, total, chunk, completed, profile = sys.argv[1:]
payload = {
    "estado": status,
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "run_name": run_name,
    "run_dir": run_dir,
    "fase_actual": int(phase),
    "fase_nombre": {
        "1": "mantener_pose_xml",
        "2": "llegar_desde_suelo",
        "3": "recuperar_desde_caida",
    }.get(str(phase), "desconocida"),
    "perfil_ppo": profile,
    "total_steps_objetivo": int(total),
    "chunk_steps": int(chunk),
    "steps_lanzados_por_supervisor": int(completed),
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
PY
}

read_requested_phase() {
  if [[ ! -f "${REQUEST_FILE}" ]]; then
    return 1
  fi
  local requested
  requested="$(tr -cd '0-9' < "${REQUEST_FILE}" | head -c 1)"
  rm -f "${REQUEST_FILE}"
  case "${requested}" in
    1|2|3) printf '%s\n' "${requested}"; return 0 ;;
    *) echo "Solicitud de fase ignorada: ${requested:-<vacia>}" >&2; return 1 ;;
  esac
}

wait_for_training_end_or_request() {
  local pid_file="${RUN_DIR}/entrenamiento.pid"
  local pid=""
  for _ in $(seq 1 60); do
    if [[ -f "${pid_file}" ]]; then
      pid="$(cat "${pid_file}")"
      break
    fi
    sleep 1
  done
  if [[ -z "${pid}" ]]; then
    echo "No se encontro entrenamiento.pid; revisa ${RUN_DIR}/entrenamiento.log" >&2
    return 1
  fi

  while kill -0 "${pid}" >/dev/null 2>&1; do
    if [[ -f "${REQUEST_FILE}" && "${STOP_ON_REQUEST}" == "1" ]]; then
      echo "Solicitud manual detectada. Parando chunk actual para cambiar fase..."
      "${SCRIPT_DIR}/tarantulin_wsl.sh" stop || true
      break
    fi
    sleep 30
  done
}

phase_ready_to_advance() {
  local phase="$1"
  python3 - "${RUN_DIR}/recompensas.csv" "${phase}" <<'PY'
import csv
import math
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
phase = int(sys.argv[2])
if phase >= 3 or not csv_path.exists():
    raise SystemExit(1)

rows = []
with csv_path.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("source") == "eval":
            rows.append(row)
rows = rows[-6:]
if len(rows) < 3:
    print("Criterio: aun hay pocas evaluaciones recientes.")
    raise SystemExit(1)

def mean_field(name, default=math.nan):
    vals = []
    for row in rows:
        try:
            value = float(row.get(name, ""))
        except ValueError:
            continue
        if math.isfinite(value):
            vals.append(value)
    return sum(vals) / len(vals) if vals else default

pose = mean_field("state_pose_imitacion_reward")
height_error = abs(mean_field("state_error_altura_xml", 999.0))
level = mean_field("state_cuerpo_paralelo_reward", mean_field("state_level_gate", 0.0))
support = mean_field("state_support_gate", 0.0)
knee_pen = mean_field("rodilla_suelo_penalty_ponderado", 0.0)
height_excess = mean_field("altura_exceso_penalty_ponderado", 0.0)
contacts = mean_field("state_foot_contacts", 0.0)

if phase == 1:
    ready = (
        pose >= 0.72
        and height_error <= 0.030
        and level >= 0.60
        and knee_pen <= 0.30
        and height_excess <= 0.30
    )
    reason = (
        f"fase1 pose={pose:.3f} |err_z|={height_error:.3f} "
        f"nivel={level:.3f} rodilla={knee_pen:.3f} exceso_z={height_excess:.3f}"
    )
else:
    ready = (
        pose >= 0.62
        and height_error <= 0.040
        and level >= 0.55
        and support >= 0.35
        and contacts >= 3.0
        and knee_pen <= 0.35
        and height_excess <= 0.35
    )
    reason = (
        f"fase2 pose={pose:.3f} |err_z|={height_error:.3f} "
        f"nivel={level:.3f} support={support:.3f} contactos={contacts:.2f} "
        f"rodilla={knee_pen:.3f} exceso_z={height_excess:.3f}"
    )

print("Criterio:", reason)
raise SystemExit(0 if ready else 1)
PY
}

COMPLETED_STEPS=0
FIRST_CHUNK=1
write_state "iniciando"

cat <<EOF
Curriculum automatico TARANTULIN
  run_name: ${RUN_NAME}
  perfil_ppo: ${PERFIL_PPO}
  fase inicial: ${PHASE} - $(phase_name "${PHASE}")
  total_steps: ${TOTAL_STEPS}
  chunk_steps: ${CHUNK_STEPS}
  run_dir: ${RUN_DIR}

Puedes cambiar fase con:
  scripts/cambiar_fase_tarantulin.sh
EOF

while (( COMPLETED_STEPS < TOTAL_STEPS )); do
  if requested="$(read_requested_phase)"; then
    PHASE="${requested}"
    echo "Cambio manual aplicado antes del chunk: fase ${PHASE} - $(phase_name "${PHASE}")"
  fi

  remaining=$(( TOTAL_STEPS - COMPLETED_STEPS ))
  this_chunk="${CHUNK_STEPS}"
  if (( remaining < CHUNK_STEPS )); then
    this_chunk="${remaining}"
  fi

  echo ""
  echo "=== Chunk fase ${PHASE} - $(phase_name "${PHASE}") | ${this_chunk} steps ==="
  write_state "entrenando_chunk"

  train_args=(
    train
    --background
    --skip-test-mjx
    --run-name "${RUN_NAME}"
    --perfil-ppo "${PERFIL_PPO}"
    --fase-recompensa "${PHASE}"
    --num-timesteps "${this_chunk}"
    --append-csv
  )
  if (( FIRST_CHUNK == 1 && SETUP_FIRST == 1 )); then
    train_args+=(--setup)
  fi
  if (( FIRST_CHUNK == 1 && RESET_FIRST == 1 )); then
    train_args+=(--reset-checkpoint)
  fi

  "${SCRIPT_DIR}/tarantulin_wsl.sh" "${train_args[@]}"
  FIRST_CHUNK=0
  wait_for_training_end_or_request
  COMPLETED_STEPS=$(( COMPLETED_STEPS + this_chunk ))
  write_state "chunk_terminado"

  if requested="$(read_requested_phase)"; then
    PHASE="${requested}"
    echo "Cambio manual aplicado: fase ${PHASE} - $(phase_name "${PHASE}")"
    continue
  fi

  if phase_ready_to_advance "${PHASE}"; then
    if (( PHASE < 3 )); then
      PHASE=$(( PHASE + 1 ))
      echo "Criterio cumplido. Subiendo automaticamente a fase ${PHASE} - $(phase_name "${PHASE}")"
    fi
  else
    echo "Criterio no cumplido. Seguimos en fase ${PHASE}."
  fi
done

write_state "terminado"
echo "Curriculum automatico terminado: ${RUN_DIR}"
