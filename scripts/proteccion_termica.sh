#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOGS_DIR="${REPO_ROOT}/logs_tarantulin_mjx"
# shellcheck source=process_identity.sh
source "${SCRIPT_DIR}/process_identity.sh"

TRAIN_PID=""
TRAIN_STARTTIME=""
RUN_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pid) TRAIN_PID="$2"; shift 2 ;;
    --train-starttime) TRAIN_STARTTIME="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    *) echo "Argumento no reconocido para proteccion_termica: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${TRAIN_PID}" || -z "${TRAIN_STARTTIME}" || -z "${RUN_DIR}" ]]; then
  echo "Uso: proteccion_termica.sh --pid PID --train-starttime TICKS --run-dir DIR" >&2
  exit 1
fi

RUN_DIR_REQUESTED="${RUN_DIR}"
if [[ -L "${RUN_DIR_REQUESTED}" ]]; then
  echo "La ejecucion no puede ser un enlace simbolico: ${RUN_DIR_REQUESTED}" >&2
  exit 1
fi
RUN_DIR="$(realpath -e "${RUN_DIR_REQUESTED}")"
LOGS_DIR="$(realpath -e "${LOGS_DIR}")"
if [[ "$(dirname -- "${RUN_DIR}")" != "${LOGS_DIR}" ]]; then
  echo "Ejecucion insegura o ajena a este runtime: ${RUN_DIR}" >&2
  exit 1
fi

training_identity_valid() {
  tarantulin_training_process_matches \
    "${TRAIN_PID}" "${REPO_ROOT}" "${LOGS_DIR}" "${RUN_DIR}" "${TRAIN_STARTTIME}"
}

if ! training_identity_valid; then
  echo "El PID no corresponde al entrenamiento TARANTULIN indicado; la proteccion no se inicia." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "La proteccion termica integrada usa telemetria NVIDIA. No se activa para AMD/Intel/CPU." >&2
  exit 0
fi

TEMP_WARN_C="${TARANTULIN_GPU_TEMP_WARN_C:-78}"
TEMP_STOP_C="${TARANTULIN_GPU_TEMP_STOP_C:-84}"
TEMP_CRITICAL_C="${TARANTULIN_GPU_TEMP_CRITICAL_C:-87}"
INTERVAL_SECONDS="${TARANTULIN_THERMAL_INTERVAL_SECONDS:-5}"
MAX_CONSECUTIVE="${TARANTULIN_THERMAL_MAX_CONSECUTIVE:-3}"
STATE_PATH="${RUN_DIR}/proteccion_termica_estado.csv"
PROGRESS_PATH="${RUN_DIR}/progreso.csv"
START_EPOCH="$(date +%s)"

if [[ ! -f "${STATE_PATH}" ]]; then
  printf 'timestamp,elapsed_seconds,elapsed_hours,train_pid,latest_num_steps,steps_per_second,training_steps_per_second,gpu_temp_c,gpu_util_percent,vram_used_mib,vram_total_mib,estado,consecutive_hot\n' > "${STATE_PATH}"
fi

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

pid_alive() {
  local pid="${1:-}"
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${pid}" >/dev/null 2>&1
}

stop_training() {
  local reason="$1"
  if ! training_identity_valid; then
    log "SEGURIDAD: la identidad de PID ${TRAIN_PID} cambio; no se envia ninguna senal."
    return 1
  fi
  log "PARADA DE SEGURIDAD: ${reason}. Enviando SIGTERM a PID ${TRAIN_PID}."
  training_identity_valid || return 1
  kill "${TRAIN_PID}" >/dev/null 2>&1 || return 0
  for _ in $(seq 1 12); do
    training_identity_valid || return 0
    sleep 1
  done
  if training_identity_valid; then
    log "PID ${TRAIN_PID} sigue activo y conserva su identidad; enviando SIGKILL."
    kill -9 "${TRAIN_PID}" >/dev/null 2>&1 || true
  else
    log "SEGURIDAD: la identidad cambio antes de SIGKILL; no se envia la senal."
  fi
}

progress_snapshot() {
  if [[ ! -f "${PROGRESS_PATH}" ]]; then
    printf ',,\n'
    return
  fi
  awk -F',' '
    NR == 1 {
      for (i = 1; i <= NF; i++) {
        idx[$i] = i
      }
      next
    }
    NF {
      row = $0
    }
    END {
      if (row == "") {
        print ",,"
        exit
      }
      split(row, values, ",")
      print values[idx["num_steps"]] "," values[idx["steps_per_second"]] "," values[idx["training_steps_per_second"]]
    }
  ' "${PROGRESS_PATH}"
}

log "Proteccion termica iniciada para PID ${TRAIN_PID}."
log "Umbrales: warn=${TEMP_WARN_C}C stop=${TEMP_STOP_C}C critical=${TEMP_CRITICAL_C}C intervalo=${INTERVAL_SECONDS}s consecutivos=${MAX_CONSECUTIVE}."

consecutive_hot=0
while training_identity_valid; do
  gpu_line="$(
    nvidia-smi \
      --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits 2>/dev/null |
      head -n 1 || true
  )"
  if [[ -z "${gpu_line}" ]]; then
    log "No puedo leer nvidia-smi; sigo vigilando."
    sleep "${INTERVAL_SECONDS}"
    continue
  fi

  IFS=',' read -r temp util mem_used mem_total <<< "${gpu_line}"
  temp="${temp//[[:space:]]/}"
  util="${util//[[:space:]]/}"
  mem_used="${mem_used//[[:space:]]/}"
  mem_total="${mem_total//[[:space:]]/}"

  estado="ok"
  now_epoch="$(date +%s)"
  elapsed_seconds=$(( now_epoch - START_EPOCH ))
  elapsed_hours="$(awk -v seconds="${elapsed_seconds}" 'BEGIN { printf "%.6f", seconds / 3600 }')"
  IFS=',' read -r latest_num_steps steps_per_second training_steps_per_second <<< "$(progress_snapshot)"
  if (( temp >= TEMP_CRITICAL_C )); then
    estado="critical"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$(date --iso-8601=seconds)" "${elapsed_seconds}" "${elapsed_hours}" "${TRAIN_PID}" \
      "${latest_num_steps}" "${steps_per_second}" "${training_steps_per_second}" \
      "${temp}" "${util}" "${mem_used}" "${mem_total}" "${estado}" "${consecutive_hot}" \
      >> "${STATE_PATH}"
    stop_training "GPU a ${temp}C, supera umbral critico ${TEMP_CRITICAL_C}C"
    exit 2
  fi

  if (( temp >= TEMP_STOP_C )); then
    consecutive_hot=$(( consecutive_hot + 1 ))
    estado="hot"
  else
    consecutive_hot=0
    if (( temp >= TEMP_WARN_C )); then
      estado="warn"
    fi
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$(date --iso-8601=seconds)" "${elapsed_seconds}" "${elapsed_hours}" "${TRAIN_PID}" \
    "${latest_num_steps}" "${steps_per_second}" "${training_steps_per_second}" \
    "${temp}" "${util}" "${mem_used}" "${mem_total}" "${estado}" "${consecutive_hot}" \
    >> "${STATE_PATH}"

  if [[ "${estado}" == "warn" ]]; then
    log "Aviso: GPU a ${temp}C."
  elif [[ "${estado}" == "hot" ]]; then
    log "Temperatura alta ${temp}C (${consecutive_hot}/${MAX_CONSECUTIVE})."
  fi

  if (( consecutive_hot >= MAX_CONSECUTIVE )); then
    stop_training "GPU >= ${TEMP_STOP_C}C durante ${consecutive_hot} chequeos seguidos"
    exit 2
  fi

  sleep "${INTERVAL_SECONDS}"
done

log "El entrenamiento PID ${TRAIN_PID} termino o cambio de identidad; proteccion termica termina."
