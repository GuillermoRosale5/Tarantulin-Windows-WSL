#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOGS_DIR="${REPO_ROOT}/logs_tarantulin_mjx"

PASOS_TOTALES=200000000
PASOS_BLOQUE=5000000
PERFIL_PPO="${TARANTULIN_PERFIL_PPO:-ligero}"
NOMBRE_EJECUCION="TarantulinCurriculo-$(date +%Y%m%d-%H%M%S)"
FASE="${TARANTULIN_FASE_RECOMPENSA:-1}"
CONFIGURAR_PRIMERO=1
REINICIAR_PRIMERO=1
PARAR_AL_SOLICITAR=1

usage() {
  cat <<'EOF'
Uso:
  scripts/curriculo_automatico_tarantulin.sh [opciones]

Opciones:
  --perfil-ppo depuracion|ligero|ligero_rapido|completo
  --nombre-ejecucion NOMBRE
  --fase-inicial 1|2|3
  --pasos-totales N     predeterminado: 200000000
  --pasos-por-bloque N       predeterminado: 5000000
  --sin-preparacion
  --sin-reinicio-inicial
  --sin-parada-al-solicitar  no para el bloque actual al recibir fase_solicitada.txt

Control manual:
  scripts/cambiar_fase_tarantulin.sh

El supervisor entrena por bloques. Al terminar cada bloque evalua metricas recientes
y sube de fase si el rendimiento es suficiente. Si pides una fase manual, para
el bloque actual y relanza restaurando el ultimo checkpoint.
EOF
}

validar_fase() {
  case "${1:-}" in
    1|2|3) return 0 ;;
    *) echo "Fase invalida: ${1:-<vacia>}. Usa 1, 2 o 3." >&2; exit 1 ;;
  esac
}

validar_perfil() {
  case "${1:-}" in
    depuracion|ligero|ligero_rapido|completo) return 0 ;;
    *) echo "Perfil PPO invalido: ${1:-<vacio>}." >&2; exit 1 ;;
  esac
}

validar_nombre_ejecucion() {
  local value="${1:-}"
  if [[ ! "${value}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || [[ "${value}" == "." || "${value}" == ".." ]]; then
    echo "Nombre de ejecucion no valido: ${value:-<vacio>}" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --perfil-ppo) PERFIL_PPO="$2"; shift 2 ;;
    --perfil-ppo=*) PERFIL_PPO="${1#*=}"; shift ;;
    --nombre-ejecucion) NOMBRE_EJECUCION="$2"; shift 2 ;;
    --nombre-ejecucion=*) NOMBRE_EJECUCION="${1#*=}"; shift ;;
    --fase-inicial) FASE="$2"; shift 2 ;;
    --fase-inicial=*) FASE="${1#*=}"; shift ;;
    --pasos-totales) PASOS_TOTALES="$2"; shift 2 ;;
    --pasos-totales=*) PASOS_TOTALES="${1#*=}"; shift ;;
    --pasos-por-bloque) PASOS_BLOQUE="$2"; shift 2 ;;
    --pasos-por-bloque=*) PASOS_BLOQUE="${1#*=}"; shift ;;
    --sin-preparacion) CONFIGURAR_PRIMERO=0; shift ;;
    --sin-reinicio-inicial) REINICIAR_PRIMERO=0; shift ;;
    --sin-parada-al-solicitar) PARAR_AL_SOLICITAR=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Argumento no reconocido: $1" >&2; usage; exit 1 ;;
  esac
done

validar_perfil "${PERFIL_PPO}"
validar_fase "${FASE}"
validar_nombre_ejecucion "${NOMBRE_EJECUCION}"

DIRECTORIO_EJECUCION="${LOGS_DIR}/${NOMBRE_EJECUCION}"
repo_real="$(realpath -e "${REPO_ROOT}")"
logs_real="$(realpath -m "${LOGS_DIR}")"
run_real="$(realpath -m "${DIRECTORIO_EJECUCION}")"
if [[ -L "${LOGS_DIR}" || "${logs_real}" != "${repo_real}/logs_tarantulin_mjx" || \
      -L "${DIRECTORIO_EJECUCION}" || "$(dirname -- "${run_real}")" != "${logs_real}" ]]; then
  echo "Ruta de curriculo insegura o enlazada fuera del entorno de ejecucion: ${DIRECTORIO_EJECUCION}" >&2
  exit 1
fi
ARCHIVO_SOLICITUD="${DIRECTORIO_EJECUCION}/fase_solicitada.txt"
ARCHIVO_ESTADO="${DIRECTORIO_EJECUCION}/curriculo_automatico_estado.json"
mkdir -p "${DIRECTORIO_EJECUCION}"

nombre_fase() {
  case "$1" in
    1) printf '%s' "mantener_pose_xml" ;;
    2) printf '%s' "llegar_desde_suelo" ;;
    3) printf '%s' "recuperar_desde_caida" ;;
  esac
}

escribir_estado() {
  local estado="$1"
  python3 - "$ARCHIVO_ESTADO" "$estado" "$NOMBRE_EJECUCION" "$DIRECTORIO_EJECUCION" "$FASE" "$PASOS_TOTALES" "$PASOS_BLOQUE" "$PASOS_COMPLETADOS" "$PERFIL_PPO" <<'PY'
import json
import sys
from datetime import datetime

ruta, estado, nombre_ejecucion, directorio_ejecucion, fase, total, bloque, completados, perfil = sys.argv[1:]
payload = {
    "estado": estado,
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "run_name": nombre_ejecucion,
    "run_dir": directorio_ejecucion,
    "fase_actual": int(fase),
    "fase_nombre": {
        "1": "mantener_pose_xml",
        "2": "llegar_desde_suelo",
        "3": "recuperar_desde_caida",
    }.get(str(fase), "desconocida"),
    "perfil_ppo": perfil,
    "total_steps_objetivo": int(total),
    "chunk_steps": int(bloque),
    "steps_lanzados_por_supervisor": int(completados),
}
with open(ruta, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
PY
}

leer_fase_solicitada() {
  if [[ ! -f "${ARCHIVO_SOLICITUD}" ]]; then
    return 1
  fi
  local solicitada
  solicitada="$(tr -cd '0-9' < "${ARCHIVO_SOLICITUD}" | head -c 1)"
  rm -f "${ARCHIVO_SOLICITUD}"
  case "${solicitada}" in
    1|2|3) printf '%s\n' "${solicitada}"; return 0 ;;
    *) echo "Solicitud de fase ignorada: ${solicitada:-<vacia>}" >&2; return 1 ;;
  esac
}

esperar_fin_entrenamiento_o_solicitud() {
  local archivo_pid="${DIRECTORIO_EJECUCION}/entrenamiento.pid"
  local pid=""
  for _ in $(seq 1 60); do
    if [[ -f "${archivo_pid}" ]]; then
      pid="$(cat "${archivo_pid}")"
      break
    fi
    sleep 1
  done
  if [[ -z "${pid}" ]]; then
    echo "No se encontro entrenamiento.pid; revisa ${DIRECTORIO_EJECUCION}/entrenamiento.log" >&2
    return 1
  fi

  while kill -0 "${pid}" >/dev/null 2>&1; do
    if [[ -f "${ARCHIVO_SOLICITUD}" && "${PARAR_AL_SOLICITAR}" == "1" ]]; then
      echo "Solicitud manual detectada. Parando el bloque actual para cambiar de fase..."
      "${SCRIPT_DIR}/tarantulin.sh" parar || true
      break
    fi
    sleep 30
  done
}

fase_lista_para_avanzar() {
  local phase="$1"
  python3 - "${DIRECTORIO_EJECUCION}/recompensas.csv" "${phase}" <<'PY'
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

PASOS_COMPLETADOS=0
PRIMER_BLOQUE=1
escribir_estado "iniciando"

cat <<EOF
Curriculo automatico TARANTULIN
  nombre de ejecucion: ${NOMBRE_EJECUCION}
  perfil_ppo: ${PERFIL_PPO}
  fase inicial: ${FASE} - $(nombre_fase "${FASE}")
  pasos totales: ${PASOS_TOTALES}
  pasos por bloque: ${PASOS_BLOQUE}
  directorio: ${DIRECTORIO_EJECUCION}

Puedes cambiar fase con:
  scripts/cambiar_fase_tarantulin.sh
EOF

while (( PASOS_COMPLETADOS < PASOS_TOTALES )); do
  if solicitada="$(leer_fase_solicitada)"; then
    FASE="${solicitada}"
    echo "Cambio manual aplicado antes del bloque: fase ${FASE} - $(nombre_fase "${FASE}")"
  fi

  pasos_restantes=$(( PASOS_TOTALES - PASOS_COMPLETADOS ))
  pasos_bloque_actual="${PASOS_BLOQUE}"
  if (( pasos_restantes < PASOS_BLOQUE )); then
    pasos_bloque_actual="${pasos_restantes}"
  fi

  echo ""
  echo "=== Bloque de la fase ${FASE} - $(nombre_fase "${FASE}") | ${pasos_bloque_actual} pasos ==="
  escribir_estado "entrenando_chunk"

  train_args=(
    entrenar
    --segundo-plano
    --omitir-prueba-mjx
    --nombre-ejecucion "${NOMBRE_EJECUCION}"
    --perfil-ppo "${PERFIL_PPO}"
    --fase-recompensa "${FASE}"
    --num-timesteps "${pasos_bloque_actual}"
    --anexar-csv
  )
  if (( PRIMER_BLOQUE == 1 && CONFIGURAR_PRIMERO == 1 )); then
    train_args+=(--setup)
  fi
  if (( PRIMER_BLOQUE == 1 && REINICIAR_PRIMERO == 1 )); then
    train_args+=(--desde-cero)
  fi

  "${SCRIPT_DIR}/tarantulin.sh" "${train_args[@]}"
  PRIMER_BLOQUE=0
  esperar_fin_entrenamiento_o_solicitud
  PASOS_COMPLETADOS=$(( PASOS_COMPLETADOS + pasos_bloque_actual ))
  escribir_estado "chunk_terminado"

  if solicitada="$(leer_fase_solicitada)"; then
    FASE="${solicitada}"
    echo "Cambio manual aplicado: fase ${FASE} - $(nombre_fase "${FASE}")"
    continue
  fi

  if fase_lista_para_avanzar "${FASE}"; then
    if (( FASE < 3 )); then
      FASE=$(( FASE + 1 ))
      echo "Criterio cumplido. Subiendo automaticamente a fase ${FASE} - $(nombre_fase "${FASE}")"
    fi
  else
    echo "Criterio no cumplido. Seguimos en fase ${FASE}."
  fi
done

escribir_estado "terminado"
echo "Curriculo automatico terminado: ${DIRECTORIO_EJECUCION}"
