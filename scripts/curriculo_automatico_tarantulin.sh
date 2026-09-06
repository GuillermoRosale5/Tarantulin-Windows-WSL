#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOGS_DIR="${REPO_ROOT}/logs_tarantulin_mjx"
# Se carga como biblioteca: aporta validacion de rutas e identidad de procesos,
# pero no ejecuta main cuando BASH_SOURCE es distinto de $0.
# shellcheck source=tarantulin.sh
source "${SCRIPT_DIR}/tarantulin.sh"

PASOS_TOTALES=200000000
PASOS_BLOQUE=5000000
PERFIL_PPO="${TARANTULIN_PERFIL_PPO:-ligero}"
NOMBRE_EJECUCION="TarantulinCurriculo-$(date +%Y%m%d-%H%M%S)"
FASE="${TARANTULIN_FASE_RECOMPENSA:-1}"
CONFIGURAR_PRIMERO=1
REINICIAR_PRIMERO=1
PARAR_AL_SOLICITAR=1

# La identidad del supervisor debe quedar tambien en /proc/PID/cmdline. Si el
# usuario no da un nombre, reiniciamos el mismo proceso una sola vez incluyendo
# el nombre generado de forma explicita.
nombre_ejecucion_argumentado=0
for argumento_original in "$@"; do
  case "${argumento_original}" in
    --nombre-ejecucion|--nombre-ejecucion=*) nombre_ejecucion_argumentado=1 ;;
  esac
done
if (( nombre_ejecucion_argumentado == 0 )); then
  exec bash "$0" --nombre-ejecucion "${NOMBRE_EJECUCION}" "$@"
fi
unset nombre_ejecucion_argumentado argumento_original

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

validar_entero_positivo() {
  local nombre="$1" valor="$2"
  if [[ ! "${valor}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${nombre} debe ser un entero positivo: ${valor:-<vacio>}." >&2
    exit 1
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
validar_entero_positivo "--pasos-totales" "${PASOS_TOTALES}"
validar_entero_positivo "--pasos-por-bloque" "${PASOS_BLOQUE}"

if [[ -n "${TARANTULIN_RUNTIME_LOCK_HELD:-}" && "${CONFIGURAR_PRIMERO}" == "1" ]]; then
  if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    echo "El runtime no tiene entorno Python. Cierra esta terminal y repite INSTALAR_WINDOWS.cmd." >&2
    exit 1
  fi
  CONFIGURAR_PRIMERO=0
fi

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
ARCHIVO_PID_SUPERVISOR="${DIRECTORIO_EJECUCION}/curriculo_automatico.pid"
ARCHIVO_START_SUPERVISOR="${DIRECTORIO_EJECUCION}/curriculo_automatico.starttime"
ARCHIVO_PARADA_TOTAL="${DIRECTORIO_EJECUCION}/parada_total_solicitada"
ARCHIVO_PARADA_ENTRENAMIENTO="${DIRECTORIO_EJECUCION}/parada_entrenamiento_solicitada"
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
  local detalle="${2:-}"
  python3 - "$ARCHIVO_ESTADO" "$estado" "$NOMBRE_EJECUCION" "$DIRECTORIO_EJECUCION" "$FASE" "$PASOS_TOTALES" "$PASOS_BLOQUE" "$PASOS_COMPLETADOS" "$PERFIL_PPO" "$detalle" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ruta, estado, nombre_ejecucion, directorio_ejecucion, fase, total, bloque, completados, perfil, detalle = sys.argv[1:]
ruta = Path(ruta)
if ruta.is_symlink():
    raise SystemExit(f"El estado curricular no puede ser un enlace simbolico: {ruta}")
payload = {}
if ruta.is_file():
    try:
        anterior = json.loads(ruta.read_text(encoding="utf-8"))
        if isinstance(anterior, dict):
            payload.update(anterior)
    except (OSError, ValueError, TypeError):
        pass
if estado == "iniciando":
    payload.pop("detalle", None)
payload.pop("pasos_lanzados_por_supervisor", None)
payload.update({
    "estado": estado,
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "pid_supervisor": (
        None
        if estado in {"terminado", "cancelado", "error"}
        else int(os.environ.get("TARANTULIN_CURRICULUM_PID", "0") or 0)
    ),
    "nombre_ejecucion": nombre_ejecucion,
    "directorio_ejecucion": directorio_ejecucion,
    "fase_actual": int(fase),
    "fase_nombre": {
        "1": "mantener_pose_xml",
        "2": "llegar_desde_suelo",
        "3": "recuperar_desde_caida",
    }.get(str(fase), "desconocida"),
    "perfil_ppo": perfil,
    "pasos_totales_objetivo": int(total),
    "pasos_por_bloque": int(bloque),
    "pasos_confirmados_por_supervisor": int(completados),
})
if detalle:
    payload["detalle"] = detalle
descriptor, nombre_temporal = tempfile.mkstemp(
    prefix=f".{ruta.name}.", suffix=".tmp", dir=ruta.parent
)
temporal = Path(nombre_temporal)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as archivo:
        json.dump(payload, archivo, indent=2, sort_keys=True)
        archivo.write("\n")
        archivo.flush()
        os.fsync(archivo.fileno())
    os.replace(temporal, ruta)
finally:
    temporal.unlink(missing_ok=True)
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

estado_entrenamiento() {
  python3 - "${DIRECTORIO_EJECUCION}/estado.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except (OSError, ValueError, TypeError):
    print("desconocido")
else:
    print(data.get("estado", "desconocido"))
PY
}

contar_filas_progreso() {
  python3 - "${DIRECTORIO_EJECUCION}/progreso.csv" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print(0)
else:
    with path.open(newline="", encoding="utf-8") as handle:
        print(sum(1 for _ in csv.DictReader(handle)))
PY
}

pasos_confirmados_desde_fila() {
  local primera_fila="$1"
  python3 - "${DIRECTORIO_EJECUCION}/progreso.csv" "${primera_fila}" <<'PY'
import csv
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
offset = int(sys.argv[2])
if not path.is_file():
    print(0)
    raise SystemExit
with path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))[offset:]
steps = []
for row in rows:
    try:
        value = float(row.get("num_steps", ""))
    except (TypeError, ValueError):
        continue
    if math.isfinite(value) and value >= 0:
        steps.append(int(value))
print(max(steps, default=0))
PY
}

esperar_fin_entrenamiento_o_solicitud() {
  local pid="" launcher_pid="" parada_por_fase_enviada=0
  while :; do
    if [[ -f "${ARCHIVO_PARADA_TOTAL}" ]]; then
      echo "Parada total detectada; no se lanzara ningun bloque nuevo."
      "${SCRIPT_DIR}/tarantulin.sh" parar \
        --solo-entrenamiento --ejecucion "${NOMBRE_EJECUCION}" || true
      return 130
    fi
    if [[ -f "${ARCHIVO_SOLICITUD}" && "${PARAR_AL_SOLICITAR}" == "1" &&
          "${parada_por_fase_enviada}" == "0" ]]; then
      echo "Solicitud manual detectada. Parando el bloque actual para cambiar de fase..."
      "${SCRIPT_DIR}/tarantulin.sh" parar \
        --solo-entrenamiento --ejecucion "${NOMBRE_EJECUCION}" || true
      parada_por_fase_enviada=1
    fi

    pid="$(tarantulin_read_safe_pid_file "${DIRECTORIO_EJECUCION}/entrenamiento.pid" 2>/dev/null || true)"
    launcher_pid="$(tarantulin_read_safe_pid_file "${DIRECTORIO_EJECUCION}/lanzador.pid" 2>/dev/null || true)"
    if ! proceso_entrenamiento_corresponde_ejecucion \
        "${pid}" "${DIRECTORIO_EJECUCION}" &&
        ! proceso_lanzador_corresponde_ejecucion \
          "${launcher_pid}" "${DIRECTORIO_EJECUCION}"; then
      return 0
    fi
    sleep 0.5
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
        if (
            row.get("source") == "eval"
            and row.get("fase_curriculum_recompensa") == str(phase)
        ):
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
BLOQUE_ACTIVO=0
PARADA_RECIBIDA=0
ESTADO_FINAL=""
export TARANTULIN_CURRICULUM_PID="$$"

pid_supervisor_anterior="$(tarantulin_read_safe_pid_file "${ARCHIVO_PID_SUPERVISOR}" 2>/dev/null || true)"
if proceso_curriculo_corresponde_ejecucion \
    "${pid_supervisor_anterior}" "${DIRECTORIO_EJECUCION}"; then
  echo "Ya existe un supervisor curricular activo para esta ejecucion: PID ${pid_supervisor_anterior}." >&2
  exit 1
fi
rm -f "${ARCHIVO_PID_SUPERVISOR}" "${ARCHIVO_START_SUPERVISOR}" \
  "${ARCHIVO_PARADA_TOTAL}" "${ARCHIVO_PARADA_ENTRENAMIENTO}"
escribir_inicio_proceso "$$" "${ARCHIVO_START_SUPERVISOR}"
pid_supervisor_temporal="$(mktemp "${DIRECTORIO_EJECUCION}/.curriculo_automatico.pid.XXXXXX")"
printf '%s\n' "$$" > "${pid_supervisor_temporal}"
mv -f -- "${pid_supervisor_temporal}" "${ARCHIVO_PID_SUPERVISOR}"

solicitar_parada_supervisor() {
  local nombre_senal="$1" temporal
  PARADA_RECIBIDA=1
  if [[ ! -e "${ARCHIVO_PARADA_TOTAL}" && ! -L "${ARCHIVO_PARADA_TOTAL}" ]]; then
    temporal="$(mktemp "${DIRECTORIO_EJECUCION}/.parada_total.XXXXXX")"
    printf '%s\t%s\n' "$(date --iso-8601=seconds)" "${nombre_senal}" > "${temporal}"
    mv -f -- "${temporal}" "${ARCHIVO_PARADA_TOTAL}"
  fi
  escribir_estado "cancelando" "senal ${nombre_senal} recibida"
  exit 130
}

finalizar_supervisor() {
  local codigo_salida=$? pid_guardado
  trap - EXIT TERM INT HUP
  if (( BLOQUE_ACTIVO == 1 )); then
    "${SCRIPT_DIR}/tarantulin.sh" parar \
      --solo-entrenamiento --ejecucion "${NOMBRE_EJECUCION}" || true
  fi
  if (( PARADA_RECIBIDA == 1 )) || [[ -f "${ARCHIVO_PARADA_TOTAL}" ]]; then
    escribir_estado "cancelado" "supervisor y entrenamiento detenidos"
  elif [[ "${ESTADO_FINAL}" != "terminado" && "${codigo_salida}" -ne 0 ]]; then
    escribir_estado "error" "el supervisor termino con codigo ${codigo_salida}"
  fi
  pid_guardado="$(tarantulin_read_safe_pid_file "${ARCHIVO_PID_SUPERVISOR}" 2>/dev/null || true)"
  if [[ "${pid_guardado}" == "$$" ]]; then
    rm -f "${ARCHIVO_PID_SUPERVISOR}" "${ARCHIVO_START_SUPERVISOR}"
  fi
  exit "${codigo_salida}"
}

trap finalizar_supervisor EXIT
trap 'solicitar_parada_supervisor SIGTERM' TERM
trap 'solicitar_parada_supervisor SIGINT' INT
trap 'solicitar_parada_supervisor SIGHUP' HUP

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
  if [[ -f "${ARCHIVO_PARADA_TOTAL}" ]]; then
    PARADA_RECIBIDA=1
    exit 130
  fi
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
  escribir_estado "entrenando_bloque"
  filas_progreso_antes="$(contar_filas_progreso)"
  rm -f "${ARCHIVO_PARADA_ENTRENAMIENTO}"

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

  BLOQUE_ACTIVO=1
  TARANTULIN_CURRICULUM_CHILD=1 \
    "${SCRIPT_DIR}/tarantulin.sh" "${train_args[@]}"
  PRIMER_BLOQUE=0
  set +e
  esperar_fin_entrenamiento_o_solicitud
  estado_espera=$?
  set -e
  pasos_confirmados_bloque="$(pasos_confirmados_desde_fila "${filas_progreso_antes}")"
  PASOS_COMPLETADOS=$(( PASOS_COMPLETADOS + pasos_confirmados_bloque ))
  BLOQUE_ACTIVO=0
  estado_bloque="$(estado_entrenamiento)"

  if [[ "${estado_espera}" -eq 130 || -f "${ARCHIVO_PARADA_TOTAL}" ]]; then
    PARADA_RECIBIDA=1
    exit 130
  fi

  if solicitada="$(leer_fase_solicitada)"; then
    case "${estado_bloque}" in
      terminado|cancelado) ;;
      *)
        echo "El bloque no termino de forma recuperable: estado ${estado_bloque}." >&2
        exit 1
        ;;
    esac
    FASE="${solicitada}"
    rm -f "${ARCHIVO_PARADA_ENTRENAMIENTO}"
    escribir_estado \
      "bloque_cancelado_por_cambio_de_fase" \
      "${pasos_confirmados_bloque} pasos confirmados antes del cambio"
    echo "Cambio manual aplicado: fase ${FASE} - $(nombre_fase "${FASE}"); ${pasos_confirmados_bloque} pasos confirmados."
    continue
  fi

  if [[ "${estado_bloque}" != "terminado" ]]; then
    echo "El entrenamiento termino con estado ${estado_bloque}; el supervisor no relanzara otro bloque." >&2
    exit 1
  fi
  if (( pasos_confirmados_bloque <= 0 )); then
    echo "El bloque figura como terminado, pero no registro ningun paso positivo; no se contabiliza ni se relanza." >&2
    exit 1
  fi
  rm -f "${ARCHIVO_PARADA_ENTRENAMIENTO}"
  escribir_estado \
    "bloque_terminado" \
    "${pasos_confirmados_bloque} pasos confirmados en el ultimo bloque"

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
ESTADO_FINAL="terminado"
echo "Curriculo automatico terminado: ${DIRECTORIO_EJECUCION}"
