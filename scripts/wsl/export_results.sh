#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runtime_common.sh
source "${SCRIPT_DIR}/runtime_common.sh"

SOURCE_ROOT=""
RUNTIME_REQUESTED=""
SOURCE_ID=""
EXPORT_ALL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE_ROOT="$2"; shift 2 ;;
    --runtime) RUNTIME_REQUESTED="$2"; shift 2 ;;
    --source-id) SOURCE_ID="$2"; shift 2 ;;
    --all) EXPORT_ALL=1; shift ;;
    --latest) EXPORT_ALL=0; shift ;;
    *) echo "Argumento no reconocido para pull-results: $1" >&2; exit 2 ;;
  esac
done

RUNTIME_ROOT="$(tarantulin_resolve_runtime "${RUNTIME_REQUESTED}")"
tarantulin_assert_safe_runtime_path "${RUNTIME_ROOT}"
tarantulin_validate_runtime_marker "${RUNTIME_ROOT}" "${SOURCE_ID}"
SOURCE_ROOT="$(realpath -e "${SOURCE_ROOT}")"
tarantulin_assert_windows_mount "${SOURCE_ROOT}"
[[ -f "${SOURCE_ROOT}/.tarantulin/source.marker" ]] || {
  echo "El destino Windows no contiene el marcador de fuente TARANTULIN." >&2
  exit 1
}
[[ "$(tr -d '\r\n' < "${SOURCE_ROOT}/.tarantulin/source.marker")" == "tarantulin-windows-source-v1" ]] || {
  echo "El marcador del destino Windows no es valido." >&2
  exit 1
}
logs="${RUNTIME_ROOT}/workspace/logs_tarantulin_mjx"
[[ -d "${logs}" ]] || { echo "Todavia no hay resultados en el runtime." >&2; exit 1; }
workspace_real="$(realpath -e "${RUNTIME_ROOT}/workspace")"
logs_real="$(realpath -e "${logs}")"
if [[ -L "${logs}" || "${logs_real}" != "${workspace_real}/logs_tarantulin_mjx" ]]; then
  echo "Directorio de resultados inseguro o enlazado fuera del workspace: ${logs}" >&2
  exit 1
fi
logs="${logs_real}"
destination="${SOURCE_ROOT}/artifacts/logs_tarantulin_mjx"
if [[ -L "${SOURCE_ROOT}/artifacts" || -L "${destination}" ]]; then
  echo "El destino de resultados no puede ser un enlace simbolico." >&2
  exit 1
fi
mkdir -p "${destination}"

command -v flock >/dev/null 2>&1 || {
  echo "Falta flock (paquete util-linux). Repite la instalacion completa desde Windows." >&2
  exit 1
}
rsync_args=(-a --human-readable --info=stats2 --safe-links)
exec {TRAINING_LOCK_FD}> "${RUNTIME_ROOT}/training.lock"
if ! flock --exclusive --nonblock "${TRAINING_LOCK_FD}"; then
  rsync_args+=(--exclude='checkpoints/***' --exclude='*/checkpoints/***')
  echo "Aviso: hay un entrenamiento activo. Se exportan registros y metricas, pero no checkpoints que podrian estar escribiendose." >&2
  echo "Para copiar tambien los checkpoints, deten el entrenamiento o espera a que termine y repite." >&2
fi

if (( EXPORT_ALL == 1 )); then
  rsync "${rsync_args[@]}" "${logs}/" "${destination}/"
  echo "Todos los resultados se copiaron a: ${destination}"
  exit 0
fi

run_dir=""
if [[ -f "${logs}/ultima_run.txt" ]]; then
  candidate="$(tr -d '\r\n' < "${logs}/ultima_run.txt")"
  if [[ -n "${candidate}" && -d "${candidate}" ]]; then run_dir="$(realpath -e "${candidate}")"; fi
fi
if [[ -z "${run_dir}" ]]; then
  run_dir="$(find "${logs}" -mindepth 1 -maxdepth 1 -type d -printf '%T@\t%p\n' | sort -nr | head -1 | cut -f2-)"
fi
[[ -n "${run_dir}" ]] || { echo "No se encontro ninguna ejecucion para exportar." >&2; exit 1; }
case "${run_dir}" in "${logs}/"*) ;; *) echo "Ruta de resultados rechazada: ${run_dir}" >&2; exit 1 ;; esac
rsync "${rsync_args[@]}" "${run_dir}/" "${destination}/$(basename "${run_dir}")/"
echo "Ultimo resultado copiado a: ${destination}/$(basename "${run_dir}")"
