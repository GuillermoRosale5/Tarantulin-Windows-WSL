#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runtime_common.sh
source "${SCRIPT_DIR}/runtime_common.sh"

SOURCE_ROOT=""
RUNTIME_REQUESTED=""
SOURCE_ID=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Uso interno:
  sync_runtime.sh --source /mnt/c/... --runtime ~/.local/share/... --source-id HEX
                  [--dry-run]

Solo sincroniza desde la fuente Windows al workspace WSL. Nunca sincroniza en
sentido inverso y nunca toca logs, checkpoints, external ni entornos virtuales.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE_ROOT="$2"; shift 2 ;;
    --runtime) RUNTIME_REQUESTED="$2"; shift 2 ;;
    --source-id) SOURCE_ID="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Argumento no reconocido: $1" >&2; usage >&2; exit 2 ;;
  esac
done

tarantulin_require_wsl
tarantulin_validate_source_id "${SOURCE_ID}"
[[ -n "${SOURCE_ROOT}" && -n "${RUNTIME_REQUESTED}" ]] || {
  usage >&2
  exit 2
}
SOURCE_ROOT="$(realpath -e "${SOURCE_ROOT}")"
RUNTIME_ROOT="$(tarantulin_resolve_runtime "${RUNTIME_REQUESTED}")"
tarantulin_assert_safe_runtime_path "${RUNTIME_ROOT}"

tarantulin_assert_windows_mount "${SOURCE_ROOT}"
[[ -f "${SOURCE_ROOT}/.tarantulin/source.marker" ]] || {
  echo "La carpeta fuente no contiene .tarantulin/source.marker." >&2
  exit 1
}
[[ "$(tr -d '\r\n' < "${SOURCE_ROOT}/.tarantulin/source.marker")" == "tarantulin-windows-source-v1" ]] || {
  echo "El marcador de fuente no es valido." >&2
  exit 1
}
command -v rsync >/dev/null 2>&1 || {
  echo "Falta rsync dentro de WSL. Ejecuta .\\install.ps1 (sin -SyncOnly)." >&2
  exit 1
}
command -v flock >/dev/null 2>&1 || {
  echo "Falta flock (paquete util-linux) dentro de WSL. Ejecuta .\\install.ps1 sin -SkipSystemPackages." >&2
  exit 1
}

MARKER="${RUNTIME_ROOT}/.tarantulin-runtime"
if [[ ! -e "${RUNTIME_ROOT}" ]]; then
  if (( DRY_RUN == 1 )); then
    echo "DRY-RUN: se crearia ${RUNTIME_ROOT} y su marcador seguro."
    exit 0
  fi
  mkdir -p "${RUNTIME_ROOT}"
fi

LOCK_FILE="${RUNTIME_ROOT}/runtime.lock"
exec {RUNTIME_LOCK_FD}> "${LOCK_FILE}"
if ! flock --exclusive --nonblock "${RUNTIME_LOCK_FD}"; then
  echo "El runtime esta en uso por una simulacion, entrenamiento, visor o terminal: ${RUNTIME_ROOT}" >&2
  echo "Deten el proceso correspondiente antes de sincronizar o reinstalar." >&2
  exit 1
fi

if [[ ! -f "${MARKER}" ]]; then
  if find "${RUNTIME_ROOT}" -mindepth 1 -maxdepth 1 \
      ! -name runtime.lock -print -quit | grep -q .; then
    echo "La ruta runtime existe, contiene datos y no tiene marcador: ${RUNTIME_ROOT}" >&2
    echo "No se borra ni se sobrescribe nada." >&2
    exit 1
  fi
  (( DRY_RUN == 0 )) || {
    echo "DRY-RUN: falta el marcador en ${RUNTIME_ROOT}; no se ha modificado nada."
    exit 0
  }
  marker_tmp="${RUNTIME_ROOT}/.tarantulin-runtime.tmp.$$"
  printf '%s\n' \
    'kind=tarantulin-windows-runtime' \
    'schema=1' \
    "source_id=${SOURCE_ID}" > "${marker_tmp}"
  mv -f "${marker_tmp}" "${MARKER}"
fi

tarantulin_validate_runtime_marker "${RUNTIME_ROOT}" "${SOURCE_ID}"
WORKSPACE="${RUNTIME_ROOT}/workspace"
if [[ -L "${WORKSPACE}" ]]; then
  echo "El workspace no puede ser un enlace simbolico: ${WORKSPACE}" >&2
  exit 1
fi
if (( DRY_RUN == 0 )); then
  mkdir -p "${WORKSPACE}"
fi

rsync_args=(
  --archive
  --delete-delay
  --safe-links
  --human-readable
  --itemize-changes
  --exclude=/.git/
  --exclude=/.venv/
  --exclude=/external/
  --exclude=/logs_tarantulin_mjx/
  --exclude=/checkpoints/
  --exclude='/checkpoints buenos/'
  --exclude=/artifacts/
  --exclude=/.tarantulin-local/
  --exclude=/.ruff_cache/
  --exclude=/.pytest_cache/
  --exclude=/.mypy_cache/
  --exclude=/.coverage
  --exclude='**/__pycache__/'
  --exclude='*.pyc'
  --exclude='*.pid'
)
if (( DRY_RUN == 1 )); then
  rsync_args+=(--dry-run)
fi

echo "Fuente : ${SOURCE_ROOT}"
echo "Runtime: ${WORKSPACE}"
rsync "${rsync_args[@]}" "${SOURCE_ROOT}/" "${WORKSPACE}/"

if (( DRY_RUN == 1 )); then
  echo "Sincronizacion simulada; no se ha modificado el runtime."
  exit 0
fi

chmod +x "${WORKSPACE}"/scripts/*.sh "${WORKSPACE}"/scripts/wsl/*.sh 2>/dev/null || true
manifest_tmp="${RUNTIME_ROOT}/sync-manifest.sha256.tmp.$$"
(
  cd "${WORKSPACE}"
  find . \
    \( -path './.git' -o -path './.venv' -o -path './external' \
       -o -path './logs_tarantulin_mjx' -o -path './artifacts' \
       -o -path './checkpoints' -o -path './checkpoints buenos' \
       -o -path './.tarantulin-local' -o -path './.ruff_cache' \
       -o -path './.pytest_cache' -o -path './.mypy_cache' \
       -o -name '__pycache__' \) -prune \
    -o -type f ! -name '*.pyc' ! -name '*.pid' -print0 |
    sort -z |
    while IFS= read -r -d '' file; do
      hash="$(sha256sum -- "${file}" | awk '{print $1}')"
      printf '%s  %s\n' "${hash}" "${file#./}"
    done
) > "${manifest_tmp}"
mv -f "${manifest_tmp}" "${RUNTIME_ROOT}/sync-manifest.sha256"
printf '%s\n' "$(date --iso-8601=seconds)" > "${RUNTIME_ROOT}/last-sync.txt"
echo "Sincronizacion segura completada."
