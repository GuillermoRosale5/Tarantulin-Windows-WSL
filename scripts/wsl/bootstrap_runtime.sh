#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT=""
RUNTIME_REQUESTED=""
SOURCE_ID=""
ACCELERATOR="auto"
RUN_SETUP=1
SYNC_ONLY=0
INSTALL_SYSTEM_PACKAGES=1
SKIP_GPU_CHECK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE_ROOT="$2"; shift 2 ;;
    --runtime) RUNTIME_REQUESTED="$2"; shift 2 ;;
    --source-id) SOURCE_ID="$2"; shift 2 ;;
    --accelerator) ACCELERATOR="$2"; shift 2 ;;
    --no-setup) RUN_SETUP=0; shift ;;
    --sync-only) SYNC_ONLY=1; RUN_SETUP=0; INSTALL_SYSTEM_PACKAGES=0; shift ;;
    --skip-system-packages) INSTALL_SYSTEM_PACKAGES=0; shift ;;
    --skip-gpu-check) SKIP_GPU_CHECK=1; shift ;;
    *) echo "Argumento no reconocido: $1" >&2; exit 2 ;;
  esac
done

# shellcheck source=runtime_common.sh
source "${SCRIPT_DIR}/runtime_common.sh"
tarantulin_require_wsl
case "${ACCELERATOR}" in auto|nvidia|amd|intel|cpu) ;; *) echo "Acelerador no valido: ${ACCELERATOR}" >&2; exit 2 ;; esac

if (( INSTALL_SYSTEM_PACKAGES == 1 )); then
  [[ -r /etc/os-release ]] || { echo "No se puede identificar la distribucion Linux." >&2; exit 1; }
  # shellcheck disable=SC1091
  source /etc/os-release
  [[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "24.04" ]] || {
    echo "La variante Windows esta validada con Ubuntu 24.04 bajo WSL; detectado ${PRETTY_NAME:-desconocido}." >&2
    exit 1
  }
  command -v sudo >/dev/null 2>&1 || { echo "Falta sudo en WSL." >&2; exit 1; }
  echo "==> Instalando dependencias base en ${PRETTY_NAME}"
  echo "apt puede tardar varios minutos; mientras muestre actividad no esta bloqueado."
  apt_options=(
    -o Acquire::Retries=3
    -o Acquire::http::Timeout=30
    -o Acquire::https::Timeout=30
    -o DPkg::Lock::Timeout=120
  )
  sudo -v
  sudo apt-get "${apt_options[@]}" update
  sudo env DEBIAN_FRONTEND=noninteractive apt-get "${apt_options[@]}" install -y \
    ca-certificates curl git rsync build-essential python3 python3-venv python3-pip \
    pkg-config libgl1 libegl1 libglfw3 libglew2.2 pciutils util-linux
fi

bash "${SCRIPT_DIR}/sync_runtime.sh" \
  --source "${SOURCE_ROOT}" \
  --runtime "${RUNTIME_REQUESTED}" \
  --source-id "${SOURCE_ID}"

if (( SYNC_ONLY == 1 )); then
  echo "Runtime sincronizado; instalacion y setup omitidos por -SyncOnly."
  exit 0
fi
RUNTIME_ROOT="$(tarantulin_resolve_runtime "${RUNTIME_REQUESTED}")"
workspace="${RUNTIME_ROOT}/workspace"
command -v flock >/dev/null 2>&1 || {
  echo "Falta flock (paquete util-linux). Repite sin -SkipSystemPackages." >&2
  exit 1
}
exec {RUNTIME_LOCK_FD}> "${RUNTIME_ROOT}/runtime.lock"
if ! flock --exclusive --nonblock "${RUNTIME_LOCK_FD}"; then
  echo "El runtime esta en uso; detiene el entrenamiento, visor o terminal antes de instalar." >&2
  exit 1
fi
export TARANTULIN_RUNTIME_ROOT="${RUNTIME_ROOT}"
export TARANTULIN_RUNTIME_LOCK_HELD=exclusive
export TARANTULIN_BACKEND_PROFILE="${ACCELERATOR}"
if (( SKIP_GPU_CHECK == 1 )); then export TARANTULIN_SKIP_BACKEND_CHECK=1; fi
# shellcheck source=../platform.sh
source "${workspace}/scripts/platform.sh"
resolved_accelerator="$(tarantulin_resolve_accelerator)"
case "${resolved_accelerator}" in auto|nvidia|amd|intel|cpu) ;; *) echo "Perfil resuelto no valido." >&2; exit 2 ;; esac
tarantulin_accelerator_preflight "${resolved_accelerator}"
if (( RUN_SETUP == 0 )); then
  profile_tmp="${RUNTIME_ROOT}/accelerator.profile.tmp.$$"
  printf '%s\n' "${resolved_accelerator}" > "${profile_tmp}"
  mv -f "${profile_tmp}" "${RUNTIME_ROOT}/accelerator.profile"
  rm -f -- "${RUNTIME_ROOT}/amd-wsl-experimental.opt-in"
  echo "Perfil persistido para este runtime: ${resolved_accelerator}"
  echo "Runtime creado; setup Python/MJX omitido por -NoSetup."
  exit 0
fi

install_args=(--skip-system-packages --accelerator "${resolved_accelerator}")
if (( SKIP_GPU_CHECK == 1 )); then install_args+=(--skip-gpu-check); fi
bash "${workspace}/scripts/install_wsl.sh" "${install_args[@]}"

profile_tmp="${RUNTIME_ROOT}/accelerator.profile.tmp.$$"
printf '%s\n' "${resolved_accelerator}" > "${profile_tmp}"
mv -f "${profile_tmp}" "${RUNTIME_ROOT}/accelerator.profile"
rm -f -- "${RUNTIME_ROOT}/amd-wsl-experimental.opt-in"
echo "Perfil persistido para este runtime: ${resolved_accelerator}"

echo "Runtime preparado en: ${workspace}"
