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
ENABLE_EXPERIMENTAL_AMD_WSL=0

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
    --enable-experimental-amd-wsl) ENABLE_EXPERIMENTAL_AMD_WSL=1; shift ;;
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
  [[ "${ID:-}" == "ubuntu" ]] || {
    echo "La variante Windows esta validada con Ubuntu bajo WSL; detectado ${PRETTY_NAME:-desconocido}." >&2
    exit 1
  }
  command -v sudo >/dev/null 2>&1 || { echo "Falta sudo en WSL." >&2; exit 1; }
  echo "==> Instalando dependencias base en ${PRETTY_NAME}"
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates curl git rsync build-essential python3 python3-venv python3-pip \
    pkg-config libgl1 libegl1 libglfw3 libglew2.2 pciutils
fi

"${SCRIPT_DIR}/sync_runtime.sh" \
  --source "${SOURCE_ROOT}" \
  --runtime "${RUNTIME_REQUESTED}" \
  --source-id "${SOURCE_ID}"

if (( SYNC_ONLY == 1 )); then
  echo "Runtime sincronizado; instalacion y setup omitidos por -SyncOnly."
  exit 0
fi
RUNTIME_ROOT="$(tarantulin_resolve_runtime "${RUNTIME_REQUESTED}")"
workspace="${RUNTIME_ROOT}/workspace"
export TARANTULIN_BACKEND_PROFILE="${ACCELERATOR}"
if (( ENABLE_EXPERIMENTAL_AMD_WSL == 1 )); then export TARANTULIN_ENABLE_EXPERIMENTAL_AMD_WSL=1; fi
if (( SKIP_GPU_CHECK == 1 )); then export TARANTULIN_SKIP_BACKEND_CHECK=1; fi
# shellcheck source=../platform.sh
source "${workspace}/scripts/platform.sh"
resolved_accelerator="$(tarantulin_resolve_accelerator)"
case "${resolved_accelerator}" in auto|nvidia|amd|intel|cpu) ;; *) echo "Perfil resuelto no valido." >&2; exit 2 ;; esac
tarantulin_accelerator_preflight "${resolved_accelerator}"
profile_tmp="${RUNTIME_ROOT}/accelerator.profile.tmp.$$"
printf '%s\n' "${resolved_accelerator}" > "${profile_tmp}"
mv -f "${profile_tmp}" "${RUNTIME_ROOT}/accelerator.profile"
echo "Perfil persistido para este runtime: ${resolved_accelerator}"
if [[ "${resolved_accelerator}" == "amd" && "${ENABLE_EXPERIMENTAL_AMD_WSL}" == "1" ]]; then
  printf '%s\n' 'accepted=amd-wsl-experimental-v1' > "${RUNTIME_ROOT}/amd-wsl-experimental.opt-in"
else
  rm -f -- "${RUNTIME_ROOT}/amd-wsl-experimental.opt-in"
fi

if (( RUN_SETUP == 0 )); then
  echo "Runtime creado; setup Python/MJX omitido por -NoSetup."
  exit 0
fi

install_args=(--skip-system-packages --accelerator "${resolved_accelerator}")
if (( SKIP_GPU_CHECK == 1 )); then install_args+=(--skip-gpu-check); fi
if (( ENABLE_EXPERIMENTAL_AMD_WSL == 1 )); then install_args+=(--enable-experimental-amd-wsl); fi
"${workspace}/scripts/install_wsl.sh" "${install_args[@]}"

echo "Runtime preparado en: ${workspace}"
