#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runtime_common.sh
source "${SCRIPT_DIR}/runtime_common.sh"

SOURCE_ROOT=""
RUNTIME_REQUESTED=""
SOURCE_ID=""
ACCELERATOR="auto"
SKIP_GPU_CHECK=0
ENABLE_EXPERIMENTAL_AMD_WSL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE_ROOT="$2"; shift 2 ;;
    --runtime) RUNTIME_REQUESTED="$2"; shift 2 ;;
    --source-id) SOURCE_ID="$2"; shift 2 ;;
    --accelerator) ACCELERATOR="$2"; shift 2 ;;
    --skip-gpu-check) SKIP_GPU_CHECK=1; shift ;;
    --enable-experimental-amd-wsl) ENABLE_EXPERIMENTAL_AMD_WSL=1; shift ;;
    --) shift; break ;;
    *) echo "Argumento interno no reconocido: $1" >&2; exit 2 ;;
  esac
done

[[ $# -gt 0 ]] || { echo "Falta el comando TARANTULIN." >&2; exit 2; }
tarantulin_require_wsl
tarantulin_validate_source_id "${SOURCE_ID}"
RUNTIME_ROOT="$(tarantulin_resolve_runtime "${RUNTIME_REQUESTED}")"
tarantulin_assert_safe_runtime_path "${RUNTIME_ROOT}"
tarantulin_validate_runtime_marker "${RUNTIME_ROOT}" "${SOURCE_ID}"
WORKSPACE="${RUNTIME_ROOT}/workspace"
[[ -d "${WORKSPACE}" && ! -L "${WORKSPACE}" ]] || {
  echo "Workspace runtime ausente o inseguro: ${WORKSPACE}" >&2
  exit 1
}

if [[ "${ACCELERATOR}" == "auto" && -f "${RUNTIME_ROOT}/accelerator.profile" ]]; then
  persisted="$(tr -d '\r\n' < "${RUNTIME_ROOT}/accelerator.profile")"
  case "${persisted}" in
    nvidia|amd|intel|cpu) ACCELERATOR="${persisted}" ;;
    *) echo "Perfil persistido no valido: ${persisted}" >&2; exit 1 ;;
  esac
fi
export TARANTULIN_BACKEND_PROFILE="${ACCELERATOR}"
if (( SKIP_GPU_CHECK == 1 )); then export TARANTULIN_SKIP_BACKEND_CHECK=1; fi
if (( ENABLE_EXPERIMENTAL_AMD_WSL == 1 )); then export TARANTULIN_ENABLE_EXPERIMENTAL_AMD_WSL=1; fi
if [[ "${ACCELERATOR}" == "amd" && -f "${RUNTIME_ROOT}/amd-wsl-experimental.opt-in" ]] && \
   grep -qx 'accepted=amd-wsl-experimental-v1' "${RUNTIME_ROOT}/amd-wsl-experimental.opt-in"; then
  export TARANTULIN_ENABLE_EXPERIMENTAL_AMD_WSL=1
fi

command="$1"
shift
case "${command}" in
  doctor)
    exec "${WORKSPACE}/scripts/doctor.sh" "$@"
    ;;
  pull-results)
    exec "${WORKSPACE}/scripts/wsl/export_results.sh" \
      --source "${SOURCE_ROOT}" --runtime "${RUNTIME_ROOT}" --source-id "${SOURCE_ID}" "$@"
    ;;
  *)
    cd "${WORKSPACE}"
    exec "${WORKSPACE}/scripts/tarantulin_wsl.sh" "${command}" "$@"
    ;;
esac
