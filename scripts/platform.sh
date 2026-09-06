#!/usr/bin/env bash

tarantulin_is_wsl() {
  [[ -n "${WSL_INTEROP:-}" ]] || grep -qi microsoft /proc/version 2>/dev/null
}

tarantulin_path_is_windows_mount() {
  local path="$1"
  local fstype
  command -v findmnt >/dev/null 2>&1 || return 1
  fstype="$(findmnt --target "${path}" --noheadings --output FSTYPE 2>/dev/null | head -n 1 | tr -d '[:space:]')"
  case "${fstype}" in
    9p|drvfs|v9fs|virtiofs) return 0 ;;
    *) return 1 ;;
  esac
}

tarantulin_requested_accelerator() {
  local requested="${TARANTULIN_BACKEND_PROFILE:-auto}"
  requested="${requested,,}"
  case "${requested}" in
    auto|nvidia|amd|intel|cpu) printf '%s\n' "${requested}" ;;
    *)
      echo "Perfil de acelerador no valido: ${requested}" >&2
      echo "Usa: auto, nvidia, amd, intel o cpu." >&2
      return 2
      ;;
  esac
}

tarantulin_has_nvidia() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

tarantulin_has_amd_rocm() {
  command -v rocminfo >/dev/null 2>&1 && rocminfo 2>/dev/null | grep -Eq 'Name:[[:space:]]+gfx|Marketing Name:'
}

tarantulin_resolve_accelerator() {
  local requested
  requested="$(tarantulin_requested_accelerator)" || return
  if [[ "${requested}" != "auto" ]]; then
    printf '%s\n' "${requested}"
    return
  fi
  if tarantulin_has_nvidia; then
    printf '%s\n' nvidia
  else
    printf '%s\n' cpu
  fi
}

tarantulin_accelerator_preflight() {
  local profile="${1:-$(tarantulin_resolve_accelerator)}"
  local skip="${TARANTULIN_SKIP_BACKEND_CHECK:-0}"
  case "${profile}" in
    nvidia)
      if ! tarantulin_has_nvidia; then
        if [[ "${skip}" == "1" ]]; then
          echo "Aviso: NVIDIA no esta visible; se omite la comprobacion por configuracion." >&2
          return 0
        fi
        echo "No se detecta una GPU NVIDIA utilizable desde WSL (nvidia-smi -L)." >&2
        echo "Instala/actualiza el driver NVIDIA de Windows con soporte WSL o usa -Accelerator cpu." >&2
        return 1
      fi
      ;;
    amd)
      echo "La aceleracion AMD/ROCm con JAX no esta admitida por esta version bajo WSL2." >&2
      echo "Usa -Accelerator cpu en Windows/WSL o el repositorio Linux/WSL desde Ubuntu 24.04 nativo." >&2
      return 1
      ;;
    intel)
      echo "La aceleracion Intel GPU no esta soportada por JAX/XLA en WSL para este proyecto." >&2
      echo "Usa -Accelerator cpu; esta pila JAX/MJX tampoco habilita la GPU Intel en Ubuntu nativo." >&2
      return 1
      ;;
    cpu) ;;
    *) echo "Perfil interno desconocido: ${profile}" >&2; return 2 ;;
  esac
}
