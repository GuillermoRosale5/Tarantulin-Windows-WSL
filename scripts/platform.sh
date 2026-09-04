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
      if [[ "${TARANTULIN_ENABLE_EXPERIMENTAL_AMD_WSL:-0}" != "1" ]]; then
        echo "AMD/ROCm con JAX no esta habilitado ni validado oficialmente bajo WSL." >&2
        echo "Usa -Accelerator cpu. Solo para una prueba consciente puedes anadir -EnableExperimentalAmdWsl." >&2
        return 1
      fi
      echo "Aviso: modo AMD/ROCm experimental habilitado expresamente; puede no instalar, inicializar o ejecutar." >&2
      if ! tarantulin_is_wsl; then
        echo "Este repositorio espera WSL; usa el repositorio Linux-WSL para Linux nativo." >&2
        return 1
      fi
      if ! tarantulin_has_amd_rocm; then
        if [[ "${skip}" == "1" ]]; then
          echo "Aviso: ROCm no esta operativo; se omite la comprobacion por configuracion." >&2
          return 0
        fi
        echo "ROCm no expone una GPU AMD a WSL (rocminfo)." >&2
        echo "Comprueba primero la matriz oficial AMD para WSL; si tu GPU no es compatible usa -Accelerator cpu." >&2
        return 1
      fi
      ;;
    intel)
      echo "La aceleracion Intel GPU no esta soportada por JAX/XLA en WSL para este proyecto." >&2
      echo "Usa -Accelerator cpu en Windows/WSL o el repositorio Linux-WSL para probar el backend Intel experimental en Linux nativo." >&2
      return 1
      ;;
    cpu) ;;
    *) echo "Perfil interno desconocido: ${profile}" >&2; return 2 ;;
  esac
}
