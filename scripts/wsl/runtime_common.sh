#!/usr/bin/env bash

tarantulin_expand_home() {
  local value="${1:-}"
  case "${value}" in
    "~") printf '%s\n' "${HOME}" ;;
    "~/"*) printf '%s/%s\n' "${HOME}" "${value#\~/}" ;;
    *) printf '%s\n' "${value}" ;;
  esac
}

tarantulin_runtime_base() {
  realpath -m "${TARANTULIN_RUNTIME_BASE:-${HOME}/.local/share/tarantulin-windows}"
}

tarantulin_resolve_runtime() {
  local requested
  requested="$(tarantulin_expand_home "${1:-}")"
  realpath -m "${requested}"
}

tarantulin_assert_safe_runtime_path() {
  local runtime="$1"
  local base
  base="$(tarantulin_runtime_base)"
  case "${runtime}" in
    "${base}/"*) ;;
    *)
      echo "Ruta runtime rechazada por seguridad: ${runtime}" >&2
      echo "Debe estar dentro de: ${base}/" >&2
      return 1
      ;;
  esac
  [[ "${runtime}" != "/" && "${runtime}" != "${HOME}" && "${runtime}" != "${base}" ]] || {
    echo "La ruta runtime es demasiado amplia: ${runtime}" >&2
    return 1
  }
}

tarantulin_validate_source_id() {
  [[ "${1:-}" =~ ^[0-9a-f]{16}$ ]] || {
    echo "source-id no valido: ${1:-<vacio>}" >&2
    return 1
  }
}

tarantulin_validate_runtime_marker() {
  local runtime="$1"
  local source_id="$2"
  local marker="${runtime}/.tarantulin-runtime"
  [[ -f "${marker}" ]] || {
    echo "El runtime no esta inicializado o no tiene marcador seguro: ${runtime}" >&2
    echo "Ejecuta .\\install.ps1 desde la carpeta Windows." >&2
    return 1
  }
  grep -qx 'kind=tarantulin-windows-runtime' "${marker}" || {
    echo "Marcador runtime desconocido: ${marker}" >&2
    return 1
  }
  grep -qx "source_id=${source_id}" "${marker}" || {
    echo "El runtime pertenece a otra carpeta Windows; se cancela la operacion." >&2
    return 1
  }
}

tarantulin_require_wsl() {
  if [[ -z "${WSL_INTEROP:-}" ]] && ! grep -qi microsoft /proc/version 2>/dev/null; then
    echo "Este componente debe ejecutarse dentro de WSL." >&2
    return 1
  fi
}

tarantulin_assert_windows_mount() {
  local path="$1"
  local fstype
  command -v findmnt >/dev/null 2>&1 || {
    echo "Falta findmnt (util-linux) para validar el filesystem Windows." >&2
    return 1
  }
  fstype="$(findmnt --target "${path}" --noheadings --output FSTYPE 2>/dev/null | head -n 1 | tr -d '[:space:]')"
  case "${fstype}" in
    9p|drvfs|v9fs|virtiofs) ;;
    *)
      echo "La ruta debe estar en un filesystem Windows montado por WSL." >&2
      echo "Ruta: ${path}; tipo detectado: ${fstype:-desconocido}" >&2
      return 1
      ;;
  esac
}
