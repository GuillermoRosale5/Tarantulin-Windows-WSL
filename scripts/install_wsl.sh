#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUN_SETUP=1
INSTALL_SYSTEM_PACKAGES=1
SKIP_GPU_CHECK=0
ACCELERATOR="${TARANTULIN_BACKEND_PROFILE:-auto}"

usage() {
  cat <<'EOF'
Uso:
  scripts/install_wsl.sh [--no-setup] [--skip-system-packages]
                         [--accelerator auto|nvidia|amd|intel|cpu]
                         [--skip-gpu-check]

Instala paquetes base de Ubuntu/WSL y, por defecto, ejecuta:
  scripts/tarantulin.sh setup

Opciones:
  --no-setup             Solo instala paquetes del sistema y permisos.
  --skip-system-packages No ejecuta apt (lo usa el bootstrap Windows).
  --accelerator PERFIL   Seleccion de backend JAX.
  --skip-gpu-check       Instala aunque el backend solicitado aun no sea visible.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-setup) RUN_SETUP=0; shift ;;
    --skip-system-packages) INSTALL_SYSTEM_PACKAGES=0; shift ;;
    --accelerator) ACCELERATOR="$2"; shift 2 ;;
    --skip-gpu-check) SKIP_GPU_CHECK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Argumento no reconocido: $1" >&2; usage >&2; exit 1 ;;
  esac
done

ensure_wsl() {
  if [[ -z "${WSL_INTEROP:-}" ]] && ! grep -qi microsoft /proc/version 2>/dev/null; then
    echo "Este instalador esta pensado para ejecutarse dentro de WSL." >&2
    exit 1
  fi
}

ensure_linux_fs() {
  local path
  path="$(realpath "${REPO_ROOT}")"
  # shellcheck source=platform.sh
  source "${SCRIPT_DIR}/platform.sh"
  if tarantulin_path_is_windows_mount "${path}"; then
    echo "No ejecutes el proyecto desde un filesystem Windows: ${path}" >&2
    echo "Ejecuta install.ps1 desde Windows para crear automaticamente el runtime ext4." >&2
    exit 1
  fi
}

install_apt_packages() {
  local packages=(
    ca-certificates
    curl
    git
    rsync
    build-essential
    python3
    python3-venv
    python3-pip
    pkg-config
    util-linux
    libgl1
    libegl1
    libglfw3
    libglew2.2
    pciutils
  )

  if ! command -v sudo >/dev/null 2>&1; then
    echo "No encuentro sudo. Instala sudo o ejecuta como usuario con permisos." >&2
    exit 1
  fi

  local apt_options=(
    -o Acquire::Retries=3
    -o Acquire::http::Timeout=30
    -o Acquire::https::Timeout=30
    -o DPkg::Lock::Timeout=120
  )
  echo "apt puede tardar varios minutos; mientras muestre actividad no esta bloqueado."
  sudo -v
  sudo apt-get "${apt_options[@]}" update
  sudo env DEBIAN_FRONTEND=noninteractive apt-get "${apt_options[@]}" install -y "${packages[@]}"
}

ensure_wsl
ensure_linux_fs
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
fi
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
  echo "Esta variante necesita Ubuntu 24.04 bajo WSL; detectado ${PRETTY_NAME:-desconocido}." >&2
  exit 1
fi
case "${ACCELERATOR}" in auto|nvidia|amd|intel|cpu) ;; *) echo "Acelerador no valido: ${ACCELERATOR}" >&2; exit 2 ;; esac
if (( INSTALL_SYSTEM_PACKAGES == 1 )); then install_apt_packages; fi
chmod +x "${REPO_ROOT}"/scripts/*.sh
chmod +x "${REPO_ROOT}"/scripts/wsl/*.sh 2>/dev/null || true

if (( RUN_SETUP == 1 )); then
  export TARANTULIN_BACKEND_PROFILE="${ACCELERATOR}"
  if (( SKIP_GPU_CHECK == 1 )); then export TARANTULIN_SKIP_BACKEND_CHECK=1; fi
  "${REPO_ROOT}/scripts/tarantulin.sh" setup
else
  echo "Instalacion base OK. Setup completo omitido por --no-setup."
fi
