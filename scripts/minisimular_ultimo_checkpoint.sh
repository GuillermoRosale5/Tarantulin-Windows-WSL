#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
XML_IDEAL="${REPO_ROOT}/tarantulin/xmls/TARANTULIN_POSE_IDEAL.xml"

cd "${REPO_ROOT}"

select_from_list() {
  local prompt="$1"
  local count="$2"
  local selected
  read -r -p "${prompt}" selected
  if [[ ! "${selected}" =~ ^[0-9]+$ ]] || (( selected < 1 || selected > count )); then
    echo "Opcion no valida: ${selected}. Elige un numero entre 1 y ${count}." >&2
    exit 1
  fi
  printf '%s\n' "${selected}"
}

if [[ ! -f "${XML_IDEAL}" ]]; then
  echo "No encuentro el XML ideal: ${XML_IDEAL}" >&2
  exit 1
fi

echo ""
echo "Pose inicial de la minisimulacion:"
echo "  1) Actual del checkpoint: random a cierta altura"
echo "  2) Tumbada en suelo, tipo POSE_SUELO2"
echo "  3) Pose ideal de TARANTULIN_POSE_IDEAL.xml"
echo "  4) Caida de lado, muy inclinado y random"
echo "  5) Boca abajo en pose random"
echo ""

pose_choice="$(select_from_list "Elige pose inicial [1-5]: " 5)"
case "${pose_choice}" in
  1) reset_preset="actual" ;;
  2) reset_preset="suelo2" ;;
  3) reset_preset="ideal" ;;
  4) reset_preset="caida_lateral" ;;
  5) reset_preset="boca_abajo" ;;
esac

extra_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --xml-path)
      shift 2
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

echo "Minisimulando ultimo checkpoint con XML fijo: ${XML_IDEAL}"
echo "Pose inicial: ${reset_preset}"
exec ./scripts/tarantulin_wsl.sh mini-sim \
  --xml-path "${XML_IDEAL}" \
  --reset-preset "${reset_preset}" \
  "${extra_args[@]}"
