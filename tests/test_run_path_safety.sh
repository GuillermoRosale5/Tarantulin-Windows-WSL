#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${1:?uso: test_run_path_safety.sh SOURCE_ROOT}"
BASE="${HOME}/.local/share/tarantulin-windows"
mkdir -p "${BASE}"
TEST_ROOT="$(mktemp -d -p "${BASE}" codex-run-safety-test.XXXXXX)"

cleanup() {
  case "${TEST_ROOT}" in
    "${BASE}"/codex-run-safety-test.*) rm -rf -- "${TEST_ROOT}" ;;
  esac
}
trap cleanup EXIT

# shellcheck source=../scripts/tarantulin.sh
source "${SOURCE_ROOT}/scripts/tarantulin.sh"

REPO_ROOT="${TEST_ROOT}/workspace"
PROJECT_DIR="${REPO_ROOT}"
LOGS_DIR="${REPO_ROOT}/logs_tarantulin_mjx"
LAST_RUN="${LOGS_DIR}/ultima_run.txt"
mkdir -p "${LOGS_DIR}" "${TEST_ROOT}/outside"
printf 'intacto\n' > "${TEST_ROOT}/outside/sentinel.txt"

if (validar_nombre_ejecucion "../escape") >/dev/null 2>&1; then
  echo "ERROR: se acepto una ruta insegura en el nombre de ejecución" >&2
  exit 1
fi
validar_nombre_ejecucion "run-segura_01.2"

ln -s "${TEST_ROOT}/outside" "${LOGS_DIR}/run-enlazada"
if comprobar_ruta_ejecucion_segura "${LOGS_DIR}/run-enlazada" >/dev/null 2>&1; then
  echo "ERROR: se acepto una ejecucion enlazada fuera de logs" >&2
  exit 1
fi
test "$(cat "${TEST_ROOT}/outside/sentinel.txt")" = intacto

printf '%s\n' "${TEST_ROOT}/outside" > "${LAST_RUN}"
test -z "$(ejecucion_actual 2>/dev/null)"

mkdir -p "${LOGS_DIR}/run-segura"
rm -f -- "${LAST_RUN}"
ln -s "${TEST_ROOT}/outside/sentinel.txt" "${LAST_RUN}"
escribir_ultima_ejecucion "${LOGS_DIR}/run-segura"
test ! -L "${LAST_RUN}"
test "$(cat "${TEST_ROOT}/outside/sentinel.txt")" = intacto
test "$(cat "${LAST_RUN}")" = "${LOGS_DIR}/run-segura"

echo "RUN_PATH_SAFETY_TESTS_OK"
