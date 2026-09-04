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

# shellcheck source=../scripts/tarantulin_wsl.sh
source "${SOURCE_ROOT}/scripts/tarantulin_wsl.sh"

REPO_ROOT="${TEST_ROOT}/workspace"
PROJECT_DIR="${REPO_ROOT}"
LOGS_DIR="${REPO_ROOT}/logs_tarantulin_mjx"
LAST_RUN="${LOGS_DIR}/ultima_run.txt"
mkdir -p "${LOGS_DIR}" "${TEST_ROOT}/outside"
printf 'intacto\n' > "${TEST_ROOT}/outside/sentinel.txt"

if (validate_run_name "../escape") >/dev/null 2>&1; then
  echo "ERROR: se acepto traversal en run-name" >&2
  exit 1
fi
validate_run_name "run-segura_01.2"

ln -s "${TEST_ROOT}/outside" "${LOGS_DIR}/run-enlazada"
if assert_run_path_safe "${LOGS_DIR}/run-enlazada" >/dev/null 2>&1; then
  echo "ERROR: se acepto una run enlazada fuera de logs" >&2
  exit 1
fi
test "$(cat "${TEST_ROOT}/outside/sentinel.txt")" = intacto

printf '%s\n' "${TEST_ROOT}/outside" > "${LAST_RUN}"
test -z "$(current_run 2>/dev/null)"

mkdir -p "${LOGS_DIR}/run-segura"
rm -f -- "${LAST_RUN}"
ln -s "${TEST_ROOT}/outside/sentinel.txt" "${LAST_RUN}"
write_last_run "${LOGS_DIR}/run-segura"
test ! -L "${LAST_RUN}"
test "$(cat "${TEST_ROOT}/outside/sentinel.txt")" = intacto
test "$(cat "${LAST_RUN}")" = "${LOGS_DIR}/run-segura"

echo "RUN_PATH_SAFETY_TESTS_OK"
