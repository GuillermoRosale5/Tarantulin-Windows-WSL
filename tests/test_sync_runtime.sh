#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${1:?uso: test_sync_runtime.sh SOURCE_ROOT SOURCE_ID}"
SOURCE_ID="${2:?uso: test_sync_runtime.sh SOURCE_ROOT SOURCE_ID}"
BASE="${HOME}/.local/share/tarantulin-windows"
mkdir -p "${BASE}"
TEST_RUNTIME="$(mktemp -d -p "${BASE}" codex-sync-test.XXXXXX)"
MARKERLESS="$(mktemp -d -p "${BASE}" codex-markerless-test.XXXXXX)"

cleanup() {
  case "${TEST_RUNTIME}" in "${BASE}"/codex-sync-test.*) rm -rf -- "${TEST_RUNTIME}" ;; esac
  case "${MARKERLESS}" in "${BASE}"/codex-markerless-test.*) rm -rf -- "${MARKERLESS}" ;; esac
}
trap cleanup EXIT

bash "${SOURCE_ROOT}/scripts/wsl/sync_runtime.sh" \
  --source "${SOURCE_ROOT}" --runtime "${TEST_RUNTIME}" --source-id "${SOURCE_ID}" >/dev/null
test -f "${TEST_RUNTIME}/.tarantulin-runtime"
test -f "${TEST_RUNTIME}/sync-manifest.sha256"
test -f "${TEST_RUNTIME}/workspace/pyproject.toml"

mkdir -p \
  "${TEST_RUNTIME}/workspace/external" \
  "${TEST_RUNTIME}/workspace/logs_tarantulin_mjx" \
  "${TEST_RUNTIME}/workspace/.venv"
printf keep > "${TEST_RUNTIME}/workspace/external/keep"
printf keep > "${TEST_RUNTIME}/workspace/logs_tarantulin_mjx/keep"
printf keep > "${TEST_RUNTIME}/workspace/.venv/keep"
printf stale > "${TEST_RUNTIME}/workspace/stale.tmp"

bash "${SOURCE_ROOT}/scripts/wsl/sync_runtime.sh" \
  --source "${SOURCE_ROOT}" --runtime "${TEST_RUNTIME}" --source-id "${SOURCE_ID}" >/dev/null
test -f "${TEST_RUNTIME}/workspace/external/keep"
test -f "${TEST_RUNTIME}/workspace/logs_tarantulin_mjx/keep"
test -f "${TEST_RUNTIME}/workspace/.venv/keep"
test ! -e "${TEST_RUNTIME}/workspace/stale.tmp"

if bash "${SOURCE_ROOT}/scripts/wsl/sync_runtime.sh" \
  --source "${SOURCE_ROOT}" --runtime "${TEST_RUNTIME}" --source-id deadbeefdeadbeef >/dev/null 2>&1; then
  echo "ERROR: se acepto un source-id incorrecto" >&2
  exit 1
fi

printf precious > "${MARKERLESS}/precious.txt"
if bash "${SOURCE_ROOT}/scripts/wsl/sync_runtime.sh" \
  --source "${SOURCE_ROOT}" --runtime "${MARKERLESS}" --source-id "${SOURCE_ID}" >/dev/null 2>&1; then
  echo "ERROR: se acepto un destino no vacio sin marcador" >&2
  exit 1
fi
test "$(cat "${MARKERLESS}/precious.txt")" = precious

if bash "${SOURCE_ROOT}/scripts/wsl/sync_runtime.sh" \
  --source "${SOURCE_ROOT}" --runtime "${HOME}" --source-id "${SOURCE_ID}" >/dev/null 2>&1; then
  echo "ERROR: se acepto HOME como destino" >&2
  exit 1
fi

# Este bloque comprueba la coordinación del runtime, no el kernel de WSL.
# En GitHub Actions simulamos únicamente la señal WSL_INTEROP; las pruebas
# reales Windows->WSL se ejecutan además sobre el equipo del proyecto.
WSL_INTEROP="${WSL_INTEROP:-/tmp/tarantulin-ci-wsl-interop}" \
  bash "${SOURCE_ROOT}/scripts/wsl/bootstrap_runtime.sh" \
  --source "${SOURCE_ROOT}" --runtime "${TEST_RUNTIME}" --source-id "${SOURCE_ID}" \
  --accelerator cpu --no-setup --skip-system-packages >/dev/null
test "$(cat "${TEST_RUNTIME}/accelerator.profile")" = cpu

echo "SYNC_SAFETY_TESTS_OK"
