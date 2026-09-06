#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${1:?uso: test_process_identity.sh SOURCE_ROOT}"
BASE="${HOME}/.local/share/tarantulin-windows"
mkdir -p "${BASE}"
TEST_ROOT="$(mktemp -d -p "${BASE}" codex-process-test.XXXXXX)"
spawned_identities=()

cleanup() {
  local identity pid starttime
  for identity in "${spawned_identities[@]}"; do
    IFS=: read -r pid starttime <<< "${identity}"
    if tarantulin_process_has_starttime "${pid}" "${starttime}" 2>/dev/null; then
      kill "${pid}" >/dev/null 2>&1 || true
      wait "${pid}" >/dev/null 2>&1 || true
    fi
  done
  case "${TEST_ROOT}" in
    "${BASE}"/codex-process-test.*) rm -rf -- "${TEST_ROOT}" ;;
  esac
}
trap cleanup EXIT

# shellcheck source=../scripts/tarantulin.sh
source "${SOURCE_ROOT}/scripts/tarantulin.sh"

REPO_ROOT="${TEST_ROOT}/workspace"
LOGS_DIR="${REPO_ROOT}/logs_tarantulin_mjx"
LAST_RUN="${LOGS_DIR}/ultima_run.txt"
RUN_DIR="${LOGS_DIR}/run-identidad"
mkdir -p "${REPO_ROOT}/tarantulin" "${RUN_DIR}"
printf '%s\n' 'import time' 'time.sleep(60)' > "${REPO_ROOT}/tarantulin/entrenar_ppo_mjx.py"
printf '%s\n' "${RUN_DIR}" > "${LAST_RUN}"

# Un PID vivo pero ajeno nunca debe recibir SIGTERM.
sleep 60 &
decoy_pid=$!
decoy_start="$(tarantulin_process_starttime "${decoy_pid}")"
spawned_identities+=("${decoy_pid}:${decoy_start}")
printf '%s\n' "${decoy_pid}" > "${RUN_DIR}/entrenamiento.pid"
printf '%s\n' "${decoy_start}" > "${RUN_DIR}/entrenamiento.starttime"
parar_entrenamiento > "${TEST_ROOT}/stale-stop.log" 2>&1
kill -0 "${decoy_pid}"
grep -q 'No hay PID guardado ni proceso de TARANTULIN activo' "${TEST_ROOT}/stale-stop.log"

# Incluso con comando/cwd correctos, un starttime distinto simula PID reciclado
# y debe provocar rechazo sin senal.
(
  cd "${REPO_ROOT}"
  exec python3 "${REPO_ROOT}/tarantulin/entrenar_ppo_mjx.py" \
    --logdir "${LOGS_DIR}" --run_name "$(basename -- "${RUN_DIR}")"
) &
reused_pid=$!
printf '%s\n' "${reused_pid}" > "${RUN_DIR}/entrenamiento.pid"
reused_start="$(tarantulin_process_starttime "${reused_pid}")"
spawned_identities+=("${reused_pid}:${reused_start}")
printf '%s\n' "$((reused_start + 1))" > "${RUN_DIR}/entrenamiento.starttime"
parar_entrenamiento > "${TEST_ROOT}/reused-stop.log" 2>&1
kill -0 "${reused_pid}"

# Un trainer con script, cwd, logdir, run_name y starttime exactos si se detiene.
(
  cd "${REPO_ROOT}"
  exec python3 "${REPO_ROOT}/tarantulin/entrenar_ppo_mjx.py" \
    --logdir "${LOGS_DIR}" --run_name "$(basename -- "${RUN_DIR}")"
) &
trainer_pid=$!
printf '%s\n' "${trainer_pid}" > "${RUN_DIR}/entrenamiento.pid"
trainer_start="$(tarantulin_process_starttime "${trainer_pid}")"
spawned_identities+=("${trainer_pid}:${trainer_start}")
printf '%s\n' "${trainer_start}" > "${RUN_DIR}/entrenamiento.starttime"
# El puntero visible puede estar obsoleto: parar debe localizar el par validado
# de la otra ejecucion sin mezclar ruta y PID.
STALE_RUN="${LOGS_DIR}/run-puntero-stale"
mkdir -p "${STALE_RUN}"
printf '%s\n' "${STALE_RUN}" > "${LAST_RUN}"
parar_entrenamiento > "${TEST_ROOT}/valid-stop.log" 2>&1
wait "${trainer_pid}" >/dev/null 2>&1 || true
if kill -0 "${trainer_pid}" >/dev/null 2>&1; then
  echo "ERROR: el trainer validado sigue activo" >&2
  exit 1
fi

# Un PID ajeno guardado como proteccion termica tampoco se senaliza.
(
  cd "${REPO_ROOT}"
  exec python3 "${REPO_ROOT}/tarantulin/entrenar_ppo_mjx.py" \
    --logdir "${LOGS_DIR}" --run_name "$(basename -- "${RUN_DIR}")"
) &
trainer_pid_2=$!
printf '%s\n' "${trainer_pid_2}" > "${RUN_DIR}/entrenamiento.pid"
trainer_start_2="$(tarantulin_process_starttime "${trainer_pid_2}")"
spawned_identities+=("${trainer_pid_2}:${trainer_start_2}")
printf '%s\n' "${trainer_start_2}" > "${RUN_DIR}/entrenamiento.starttime"
sleep 60 &
guard_decoy_pid=$!
guard_decoy_start="$(tarantulin_process_starttime "${guard_decoy_pid}")"
spawned_identities+=("${guard_decoy_pid}:${guard_decoy_start}")
printf '%s\n' "${guard_decoy_pid}" > "${RUN_DIR}/proteccion_termica.pid"
printf '%s\n' "${guard_decoy_start}" > "${RUN_DIR}/proteccion_termica.starttime"
parar_entrenamiento > "${TEST_ROOT}/guard-stop.log" 2>&1
kill -0 "${guard_decoy_pid}"
grep -q 'no corresponde a la proteccion' "${TEST_ROOT}/guard-stop.log"

# La identidad del supervisor incluye el nombre exacto de ejecucion. Otro
# curriculo con el mismo script no se puede confundir con este.
mkdir -p "${REPO_ROOT}/scripts"
printf '%s\n' '#!/usr/bin/env bash' 'sleep 60' > \
  "${REPO_ROOT}/scripts/curriculo_automatico_tarantulin.sh"
chmod +x "${REPO_ROOT}/scripts/curriculo_automatico_tarantulin.sh"
(
  cd "${REPO_ROOT}"
  exec bash "${REPO_ROOT}/scripts/curriculo_automatico_tarantulin.sh" \
    --nombre-ejecucion "$(basename -- "${RUN_DIR}")"
) &
curriculum_pid=$!
curriculum_start="$(tarantulin_process_starttime "${curriculum_pid}")"
spawned_identities+=("${curriculum_pid}:${curriculum_start}")
tarantulin_curriculum_process_matches \
  "${curriculum_pid}" "${REPO_ROOT}/scripts/curriculo_automatico_tarantulin.sh" \
  "${RUN_DIR}" "${curriculum_start}"

(
  cd "${REPO_ROOT}"
  exec bash "${REPO_ROOT}/scripts/curriculo_automatico_tarantulin.sh" \
    --nombre-ejecucion otra-ejecucion
) &
curriculum_decoy_pid=$!
curriculum_decoy_start="$(tarantulin_process_starttime "${curriculum_decoy_pid}")"
spawned_identities+=("${curriculum_decoy_pid}:${curriculum_decoy_start}")
if tarantulin_curriculum_process_matches \
  "${curriculum_decoy_pid}" "${REPO_ROOT}/scripts/curriculo_automatico_tarantulin.sh" \
  "${RUN_DIR}" "${curriculum_decoy_start}"; then
  echo "ERROR: se acepto el supervisor de otra ejecucion" >&2
  exit 1
fi

echo "PROCESS_IDENTITY_TESTS_OK"
