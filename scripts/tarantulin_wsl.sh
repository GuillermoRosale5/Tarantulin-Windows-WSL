#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="${REPO_ROOT}"
VENV_PYTHON="${PROJECT_DIR}/.venv/bin/python"
PLAYGROUND_COMMIT="9c2dce4a3519cd4bb9d299bf28a6ef3f5086844b"
UV_VERSION="0.11.8"
LOGS_DIR="${REPO_ROOT}/logs_tarantulin_mjx"
LAST_RUN="${LOGS_DIR}/ultima_run.txt"
DEFAULT_RUN_PREFIX="TarantulinStandup"
DEFAULT_NUM_TIMESTEPS=30000000
DEFAULT_EPISODE_LENGTH=1000
PRETRAINED_MODEL_DIR="${REPO_ROOT}/pretrained/tarantulin_standup_fase2_45932544"
PRETRAINED_CHECKPOINT="${PRETRAINED_MODEL_DIR}/checkpoints/000045932544"
PRETRAINED_EPISODE_LENGTH=1500

export PATH="${HOME}/.local/bin:${PATH}"
# shellcheck source=platform.sh
source "${SCRIPT_DIR}/platform.sh"
# shellcheck source=process_identity.sh
source "${SCRIPT_DIR}/process_identity.sh"

ensure_wsl() {
  if ! tarantulin_is_wsl; then
    echo "Este script debe ejecutarse dentro de WSL." >&2
    exit 1
  fi
}

ensure_linux_fs() {
  local path
  path="$(realpath "${REPO_ROOT}")"
  if tarantulin_path_is_windows_mount "${path}"; then
    echo "No ejecutes este proyecto desde un filesystem Windows: ${path}" >&2
    echo "Usa .\\tarantulin.ps1 desde la carpeta Windows; el runtime ext4 se crea automaticamente." >&2
    exit 1
  fi
}

ensure_uv() {
  local actual="" actual_version=""
  if command -v uv >/dev/null 2>&1; then
    actual="$(uv --version 2>/dev/null || true)"
    actual_version="$(awk '{print $2}' <<< "${actual}")"
  fi
  if [[ "${actual_version}" == "${UV_VERSION}" ]]; then
    return
  fi
  if [[ -n "${actual}" ]]; then
    echo "Se encontro ${actual}; instalando la version reproducible uv ${UV_VERSION}..." >&2
  else
    echo "Instalando uv ${UV_VERSION}..."
  fi
  UV_NO_MODIFY_PATH=1 curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | \
    env UV_NO_MODIFY_PATH=1 sh
  export PATH="${HOME}/.local/bin:${PATH}"
  hash -r
  actual="$(uv --version 2>/dev/null || true)"
  actual_version="$(awk '{print $2}' <<< "${actual}")"
  [[ "${actual_version}" == "${UV_VERSION}" ]] || {
    echo "Se esperaba uv ${UV_VERSION}, pero se obtuvo '${actual:-no disponible}'." >&2
    exit 1
  }
}

ensure_gpu_visible() {
  local profile
  profile="$(tarantulin_resolve_accelerator)"
  tarantulin_accelerator_preflight "${profile}"
}

ensure_playground() {
  ensure_uv
  [[ -f "${PROJECT_DIR}/pyproject.toml" && -f "${PROJECT_DIR}/uv.lock" ]] || {
    echo "Faltan pyproject.toml o uv.lock en ${PROJECT_DIR}." >&2
    exit 1
  }
}

sync_env() {
  ensure_playground
  cd "${PROJECT_DIR}"
  local profile
  profile="$(tarantulin_resolve_accelerator)"
  tarantulin_accelerator_preflight "${profile}"
  case "${profile}" in
    nvidia)
      uv sync --frozen --extra nvidia --no-dev
      ;;
    cpu)
      uv sync --frozen --no-dev
      ;;
    amd)
      # Primero vuelve exactamente al lock CPU para retirar overlays CUDA/ROCm
      # de perfiles anteriores; despues instala las tres wheels ROCm fijadas.
      uv sync --frozen --no-dev
      echo "Instalando overlay ROCm fijado y solicitado expresamente (modo WSL experimental)..." >&2
      uv pip install --python "${VENV_PYTHON}" --no-deps --reinstall \
        --requirements "${PROJECT_DIR}/requirements/amd-rocm70-py312.txt"
      ;;
    intel)
      tarantulin_accelerator_preflight intel
      ;;
  esac
}

python_base_env() {
  export PYTHONPATH="${REPO_ROOT}"
}

train_env() {
  python_base_env
  local profile
  profile="$(tarantulin_resolve_accelerator)"
  export TARANTULIN_RESOLVED_BACKEND_PROFILE="${profile}"
  if [[ "${profile}" == "cpu" ]]; then
    export JAX_PLATFORMS=cpu
    export JAX_PLATFORM_NAME=cpu
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
  else
    unset JAX_PLATFORMS || true
    unset JAX_PLATFORM_NAME || true
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"
  fi
  unset XLA_PYTHON_CLIENT_ALLOCATOR || true
  unset TF_GPU_ALLOCATOR || true
  export MUJOCO_GL=egl
  export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-high}"
}

viewer_env() {
  python_base_env
  local profile
  profile="$(tarantulin_resolve_accelerator)"
  export TARANTULIN_RESOLVED_BACKEND_PROFILE="${profile}"
  if [[ "${profile}" == "cpu" ]]; then
    export JAX_PLATFORMS=cpu
    export JAX_PLATFORM_NAME=cpu
  fi
  export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.10}"
  unset TF_GPU_ALLOCATOR || true
  export MUJOCO_GL="${MUJOCO_VIEWER_GL:-glfw}"
  export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-high}"
}

render_env() {
  python_base_env
  export JAX_PLATFORM_NAME="${TARANTULIN_RENDER_JAX_PLATFORM:-cpu}"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.05}"
  unset XLA_PYTHON_CLIENT_ALLOCATOR || true
  unset TF_GPU_ALLOCATOR || true
  export MUJOCO_GL="${MUJOCO_RENDER_GL:-egl}"
  export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-high}"
}

uv_python() {
  [[ -x "${VENV_PYTHON}" ]] || {
    echo "Entorno Python ausente: ${VENV_PYTHON}. Ejecuta setup/install.ps1." >&2
    return 1
  }
  "${VENV_PYTHON}" "$@"
}

print_backend() {
  python_base_env
  echo "Perfil solicitado: $(tarantulin_requested_accelerator)"
  echo "Perfil resuelto: $(tarantulin_resolve_accelerator)"
  uv_python - <<'PY'
import jax
print("Backend JAX:", jax.default_backend())
print("Dispositivos JAX:", jax.devices())
PY
}

print_ppo_profiles() {
  cat <<'EOF'
Perfiles PPO disponibles:
  debug      5M steps   20 evals   512 envs   ep 1000   red 256-128       mb 4   upd 1   gamma 0.95   ent 0.015
  lite       100M steps  200 evals  512 envs   ep 1500   red 256-256       mb 4   upd 2   gamma 0.95   ent 0.01   default
  lite_fast  100M steps  200 evals  1024 envs  ep 1500   red 256-256       mb 4   upd 2   gamma 0.95   ent 0.01
  full       50M steps  200 evals  512 envs   ep 3000   red 512-256-128   mb 8   upd 2   gamma 0.97   ent 0.02

Uso rapido:
  ./scripts/lanzar_tarantulin.sh --perfil-ppo lite
  ./scripts/lanzar_tarantulin.sh --perfil-ppo debug
  ./scripts/tarantulin_wsl.sh train --background --setup --perfil-ppo lite_fast --fase-recompensa 1

Overrides siguen disponibles:
  --num-timesteps --num-envs --num-evals --episode-length --batch-size
  --unroll-length --num-minibatches --num-updates-per-batch --num-eval-envs
EOF
}

print_reward_phases() {
  cat <<'EOF'
Fases curriculares de recompensa:
  0  base_actual              sin curriculum; solo compatibilidad/debug
  1  mantener_pose_xml        cerca de qpos0 del XML ideal; aprende a quedarse ahi
  2  llegar_desde_suelo       desde suelo/estado peor hasta la pose XML
  3  recuperar_desde_caida    caidas/perturbaciones y volver estable a la pose XML
EOF
}

validate_ppo_profile() {
  case "${1:-}" in
    debug|lite|lite_fast|full) return 0 ;;
    *)
      echo "Perfil PPO no reconocido: ${1:-<vacio>}" >&2
      echo "Opciones validas: debug, lite, lite_fast, full" >&2
      exit 1
      ;;
  esac
}

validate_reward_phase() {
  case "${1:-}" in
    0|1|2|3) return 0 ;;
    *)
      echo "Fase curricular no reconocida: ${1:-<vacio>}" >&2
      echo "Opciones validas: 0, 1, 2, 3" >&2
      exit 1
      ;;
  esac
}

validate_run_name() {
  local value="${1:-}"
  if [[ ! "${value}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || [[ "${value}" == "." || "${value}" == ".." ]]; then
    echo "Nombre de run no valido: ${value:-<vacio>}" >&2
    echo "Usa un unico componente (maximo 128 caracteres): letras, numeros, punto, guion o guion bajo." >&2
    exit 2
  fi
}

assert_logs_root_safe() {
  local repo_real logs_real expected
  repo_real="$(realpath -e "${REPO_ROOT}")"
  expected="${repo_real}/logs_tarantulin_mjx"
  logs_real="$(realpath -m "${LOGS_DIR}")"
  if [[ -L "${LOGS_DIR}" || "${logs_real}" != "${expected}" ]]; then
    echo "Directorio de logs inseguro o enlazado fuera del runtime: ${LOGS_DIR}" >&2
    return 1
  fi
}

assert_run_path_safe() {
  local candidate="$1"
  local logs_real candidate_real name
  assert_logs_root_safe || return
  logs_real="$(realpath -m "${LOGS_DIR}")"
  candidate_real="$(realpath -m "${candidate}")"
  name="$(basename -- "${candidate_real}")"
  validate_run_name "${name}"
  if [[ -L "${candidate}" || "$(dirname -- "${candidate_real}")" != "${logs_real}" ]]; then
    echo "Ruta de run insegura o fuera de logs_tarantulin_mjx: ${candidate}" >&2
    return 1
  fi
}

write_last_run() {
  local run_dir="$1"
  local temp
  assert_run_path_safe "${run_dir}"
  mkdir -p "${LOGS_DIR}"
  temp="${LOGS_DIR}/.ultima_run.tmp.$$"
  printf '%s\n' "$(realpath -m "${run_dir}")" > "${temp}"
  mv -f -- "${temp}" "${LAST_RUN}"
}

verify_versions() {
  python_base_env
  uv_python - <<'PY'
import importlib.metadata as md
import os
from tarantulin.hiperparametros import VERSIONES_ESPERADAS

errors = []
profile = os.environ.get("TARANTULIN_RESOLVED_BACKEND_PROFILE", "")
for package, expected in VERSIONES_ESPERADAS.items():
  try:
    got = md.version(package)
  except md.PackageNotFoundError:
    errors.append(f"{package}: no instalado")
    continue
  print(f"{package}=={got}")
  if profile == "amd" and package in {"jax", "jaxlib"}:
    print(f"  nota: version gestionada por el plugin ROCm experimental (referencia original {expected})")
  elif got != expected:
    errors.append(f"{package}: esperado {expected}, encontrado {got}")
if errors:
  raise SystemExit("Versiones no fijadas:\n" + "\n".join(errors))
PY
}

require_compute_backend() {
  train_env
  if [[ "${TARANTULIN_SKIP_BACKEND_CHECK:-0}" == "1" ]]; then
    echo "Comprobacion final del backend omitida por configuracion."
    return 0
  fi
  uv_python - <<'PY'
import jax
import os
print("Backend JAX:", jax.default_backend())
print("Dispositivos JAX:", jax.devices())
expected = os.environ.get("TARANTULIN_RESOLVED_BACKEND_PROFILE", "cpu")
backend = jax.default_backend()
if expected in {"nvidia", "amd"} and backend != "gpu":
  raise SystemExit(f"Se esperaba backend GPU para {expected}; JAX usa {backend}.")
if expected == "cpu" and backend != "cpu":
  raise SystemExit(f"Se esperaba backend CPU; JAX usa {backend}.")
PY
}

require_portable_impl() {
  if [[ "${1:-jax}" != "jax" ]]; then
    echo "Esta distribucion portable admite --impl jax. Warp es NVIDIA-only y queda deshabilitado." >&2
    exit 2
  fi
}

setup_all() {
  ensure_wsl
  ensure_linux_fs
  ensure_gpu_visible
  sync_env
  train_env
  verify_versions
  require_compute_backend
  echo "setup OK."
}

current_run() {
  assert_logs_root_safe || return 1
  if [[ -L "${LAST_RUN}" ]]; then
    echo "Se ignora ultima_run.txt porque es un enlace simbolico inseguro." >&2
  elif [[ -f "${LAST_RUN}" ]]; then
    local saved
    saved="$(<"${LAST_RUN}")"
    if [[ -n "${saved}" && -d "${saved}" ]] && assert_run_path_safe "${saved}"; then
      printf '%s\n' "${saved}"
      return
    fi
  fi
  find "${LOGS_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%T@\t%p\n' 2>/dev/null |
    sort -nr |
    head -1 |
    cut -f2- || true
}

training_process_matches_run() {
  local pid="$1"
  local run_dir="$2"
  local pid_path="${run_dir}/entrenamiento.pid"
  local start_path="${run_dir}/entrenamiento.starttime"
  local saved_pid saved_start
  assert_run_path_safe "${run_dir}" || return 1
  [[ -f "${pid_path}" && ! -L "${pid_path}" && -f "${start_path}" && ! -L "${start_path}" ]] || return 1
  saved_pid="$(tr -d '\r\n' < "${pid_path}")"
  saved_start="$(tr -d '\r\n' < "${start_path}")"
  [[ "${saved_pid}" == "${pid}" ]] || return 1
  tarantulin_training_process_matches \
    "${pid}" "${REPO_ROOT}" "${LOGS_DIR}" "${run_dir}" "${saved_start}"
}

guard_process_matches_run() {
  local guard_pid="$1"
  local train_pid="$2"
  local run_dir="$3"
  local guard_pid_path="${run_dir}/proteccion_termica.pid"
  local guard_start_path="${run_dir}/proteccion_termica.starttime"
  local train_start_path="${run_dir}/entrenamiento.starttime"
  local saved_guard_pid guard_start train_start
  assert_run_path_safe "${run_dir}" || return 1
  [[ -f "${guard_pid_path}" && ! -L "${guard_pid_path}" \
     && -f "${guard_start_path}" && ! -L "${guard_start_path}" \
     && -f "${train_start_path}" && ! -L "${train_start_path}" ]] || return 1
  saved_guard_pid="$(tr -d '\r\n' < "${guard_pid_path}")"
  guard_start="$(tr -d '\r\n' < "${guard_start_path}")"
  train_start="$(tr -d '\r\n' < "${train_start_path}")"
  [[ "${saved_guard_pid}" == "${guard_pid}" ]] || return 1
  tarantulin_guard_process_matches \
    "${guard_pid}" "${SCRIPT_DIR}/proteccion_termica.sh" \
    "${train_pid}" "${train_start}" "${run_dir}" "${guard_start}"
}

write_process_starttime() {
  local pid="$1"
  local destination="$2"
  local starttime temporary
  starttime="$(tarantulin_process_starttime "${pid}")" || return 1
  temporary="${destination}.tmp.$$"
  printf '%s\n' "${starttime}" > "${temporary}"
  mv -f -- "${temporary}" "${destination}"
}

current_active_run_with_pid() {
  local run_dir
  run_dir="$(current_run)"
  if [[ -n "${run_dir}" && -f "${run_dir}/entrenamiento.pid" ]]; then
    local saved_pid
    saved_pid="$(cat "${run_dir}/entrenamiento.pid")"
    if training_process_matches_run "${saved_pid}" "${run_dir}" 2>/dev/null; then
      printf '%s\t%s\n' "${run_dir}" "${saved_pid}"
      return
    fi
  fi

  find "${LOGS_DIR}" -mindepth 2 -maxdepth 2 -name entrenamiento.pid -print 2>/dev/null |
    while IFS= read -r pid_file; do
      local candidate
      candidate="$(cat "${pid_file}")"
      if training_process_matches_run "${candidate}" "$(dirname "${pid_file}")" 2>/dev/null; then
        printf '%s\t%s\t%s\n' "$(stat -c %Y "${pid_file}")" "$(dirname "${pid_file}")" "${candidate}"
      fi
    done |
    sort -nr |
    head -1 |
    awk -F '\t' 'NF >= 3 {print $2 "\t" $3}'
}

current_pid() {
  current_active_run_with_pid | awk -F '\t' 'NF >= 2 {print $2}'
}

pid_alive() {
  local pid="${1:-}"
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${pid}" >/dev/null 2>&1
}

format_duration_seconds() {
  local total="${1:-0}"
  if (( total < 0 )); then
    total=0
  fi
  local days=$(( total / 86400 ))
  local hours=$(( (total % 86400) / 3600 ))
  local minutes=$(( (total % 3600) / 60 ))
  local seconds=$(( total % 60 ))
  if (( days > 0 )); then
    printf '%dd %02dh %02dm %02ds' "${days}" "${hours}" "${minutes}" "${seconds}"
  elif (( hours > 0 )); then
    printf '%dh %02dm %02ds' "${hours}" "${minutes}" "${seconds}"
  else
    printf '%dm %02ds' "${minutes}" "${seconds}"
  fi
}

file_age_text() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    printf 'n/a'
    return
  fi
  local now mtime
  now="$(date +%s)"
  mtime="$(stat -c %Y "${path}")"
  format_duration_seconds "$(( now - mtime ))"
}

file_time_text() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    printf 'n/a'
    return
  fi
  date -d "@$(stat -c %Y "${path}")" '+%H:%M:%S'
}

latest_metrics_file() {
  local run_dir="$1"
  local newest_path=""
  local newest_mtime=-1
  local candidate mtime
  for candidate in "${run_dir}/progreso.csv" "${run_dir}/recompensas.csv"; do
    if [[ -e "${candidate}" ]]; then
      mtime="$(stat -c %Y "${candidate}")"
      if (( mtime > newest_mtime )); then
        newest_mtime="${mtime}"
        newest_path="${candidate}"
      fi
    fi
  done
  printf '%s\n' "${newest_path}"
}

run_test_mjx() {
  local steps=150
  local iterations=12
  local ls_iterations=4
  local impl="jax"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --steps) steps="$2"; shift 2 ;;
      --iterations) iterations="$2"; shift 2 ;;
      --ls-iterations) ls_iterations="$2"; shift 2 ;;
      --impl) impl="$2"; shift 2 ;;
      *) echo "Argumento no reconocido para test-mjx: $1" >&2; exit 1 ;;
    esac
  done
  require_portable_impl "${impl}"
  ensure_wsl
  ensure_linux_fs
  train_env
  require_compute_backend
  uv_python "${REPO_ROOT}/tarantulin/test_mjx.py" \
    --steps "${steps}" \
    --iterations "${iterations}" \
    --ls_iterations "${ls_iterations}" \
    --impl "${impl}"
}

run_benchmark() {
  local run_name="benchmark-$(date +%Y%m%d-%H%M%S)"
  local warmup_steps=64
  local measure_steps=256
  local impl="jax"
  local envs="128 256 512 768 1024 1536 2048"
  local precisions="default high highest"
  local allocators="preallocate cuda_malloc_async"
  local solver_pairs="8:4 12:4 16:4"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-name) run_name="$2"; shift 2 ;;
      --warmup-steps) warmup_steps="$2"; shift 2 ;;
      --measure-steps) measure_steps="$2"; shift 2 ;;
      --envs) envs="$2"; shift 2 ;;
      --precisions) precisions="$2"; shift 2 ;;
      --allocators) allocators="$2"; shift 2 ;;
      --solver-pairs) solver_pairs="$2"; shift 2 ;;
      --impl) impl="$2"; shift 2 ;;
      *) echo "Argumento no reconocido para benchmark: $1" >&2; exit 1 ;;
    esac
  done
  require_portable_impl "${impl}"
  validate_run_name "${run_name}"

  ensure_wsl
  ensure_linux_fs
  ensure_gpu_visible
  local resolved_profile
  resolved_profile="$(tarantulin_resolve_accelerator)"
  if [[ "${resolved_profile}" == "cpu" ]]; then
    [[ "${envs}" != "128 256 512 768 1024 1536 2048" ]] || envs="1 8 32"
    [[ "${allocators}" != "preallocate cuda_malloc_async" ]] || allocators="platform"
  elif [[ "${resolved_profile}" == "amd" ]]; then
    [[ "${allocators}" != "preallocate cuda_malloc_async" ]] || allocators="preallocate"
    if [[ " ${allocators} " == *" cuda_malloc_async "* ]]; then
      echo "cuda_malloc_async solo esta permitido con el perfil NVIDIA." >&2
      exit 2
    fi
  fi
  assert_run_path_safe "${LOGS_DIR}/${run_name}"
  mkdir -p "${LOGS_DIR}/${run_name}"
  write_last_run "${LOGS_DIR}/${run_name}"
  local csv_path="${LOGS_DIR}/${run_name}/benchmark.csv"
  : > "${LOGS_DIR}/${run_name}/benchmark.log"

  for pair in ${solver_pairs}; do
    local iterations="${pair%%:*}"
    local ls_iterations="${pair##*:}"
    for allocator in ${allocators}; do
      for precision in ${precisions}; do
        for num_envs in ${envs}; do
          echo "Benchmark: envs=${num_envs}, allocator=${allocator}, precision=${precision}, iterations=${iterations}, ls=${ls_iterations}" | tee -a "${LOGS_DIR}/${run_name}/benchmark.log"
          python_base_env
          export MUJOCO_GL=egl
          export JAX_DEFAULT_MATMUL_PRECISION="${precision}"
          if [[ "${allocator}" == "preallocate" ]]; then
            export XLA_PYTHON_CLIENT_PREALLOCATE=true
            export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"
            unset TF_GPU_ALLOCATOR || true
            unset XLA_PYTHON_CLIENT_ALLOCATOR || true
          elif [[ "${allocator}" == "cuda_malloc_async" ]]; then
            export XLA_PYTHON_CLIENT_PREALLOCATE=false
            export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"
            export TF_GPU_ALLOCATOR=cuda_malloc_async
            unset XLA_PYTHON_CLIENT_ALLOCATOR || true
          else
            export XLA_PYTHON_CLIENT_PREALLOCATE=false
            unset TF_GPU_ALLOCATOR || true
            unset XLA_PYTHON_CLIENT_ALLOCATOR || true
          fi
          set +e
          uv_python "${REPO_ROOT}/tarantulin/benchmark_mjx.py" \
            --csv_path "${csv_path}" \
            --num_envs "${num_envs}" \
            --warmup_steps "${warmup_steps}" \
            --measure_steps "${measure_steps}" \
            --iterations "${iterations}" \
            --ls_iterations "${ls_iterations}" \
            --matmul_precision "${precision}" \
            --allocator "${allocator}" \
            --impl "${impl}" >> "${LOGS_DIR}/${run_name}/benchmark.log" 2>&1
          local status=$?
          set -e
          if [[ "${status}" -ne 0 ]]; then
            echo "Benchmark proceso fallo con codigo ${status}; continuo con la siguiente configuracion." | tee -a "${LOGS_DIR}/${run_name}/benchmark.log"
          fi
          check_swap || true
        done
      done
    done
  done
  echo "Benchmark terminado: ${csv_path}"
}

latest_checkpoint() {
  local run_dir="${1:-}"
  if [[ -z "${run_dir}" ]]; then
    run_dir="$(current_run)"
  fi
  [[ -n "${run_dir}" && -d "${run_dir}/checkpoints" ]] || return 0
  find "${run_dir}/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%P\n' 2>/dev/null |
    awk '/^[0-9]+$/' | sort -n | tail -1 |
    awk -v root="${run_dir}/checkpoints" '{ if ($0 != "") print root "/" $0 }'
}

start_thermal_guard() {
  local train_pid="$1"
  local run_dir="$2"
  local guard_log="${run_dir}/proteccion_termica.log"
  local train_starttime
  if [[ "${TARANTULIN_THERMAL_GUARD:-1}" == "0" ]]; then
    echo "Proteccion termica desactivada por TARANTULIN_THERMAL_GUARD=0."
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "No encuentro nvidia-smi; no puedo activar proteccion termica." | tee -a "${guard_log}" >&2
    return 0
  fi
  if [[ ! -x "${SCRIPT_DIR}/proteccion_termica.sh" ]]; then
    echo "No encuentro scripts/proteccion_termica.sh ejecutable." | tee -a "${guard_log}" >&2
    return 0
  fi
  if [[ ! -f "${run_dir}/entrenamiento.starttime" ]]; then
    write_process_starttime "${train_pid}" "${run_dir}/entrenamiento.starttime" || {
      echo "No se pudo registrar la identidad del proceso de entrenamiento; no activo la proteccion." >&2
      return 1
    }
  fi
  if ! training_process_matches_run "${train_pid}" "${run_dir}"; then
    echo "El PID ${train_pid} no corresponde al entrenamiento de esta run; no activo la proteccion." >&2
    return 1
  fi
  train_starttime="$(tr -d '\r\n' < "${run_dir}/entrenamiento.starttime")"
  nohup "${SCRIPT_DIR}/proteccion_termica.sh" \
    --pid "${train_pid}" \
    --train-starttime "${train_starttime}" \
    --run-dir "${run_dir}" \
    >> "${guard_log}" 2>&1 < /dev/null &
  local guard_pid=$!
  local guard_pid_tmp="${run_dir}/proteccion_termica.pid.tmp.$$"
  printf '%s\n' "${guard_pid}" > "${guard_pid_tmp}"
  mv -f -- "${guard_pid_tmp}" "${run_dir}/proteccion_termica.pid"
  write_process_starttime "${guard_pid}" "${run_dir}/proteccion_termica.starttime" || {
    echo "No se pudo registrar la identidad del guard termico; no se le enviaran senales automaticas." >&2
    return 1
  }
  echo "Proteccion termica activa. PID guard: ${guard_pid}. Log: ${guard_log}"
}

start_thermal_guard_current() {
  ensure_wsl
  ensure_linux_fs
  ensure_gpu_visible
  local run_info run_dir pid guard_pid
  run_info="$(current_active_run_with_pid || true)"
  IFS=$'\t' read -r run_dir pid <<< "${run_info}"
  if [[ -z "${run_dir}" || -z "${pid}" ]]; then
    echo "No encuentro una run activa con identidad de proceso validada para proteger." >&2
    exit 1
  fi
  if ! pid_alive "${pid}"; then
    echo "El PID guardado no esta activo: ${pid}" >&2
    exit 1
  fi
  if [[ -f "${run_dir}/proteccion_termica.pid" ]]; then
    guard_pid="$(cat "${run_dir}/proteccion_termica.pid")"
    if guard_process_matches_run "${guard_pid}" "${pid}" "${run_dir}"; then
      echo "La proteccion termica ya esta activa con PID ${guard_pid}."
      return 0
    elif pid_alive "${guard_pid}"; then
      echo "El PID guardado para proteccion termica pertenece a otro proceso; no se senaliza." >&2
    fi
  fi
  start_thermal_guard "${pid}" "${run_dir}"
}

run_train() {
  local run_name="${DEFAULT_RUN_PREFIX}-$(date +%Y%m%d-%H%M%S)"
  local num_timesteps=""
  local num_envs=""
  local num_evals=""
  local num_eval_envs=""
  local episode_length=""
  local batch_size=""
  local unroll_length=""
  local num_minibatches=""
  local num_updates_per_batch=""
  local intervalo_log_recompensas=""
  local training_metrics_steps=""
  local sin_metricas_recompensa_entrenamiento=0
  local metricas_recompensa_entrenamiento=0
  local metricas_fisicas_completas=0
  local curriculum_penalizaciones=""
  local fase_recompensa=0
  local perfil_ppo="${TARANTULIN_PERFIL_PPO:-lite}"
  local seed=42
  local impl="jax"
  local background=0
  local setup_first=0
  local skip_test=0
  local resume_latest=0
  local load_checkpoint_path=""
  local reset_checkpoint=0
  local append_csv=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-name) run_name="$2"; shift 2 ;;
      --num-timesteps) num_timesteps="$2"; shift 2 ;;
      --num-envs) num_envs="$2"; shift 2 ;;
      --num-evals) num_evals="$2"; shift 2 ;;
      --num-eval-envs) num_eval_envs="$2"; shift 2 ;;
      --episode-length) episode_length="$2"; shift 2 ;;
      --batch-size) batch_size="$2"; shift 2 ;;
      --unroll-length) unroll_length="$2"; shift 2 ;;
      --num-minibatches) num_minibatches="$2"; shift 2 ;;
      --num-updates-per-batch) num_updates_per_batch="$2"; shift 2 ;;
      --intervalo-log-recompensas|--reward-log-policy-updates-interval) intervalo_log_recompensas="$2"; shift 2 ;;
      --training-metrics-steps) training_metrics_steps="$2"; shift 2 ;;
      --sin-metricas-recompensa-entrenamiento|--no-training-reward-metrics) sin_metricas_recompensa_entrenamiento=1; shift ;;
      --metricas-recompensa-entrenamiento|--training-reward-metrics) metricas_recompensa_entrenamiento=1; shift ;;
      --metricas-fisicas-completas|--full-physical-metrics) metricas_fisicas_completas=1; shift ;;
      --curriculum-penalizaciones|--curriculum_penalizaciones) curriculum_penalizaciones="$2"; shift 2 ;;
      --fase-recompensa|--fase_recompensa|--reward-phase|--reward_phase) fase_recompensa="$2"; shift 2 ;;
      --fase-recompensa=*|--fase_recompensa=*|--reward-phase=*|--reward_phase=*) fase_recompensa="${1#*=}"; shift ;;
      --perfil-ppo|--perfil_ppo|--ppo-profile) perfil_ppo="$2"; shift 2 ;;
      --perfil-ppo=*|--perfil_ppo=*|--ppo-profile=*) perfil_ppo="${1#*=}"; shift ;;
      --seed) seed="$2"; shift 2 ;;
      --impl) impl="$2"; shift 2 ;;
      --background) background=1; shift ;;
      --setup) setup_first=1; shift ;;
      --skip-test-mjx) skip_test=1; shift ;;
      --resume-latest) resume_latest=1; shift ;;
      --load-checkpoint-path) load_checkpoint_path="$2"; shift 2 ;;
      --reset-checkpoint|--desde-cero|--fresh) reset_checkpoint=1; shift ;;
      --append-csv) append_csv=1; shift ;;
      *) echo "Argumento no reconocido para train: $1" >&2; exit 1 ;;
    esac
  done
  validate_ppo_profile "${perfil_ppo}"
  validate_reward_phase "${fase_recompensa}"
  require_portable_impl "${impl}"
  validate_run_name "${run_name}"

  if [[ "${TARANTULIN_SHOW_PPO_PROFILES:-1}" == "1" ]]; then
    print_ppo_profiles
    echo ""
  fi
  if [[ "${TARANTULIN_SHOW_REWARD_PHASES:-1}" == "1" ]]; then
    print_reward_phases
    echo ""
  fi
  echo "Configuracion seleccionada:"
  echo "  perfil_ppo: ${perfil_ppo}"
  echo "  fase_recompensa: ${fase_recompensa}"
  echo "  impl: ${impl}"
  echo "  run_name: ${run_name}"
  echo "  logdir: ${LOGS_DIR}"
  if [[ -n "${num_timesteps}" || -n "${num_envs}" || -n "${num_evals}" || -n "${episode_length}" || -n "${batch_size}" || -n "${unroll_length}" || -n "${num_minibatches}" || -n "${num_updates_per_batch}" || -n "${num_eval_envs}" ]]; then
    echo "  overrides PPO:"
    [[ -n "${num_timesteps}" ]] && echo "    num_timesteps=${num_timesteps}"
    [[ -n "${num_envs}" ]] && echo "    num_envs=${num_envs}"
    [[ -n "${num_evals}" ]] && echo "    num_evals=${num_evals}"
    [[ -n "${num_eval_envs}" ]] && echo "    num_eval_envs=${num_eval_envs}"
    [[ -n "${episode_length}" ]] && echo "    episode_length=${episode_length}"
    [[ -n "${batch_size}" ]] && echo "    batch_size=${batch_size}"
    [[ -n "${unroll_length}" ]] && echo "    unroll_length=${unroll_length}"
    [[ -n "${num_minibatches}" ]] && echo "    num_minibatches=${num_minibatches}"
    [[ -n "${num_updates_per_batch}" ]] && echo "    num_updates_per_batch=${num_updates_per_batch}"
  else
    echo "  overrides PPO: ninguno"
  fi
  echo ""

  ensure_wsl
  ensure_linux_fs
  ensure_gpu_visible
  if (( setup_first == 1 )) || [[ ! -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
    setup_all
  fi
  train_env
  require_compute_backend

  local active_pid
  active_pid="$(current_pid || true)"
  if pid_alive "${active_pid}"; then
    echo "Ya hay entrenamiento activo con PID ${active_pid}." >&2
    exit 1
  fi

  if (( skip_test == 0 )); then
    run_test_mjx --steps 150 --impl "${impl}"
  fi

  if [[ ! -f "${LOGS_DIR}"/benchmark-*/benchmark.csv ]]; then
    echo "Aviso: no encuentro benchmark.csv reciente. Recomendado: scripts/tarantulin_wsl.sh benchmark"
  fi

  if (( resume_latest == 1 )) && [[ -z "${load_checkpoint_path}" ]]; then
    load_checkpoint_path="$(current_run)"
  fi
  if (( reset_checkpoint == 1 )) && [[ -n "${load_checkpoint_path}" ]]; then
    echo "No puedo usar --reset-checkpoint y restaurar checkpoint a la vez." >&2
    exit 1
  fi

  local run_dir="${LOGS_DIR}/${run_name}"
  assert_run_path_safe "${run_dir}"
  mkdir -p "${run_dir}"
  if (( reset_checkpoint == 1 )); then
    rm -rf "${run_dir}/checkpoints"
  fi
  rm -f \
    "${run_dir}/entrenamiento.pid" \
    "${run_dir}/entrenamiento.starttime" \
    "${run_dir}/proteccion_termica.pid" \
    "${run_dir}/proteccion_termica.starttime"
  write_last_run "${run_dir}"

  local -a cmd=(
    "${REPO_ROOT}/tarantulin/entrenar_ppo_mjx.py"
    --perfil_ppo "${perfil_ppo}"
    --fase_recompensa "${fase_recompensa}"
    --impl "${impl}"
    --seed "${seed}"
    --logdir "${LOGS_DIR}"
    --run_name "${run_name}"
  )
  if [[ -n "${num_timesteps}" ]]; then
    cmd+=(--num_timesteps "${num_timesteps}")
  fi
  if [[ -n "${num_envs}" ]]; then
    cmd+=(--num_envs "${num_envs}")
  fi
  if [[ -n "${num_evals}" ]]; then
    cmd+=(--num_evals "${num_evals}")
  fi
  if [[ -n "${num_eval_envs}" ]]; then
    cmd+=(--num_eval_envs "${num_eval_envs}")
  fi
  if [[ -n "${episode_length}" ]]; then
    cmd+=(--episode_length "${episode_length}")
  fi
  if [[ -n "${batch_size}" ]]; then
    cmd+=(--batch_size "${batch_size}")
  fi
  if [[ -n "${unroll_length}" ]]; then
    cmd+=(--unroll_length "${unroll_length}")
  fi
  if [[ -n "${num_minibatches}" ]]; then
    cmd+=(--num_minibatches "${num_minibatches}")
  fi
  if [[ -n "${num_updates_per_batch}" ]]; then
    cmd+=(--num_updates_per_batch "${num_updates_per_batch}")
  fi
  if [[ -n "${intervalo_log_recompensas}" ]]; then
    cmd+=(--intervalo_log_recompensas "${intervalo_log_recompensas}")
  fi
  if [[ -n "${load_checkpoint_path}" ]]; then
    cmd+=(--load_checkpoint_path "${load_checkpoint_path}")
  fi
  if [[ -n "${training_metrics_steps}" ]]; then
    cmd+=(--training_metrics_steps "${training_metrics_steps}")
  fi
  if [[ -n "${curriculum_penalizaciones}" ]]; then
    cmd+=(--curriculum_penalizaciones "${curriculum_penalizaciones}")
  fi
  if (( sin_metricas_recompensa_entrenamiento == 1 )); then
    cmd+=(--sin_metricas_recompensa_entrenamiento)
  fi
  if (( metricas_recompensa_entrenamiento == 1 )); then
    cmd+=(--metricas_recompensa_entrenamiento)
  fi
  if (( metricas_fisicas_completas == 1 )); then
    cmd+=(--metricas_fisicas_completas)
  fi
  if (( append_csv == 1 )); then
    cmd+=(--append_csv)
  fi
  if (( reset_checkpoint == 1 )); then
    cmd+=(--reset_checkpoint)
  fi

  if (( background == 1 )); then
    : > "${run_dir}/entrenamiento.log"
    {
      printf 'cd %q\n' "${REPO_ROOT}"
      printf 'env PYTHONUNBUFFERED=1 PYTHONPATH=%q XLA_PYTHON_CLIENT_PREALLOCATE=%q XLA_PYTHON_CLIENT_MEM_FRACTION=%q MUJOCO_GL=egl JAX_DEFAULT_MATMUL_PRECISION=%q %q' \
        "${REPO_ROOT}" \
        "${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
        "${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}" \
        "${JAX_DEFAULT_MATMUL_PRECISION:-high}" \
        "${VENV_PYTHON}"
      printf ' %q' "${cmd[@]}"
      printf '\n'
    } > "${run_dir}/comando_lanzador.sh"
    nohup env \
      PYTHONUNBUFFERED=1 \
      PYTHONPATH="${REPO_ROOT}" \
      XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
      XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}" \
      MUJOCO_GL=egl \
      JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-high}" \
      "${VENV_PYTHON}" "${cmd[@]}" \
      >> "${run_dir}/entrenamiento.log" 2>&1 < /dev/null &
    local launcher_pid=$!
    printf '%s\n' "${launcher_pid}" > "${run_dir}/lanzador.pid"
    echo "Entrenamiento lanzado. Run: ${run_dir}"
    echo "PID lanzador: ${launcher_pid}"
    echo "Log: ${run_dir}/entrenamiento.log"
    echo "Esperando a que Python escriba PID real..."
    for _ in $(seq 1 30); do
      [[ -f "${run_dir}/entrenamiento.pid" ]] && break
      pid_alive "${launcher_pid}" || break
      sleep 1
    done
    if [[ -f "${run_dir}/entrenamiento.pid" ]]; then
      local real_pid
      real_pid="$(cat "${run_dir}/entrenamiento.pid")"
      echo "PID real: ${real_pid}"
      if pid_alive "${real_pid}"; then
        start_thermal_guard "${real_pid}" "${run_dir}"
      else
        echo "ERROR: Python escribio entrenamiento.pid, pero el proceso ya no esta activo." >&2
        echo "Ultimas lineas del log:" >&2
        tail -80 "${run_dir}/entrenamiento.log" >&2 || true
        exit 1
      fi
    elif pid_alive "${launcher_pid}"; then
      echo "Aviso: Python no ha escrito entrenamiento.pid aun; el lanzador sigue vivo." >&2
      echo "Puedes revisar el log mientras termina de importar/compilar." >&2
    else
      echo "ERROR: el proceso de lanzamiento termino antes de escribir entrenamiento.pid." >&2
      echo "Ultimas lineas del log:" >&2
      tail -80 "${run_dir}/entrenamiento.log" >&2 || true
      exit 1
    fi
  else
    : > "${run_dir}/entrenamiento.log"
    env \
      PYTHONUNBUFFERED=1 \
      PYTHONPATH="${REPO_ROOT}" \
      XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
      XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}" \
      MUJOCO_GL=egl \
      JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-high}" \
      "${VENV_PYTHON}" "${cmd[@]}" &
    local launcher_pid=$!
    for _ in $(seq 1 30); do
      [[ -f "${run_dir}/entrenamiento.pid" ]] && break
      sleep 1
    done
    if [[ -f "${run_dir}/entrenamiento.pid" ]]; then
      start_thermal_guard "$(cat "${run_dir}/entrenamiento.pid")" "${run_dir}"
    else
      echo "Aviso: Python no ha escrito entrenamiento.pid aun; no activo proteccion termica."
    fi
    wait "${launcher_pid}"
  fi
}

stop_training() {
  local run_info run_dir pid guard_pid
  run_info="$(current_active_run_with_pid || true)"
  IFS=$'\t' read -r run_dir pid <<< "${run_info}"
  if [[ -z "${run_dir}" || -z "${pid}" ]]; then
    echo "No hay entrenamiento activo con identidad validada."
    return 0
  fi
  assert_run_path_safe "${run_dir}"
  if ! training_process_matches_run "${pid}" "${run_dir}"; then
    echo "SEGURIDAD: PID ${pid} no corresponde al entrenamiento de esta run/runtime." >&2
    echo "No se envia ninguna senal; el marcador puede estar obsoleto o el PID fue reutilizado." >&2
    return 1
  fi
  if [[ -n "${run_dir}" && -f "${run_dir}/proteccion_termica.pid" ]]; then
    guard_pid="$(tr -d '\r\n' < "${run_dir}/proteccion_termica.pid")"
    if pid_alive "${guard_pid}"; then
      if guard_process_matches_run "${guard_pid}" "${pid}" "${run_dir}"; then
        echo "Parando proteccion termica PID ${guard_pid}..."
        guard_process_matches_run "${guard_pid}" "${pid}" "${run_dir}" && kill "${guard_pid}" || true
      else
        echo "SEGURIDAD: PID ${guard_pid} no corresponde al guard de esta run; no se senaliza." >&2
      fi
    fi
  fi
  echo "Parando entrenamiento PID ${pid}..."
  if ! training_process_matches_run "${pid}" "${run_dir}"; then
    echo "La identidad del proceso cambio antes de SIGTERM; no se envia la senal." >&2
    return 1
  fi
  kill "${pid}" || return 0
  for _ in $(seq 1 10); do
    training_process_matches_run "${pid}" "${run_dir}" || return 0
    sleep 1
  done
  if training_process_matches_run "${pid}" "${run_dir}"; then
    echo "PID ${pid} sigue activo y conserva su identidad; enviando SIGKILL."
    kill -9 "${pid}" || true
  else
    echo "La identidad de PID ${pid} cambio; no se envia SIGKILL." >&2
  fi
}

check_swap() {
  free -m
  awk '/Swap:/ { if ($3 > 0) { printf "Aviso: WSL esta usando swap: %s MiB\n", $3; exit 1 } }' < <(free -m)
}

print_run_details() {
  local run_dir="$1"
  local pid="${2:-}"
  python3 - "${run_dir}" "${pid}" <<'PY'
import csv
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
pid = sys.argv[2] if len(sys.argv) > 2 else ""

def load_json(name):
  path = run / name
  if not path.exists():
    return {}
  try:
    return json.loads(path.read_text(encoding="utf-8"))
  except Exception as exc:
    return {"_error": str(exc)}

hiper = load_json("hiperparametros.json")
cfg = load_json("config_entorno.json")
estado = load_json("estado.json")
curriculum = load_json("config_curriculum_recompensa.json")

def load_csv_rows(name):
  path = run / name
  if not path.exists():
    return []
  try:
    with path.open(newline="", encoding="utf-8") as handle:
      return list(csv.DictReader(handle))
  except Exception:
    return []

progress_path = run / "progreso.csv"
reward_path = run / "recompensas.csv"
rows = load_csv_rows("progreso.csv")
reward_rows = load_csv_rows("recompensas.csv")
last = rows[-1] if rows else {}
last_reward = reward_rows[-1] if reward_rows else {}

def get(data, key, default="n/a"):
  value = data.get(key, default)
  return default if value == "" else value

def first_value(*values, default="n/a"):
  for value in values:
    if value not in (None, "", "n/a"):
      return value
  return default

def last_nonempty(key, *rowsets, default="n/a"):
  for rowset in rowsets:
    for row in reversed(rowset):
      value = row.get(key)
      if value not in (None, "", "n/a"):
        return value
  return default

num_envs = get(hiper, "num_envs")
perfil_ppo = first_value(get(estado, "perfil_ppo"), default="n/a")
total = get(hiper, "num_timesteps")
batch = get(hiper, "batch_size")
unroll = get(hiper, "unroll_length")
episode = get(hiper, "episode_length")
evals = get(hiper, "num_evals")
eval_envs = get(hiper, "num_eval_envs")
updates = get(hiper, "num_updates_per_batch")
minibatches = get(hiper, "num_minibatches")
log_metricas_entrenamiento = get(hiper, "log_training_metrics")
training_metrics_steps = get(hiper, "training_metrics_steps")
intervalo_log_recompensas = get(hiper, "reward_log_policy_updates_interval")
iterations = get(cfg, "solver_iterations")
ls_iterations = get(cfg, "solver_ls_iterations")
xml_path = str(get(cfg, "xml_path", ""))
xml_name = Path(xml_path).name if xml_path != "n/a" else "n/a"

num_steps = first_value(get(last, "num_steps"), get(last_reward, "num_steps"))
percent = first_value(get(last, "percent"), get(last_reward, "percent"))
eval_reward = first_value(
    get(last, "eval_episode_reward"),
    get(last_reward, "eval_episode_reward"),
)
reward_total = get(last_reward, "reward_total")
positive_reward = get(last_reward, "positive_reward")
penalties = get(last_reward, "penalties")
eval_reward_per_step = first_value(
    get(last, "eval_reward_per_step"),
    get(last_reward, "eval_reward_per_step"),
)
length = first_value(get(last, "eval_episode_length"), get(last_reward, "eval_episode_length"))
sps = last_nonempty("steps_per_second", rows, reward_rows)
wall_sps = last_nonempty("wall_steps_per_second", rows, reward_rows)
training_sps = last_nonempty("training_steps_per_second", rows, reward_rows)
if sps == "n/a":
  sps = first_value(training_sps, wall_sps)
elapsed = first_value(get(last, "elapsed_seconds"), get(last_reward, "elapsed_seconds"))
elapsed_hours = first_value(get(last, "elapsed_hours"), get(last_reward, "elapsed_hours"))
fase_curriculum = first_value(
    get(last_reward, "fase_curriculum_recompensa"),
    get(estado, "fase_curriculum_recompensa"),
    get(cfg, "fase_curriculum_recompensa"),
    get(curriculum, "fase"),
)
nombre_curriculum = first_value(
    get(last_reward, "nombre_curriculum_recompensa"),
    get(estado, "nombre_curriculum_recompensa"),
    get(cfg, "nombre_curriculum_recompensa"),
    get(curriculum, "nombre"),
)

eta = "n/a"
estimated_total = "n/a"
next_eval = "n/a"
next_eval_eta = "n/a"
try:
  if sps != "n/a" and float(sps) > 0 and num_steps != "n/a":
    sps_float = float(sps)
    steps_float = float(num_steps)
    total_float = float(total)
    remaining = max(0.0, (total_float - steps_float) / sps_float)
    hours = int(remaining // 3600)
    minutes = int((remaining % 3600) // 60)
    seconds = int(remaining % 60)
    eta = f"{hours}h {minutes:02d}m {seconds:02d}s" if hours else f"{minutes}m {seconds:02d}s"

    total_seconds = total_float / sps_float
    total_hours = int(total_seconds // 3600)
    total_minutes = int((total_seconds % 3600) // 60)
    total_sec = int(total_seconds % 60)
    estimated_total = (
        f"{total_hours}h {total_minutes:02d}m {total_sec:02d}s"
        if total_hours
        else f"{total_minutes}m {total_sec:02d}s"
    )

    if len(rows) >= 2:
      prev_step = float(rows[-2].get("num_steps") or 0)
      step_delta = max(1.0, steps_float - prev_step)
    else:
      step_delta = max(1.0, total_float / max(1.0, float(evals)))
    next_step = min(total_float, steps_float + step_delta)
    next_eval = str(int(next_step))
    next_remaining = max(0.0, (next_step - steps_float) / sps_float)
    next_minutes = int(next_remaining // 60)
    next_seconds = int(next_remaining % 60)
    next_eval_eta = f"{next_minutes}m {next_seconds:02d}s"
except Exception:
  pass

log_path = run / "entrenamiento.log"
ckpt_root = run / "checkpoints"
ckpts = []
if ckpt_root.exists():
  ckpts = sorted(
      [path for path in ckpt_root.iterdir() if path.is_dir() and path.name.isdigit()],
      key=lambda path: int(path.name),
  )

estado_json = estado.get("estado", "n/a")
pid_estado = str(estado.get("pid", "") or "")
pid_real = pid or "n/a"
guard_path = run / "proteccion_termica.pid"
guard_pid = guard_path.read_text(encoding="utf-8").strip() if guard_path.exists() else "n/a"
thermal_log = run / "proteccion_termica.log"

def to_float(value):
  try:
    if value in (None, "", "n/a"):
      return None
    return float(value)
  except Exception:
    return None

def fmt(value, digits=2, width=0):
  number = to_float(value)
  if number is None:
    text = "n/a"
  elif abs(number) >= 100000:
    text = f"{number:.2e}"
  elif abs(number) >= 1000:
    text = f"{number:.0f}"
  else:
    text = f"{number:.{digits}f}"
  return text.rjust(width) if width else text

def fmt_int(value):
  number = to_float(value)
  return "n/a" if number is None else f"{int(number):,}".replace(",", ".")

def bar(value, width=28):
  number = to_float(value)
  if number is None:
    return "[" + "." * width + "]"
  filled = max(0, min(width, int(round(width * number / 100.0))))
  return "[" + "#" * filled + "." * (width - filled) + "]"

def clipped_bar(value, max_abs=20.0, width=18):
  number = to_float(value)
  if number is None:
    return "." * width
  magnitude = min(abs(number) / max_abs, 1.0)
  filled = int(round(width * magnitude))
  mark = "+" if number >= 0 else "-"
  return mark * filled + "." * (width - filled)

def row(label, value, extra=""):
  print(f"  {label:<28} {fmt(value, 3, 10)}  {extra}")

def latest_value(*keys):
  for key in keys:
    value = last_reward.get(key)
    if value not in (None, "", "n/a"):
      return value
  return "n/a"

reward_trend_values = []
for reward_row in reward_rows[-8:]:
  value = reward_row.get("eval_episode_reward") or reward_row.get("reward_total")
  number = to_float(value)
  if number is not None:
    reward_trend_values.append(number)
trend = "n/a"
if reward_trend_values:
  trend = " -> ".join(fmt(value, 1) for value in reward_trend_values[-5:])

reward_now = first_value(eval_reward, reward_total)
reward_items = [
    ("eval PPO", eval_reward, "por paso " + fmt(eval_reward_per_step, 4)),
    ("total raw", reward_total, "pos " + fmt(positive_reward, 2) + " / pen " + fmt(penalties, 2)),
    ("pose articular", latest_value("pose_articular_reward_ponderado")),
    ("altura", latest_value("altura_reward_ponderado", "altura_legacy_reward")),
    ("soporte/contactos", latest_value("soporte_estatico_reward_ponderado", "contactos_suelo_reward_ponderado")),
    ("poligono CoG", latest_value("poligono_CoG_reward_ponderado")),
    ("simetria patas", latest_value("simetria_patas_reward_ponderado", "simetria_patas_legacy_reward")),
    ("vel lineal cero", latest_value("velocidad_lineal_cero_reward_ponderado")),
    ("vel angular cero", latest_value("velocidad_angular_cero_reward_ponderado")),
    ("control penalty", latest_value("control_penalty_ponderado")),
    ("cambio accion penalty", latest_value("cambio_accion_penalty_ponderado")),
    ("contacto invalido pen", latest_value("contacto_invalido_penalty_ponderado")),
]

print("+------------------------------------------------------------------------------+")
print("| RESUMEN                                                                      |")
print("+------------------------------------------------------------------------------+")
print(f"  estado {estado_json:<14} pid {pid_real:<8} perfil {perfil_ppo:<10} fase {fase_curriculum} - {nombre_curriculum}")
if estado_json == "entrenando" and not pid:
  print("  aviso: estado.json dice entrenando, pero ese proceso ya no existe.")
print(f"  run    {run.name}")
print(f"  xml    {xml_name:<34} solver {iterations}/{ls_iterations}  thermal {guard_pid}")
print("")
print("+------------------------------------------------------------------------------+")
print("| PROGRESO                                                                     |")
print("+------------------------------------------------------------------------------+")
print(f"  steps  {fmt_int(num_steps):>14} / {fmt_int(total):<14} {bar(percent)} {fmt(percent, 2)}%")
print(f"  speed  {fmt(sps, 1):>10} steps/s   train {fmt(training_sps, 1):>10}   wall {fmt(wall_sps, 1):>10}")
print(f"  time   elapsed {elapsed_hours} h   ETA {eta:<12} total est. {estimated_total}")
print(f"  next   eval/check aprox step {next_eval} en {next_eval_eta}   episode_len {length}")
print("")
print("+------------------------------------------------------------------------------+")
print("| PPO / CONFIG                                                                  |")
print("+------------------------------------------------------------------------------+")
print(f"  envs {num_envs:<6} eval_envs {eval_envs:<4} evals {evals:<5} episode {episode:<6} batch {batch:<5} unroll {unroll}")
print(f"  minibatches {minibatches:<4} updates/batch {updates:<4} log_train_metrics {log_metricas_entrenamiento}")
print(f"  reward_log_interval {intervalo_log_recompensas:<6} training_metrics_steps {training_metrics_steps}")
print("")
print("+------------------------------------------------------------------------------+")
print("| REWARD                                                                       |")
print("+------------------------------------------------------------------------------+")
print(f"  >>> reward ahora: {fmt(reward_now, 3)}   eval: {fmt(eval_reward, 3)}   raw: {fmt(reward_total, 3)}   per_step: {fmt(eval_reward_per_step, 5)}")
print(f"  tendencia reciente: {trend}")
print("  componentes principales:")
for label, value, *rest in reward_items:
  extra = rest[0] if rest else ""
  print(f"    {label:<22} {fmt(value, 3, 11)}  {clipped_bar(value)} {extra}")
if not rows and not reward_rows:
  print("  nota: sin metricas aun; probablemente compilando JIT o inicializando.")
elif len(rows) <= 1 and str(num_steps) == "0":
  print("  nota: solo hay evaluacion inicial; el primer rollout/compilacion puede tardar.")
print("")
print("+------------------------------------------------------------------------------+")
print("| ARTEFACTOS                                                                   |")
print("+------------------------------------------------------------------------------+")
print(f"  progreso.csv {len(rows):>5} filas  recompensas.csv {len(reward_rows):>5} filas  checkpoints {len(ckpts):>3}")
print(f"  log {log_path.stat().st_size if log_path.exists() else 0} bytes   ultimo_ckpt {ckpts[-1].name if ckpts else 'n/a'}")
PY
}

auto_clean_jit() {
  local cache_dir="${HOME}/.cache/jax_cache"
  if [[ -d "${cache_dir}" ]]; then
    echo "Limpiando ${cache_dir}"
    rm -rf "${cache_dir}"
  else
    echo "No existe ${cache_dir}"
  fi
}

reset_checkpoint_state() {
  ensure_wsl
  ensure_linux_fs
  local run_dir
  run_dir="$(current_run)"
  if [[ -z "${run_dir}" ]]; then
    echo "No hay run actual para resetear."
    return 0
  fi
  local resolved_logs resolved_run
  resolved_logs="$(realpath "${LOGS_DIR}")"
  resolved_run="$(realpath -m "${run_dir}")"
  if [[ "${resolved_run}" != "${resolved_logs}"/* ]]; then
    echo "Ruta fuera de logs_tarantulin_mjx, no borro nada: ${resolved_run}" >&2
    exit 1
  fi
  local pid guard_pid recorded_train_pid
  pid="$(current_pid || true)"
  if pid_alive "${pid}"; then
    echo "Hay entrenamiento activo con PID ${pid}; paralo antes de resetear checkpoint." >&2
    exit 1
  fi
  guard_pid=""
  recorded_train_pid=""
  if [[ -f "${run_dir}/entrenamiento.pid" && ! -L "${run_dir}/entrenamiento.pid" ]]; then
    recorded_train_pid="$(tr -d '\r\n' < "${run_dir}/entrenamiento.pid")"
  fi
  if [[ -f "${run_dir}/proteccion_termica.pid" ]]; then
    guard_pid="$(cat "${run_dir}/proteccion_termica.pid")"
  fi
  if pid_alive "${guard_pid}"; then
    if guard_process_matches_run "${guard_pid}" "${recorded_train_pid}" "${run_dir}"; then
      echo "Parando proteccion termica PID ${guard_pid}..."
      guard_process_matches_run "${guard_pid}" "${recorded_train_pid}" "${run_dir}" && kill "${guard_pid}" || true
    else
      echo "SEGURIDAD: PID ${guard_pid} no es el guard de esta run; no se senaliza." >&2
    fi
  fi
  rm -rf "${run_dir}/checkpoints"
  mkdir -p "${run_dir}/checkpoints"
  rm -f \
    "${run_dir}/entrenamiento.pid" \
    "${run_dir}/entrenamiento.starttime" \
    "${run_dir}/lanzador.pid" \
    "${run_dir}/proteccion_termica.pid" \
    "${run_dir}/proteccion_termica.starttime"
  python3 - "${run_dir}" <<'PY'
import datetime as dt
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
estado_path = run / "estado.json"
estado = {}
if estado_path.exists():
  try:
    estado = json.loads(estado_path.read_text(encoding="utf-8"))
  except Exception:
    estado = {}
estado.update({
    "estado": "checkpoint_reseteado_desde_cero",
    "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    "checkpoint_reseteado": True,
    "pid": None,
})
estado_path.write_text(
    json.dumps(estado, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY
  write_last_run "${run_dir}"
  echo "Checkpoint reseteado en: ${run_dir}"
  echo "La proxima run curricular debe lanzarse sin --resume-latest y, si reutilizas nombre, con --reset-checkpoint."
}

monitor() {
  mkdir -p "${LOGS_DIR}"
  while true; do
    clear
    printf '%s\n' "+==============================================================================+"
    printf '%s\n' "| TARANTULIN MJX/JAX PPO - MONITOR                                             |"
    printf '%s\n' "+==============================================================================+"
    printf 'Workspace: %s\n' "${REPO_ROOT}"
    printf 'Playground: dependencia fijada | commit %.12s\n' "${PLAYGROUND_COMMIT}"
    if [[ -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
      printf '%s\n' "Venv: OK"
    else
      printf '%s\n' "Venv: no creado aun; ejecuta ./scripts/tarantulin_wsl.sh setup"
    fi
    printf '%s\n' "+------------------------------------------------------------------------------+"
    printf 'Compute: '
    if tarantulin_has_nvidia; then
      nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null |
        awk -F ', ' '{printf "%s | util %s%% | VRAM %s/%s MiB | %s C\n", $1, $2, $3, $4, $5}'
    elif tarantulin_has_amd_rocm; then
      printf 'AMD ROCm (telemetria termica no disponible en este monitor)\n'
    else
      printf 'CPU\n'
    fi
    printf 'RAM: '
    free -h | awk '/Mem:/ {printf "mem %s/%s | avail %s  ", $3, $2, $7} /Swap:/ {printf "swap %s/%s\n", $3, $2}'
    printf '%s\n' "+------------------------------------------------------------------------------+"
    local run_info run_dir pid latest_metrics
    run_info="$(current_active_run_with_pid || true)"
    if [[ -n "${run_info}" ]]; then
      run_dir="$(printf '%s\n' "${run_info}" | awk -F '\t' 'NR == 1 {print $1}')"
      pid="$(printf '%s\n' "${run_info}" | awk -F '\t' 'NR == 1 {print $2}')"
    else
      run_dir="$(current_run)"
      pid=""
    fi
    if [[ -n "${run_dir}" && -z "${pid}" && -f "${run_dir}/estado.json" ]] && \
      python3 - "${run_dir}/estado.json" <<'PY'
import json
import sys
from pathlib import Path

try:
  estado = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except Exception:
  raise SystemExit(1)
raise SystemExit(0 if estado.get("estado") == "entrenando" else 1)
PY
    then
      printf '%s\n' "Aviso: estado.json sigue marcado como entrenando, pero el PID ya no existe."
    fi
    if [[ -n "${run_dir}" ]]; then
      latest_metrics="$(latest_metrics_file "${run_dir}")"
      printf 'Run: %s\n' "${run_dir}"
      printf 'PID: %s | Estado: ' "${pid:-n/a}"
      if pid_alive "${pid}"; then
        printf 'activo | vivo %s\n' "$(ps -p "${pid}" -o etime= | awk '{$1=$1; print}')"
      else
        printf 'sin proceso activo\n'
      fi
      printf 'CSV: %s | %s | hace %s    Log: %s | hace %s\n' \
        "$(basename "${latest_metrics:-n/a}")" \
        "$(file_time_text "${latest_metrics}")" \
        "$(file_age_text "${latest_metrics}")" \
        "$(file_time_text "${run_dir}/entrenamiento.log")" \
        "$(file_age_text "${run_dir}/entrenamiento.log")"
    else
      printf 'Run: n/a | Estado: sin entrenamiento activo\n'
    fi
    printf '%s\n' "+------------------------------------------------------------------------------+"
    if [[ -n "${run_dir}" ]]; then
      print_run_details "${run_dir}" "${pid}"
      local ckpt
      ckpt="$(latest_checkpoint "${run_dir}")"
      printf '\n+------------------------------------------------------------------------------+\n'
      printf '| CHECKPOINT / TERMICA / LOG                                                    |\n'
      printf '+------------------------------------------------------------------------------+\n'
      printf '  ultimo_checkpoint: %s\n' "${ckpt:-n/a}"
      printf '  termica: '
      if [[ -f "${run_dir}/proteccion_termica_estado.csv" ]]; then
        tail -n 1 "${run_dir}/proteccion_termica_estado.csv" || true
      else
        printf '%s\n' "sin muestras"
      fi
      printf '\n  ultimas lineas del log:\n'
      if [[ -f "${run_dir}/entrenamiento.log" ]]; then
        tail -n 12 "${run_dir}/entrenamiento.log" | sed 's/^/    /' || true
      elif [[ -f "${run_dir}/benchmark.log" ]]; then
        tail -n 12 "${run_dir}/benchmark.log" | sed 's/^/    /' || true
      else
        printf '%s\n' "    no hay entrenamiento.log ni benchmark.log todavia."
      fi
    else
      printf '%s\n' "No hay ninguna run creada todavia."
      printf '\nComandos normales:\n'
      printf '  ./scripts/lanzar_tarantulin.sh              # setup + test-mjx + train en background\n'
      printf '  ./scripts/tarantulin_wsl.sh test-mjx        # prueba MJX antes de entrenar\n'
      printf '  ./scripts/tarantulin_wsl.sh benchmark       # benchmark amplio\n'
      printf '  ./scripts/tarantulin_wsl.sh train --background --setup\n'
      printf '\nUltimas carpetas en logs_tarantulin_mjx:\n'
      find "${LOGS_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '  %TY-%Tm-%Td %TH:%TM  %p\n' 2>/dev/null |
        sort |
        tail -10 || true
    fi
    printf '%s\n' "+==============================================================================+"
    printf '%s\n' "Ctrl+C para salir. Refresco cada 2s."
    sleep 2
  done
}

view_results() {
  local impl="jax"
  local episode_length="${DEFAULT_EPISODE_LENGTH}"
  local checkpoint_path=""
  local checkpoint_index=0
  local force_cpu=0
  local dry_run=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --impl) impl="$2"; shift 2 ;;
      --episode-length) episode_length="$2"; shift 2 ;;
      --checkpoint-path) checkpoint_path="$2"; shift 2 ;;
      --checkpoint-index) checkpoint_index="$2"; shift 2 ;;
      --previous-checkpoint) checkpoint_index=1; shift ;;
      --cpu) force_cpu=1; shift ;;
      --dry-run) dry_run=1; shift ;;
      *) echo "Argumento no reconocido para view-results: $1" >&2; exit 1 ;;
    esac
  done
  require_portable_impl "${impl}"
  ensure_wsl
  ensure_linux_fs
  viewer_env
  local -a cmd=(
    "${REPO_ROOT}/scripts/visualizar_resultados_mjx.py"
    --impl "${impl}"
    --logs_dir "${LOGS_DIR}"
    --checkpoint_index "${checkpoint_index}"
    --episode_length "${episode_length}"
  )
  if [[ -n "${checkpoint_path}" ]]; then
    cmd+=(--checkpoint_path "${checkpoint_path}")
  fi
  if (( dry_run == 1 )); then
    cmd+=(--dry_run)
  fi
  if (( force_cpu == 1 )); then
    export JAX_PLATFORM_NAME=cpu
  fi
  uv_python "${cmd[@]}"
}

verify_pretrained_model() {
  if [[ ! -d "${PRETRAINED_MODEL_DIR}" || -L "${PRETRAINED_MODEL_DIR}" ]]; then
    echo "No encuentro el modelo preentrenado versionado: ${PRETRAINED_MODEL_DIR}" >&2
    return 1
  fi
  if [[ ! -d "${PRETRAINED_CHECKPOINT}" || -L "${PRETRAINED_CHECKPOINT}" ]]; then
    echo "Checkpoint preentrenado ausente o inseguro: ${PRETRAINED_CHECKPOINT}" >&2
    return 1
  fi
  [[ -f "${PRETRAINED_MODEL_DIR}/SHA256SUMS" ]] || {
    echo "Falta el manifiesto SHA-256 del modelo preentrenado." >&2
    return 1
  }
  (
    cd "${PRETRAINED_MODEL_DIR}"
    sha256sum --quiet --check SHA256SUMS
  ) || {
    echo "La red preentrenada no coincide con el manifiesto versionado." >&2
    return 1
  }
}

view_pretrained() {
  local -a viewer_args=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        cat <<'EOF'
Uso:
  .\tarantulin.ps1 view-pretrained [-- --impl jax|warp] [-- --cpu] [-- --dry-run]

Carga siempre la red de referencia versionada:
  fase 2 · paso 45.932.544 · episodio 1500

No consulta logs_tarantulin_mjx/ultima_run.txt. Para ver un checkpoint local usa
"view-results" o "visualizar_ultimo_checkpoint.sh".
EOF
        return 0
        ;;
      --impl)
        [[ $# -ge 2 ]] || {
          echo "Falta el valor de --impl para view-pretrained." >&2
          return 2
        }
        viewer_args+=("$1" "$2")
        shift 2
        ;;
      --cpu|--dry-run)
        viewer_args+=("$1")
        shift
        ;;
      --checkpoint-path|--checkpoint-index|--previous-checkpoint|--episode-length)
        echo "view-pretrained fija la red 45.932.544 y el episodio 1500; '$1' no se puede cambiar." >&2
        return 2
        ;;
      *)
        echo "Argumento no reconocido para view-pretrained: $1" >&2
        return 2
        ;;
    esac
  done
  verify_pretrained_model
  view_results "${viewer_args[@]}" \
    --checkpoint-path "${PRETRAINED_CHECKPOINT}" \
    --episode-length "${PRETRAINED_EPISODE_LENGTH}"
}

mini_sim() {
  local impl="jax"
  local episode_length="${DEFAULT_EPISODE_LENGTH}"
  local checkpoint_path=""
  local checkpoint_index=0
  local xml_path=""
  local reset_preset="actual"
  local dry_run=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --impl) impl="$2"; shift 2 ;;
      --episode-length) episode_length="$2"; shift 2 ;;
      --checkpoint-path) checkpoint_path="$2"; shift 2 ;;
      --checkpoint-index) checkpoint_index="$2"; shift 2 ;;
      --xml-path) xml_path="$2"; shift 2 ;;
      --reset-preset) reset_preset="$2"; shift 2 ;;
      --previous-checkpoint) checkpoint_index=1; shift ;;
      --dry-run) dry_run=1; shift ;;
      *) echo "Argumento no reconocido para mini-sim: $1" >&2; exit 1 ;;
    esac
  done
  require_portable_impl "${impl}"
  ensure_wsl
  ensure_linux_fs
  viewer_env
  # Mini-sim usa viewer_env (XLA_PYTHON_CLIENT_MEM_FRACTION=0.10, ~600 MB).
  # NO forzamos CPU: MJX requiere GPU para compilar la deteccion de colisiones.
  # Si el training esta activo en la misma GPU, el 10% de VRAM es suficiente
  # para 1 entorno de simulacion mas inferencia de la red.
  local -a cmd=(
    "${REPO_ROOT}/scripts/visualizar_resultados_mjx.py"
    --impl "${impl}"
    --logs_dir "${LOGS_DIR}"
    --checkpoint_index "${checkpoint_index}"
    --episode_length "${episode_length}"
  )
  if [[ -n "${checkpoint_path}" ]]; then
    cmd+=(--checkpoint_path "${checkpoint_path}")
  fi
  if [[ -n "${xml_path}" ]]; then
    cmd+=(--xml_path "${xml_path}")
  fi
  cmd+=(--reset_preset "${reset_preset}")
  if (( dry_run == 1 )); then
    cmd+=(--dry_run)
  fi
  uv_python "${cmd[@]}"
}

usage() {
  cat <<'EOF'
Uso:
  scripts/tarantulin_wsl.sh setup
  scripts/tarantulin_wsl.sh doctor
  scripts/tarantulin_wsl.sh backend
  scripts/tarantulin_wsl.sh profile
  scripts/tarantulin_wsl.sh perfiles-ppo
  scripts/tarantulin_wsl.sh fases-recompensa
  scripts/tarantulin_wsl.sh curriculum-auto
  scripts/tarantulin_wsl.sh test-mjx [--steps 150]
  scripts/tarantulin_wsl.sh benchmark
  scripts/tarantulin_wsl.sh train [--background] [--setup] [--skip-test-mjx]
    [--run-name NOMBRE_FIJO] [--resume-latest] [--reset-checkpoint] [--append-csv]
   [--perfil-ppo debug|lite|lite_fast|full]
   [--fase-recompensa 0|1|2|3]
   [--intervalo-log-recompensas 5] [--training-metrics-steps N]
   [--metricas-recompensa-entrenamiento] [--metricas-fisicas-completas]
   [--curriculum-penalizaciones 0.0-1.0]
 scripts/tarantulin_wsl.sh monitor
  scripts/tarantulin_wsl.sh stop
  scripts/tarantulin_wsl.sh reset-checkpoint
  scripts/tarantulin_wsl.sh check-swap
  scripts/tarantulin_wsl.sh start-thermal-guard
  scripts/tarantulin_wsl.sh auto-clean-jit
  scripts/tarantulin_wsl.sh view-pretrained [--impl jax|warp] [--cpu] [--dry-run]
  scripts/tarantulin_wsl.sh view-results [--cpu] [--previous-checkpoint] [--dry-run]
  scripts/tarantulin_wsl.sh mini-sim [--xml-path XML] [--reset-preset actual|suelo2|ideal|caida_lateral|boca_abajo] [--previous-checkpoint] [--dry-run]
  scripts/graficar_recompensas.sh [--show]     # genera una grafica por recompensas*.csv

Atajos Bash:
  scripts/lanzar_tarantulin.sh [opciones train]  # por defecto usa --perfil-ppo lite
  scripts/monitor_tarantulin.sh
  scripts/parar_tarantulin.sh
  scripts/visualizar_red_preentrenada.sh [--dry-run]
  scripts/visualizar_ultimo_checkpoint.sh [opciones view-results]
  scripts/cambiar_fase_tarantulin.sh [1|2|3]
  scripts/minisimular_ultimo_checkpoint.sh [opciones mini-sim]
  scripts/graficar_recompensas.sh [--csv_mode active|all] [opciones grafica]
EOF
}

main() {
  local command="${1:-}"
  if [[ -z "${command}" ]]; then
    usage
    exit 1
  fi
  shift || true
  case "${command}" in
    setup) setup_all "$@" ;;
    doctor) "${SCRIPT_DIR}/doctor.sh" "$@" ;;
    backend) ensure_wsl; ensure_linux_fs; train_env; print_backend ;;
    profile) printf '%s\n' "$(tarantulin_resolve_accelerator)" ;;
    perfiles-ppo|ppo-profiles|profiles) print_ppo_profiles "$@" ;;
    fases-recompensa|reward-phases|fases) print_reward_phases "$@" ;;
    curriculum-auto|auto-curriculum) "${SCRIPT_DIR}/curriculum_auto_tarantulin.sh" "$@" ;;
    test-mjx) run_test_mjx "$@" ;;
    benchmark) run_benchmark "$@" ;;
    train) run_train "$@" ;;
    monitor) monitor "$@" ;;
    stop) stop_training "$@" ;;
    reset-checkpoint) reset_checkpoint_state "$@" ;;
    check-swap) check_swap "$@" ;;
    start-thermal-guard) start_thermal_guard_current "$@" ;;
    auto-clean-jit) auto_clean_jit "$@" ;;
    view-pretrained|visualizar-red) view_pretrained "$@" ;;
    view-results) view_results "$@" ;;
    mini-sim) mini_sim "$@" ;;
    *) usage; exit 1 ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
