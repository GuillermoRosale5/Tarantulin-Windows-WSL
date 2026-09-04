#!/usr/bin/env bash

# Helpers de solo lectura para que un PID reciclado nunca reciba senales.

tarantulin_process_starttime() {
  local pid="${1:-}"
  local stat_line remainder
  [[ "${pid}" =~ ^[1-9][0-9]*$ && -r "/proc/${pid}/stat" ]] || return 1
  IFS= read -r stat_line < "/proc/${pid}/stat" || return 1
  # /proc/PID/stat puede contener espacios y parentesis en comm. El campo 22
  # (starttime) es el campo 20 despues del ultimo ') '.
  remainder="${stat_line##*) }"
  awk '{print $20}' <<< "${remainder}"
}

tarantulin_process_has_starttime() {
  local pid="${1:-}"
  local expected="${2:-}"
  local actual
  [[ "${expected}" =~ ^[0-9]+$ ]] || return 1
  actual="$(tarantulin_process_starttime "${pid}" 2>/dev/null)" || return 1
  [[ "${actual}" == "${expected}" ]]
}

tarantulin_training_process_matches() {
  local pid="${1:-}"
  local repo_root="$2"
  local logs_dir="$3"
  local run_dir="$4"
  local expected_starttime="$5"
  local repo_real logs_real run_real cwd_real trainer_real arg_real run_name
  local found_trainer=0 found_logdir=0 found_run=0 i
  local -a argv=()

  tarantulin_process_has_starttime "${pid}" "${expected_starttime}" || return 1
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  mapfile -d '' -t argv < "/proc/${pid}/cmdline" || return 1
  (( ${#argv[@]} > 0 )) || return 1

  repo_real="$(realpath -e "${repo_root}")" || return 1
  logs_real="$(realpath -e "${logs_dir}")" || return 1
  run_real="$(realpath -e "${run_dir}")" || return 1
  [[ ! -L "${run_dir}" && "$(dirname -- "${run_real}")" == "${logs_real}" ]] || return 1
  run_name="$(basename -- "${run_real}")"
  trainer_real="${repo_real}/tarantulin/entrenar_ppo_mjx.py"
  cwd_real="$(readlink -f "/proc/${pid}/cwd" 2>/dev/null)" || return 1
  [[ "${cwd_real}" == "${repo_real}" ]] || return 1

  for ((i = 0; i < ${#argv[@]}; i++)); do
    if [[ "${argv[i]}" == "${trainer_real}" ]]; then
      found_trainer=1
    elif [[ "${argv[i]}" == "--logdir" && $((i + 1)) -lt ${#argv[@]} ]]; then
      arg_real="$(realpath -m "${argv[i + 1]}")"
      [[ "${arg_real}" == "${logs_real}" ]] && found_logdir=1
    elif [[ "${argv[i]}" == "--run_name" && $((i + 1)) -lt ${#argv[@]} ]]; then
      [[ "${argv[i + 1]}" == "${run_name}" ]] && found_run=1
    fi
  done
  (( found_trainer == 1 && found_logdir == 1 && found_run == 1 ))
}

tarantulin_guard_process_matches() {
  local pid="${1:-}"
  local guard_script="$2"
  local train_pid="$3"
  local train_starttime="$4"
  local run_dir="$5"
  local expected_starttime="$6"
  local guard_real run_real arg_real
  local found_script=0 found_pid=0 found_train_start=0 found_run=0 i
  local -a argv=()

  tarantulin_process_has_starttime "${pid}" "${expected_starttime}" || return 1
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  mapfile -d '' -t argv < "/proc/${pid}/cmdline" || return 1
  (( ${#argv[@]} > 0 )) || return 1
  guard_real="$(realpath -e "${guard_script}")" || return 1
  run_real="$(realpath -e "${run_dir}")" || return 1

  for ((i = 0; i < ${#argv[@]}; i++)); do
    if [[ "${argv[i]}" == "${guard_real}" ]]; then
      found_script=1
    elif [[ "${argv[i]}" == "--pid" && $((i + 1)) -lt ${#argv[@]} ]]; then
      [[ "${argv[i + 1]}" == "${train_pid}" ]] && found_pid=1
    elif [[ "${argv[i]}" == "--train-starttime" && $((i + 1)) -lt ${#argv[@]} ]]; then
      [[ "${argv[i + 1]}" == "${train_starttime}" ]] && found_train_start=1
    elif [[ "${argv[i]}" == "--run-dir" && $((i + 1)) -lt ${#argv[@]} ]]; then
      arg_real="$(realpath -m "${argv[i + 1]}")"
      [[ "${arg_real}" == "${run_real}" ]] && found_run=1
    fi
  done
  (( found_script == 1 && found_pid == 1 && found_train_start == 1 && found_run == 1 ))
}
