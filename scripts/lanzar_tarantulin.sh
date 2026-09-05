#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

FASE_RECOMPENSA="${TARANTULIN_FASE_RECOMPENSA:-}"
PERFIL_PPO="${TARANTULIN_PERFIL_PPO:-}"
PERFIL_PPO_PASADO=0
FASE_RECOMPENSA_PASADA=0

usage() {
  cat <<'EOF'
Uso:
  scripts/lanzar_tarantulin.sh [opciones]

Sin opciones abre menus por numeros para perfil PPO y fase curricular.

Opciones utiles:
  --perfil-ppo depuracion|ligero|ligero_rapido|completo
  --fase-recompensa 1|2|3|auto

Fase auto lanza scripts/curriculo_automatico_tarantulin.sh:
  fases 1->2->3, 200M steps totales, cambio automatico por rendimiento.
EOF
}

for arg in "$@"; do
  case "${arg}" in
    --help|-h)
      usage
      exit 0
      ;;
    --perfil-ppo|--perfil-ppo=*)
      PERFIL_PPO_PASADO=1
      ;;
    --fase-recompensa|--fase-recompensa=*)
      FASE_RECOMPENSA_PASADA=1
      ;;
  esac
done

prev_arg=""
for arg in "$@"; do
  case "${prev_arg}" in
    --perfil-ppo)
      PERFIL_PPO="${arg}"
      prev_arg=""
      continue
      ;;
    --fase-recompensa)
      FASE_RECOMPENSA="${arg}"
      prev_arg=""
      continue
      ;;
  esac
  case "${arg}" in
    --perfil-ppo|--fase-recompensa)
      prev_arg="${arg}"
      ;;
    --perfil-ppo=*)
      PERFIL_PPO="${arg#*=}"
      ;;
    --fase-recompensa=*)
      FASE_RECOMPENSA="${arg#*=}"
      ;;
  esac
done

elegir_perfil_ppo() {
  cat >&2 <<'EOF'
Perfiles PPO:
  1) depuracion      5M steps, 20 evals, ep 1000, red 256-128       | prueba rapida
  2) ligero          100M steps, 200 evals, ep 1500, red 256-256    | recomendado
  3) ligero_rapido   100M steps, 200 evals, ep 1500, red 256-256, 1024 envs

  completo existe para ejecuciones finales: usa --perfil-ppo completo si lo quieres.
EOF
  local opcion
  read -r -p "Elige perfil PPO [2]: " opcion < /dev/tty
  case "${opcion:-2}" in
    1|depuracion) printf '%s\n' "depuracion" ;;
    2|ligero) printf '%s\n' "ligero" ;;
    3|ligero_rapido) printf '%s\n' "ligero_rapido" ;;
    *)
      echo "Opcion no reconocida: ${opcion}. Usa 1, 2 o 3." >&2
      exit 1
      ;;
  esac
}

elegir_fase_recompensa() {
  cat >&2 <<'EOF'
Fases curriculares de recompensa:
  1) mantener_pose_xml        empieza cerca de la pose ideal y aprende a quedarse ahi
  2) llegar_desde_suelo       empieza peor/en suelo y debe volver a la pose XML
  3) recuperar_desde_caida    recuperacion robusta desde caidas/perturbaciones
  4) curriculo_3_fases_200M   supervisor: fases 1->2->3, 200M pasos, cambio automatico

  0 existe como base_actual sin curriculo: usa --fase-recompensa 0 si lo necesitas.
EOF
  local opcion
  read -r -p "Elige fase curricular [1]: " opcion < /dev/tty
  case "${opcion:-1}" in
    0|base|base_actual) printf '%s\n' "0" ;;
    1|mantener|pose|pose_xml) printf '%s\n' "1" ;;
    2|suelo|llegar) printf '%s\n' "2" ;;
    3|caida|recuperar) printf '%s\n' "3" ;;
    4|auto) printf '%s\n' "auto" ;;
    *)
      echo "Opcion no reconocida: ${opcion}. Usa 1, 2, 3 o 4." >&2
      exit 1
      ;;
  esac
}

if (( PERFIL_PPO_PASADO == 0 )) && [[ -z "${PERFIL_PPO}" ]]; then
  if [[ -t 0 && -t 1 ]]; then
    PERFIL_PPO="$(elegir_perfil_ppo)"
  else
    PERFIL_PPO="ligero"
  fi
fi

if (( FASE_RECOMPENSA_PASADA == 0 )) && [[ -z "${FASE_RECOMPENSA}" ]]; then
  if [[ -t 0 && -t 1 ]]; then
    FASE_RECOMPENSA="$(elegir_fase_recompensa)"
  else
    FASE_RECOMPENSA="1"
  fi
fi

if [[ "${FASE_RECOMPENSA:-}" == "auto" ]]; then
  cmd=(./scripts/curriculo_automatico_tarantulin.sh \
    --perfil-ppo "${PERFIL_PPO:-ligero}" \
    --pasos-totales 200000000)
  export TARANTULIN_SHOW_PPO_PROFILES=0
  exec "${cmd[@]}"
fi

cmd=(./scripts/tarantulin.sh entrenar \
  --segundo-plano \
  --setup \
  --desde-cero)

if (( FASE_RECOMPENSA_PASADA == 0 )); then
  cmd+=(--fase-recompensa "${FASE_RECOMPENSA:-1}")
fi

if (( PERFIL_PPO_PASADO == 0 )); then
  cmd+=(--perfil-ppo "${PERFIL_PPO:-ligero}")
fi

export TARANTULIN_SHOW_PPO_PROFILES=0
exec "${cmd[@]}" "$@"
