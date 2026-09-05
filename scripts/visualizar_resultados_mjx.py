from __future__ import annotations

import argparse
from collections.abc import Mapping
import functools
import json
from pathlib import Path
import re
import sys
import time
import warnings

try:
  import flax.linen as nn
  import jax
  import jax.numpy as jp
  import mujoco
  import mujoco.viewer
  from brax.training import checkpoint
  from brax.training.agents.ppo import networks as ppo_networks
except ModuleNotFoundError as exc:
  if exc.name not in {"jax", "mujoco", "brax"}:
    raise
  print(
      f"Falta {exc.name} en este Python. No ejecutes este archivo con Python de Windows.\n"
      "Desde WSL usa:\n"
      "  ./scripts/visualizar_ultimo_checkpoint.sh\n"
      "Desde PowerShell usa:\n"
      "  .\\tarantulin.ps1 visualizar-resultados",
      file=sys.stderr,
  )
  raise SystemExit(2) from exc

from mujoco_playground import wrapper

from tarantulin.hiperparametros import configuracion_ppo_ligera
from tarantulin.entorno_tarantulin_mjx import TarantulinIncorporarse
from tarantulin.entorno_tarantulin_mjx import default_config


warnings.filterwarnings(
    "ignore",
    message="overflow encountered in cast",
    category=RuntimeWarning,
)


def _inferir_tamanos_capas_ocultas(tree) -> tuple[int, ...] | None:
  if isinstance(tree, (list, tuple)):
    for item in tree:
      nested = _inferir_tamanos_capas_ocultas(item)
      if nested:
        return nested
    return None
  if not isinstance(tree, Mapping):
    return None

  hidden_layers: list[tuple[int, int]] = []
  for key, value in tree.items():
    if isinstance(key, str):
      match = re.fullmatch(r"hidden_(\d+)", key)
      if match and isinstance(value, Mapping) and "kernel" in value:
        kernel = value["kernel"]
        if hasattr(kernel, "shape") and len(kernel.shape) == 2:
          hidden_layers.append((int(match.group(1)), int(kernel.shape[1])))
    nested = _inferir_tamanos_capas_ocultas(value)
    if nested:
      return nested

  if hidden_layers:
    hidden_layers.sort(key=lambda item: item[0])
    if len(hidden_layers) == 1:
      return tuple(size for _, size in hidden_layers)
    return tuple(size for _, size in hidden_layers[:-1])
  return None


def _ultima_ejecucion_desde_puntero(logs_dir: Path) -> Path | None:
  pointer_path = logs_dir / "ultima_run.txt"
  if not pointer_path.exists() or pointer_path.is_symlink():
    return None
  try:
    run_dir = Path(pointer_path.read_text(encoding="utf-8").strip())
    resolved_logs = logs_dir.resolve(strict=True)
    resolved_run = run_dir.resolve(strict=True)
  except OSError:
    return None
  if (
      run_dir.is_dir()
      and not run_dir.is_symlink()
      and resolved_run.parent == resolved_logs
      and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", resolved_run.name)
  ):
    return resolved_run
  return None


def _ultimo_checkpoint(logs_dir: Path) -> Path:
  pointed_run = _ultima_ejecucion_desde_puntero(logs_dir)
  if pointed_run is not None:
    checkpoints_dir = pointed_run / "checkpoints"
    if checkpoints_dir.is_dir() and not checkpoints_dir.is_symlink():
      candidates = [
          ckpt for ckpt in checkpoints_dir.iterdir()
          if ckpt.is_dir() and not ckpt.is_symlink() and ckpt.name.isdigit()
      ]
      if candidates:
        return max(candidates, key=lambda path: int(path.name))

  candidates: list[Path] = []
  resolved_logs = logs_dir.resolve()
  for ckpt_root in logs_dir.glob("*/checkpoints"):
    run_dir = ckpt_root.parent
    if (
        run_dir.is_symlink()
        or run_dir.resolve().parent != resolved_logs
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_dir.name)
        or ckpt_root.is_symlink()
    ):
      continue
    for ckpt in ckpt_root.iterdir():
      if ckpt.is_dir() and not ckpt.is_symlink() and ckpt.name.isdigit():
        candidates.append(ckpt)
  if not candidates:
    raise FileNotFoundError(f"No hay checkpoints en {logs_dir}.")
  return max(
      candidates,
      key=lambda path: max(path.parent.parent.stat().st_mtime, path.stat().st_mtime),
  )


def _checkpoints_ordenados(logs_dir: Path) -> list[Path]:
  pointed_run = _ultima_ejecucion_desde_puntero(logs_dir)
  if pointed_run is not None:
    checkpoints_dir = pointed_run / "checkpoints"
    if checkpoints_dir.is_dir() and not checkpoints_dir.is_symlink():
      candidates = [
          ckpt for ckpt in checkpoints_dir.iterdir()
          if ckpt.is_dir() and not ckpt.is_symlink() and ckpt.name.isdigit()
      ]
      if candidates:
        return sorted(candidates, key=lambda path: int(path.name), reverse=True)

  candidates: list[Path] = []
  resolved_logs = logs_dir.resolve()
  for ckpt_root in logs_dir.glob("*/checkpoints"):
    run_dir = ckpt_root.parent
    if (
        run_dir.is_symlink()
        or run_dir.resolve().parent != resolved_logs
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_dir.name)
        or ckpt_root.is_symlink()
    ):
      continue
    for ckpt in ckpt_root.iterdir():
      if ckpt.is_dir() and not ckpt.is_symlink() and ckpt.name.isdigit():
        candidates.append(ckpt)
  return sorted(
      candidates,
      key=lambda path: max(path.parent.parent.stat().st_mtime, path.stat().st_mtime),
      reverse=True,
  )


def _checkpoint_por_indice(logs_dir: Path, indice_desde_ultimo: int) -> Path:
  if indice_desde_ultimo < 0:
    raise ValueError("El indice del checkpoint debe ser mayor o igual que 0.")
  checkpoints = _checkpoints_ordenados(logs_dir)
  if not checkpoints:
    raise FileNotFoundError(f"No hay checkpoints en {logs_dir}.")
  if indice_desde_ultimo >= len(checkpoints):
    raise FileNotFoundError(
        f"Solo hay {len(checkpoints)} checkpoints disponibles en {logs_dir}."
    )
  return checkpoints[indice_desde_ultimo]


def _directorio_ejecucion_desde_checkpoint(checkpoint_path: Path) -> Path | None:
  if checkpoint_path.parent.name == "checkpoints":
    return checkpoint_path.parent.parent
  if checkpoint_path.name == "checkpoints":
    return checkpoint_path.parent
  if (checkpoint_path / "checkpoints").is_dir():
    return checkpoint_path
  return None


def _aplicar_configuracion_entorno_guardada(env_cfg, checkpoint_path: Path) -> None:
  run_dir = _directorio_ejecucion_desde_checkpoint(checkpoint_path)
  if run_dir is None:
    return
  config_path = run_dir / "config_entorno.json"
  if not config_path.exists():
    return
  try:
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as exc:
    print(f"Aviso: no pude leer {config_path}: {exc}", file=sys.stderr)
    return
  for key, value in saved_config.items():
    env_cfg[key] = value


_ACTIVACIONES_DISPONIBLES: dict = {
    "swish": nn.swish,
    "silu": nn.swish,
    "tanh": nn.tanh,
    "relu": nn.relu,
    "elu": nn.elu,
}


def _crear_fabrica_redes(ppo_params):
  """Crea la fábrica de redes extrayendo activacion_red de la configuración sin pasarla
  como kwarg invalido a make_ppo_networks."""
  factory_kwargs = dict(ppo_params.network_factory)
  nombre_activacion = factory_kwargs.pop("activacion_red", "swish")
  activacion = _ACTIVACIONES_DISPONIBLES.get(nombre_activacion, nn.swish)
  factory_kwargs["activation"] = activacion
  return functools.partial(ppo_networks.make_ppo_networks, **factory_kwargs)


def _aplicar_postura_inicial(env_cfg, postura_inicial: str | None) -> None:
  if not postura_inicial or postura_inicial == "actual":
    return
  if postura_inicial == "suelo2":
    env_cfg.reset_pose_mode = "suelo2"
    env_cfg.reset_randomize_state = False
    env_cfg.reset_use_joint_base_pose = True
    env_cfg.reset_q1_base_deg = 0.0
    # El XML ideal tiene la pose horneada: qpos0 equivale a q2=+28.26,
    # q3=-93.261 en el modelo base. Aplicamos el delta para aproximar suelo2.
    env_cfg.reset_q2_base_deg = -28.26
    env_cfg.reset_q3_base_deg = 87.261
    env_cfg.reset_project_feet_to_floor = True
    env_cfg.reset_foot_ground_margin = 0.003
    env_cfg.reset_body_z_base = 0.036
    env_cfg.reset_body_xy_noise = 0.0
    env_cfg.reset_body_z_noise = 0.0
    env_cfg.reset_roll_pitch_deg = 0.0
    env_cfg.reset_yaw_deg = 0.0
    env_cfg.reset_q1_noise_deg = 0.0
    env_cfg.reset_q2_noise_deg = 0.0
    env_cfg.reset_q3_noise_deg = 0.0
    env_cfg.reset_noise_scale = 0.0
    env_cfg.reset_step_count_max = 0
    env_cfg.reset_episode_length_jitter_steps = 0
    env_cfg.reset_grace_step_jitter_steps = 0
  elif postura_inicial == "ideal":
    env_cfg.reset_pose_mode = "default"
    env_cfg.reset_randomize_state = False
    env_cfg.reset_use_joint_base_pose = False
    env_cfg.reset_project_feet_to_floor = False
    env_cfg.reset_body_z_base = -1.0
    env_cfg.reset_noise_scale = 0.0
    env_cfg.reset_step_count_max = 0
    env_cfg.reset_episode_length_jitter_steps = 0
    env_cfg.reset_grace_step_jitter_steps = 0
  elif postura_inicial in ("caida_lateral", "boca_abajo"):
    env_cfg.reset_pose_mode = postura_inicial
    env_cfg.disable_done = True
    env_cfg.reset_randomize_state = True
    env_cfg.reset_use_joint_base_pose = False
    env_cfg.reset_project_feet_to_floor = False
    env_cfg.reset_body_z_base = 0.63
    env_cfg.reset_body_xy_noise = 0.060
    env_cfg.reset_body_z_noise = 0.060
    env_cfg.reset_roll_pitch_deg = 70.0
    env_cfg.reset_yaw_deg = 180.0
    env_cfg.reset_q1_noise_deg = 45.0
    env_cfg.reset_q2_noise_deg = 50.0
    env_cfg.reset_q3_noise_deg = 50.0
    env_cfg.reset_noise_scale = 0.0
    env_cfg.reset_step_count_max = 0
  else:
    raise ValueError(f"Postura inicial no reconocida: {postura_inicial}")


def _cargar_politica(
    checkpoint_path: Path,
    impl: str,
    episode_length: int,
    xml_path: str | None,
    postura_inicial: str | None,
):
  env_cfg = default_config()
  _aplicar_configuracion_entorno_guardada(env_cfg, checkpoint_path)
  if xml_path:
    xml = Path(xml_path)
    if not xml.is_absolute():
      xml = (Path.cwd() / xml).resolve()
    if not xml.exists():
      raise FileNotFoundError(f"No existe el XML seleccionado: {xml}")
    env_cfg.xml_path = xml.as_posix()
  _aplicar_postura_inicial(env_cfg, postura_inicial)
  env_cfg.impl = impl
  if episode_length:
    env_cfg.episode_length = episode_length
  env = TarantulinIncorporarse(config=env_cfg)

  ppo_params = configuracion_ppo_ligera()
  if episode_length:
    ppo_params.episode_length = episode_length

  wrapped_env = wrapper.wrap_for_brax_training(
      env,
      episode_length=ppo_params.episode_length,
      action_repeat=ppo_params.action_repeat,
  )

  # Usa _crear_fabrica_redes para filtrar activacion_red (no es un argumento
  # válido de make_ppo_networks) y asociarla al objeto nn.* correcto.
  network_factory = _crear_fabrica_redes(ppo_params)
  params = checkpoint.load(checkpoint_path.resolve())
  tamanos_capas_ocultas = _inferir_tamanos_capas_ocultas(params)
  if tamanos_capas_ocultas:
    nombre_activacion = getattr(ppo_params.network_factory, "activacion_red", "swish")
    activacion = _ACTIVACIONES_DISPONIBLES.get(nombre_activacion, nn.swish)
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tamanos_capas_ocultas,
        value_hidden_layer_sizes=tamanos_capas_ocultas,
        policy_obs_key=ppo_params.network_factory.policy_obs_key,
        value_obs_key=ppo_params.network_factory.value_obs_key,
        activation=activacion,
    )

  ppo_network = network_factory(wrapped_env.observation_size, wrapped_env.action_size)
  make_policy = ppo_networks.make_inference_fn(ppo_network)
  policy = jax.jit(make_policy(params, deterministic=True))
  return env, wrapped_env, policy, ppo_params


def _copiar_estado_a_datos_mujoco(state, mj_data: mujoco.MjData) -> None:
  qpos = jax.device_get(state.data.qpos)
  qvel = jax.device_get(state.data.qvel)
  ctrl = jax.device_get(state.data.ctrl)
  if qpos.ndim > 1:
    qpos = qpos[0]
  if qvel.ndim > 1:
    qvel = qvel[0]
  if ctrl is not None and ctrl.ndim > 1:
    ctrl = ctrl[0]
  mj_data.qpos[:] = qpos
  mj_data.qvel[:] = qvel
  if ctrl is not None and mj_data.ctrl.size:
    mj_data.ctrl[:] = ctrl


def _reiniciar_episodio(reset_fn, rng, data, env):
  rng, reset_key = jax.random.split(rng)
  state = reset_fn(jax.random.split(reset_key, 1))
  _copiar_estado_a_datos_mujoco(state, data)
  mujoco.mj_forward(env.mj_model, data)
  return rng, state


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Visualiza en MuJoCo una politica guardada en un checkpoint."
  )
  parser.add_argument("--impl", choices=["jax"], default="jax")
  parser.add_argument(
      "--directorio-registros", dest="logs_dir", default="logs_tarantulin_mjx"
  )
  parser.add_argument("--ruta-checkpoint", dest="checkpoint_path", default=None)
  parser.add_argument("--indice-checkpoint", dest="checkpoint_index", type=int, default=0)
  parser.add_argument("--longitud-episodio", dest="episode_length", type=int, default=0)
  parser.add_argument("--espera", dest="sleep", type=float, default=0.01)
  parser.add_argument("--pausa-episodio", dest="episode_pause", type=float, default=1.0)
  parser.add_argument("--semilla", dest="seed", type=int, default=0)
  parser.add_argument("--ruta-xml", dest="xml_path", default=None)
  parser.add_argument(
      "--postura-inicial",
      dest="reset_preset",
      choices=["actual", "suelo2", "ideal", "caida_lateral", "boca_abajo"],
      default="actual",
  )
  parser.add_argument("--reinicio-automatico", dest="auto_reset", action="store_true")
  parser.add_argument("--congelar-al-terminar", dest="freeze_on_done", action="store_true")
  parser.add_argument("--solo-comprobar", dest="dry_run", action="store_true")
  args = parser.parse_args()

  if not args.freeze_on_done:
    args.auto_reset = True

  print("Backend JAX:", jax.default_backend())
  print("Dispositivos JAX:", jax.devices())

  checkpoint_path = (
      Path(args.checkpoint_path)
      if args.checkpoint_path
      else _checkpoint_por_indice(Path(args.logs_dir), args.checkpoint_index)
  )
  print("Checkpoint:", checkpoint_path.resolve())
  env, wrapped_env, policy, ppo_params = _cargar_politica(
      checkpoint_path,
      args.impl,
      args.episode_length,
      args.xml_path,
      args.reset_preset,
  )
  print("XML:", env.xml_path)
  print("Pose inicial:", args.reset_preset)

  model = mujoco.MjModel.from_xml_path(env.xml_path)
  data = mujoco.MjData(model)
  reset = jax.jit(wrapped_env.reset)
  step = jax.jit(wrapped_env.step)
  disable_done = args.reset_preset in ("caida_lateral", "boca_abajo")

  @jax.jit
  def avanzar_episodio(state, rng):
    rng, action_key = jax.random.split(rng)
    action, _ = policy(state.obs, action_key)
    next_state = step(state, action)
    if disable_done:
      next_state = next_state.replace(done=jp.zeros_like(next_state.done))
    reward = next_state.reward[0]
    done = next_state.done[0]
    z = next_state.data.xpos[0, env._main_body_id, 2]
    return next_state, rng, jp.array([reward, done, z])

  rng = jax.random.PRNGKey(args.seed)
  rng, state = _reiniciar_episodio(reset, rng, data, env)

  if args.dry_run:
    state, rng, _ = avanzar_episodio(state, rng)
    _copiar_estado_a_datos_mujoco(state, data)
    mujoco.mj_forward(model, data)
    print("Comprobacion del visualizador terminada correctamente.")
    return

  print("Abriendo mujoco.viewer. Cierra la ventana para terminar.")
  reinicio_manual_pendiente = False

  def al_pulsar_tecla(keycode: int) -> None:
    nonlocal reinicio_manual_pendiente
    if keycode in (ord("r"), ord("R")):
      reinicio_manual_pendiente = True

  with mujoco.viewer.launch_passive(model, data, key_callback=al_pulsar_tecla) as viewer:
    episode_step = 0
    episode_return = 0.0
    episode_done = False
    episode_finished_at = 0.0
    while viewer.is_running():
      loop_start = time.time()
      if reinicio_manual_pendiente:
        rng, state = _reiniciar_episodio(reset, rng, data, env)
        episode_step = 0
        episode_return = 0.0
        episode_done = False
        episode_finished_at = 0.0
        reinicio_manual_pendiente = False

      if episode_done:
        if args.auto_reset and time.time() - episode_finished_at >= args.episode_pause:
          rng, state = _reiniciar_episodio(reset, rng, data, env)
          episode_step = 0
          episode_return = 0.0
          episode_done = False
          episode_finished_at = 0.0
        viewer.sync()
      else:
        state, rng, stats = avanzar_episodio(state, rng)
        reward, done_value, z_value = jax.device_get(stats)
        episode_step += 1
        episode_return += float(reward)
        done = bool(done_value)
        time_limit = episode_step >= ppo_params.episode_length
        if done or (time_limit and not disable_done):
          print(
              f"Episodio terminado: pasos={episode_step}, "
              f"retorno={episode_return:.3f}, z={float(z_value):.3f}, "
              f"finalizado={done}",
              flush=True,
          )
          episode_done = True
          episode_finished_at = time.time()
        _copiar_estado_a_datos_mujoco(state, data)
        mujoco.mj_forward(model, data)
        viewer.sync()

      elapsed = time.time() - loop_start
      if args.sleep > elapsed:
        time.sleep(args.sleep - elapsed)


if __name__ == "__main__":
  main()
