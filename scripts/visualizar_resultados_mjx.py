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
      "  .\\tarantulin.ps1 view-results",
      file=sys.stderr,
  )
  raise SystemExit(2) from exc

from mujoco_playground._src import wrapper

from tarantulin.hiperparametros import ppo_tarantulin_standup
from tarantulin.tarantulin_mjx_env import TarantulinStandup
from tarantulin.tarantulin_mjx_env import default_config


warnings.filterwarnings(
    "ignore",
    message="overflow encountered in cast",
    category=RuntimeWarning,
)


def _infer_hidden_layer_sizes(tree) -> tuple[int, ...] | None:
  if isinstance(tree, (list, tuple)):
    for item in tree:
      nested = _infer_hidden_layer_sizes(item)
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
    nested = _infer_hidden_layer_sizes(value)
    if nested:
      return nested

  if hidden_layers:
    hidden_layers.sort(key=lambda item: item[0])
    if len(hidden_layers) == 1:
      return tuple(size for _, size in hidden_layers)
    return tuple(size for _, size in hidden_layers[:-1])
  return None


def _last_run_from_pointer(logs_dir: Path) -> Path | None:
  pointer_path = logs_dir / "ultima_run.txt"
  if not pointer_path.exists():
    return None
  try:
    run_dir = Path(pointer_path.read_text(encoding="utf-8").strip())
  except OSError:
    return None
  if run_dir.is_dir():
    return run_dir
  return None


def _latest_checkpoint(logs_dir: Path) -> Path:
  pointed_run = _last_run_from_pointer(logs_dir)
  if pointed_run is not None:
    checkpoints_dir = pointed_run / "checkpoints"
    if checkpoints_dir.is_dir():
      candidates = [
          ckpt for ckpt in checkpoints_dir.iterdir()
          if ckpt.is_dir() and ckpt.name.isdigit()
      ]
      if candidates:
        return max(candidates, key=lambda path: int(path.name))

  candidates: list[Path] = []
  for ckpt_root in logs_dir.glob("*/checkpoints"):
    for ckpt in ckpt_root.iterdir():
      if ckpt.is_dir() and ckpt.name.isdigit():
        candidates.append(ckpt)
  if not candidates:
    raise FileNotFoundError(f"No hay checkpoints en {logs_dir}.")
  return max(
      candidates,
      key=lambda path: max(path.parent.parent.stat().st_mtime, path.stat().st_mtime),
  )


def _sorted_checkpoints(logs_dir: Path) -> list[Path]:
  pointed_run = _last_run_from_pointer(logs_dir)
  if pointed_run is not None:
    checkpoints_dir = pointed_run / "checkpoints"
    if checkpoints_dir.is_dir():
      candidates = [
          ckpt for ckpt in checkpoints_dir.iterdir()
          if ckpt.is_dir() and ckpt.name.isdigit()
      ]
      if candidates:
        return sorted(candidates, key=lambda path: int(path.name), reverse=True)

  candidates: list[Path] = []
  for ckpt_root in logs_dir.glob("*/checkpoints"):
    for ckpt in ckpt_root.iterdir():
      if ckpt.is_dir() and ckpt.name.isdigit():
        candidates.append(ckpt)
  return sorted(
      candidates,
      key=lambda path: max(path.parent.parent.stat().st_mtime, path.stat().st_mtime),
      reverse=True,
  )


def _checkpoint_by_index(logs_dir: Path, index_from_latest: int) -> Path:
  if index_from_latest < 0:
    raise ValueError("checkpoint_index debe ser >= 0")
  checkpoints = _sorted_checkpoints(logs_dir)
  if not checkpoints:
    raise FileNotFoundError(f"No hay checkpoints en {logs_dir}.")
  if index_from_latest >= len(checkpoints):
    raise FileNotFoundError(
        f"Solo hay {len(checkpoints)} checkpoints disponibles en {logs_dir}."
    )
  return checkpoints[index_from_latest]


def _run_dir_from_checkpoint(checkpoint_path: Path) -> Path | None:
  if checkpoint_path.parent.name == "checkpoints":
    return checkpoint_path.parent.parent
  if checkpoint_path.name == "checkpoints":
    return checkpoint_path.parent
  if (checkpoint_path / "checkpoints").is_dir():
    return checkpoint_path
  return None


def _apply_saved_env_config(env_cfg, checkpoint_path: Path) -> None:
  run_dir = _run_dir_from_checkpoint(checkpoint_path)
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


def _crear_network_factory(ppo_params):
  """Crea la network factory extrayendo activacion_red del config sin pasarla
  como kwarg invalido a make_ppo_networks."""
  factory_kwargs = dict(ppo_params.network_factory)
  nombre_activacion = factory_kwargs.pop("activacion_red", "swish")
  activacion = _ACTIVACIONES_DISPONIBLES.get(nombre_activacion, nn.swish)
  factory_kwargs["activation"] = activacion
  return functools.partial(ppo_networks.make_ppo_networks, **factory_kwargs)


def _apply_reset_preset(env_cfg, reset_preset: str | None) -> None:
  if not reset_preset or reset_preset == "actual":
    return
  if reset_preset == "suelo2":
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
  elif reset_preset == "ideal":
    env_cfg.reset_pose_mode = "default"
    env_cfg.reset_randomize_state = False
    env_cfg.reset_use_joint_base_pose = False
    env_cfg.reset_project_feet_to_floor = False
    env_cfg.reset_body_z_base = -1.0
    env_cfg.reset_noise_scale = 0.0
    env_cfg.reset_step_count_max = 0
    env_cfg.reset_episode_length_jitter_steps = 0
    env_cfg.reset_grace_step_jitter_steps = 0
  elif reset_preset in ("caida_lateral", "boca_abajo"):
    env_cfg.reset_pose_mode = reset_preset
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
    raise ValueError(f"reset_preset no reconocido: {reset_preset}")


def _load_policy(
    checkpoint_path: Path,
    impl: str,
    episode_length: int,
    xml_path: str | None,
    reset_preset: str | None,
):
  env_cfg = default_config()
  _apply_saved_env_config(env_cfg, checkpoint_path)
  if xml_path:
    xml = Path(xml_path)
    if not xml.is_absolute():
      xml = (Path.cwd() / xml).resolve()
    if not xml.exists():
      raise FileNotFoundError(f"No existe el XML seleccionado: {xml}")
    env_cfg.xml_path = xml.as_posix()
  _apply_reset_preset(env_cfg, reset_preset)
  env_cfg.impl = impl
  if episode_length:
    env_cfg.episode_length = episode_length
  env = TarantulinStandup(config=env_cfg)

  ppo_params = ppo_tarantulin_standup()
  if episode_length:
    ppo_params.episode_length = episode_length

  wrapped_env = wrapper.wrap_for_brax_training(
      env,
      episode_length=ppo_params.episode_length,
      action_repeat=ppo_params.action_repeat,
  )

  # Usa _crear_network_factory para filtrar activacion_red (no es kwarg valido
  # de make_ppo_networks) y mapear al objeto nn.* correcto.
  network_factory = _crear_network_factory(ppo_params)
  params = checkpoint.load(checkpoint_path.resolve())
  hidden_layer_sizes = _infer_hidden_layer_sizes(params)
  if hidden_layer_sizes:
    nombre_activacion = getattr(ppo_params.network_factory, "activacion_red", "swish")
    activacion = _ACTIVACIONES_DISPONIBLES.get(nombre_activacion, nn.swish)
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=hidden_layer_sizes,
        value_hidden_layer_sizes=hidden_layer_sizes,
        policy_obs_key=ppo_params.network_factory.policy_obs_key,
        value_obs_key=ppo_params.network_factory.value_obs_key,
        activation=activacion,
    )

  ppo_network = network_factory(wrapped_env.observation_size, wrapped_env.action_size)
  make_policy = ppo_networks.make_inference_fn(ppo_network)
  policy = jax.jit(make_policy(params, deterministic=True))
  return env, wrapped_env, policy, ppo_params


def _copy_state_to_mujoco_data(state, mj_data: mujoco.MjData) -> None:
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


def _reset_episode(reset_fn, rng, data, env):
  rng, reset_key = jax.random.split(rng)
  state = reset_fn(jax.random.split(reset_key, 1))
  _copy_state_to_mujoco_data(state, data)
  mujoco.mj_forward(env.mj_model, data)
  return rng, state


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--impl", choices=["jax"], default="jax")
  parser.add_argument("--logs_dir", default="logs_tarantulin_mjx")
  parser.add_argument("--checkpoint_path", default=None)
  parser.add_argument("--checkpoint_index", type=int, default=0)
  parser.add_argument("--episode_length", type=int, default=0)
  parser.add_argument("--sleep", type=float, default=0.01)
  parser.add_argument("--episode_pause", type=float, default=1.0)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--xml_path", default=None)
  parser.add_argument(
      "--reset_preset",
      choices=["actual", "suelo2", "ideal", "caida_lateral", "boca_abajo"],
      default="actual",
  )
  parser.add_argument("--auto_reset", action="store_true")
  parser.add_argument("--freeze_on_done", action="store_true")
  parser.add_argument("--dry_run", action="store_true")
  args = parser.parse_args()

  if not args.freeze_on_done:
    args.auto_reset = True

  print("Backend JAX:", jax.default_backend())
  print("Dispositivos JAX:", jax.devices())

  checkpoint_path = (
      Path(args.checkpoint_path)
      if args.checkpoint_path
      else _checkpoint_by_index(Path(args.logs_dir), args.checkpoint_index)
  )
  print("Checkpoint:", checkpoint_path.resolve())
  env, wrapped_env, policy, ppo_params = _load_policy(
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
  def advance_episode(state, rng):
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
  rng, state = _reset_episode(reset, rng, data, env)

  if args.dry_run:
    state, rng, _ = advance_episode(state, rng)
    _copy_state_to_mujoco_data(state, data)
    mujoco.mj_forward(model, data)
    print("view-results dry-run OK.")
    return

  print("Abriendo mujoco.viewer. Cierra la ventana para terminar.")
  pending_manual_reset = False

  def on_key(keycode: int) -> None:
    nonlocal pending_manual_reset
    if keycode in (ord("r"), ord("R")):
      pending_manual_reset = True

  with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
    episode_step = 0
    episode_return = 0.0
    episode_done = False
    episode_finished_at = 0.0
    while viewer.is_running():
      loop_start = time.time()
      if pending_manual_reset:
        rng, state = _reset_episode(reset, rng, data, env)
        episode_step = 0
        episode_return = 0.0
        episode_done = False
        episode_finished_at = 0.0
        pending_manual_reset = False

      if episode_done:
        if args.auto_reset and time.time() - episode_finished_at >= args.episode_pause:
          rng, state = _reset_episode(reset, rng, data, env)
          episode_step = 0
          episode_return = 0.0
          episode_done = False
          episode_finished_at = 0.0
        viewer.sync()
      else:
        state, rng, stats = advance_episode(state, rng)
        reward, done_value, z_value = jax.device_get(stats)
        episode_step += 1
        episode_return += float(reward)
        done = bool(done_value)
        time_limit = episode_step >= ppo_params.episode_length
        if done or (time_limit and not disable_done):
          print(
              f"Episodio terminado: steps={episode_step}, "
              f"return={episode_return:.3f}, z={float(z_value):.3f}, done={done}",
              flush=True,
          )
          episode_done = True
          episode_finished_at = time.time()
        _copy_state_to_mujoco_data(state, data)
        mujoco.mj_forward(model, data)
        viewer.sync()

      elapsed = time.time() - loop_start
      if args.sleep > elapsed:
        time.sleep(args.sleep - elapsed)


if __name__ == "__main__":
  main()
