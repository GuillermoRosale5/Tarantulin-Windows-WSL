"""Regresiones hermeticas del entrenamiento y del runtime Windows/WSL."""

from __future__ import annotations

import ast
import datetime as dt
import json
import os
from pathlib import Path
import tempfile
import unittest
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


class PruebasRegresion(unittest.TestCase):

  def test_estado_atomico_conserva_la_configuracion(self) -> None:
    trainer_path = ROOT / "tarantulin/entrenar_ppo_mjx.py"
    parsed = ast.parse(trainer_path.read_text(encoding="utf-8"))
    selected = [
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_guardar_json", "_guardar_estado"}
    ]
    self.assertEqual(
        [node.name for node in selected], ["_guardar_json", "_guardar_estado"]
    )
    namespace = {
        "Any": Any,
        "Path": Path,
        "dt": dt,
        "json": json,
        "os": os,
        "tempfile": tempfile,
    }
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(trainer_path), "exec"), namespace)

    with tempfile.TemporaryDirectory() as temporary:
      state_path = Path(temporary) / "estado.json"
      namespace["_guardar_estado"](
          state_path,
          "entrenando",
          {
              "perfil_ppo": "ligero",
              "fase_curriculum_recompensa": 2,
              "nombre_curriculum_recompensa": "llegar_desde_suelo",
          },
      )
      namespace["_guardar_estado"](
          state_path, "cancelado", {"motivo_cancelacion": "prueba"}
      )
      state = json.loads(state_path.read_text(encoding="utf-8"))
      self.assertEqual(state["estado"], "cancelado")
      self.assertEqual(state["perfil_ppo"], "ligero")
      self.assertEqual(state["fase_curriculum_recompensa"], 2)
      self.assertEqual(state["nombre_curriculum_recompensa"], "llegar_desde_suelo")
      self.assertEqual(state["motivo_cancelacion"], "prueba")
      self.assertFalse(list(state_path.parent.glob(f".{state_path.name}.*.tmp")))

  def test_metricas_no_contienen_la_clave_literal_duplicada(self) -> None:
    trainer = ROOT / "tarantulin/entrenar_ppo_mjx.py"
    parsed = ast.parse(trainer.read_text(encoding="utf-8"))
    duplicates: list[str] = []
    for node in ast.walk(parsed):
      if not isinstance(node, ast.Dict):
        continue
      keys = [key.value for key in node.keys if isinstance(key, ast.Constant)]
      duplicates.extend(key for key in set(keys) if keys.count(key) > 1)
    self.assertNotIn("vector_gravedad_estabilidad_reward_ponderado", duplicates)

  def test_nohup_no_se_convierte_en_cancelacion_por_sighup(self) -> None:
    source = (ROOT / "tarantulin/entrenar_ppo_mjx.py").read_text(encoding="utf-8")
    self.assertIn("if signal.getsignal(signal.SIGHUP) != signal.SIG_IGN:", source)
    self.assertNotIn(
        "for signal_number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):",
        source,
    )

  def test_entorno_no_comparte_configuracion_mutable(self) -> None:
    path = ROOT / "tarantulin/entorno_tarantulin_mjx.py"
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    environment_class = next(
        node
        for node in parsed.body
        if isinstance(node, ast.ClassDef) and node.name == "TarantulinIncorporarse"
    )
    initializer = next(
        node
        for node in environment_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    defaults = dict(
        zip(
            [argument.arg for argument in initializer.args.args[-len(initializer.args.defaults) :]],
            initializer.args.defaults,
            strict=True,
        )
    )
    self.assertIsInstance(defaults["config"], ast.Constant)
    self.assertIsNone(defaults["config"].value)

  def test_curriculo_suma_solo_pasos_confirmados(self) -> None:
    source = (ROOT / "scripts/curriculo_automatico_tarantulin.sh").read_text(
        encoding="utf-8"
    )
    self.assertIn(
        "PASOS_COMPLETADOS=$(( PASOS_COMPLETADOS + pasos_confirmados_bloque ))",
        source,
    )
    self.assertNotIn(
        "PASOS_COMPLETADOS=$(( PASOS_COMPLETADOS + pasos_bloque_actual ))",
        source,
    )
    self.assertIn('if [[ "${estado_bloque}" != "terminado" ]]', source)

  def test_runtime_impide_sincronizar_codigo_en_uso(self) -> None:
    synchronizer = (ROOT / "scripts/wsl/sync_runtime.sh").read_text(
        encoding="utf-8"
    )
    runner = (ROOT / "scripts/wsl/run_runtime.sh").read_text(encoding="utf-8")
    powershell = (ROOT / "tarantulin.ps1").read_text(encoding="utf-8")
    self.assertIn("flock --exclusive --nonblock", synchronizer)
    self.assertIn("flock --exclusive --nonblock", runner)
    self.assertIn("exec flock --shared", runner)
    self.assertIn('"flock", "--shared"', powershell)


if __name__ == "__main__":
  unittest.main()
