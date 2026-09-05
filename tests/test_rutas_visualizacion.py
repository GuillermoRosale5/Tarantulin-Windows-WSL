"""Pruebas sin dependencias externas para las rutas automaticas de visualizacion."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import tempfile


RAIZ_REPOSITORIO = Path(__file__).resolve().parents[1]


def _symlinks_disponibles(base: Path) -> bool:
  origen = base / "prueba-origen"
  enlace = base / "prueba-enlace"
  origen.write_text("prueba", encoding="utf-8")
  try:
    enlace.symlink_to(origen)
  except OSError:
    return False
  enlace.unlink()
  origen.unlink()
  return True


def _cargar_funciones(ruta: Path, nombres: set[str]) -> dict[str, object]:
  arbol = ast.parse(ruta.read_text(encoding="utf-8"), filename=str(ruta))
  funciones = [
      nodo
      for nodo in arbol.body
      if isinstance(nodo, ast.FunctionDef) and nodo.name in nombres
  ]
  encontrados = {funcion.name for funcion in funciones}
  assert encontrados == nombres, f"Faltan funciones en {ruta}: {nombres - encontrados}"
  modulo = ast.Module(body=funciones, type_ignores=[])
  espacio: dict[str, object] = {"Path": Path, "re": re}
  exec(compile(modulo, str(ruta), "exec"), espacio)
  return espacio


def _probar_graficador(base: Path) -> None:
  funciones = _cargar_funciones(
      RAIZ_REPOSITORIO / "scripts/graficar_recompensas.py",
      {
          "_ultima_ejecucion_desde_puntero",
          "_clave_orden_ejecucion",
          "_ultima_ejecucion",
          "_csvs_recompensa_de_ejecucion",
      },
  )
  desde_puntero = funciones["_ultima_ejecucion_desde_puntero"]
  ultima_ejecucion = funciones["_ultima_ejecucion"]
  csvs_ejecucion = funciones["_csvs_recompensa_de_ejecucion"]

  registros = base / "registros"
  segura = registros / "ejecucion-segura"
  exterior = base / "exterior"
  segura.mkdir(parents=True)
  exterior.mkdir()
  puntero = registros / "ultima_run.txt"

  puntero.write_text(str(exterior), encoding="utf-8")
  assert desde_puntero(registros) is None

  puntero.unlink()
  puntero.symlink_to(base / "puntero-controlado")
  assert desde_puntero(registros) is None
  puntero.unlink()

  puntero.write_text(str(segura), encoding="utf-8")
  assert desde_puntero(registros) == segura.resolve()
  puntero.unlink()

  (registros / "ejecucion-enlazada").symlink_to(exterior, target_is_directory=True)
  assert ultima_ejecucion(registros) == segura.resolve()

  csv_real = segura / "recompensas.csv"
  csv_real.write_text("num_steps,reward_total\n1,1\n", encoding="utf-8")
  csv_enlazado = segura / "recompensas_externas.csv"
  csv_exterior = exterior / "datos.csv"
  csv_exterior.write_text("num_steps,reward_total\n9,9\n", encoding="utf-8")
  csv_enlazado.symlink_to(csv_exterior)
  assert csvs_ejecucion(segura) == [csv_real]


def _probar_visualizador(base: Path) -> None:
  funciones = _cargar_funciones(
      RAIZ_REPOSITORIO / "scripts/visualizar_resultados_mjx.py",
      {
          "_ultima_ejecucion_desde_puntero",
          "_checkpoints_ordenados",
          "_checkpoint_por_indice",
      },
  )
  desde_puntero = funciones["_ultima_ejecucion_desde_puntero"]
  checkpoints_ordenados = funciones["_checkpoints_ordenados"]
  checkpoint_por_indice = funciones["_checkpoint_por_indice"]

  registros = base / "registros-visor"
  segura = registros / "ejecucion-segura"
  checkpoint_seguro = segura / "checkpoints" / "000000000001"
  exterior = base / "checkpoint-exterior"
  checkpoint_seguro.mkdir(parents=True)
  (exterior / "checkpoints" / "999999999999").mkdir(parents=True)
  puntero = registros / "ultima_run.txt"

  puntero.write_text(str(exterior), encoding="utf-8")
  assert desde_puntero(registros) is None
  (registros / "ejecucion-enlazada").symlink_to(exterior, target_is_directory=True)
  assert checkpoints_ordenados(registros) == [checkpoint_seguro]
  assert checkpoint_por_indice(registros, 0) == checkpoint_seguro

  checkpoint_enlazado = segura / "checkpoints" / "999999999998"
  checkpoint_enlazado.symlink_to(
      exterior / "checkpoints" / "999999999999", target_is_directory=True
  )
  assert checkpoints_ordenados(registros) == [checkpoint_seguro]


def main() -> None:
  with tempfile.TemporaryDirectory(prefix="tarantulin-rutas-visualizacion-") as tmp:
    base = Path(tmp)
    if not _symlinks_disponibles(base):
      print("VISUALIZATION_PATH_SAFETY_TESTS_SKIPPED: symlinks no disponibles")
      return
    _probar_graficador(base)
    _probar_visualizador(base)
  print("VISUALIZATION_PATH_SAFETY_TESTS_OK")


if __name__ == "__main__":
  main()
