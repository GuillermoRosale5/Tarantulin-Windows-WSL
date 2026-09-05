from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

try:
  import matplotlib
  import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
  if exc.name != "matplotlib":
    raise
  print(
      "Falta matplotlib en este Python. No ejecutes este archivo con Python de Windows.\n"
      "Desde WSL usa:\n"
      "  ./scripts/ver_recompensas_en_directo.sh",
      file=sys.stderr,
  )
  raise SystemExit(2) from exc

from scripts.graficar_recompensas import ARG_COLUMNAS_PREDETERMINADAS
from scripts.graficar_recompensas import _filtrar_ultimo_segmento
from scripts.graficar_recompensas import _tiene_datos
from scripts.graficar_recompensas import _etiqueta_leyenda
from scripts.graficar_recompensas import _ultima_ejecucion
from scripts.graficar_recompensas import _elegir_filas_grafica
from scripts.graficar_recompensas import _estilo_grafica
from scripts.graficar_recompensas import _resolver_columnas
from scripts.graficar_recompensas import _serie
from scripts.graficar_recompensas import _suavizar
from scripts.graficar_recompensas import _texto_intervalo_temporal


def _leer_csv_directo(path: Path) -> list[dict[str, str]]:
  if not path.exists() or path.stat().st_size == 0:
    return []
  try:
    with path.open(newline="", encoding="utf-8") as f:
      return list(csv.DictReader(f))
  except (OSError, csv.Error):
    return []


def _resolver_rutas_directo(args: argparse.Namespace) -> tuple[Path, Path]:
  if args.csv_path:
    csv_path = Path(args.csv_path)
    return csv_path.parent, csv_path
  run_dir = Path(args.run_dir) if args.run_dir else _ultima_ejecucion(Path(args.logs_dir))
  return run_dir, run_dir / "recompensas.csv"


def _dibujar_espera(ax, message: str) -> None:
  ax.clear()
  ax.set_title(message)
  ax.grid(True, alpha=0.3)


def _dibujar_grafica(
    ax,
    rows: list[dict[str, str]],
    csv_path: Path,
    run_dir: Path,
    args: argparse.Namespace,
) -> tuple[int, str]:
  if args.segment == "ultimo":
    rows = _filtrar_ultimo_segmento(rows)
  rows, origen_usado = _elegir_filas_grafica(rows, args.source, args.columns)
  if args.tail > 0:
    rows = rows[-args.tail :]

  x_values = _serie(rows, args.x)
  if not _tiene_datos(x_values):
    _dibujar_espera(ax, f"Esperando datos validos para eje X: {args.x}")
    return 0, ""

  x_clean = [0.0 if value is None else value for value in x_values]
  columns = _resolver_columnas(rows, args.columns)

  ax.clear()
  plotted = 0
  sample_row = rows[0]
  for column in columns:
    values = _suavizar(
        _serie(rows, column, unidades_recompensa=args.unidades_recompensa),
        args.smooth,
    )
    if not _tiene_datos(values):
      continue
    y = [float("nan") if value is None else value for value in values]
    ax.plot(
        x_clean,
        y,
        label=_etiqueta_leyenda(sample_row, column),
        **_estilo_grafica(column),
    )
    plotted += 1

  if plotted == 0:
    _dibujar_espera(ax, f"Esperando columnas de recompensa con datos en {csv_path.name}")
    return 0, ""

  time_text = _texto_intervalo_temporal(csv_path, run_dir, rows)
  ax.set_title(
      f"Recompensas en directo - {run_dir.name} / {csv_path.name}\n"
      f"{time_text} | origen={origen_usado}"
  )
  ax.set_xlabel(args.x)
  unidades_y = (
      "recompensa PPO escalada"
      if args.unidades_recompensa == "escaladas"
      else "recompensa bruta"
  )
  ax.set_ylabel(
      f"recompensa ponderada ({unidades_y}); penalizaciones dibujadas en negativo"
  )
  ax.grid(True, alpha=0.3)
  ax.legend(loc="best", fontsize="small", ncols=3)
  return plotted, time_text


def _ultimo_no_vacio(rows: list[dict[str, str]], key: str) -> str:
  for row in reversed(rows):
    value = row.get(key, "")
    if value not in ("", None):
      return str(value)
  return ""


def _linea_estado(rows: list[dict[str, str]], csv_path: Path, plotted: int) -> str:
  if not rows:
    return f"Esperando {csv_path}..."
  parts = [
      f"filas={len(rows)}",
      f"curvas={plotted}",
      f"pasos={_ultimo_no_vacio(rows, 'num_steps')}",
  ]
  reward_total = _ultimo_no_vacio(rows, "reward_total")
  positive_reward = _ultimo_no_vacio(rows, "positive_reward")
  penalties = _ultimo_no_vacio(rows, "penalties")
  fase = _ultimo_no_vacio(rows, "fase_curriculum_recompensa")
  nombre_fase = _ultimo_no_vacio(rows, "nombre_curriculum_recompensa")
  if reward_total:
    parts.append(f"reward_total={reward_total}")
  if positive_reward:
    parts.append(f"positive_reward={positive_reward}")
  if penalties:
    parts.append(f"penalties={penalties}")
  if fase or nombre_fase:
    parts.append(f"fase={fase} {nombre_fase}".strip())
  return " | ".join(parts)


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Muestra en directo la evolucion de recompensas desde recompensas.csv."
  )
  parser.add_argument("--directorio-registros", dest="logs_dir", default="logs_tarantulin_mjx")
  parser.add_argument("--directorio-ejecucion", dest="run_dir", default=None)
  parser.add_argument("--ruta-csv", dest="csv_path", default=None)
  parser.add_argument(
      "--eje-x",
      dest="x",
      choices=["num_steps", "elapsed_seconds", "elapsed_hours", "percent"],
      default="num_steps",
  )
  parser.add_argument(
      "--columnas", dest="columns", default=ARG_COLUMNAS_PREDETERMINADAS
  )
  parser.add_argument(
      "--origen",
      dest="source",
      choices=["evaluacion", "entrenamiento", "todos"],
      default="todos",
  )
  parser.add_argument(
      "--segmento",
      dest="segment",
      choices=["ultimo", "todos"],
      default="ultimo",
  )
  parser.add_argument("--suavizado", dest="smooth", type=int, default=3)
  parser.add_argument(
      "--unidades-recompensa",
      dest="unidades_recompensa",
      choices=["escaladas", "brutas"],
      default="escaladas",
      help=(
          "escaladas muestra las contribuciones ya multiplicadas por reward_scale; "
          "brutas muestra las sumas internas sin reward_scale."
      ),
  )
  parser.add_argument("--intervalo", dest="interval", type=float, default=5.0)
  parser.add_argument(
      "--ultimas-filas",
      dest="tail",
      type=int,
      default=300,
      help="Numero maximo de filas recientes a dibujar. Usa 0 para todo el CSV.",
  )
  parser.add_argument(
      "--salida",
      dest="output",
      default=None,
      help="PNG que se sobrescribe en cada refresco. Opcional.",
  )
  parser.add_argument(
      "--sin-ventana",
      action="store_true",
      dest="no_window",
      help="No abre ventana; solo actualiza --salida e imprime el estado.",
  )
  parser.add_argument(
      "--una-vez",
      action="store_true",
      dest="once",
      help="Hace un unico refresco y termina. Util para probar o generar PNG puntual.",
  )
  args = parser.parse_args()

  if args.no_window and not args.output:
    raise SystemExit("--sin-ventana necesita --salida para poder ver algo.")

  show_window = not args.no_window
  if show_window:
    plt.ion()
    fig, ax = plt.subplots(figsize=(13, 7))
  else:
    matplotlib.use("Agg", force=True)
    fig, ax = plt.subplots(figsize=(13, 7))

  print("Ctrl+C para salir.", flush=True)
  last_mtime = None
  try:
    while True:
      run_dir, csv_path = _resolver_rutas_directo(args)
      rows = _leer_csv_directo(csv_path)
      mtime = csv_path.stat().st_mtime if csv_path.exists() else None
      should_redraw = mtime != last_mtime or show_window

      if should_redraw:
        last_mtime = mtime
        if not rows:
          _dibujar_espera(ax, f"Esperando datos en {csv_path}")
          plotted = 0
        else:
          plotted, _ = _dibujar_grafica(ax, rows, csv_path, run_dir, args)

        fig.tight_layout()
        if args.output:
          output_path = Path(args.output)
          output_path.parent.mkdir(parents=True, exist_ok=True)
          fig.savefig(output_path, dpi=150)
        if show_window:
          fig.canvas.draw_idle()
          plt.pause(0.05)

        print(_linea_estado(rows, csv_path, plotted), flush=True)
        if args.once:
          break

      if show_window and not plt.fignum_exists(fig.number):
        break
      time.sleep(max(args.interval, 0.5))
  except KeyboardInterrupt:
    print("\nMonitor de recompensas detenido.", flush=True)
  finally:
    plt.close(fig)


if __name__ == "__main__":
  main()
