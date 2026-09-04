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

from scripts.graficar_recompensas import DEFAULT_COLUMNS_ARG
from scripts.graficar_recompensas import _filtrar_ultimo_segmento
from scripts.graficar_recompensas import _has_data
from scripts.graficar_recompensas import _legend_label
from scripts.graficar_recompensas import _latest_run
from scripts.graficar_recompensas import _pick_rows_for_plot
from scripts.graficar_recompensas import _plot_style
from scripts.graficar_recompensas import _resolve_columns
from scripts.graficar_recompensas import _series
from scripts.graficar_recompensas import _smooth
from scripts.graficar_recompensas import _time_window_text


def _read_csv_live(path: Path) -> list[dict[str, str]]:
  if not path.exists() or path.stat().st_size == 0:
    return []
  try:
    with path.open(newline="", encoding="utf-8") as f:
      return list(csv.DictReader(f))
  except (OSError, csv.Error):
    return []


def _resolve_live_paths(args: argparse.Namespace) -> tuple[Path, Path]:
  if args.csv_path:
    csv_path = Path(args.csv_path)
    return csv_path.parent, csv_path
  run_dir = Path(args.run_dir) if args.run_dir else _latest_run(Path(args.logs_dir))
  return run_dir, run_dir / "recompensas.csv"


def _draw_waiting(ax, message: str) -> None:
  ax.clear()
  ax.set_title(message)
  ax.grid(True, alpha=0.3)


def _draw_plot(
    ax,
    rows: list[dict[str, str]],
    csv_path: Path,
    run_dir: Path,
    args: argparse.Namespace,
) -> tuple[int, str]:
  if args.segment == "last":
    rows = _filtrar_ultimo_segmento(rows)
  rows, source_used = _pick_rows_for_plot(rows, args.source, args.columns)
  if args.tail > 0:
    rows = rows[-args.tail :]

  x_values = _series(rows, args.x)
  if not _has_data(x_values):
    _draw_waiting(ax, f"Esperando datos validos para eje X: {args.x}")
    return 0, ""

  x_clean = [0.0 if value is None else value for value in x_values]
  columns = _resolve_columns(rows, args.columns)

  ax.clear()
  plotted = 0
  sample_row = rows[0]
  for column in columns:
    values = _smooth(
        _series(rows, column, unidades_recompensa=args.unidades_recompensa),
        args.smooth,
    )
    if not _has_data(values):
      continue
    y = [float("nan") if value is None else value for value in values]
    ax.plot(
        x_clean,
        y,
        label=_legend_label(sample_row, column),
        **_plot_style(column),
    )
    plotted += 1

  if plotted == 0:
    _draw_waiting(ax, f"Esperando columnas de recompensa con datos en {csv_path.name}")
    return 0, ""

  time_text = _time_window_text(csv_path, run_dir, rows)
  ax.set_title(
      f"Recompensas en directo - {run_dir.name} / {csv_path.name}\n"
      f"{time_text} | origen={source_used}"
  )
  ax.set_xlabel(args.x)
  unidades_y = (
      "recompensa PPO escalada"
      if args.unidades_recompensa == "scaled"
      else "recompensa raw"
  )
  ax.set_ylabel(f"reward ponderada ({unidades_y}); penalties dibujadas en negativo")
  ax.grid(True, alpha=0.3)
  ax.legend(loc="best", fontsize="small", ncols=3)
  return plotted, time_text


def _last_nonempty(rows: list[dict[str, str]], key: str) -> str:
  for row in reversed(rows):
    value = row.get(key, "")
    if value not in ("", None):
      return str(value)
  return ""


def _status_line(rows: list[dict[str, str]], csv_path: Path, plotted: int) -> str:
  if not rows:
    return f"Esperando {csv_path}..."
  parts = [
      f"filas={len(rows)}",
      f"curvas={plotted}",
      f"steps={_last_nonempty(rows, 'num_steps')}",
  ]
  reward_total = _last_nonempty(rows, "reward_total")
  positive_reward = _last_nonempty(rows, "positive_reward")
  penalties = _last_nonempty(rows, "penalties")
  fase = _last_nonempty(rows, "fase_curriculum_recompensa")
  nombre_fase = _last_nonempty(rows, "nombre_curriculum_recompensa")
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
  parser.add_argument("--logs_dir", default="logs_tarantulin_mjx")
  parser.add_argument("--run_dir", default=None)
  parser.add_argument("--csv_path", default=None)
  parser.add_argument(
      "--x",
      choices=["num_steps", "elapsed_seconds", "elapsed_hours", "percent"],
      default="num_steps",
  )
  parser.add_argument("--columns", default=DEFAULT_COLUMNS_ARG)
  parser.add_argument("--source", choices=["eval", "train", "all"], default="all")
  parser.add_argument("--segment", choices=["last", "all"], default="last")
  parser.add_argument("--smooth", type=int, default=3)
  parser.add_argument(
      "--unidades_recompensa",
      "--reward_units",
      dest="unidades_recompensa",
      choices=["scaled", "raw"],
      default="scaled",
      help=(
          "scaled muestra las contribuciones ya multiplicadas por reward_scale; "
          "raw muestra las sumas internas sin reward_scale."
      ),
  )
  parser.add_argument("--interval", type=float, default=5.0)
  parser.add_argument(
      "--tail",
      type=int,
      default=300,
      help="Numero maximo de filas recientes a dibujar. Usa 0 para todo el CSV.",
  )
  parser.add_argument(
      "--output",
      default=None,
      help="PNG que se sobrescribe en cada refresco. Opcional.",
  )
  parser.add_argument(
      "--no_window",
      action="store_true",
      help="No abre ventana; solo actualiza --output e imprime estado.",
  )
  parser.add_argument(
      "--once",
      action="store_true",
      help="Hace un unico refresco y termina. Util para probar o generar PNG puntual.",
  )
  args = parser.parse_args()

  if args.no_window and not args.output:
    raise SystemExit("--no_window necesita --output para poder ver algo.")

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
      run_dir, csv_path = _resolve_live_paths(args)
      rows = _read_csv_live(csv_path)
      mtime = csv_path.stat().st_mtime if csv_path.exists() else None
      should_redraw = mtime != last_mtime or show_window

      if should_redraw:
        last_mtime = mtime
        if not rows:
          _draw_waiting(ax, f"Esperando datos en {csv_path}")
          plotted = 0
        else:
          plotted, _ = _draw_plot(ax, rows, csv_path, run_dir, args)

        fig.tight_layout()
        if args.output:
          output_path = Path(args.output)
          output_path.parent.mkdir(parents=True, exist_ok=True)
          fig.savefig(output_path, dpi=150)
        if show_window:
          fig.canvas.draw_idle()
          plt.pause(0.05)

        print(_status_line(rows, csv_path, plotted), flush=True)
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
