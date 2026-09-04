from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
import re
import sys

try:
  import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
  if exc.name != "matplotlib":
    raise
  print(
      "Falta matplotlib en este Python. No ejecutes este archivo con Python de Windows.\n"
      "Desde WSL usa:\n"
      "  ./scripts/graficar_recompensas.sh\n"
      "Desde PowerShell usa:\n"
      "  .\\tarantulin.ps1 shell  # despues: ./scripts/graficar_recompensas.sh",
      file=sys.stderr,
  )
  raise SystemExit(2) from exc


DEFAULT_COLUMNS = [
    "reward_total",
    "positive_reward",
    "penalties",
    "supervivencia_reward_ponderado",
    "vector_gravedad_reward_ponderado",
    "soporte_estatico_reward_ponderado",
    "altura_reward_ponderado",
    "plano_CoG_arana_reward_ponderado",
    "contactos_suelo_reward_ponderado",
    "poligono_CoG_reward_ponderado",
    "apertura_efectores_q3_cerca_reward_ponderado",
    "q1_centrados_reward_ponderado",
    "q3_separados_centro_reward_ponderado",
    "simetria_patas_reward_ponderado",
    "contacto_invalido_penalty_ponderado",
    "contacto_invalido_persistente_penalty_ponderado",
    "efector_encima_q3_penalty_ponderado",
    "velocidad_vertical_cuerpo_penalty_ponderado",
    "velocidad_angular_cuerpo_penalty_ponderado",
    "control_penalty_ponderado",
    "cambio_accion_penalty_ponderado",
    "limite_articular_penalty_ponderado",
]

DEFAULT_COLUMNS_ARG = "default"

BASE_COLUMN_ALIASES = {
    "eval_episode_reward": ("eval_episode_reward",),
    "eval_reward_per_step": ("eval_reward_per_step",),
    "positive_reward": ("positive_reward", "reward_positive_total"),
    "penalties": ("penalties", "penalty_total"),
    "supervivencia_reward_ponderado": (
        "supervivencia_reward_ponderado",
        "reward_supervivencia_reward",
        "reward_supervivencia_gate",
        "supervivencia_reward_gate",
    ),
    "vector_gravedad_reward_ponderado": ("vector_gravedad_reward_ponderado", "reward_vector_gravedad"),
    "poligono_CoG_reward_ponderado": ("poligono_CoG_reward_ponderado", "reward_poligono_CoG"),
    "altura_reward_ponderado": ("altura_reward_ponderado", "reward_altura"),
    "plano_CoG_arana_reward_ponderado": (
        "plano_CoG_arana_reward_ponderado",
        "plano_CoG_ara?a_reward",
        "reward_plano_CoG_ara?a",
    ),
    "contactos_suelo_reward_ponderado": ("contactos_suelo_reward_ponderado", "reward_contactos_suelo"),
    "soporte_estatico_reward_ponderado": (
        "soporte_estatico_reward_ponderado",
        "reward_soporte_estatico",
        "reward/soporte_estatico",
    ),
    "apertura_efectores_q3_cerca_reward_ponderado": ("apertura_efectores_q3_cerca_reward_ponderado",),
    "q1_centrados_reward_ponderado": ("q1_centrados_reward_ponderado",),
    "q3_separados_centro_reward_ponderado": (
        "q3_separados_centro_reward_ponderado",
        "reward_q3_separados_centro",
    ),
    "simetria_patas_reward_ponderado": ("simetria_patas_reward_ponderado", "reward_simetria_patas"),
    "contacto_invalido_penalty_ponderado": (
        "contacto_invalido_penalty_ponderado",
        "invalid_contact_penalty",
        "penalty_invalid_contact",
        "penalty_invalid_ground_contact",
        "contacto_invalido_legacy_penalty",
        "contacto_suelo_invalido_penalty",
    ),
    "contacto_invalido_persistente_penalty_ponderado": (
        "contacto_invalido_persistente_penalty_ponderado",
        "invalid_contact_persistence_penalty",
    ),
    "efector_encima_q3_penalty_ponderado": (
        "efector_encima_q3_penalty_ponderado",
        "penalty_efector_encima_q3",
    ),
    "velocidad_vertical_cuerpo_penalty_ponderado": (
        "velocidad_vertical_cuerpo_penalty_ponderado",
        "penalty_velocidad_vertical_cuerpo",
    ),
    "velocidad_angular_cuerpo_penalty_ponderado": (
        "velocidad_angular_cuerpo_penalty_ponderado",
        "penalty_velocidad_angular_cuerpo",
    ),
    "control_penalty_ponderado": ("control_penalty_ponderado", "penalty_control"),
    "cambio_accion_penalty_ponderado": (
        "cambio_accion_penalty_ponderado",
        "action_rate_penalty",
        "penalty_action_rate",
    ),
    "limite_articular_penalty_ponderado": (
        "limite_articular_penalty_ponderado",
        "joint_limit_penalty",
        "penalty_joint_limit",
    ),
}

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {}

NEGATED_PLOT_COLUMNS = {
    "contacto_invalido_penalty_ponderado",
    "contacto_invalido_persistente_penalty_ponderado",
    "efector_encima_q3_penalty_ponderado",
    "velocidad_vertical_cuerpo_penalty_ponderado",
    "velocidad_angular_cuerpo_penalty_ponderado",
    "control_penalty_ponderado",
    "cambio_accion_penalty_ponderado",
    "limite_articular_penalty_ponderado",
    "penalties",
}

PLOT_SKIP_COLUMNS = {
    "num_steps",
    "elapsed_seconds",
    "elapsed_hours",
    "percent",
    "source",
    "eval_episode_reward",
    "eval_episode_reward",
    "eval_episode_length",
    "eval_reward_per_step",
    "eval_reward_per_step",
    "wall_steps_per_second",
    "steps_per_second",
    "training_steps_per_second",
}

REWARD_COLORS = [
    "#1f77b4",
    "#2ca02c",
    "#9467bd",
    "#17becf",
    "#bcbd22",
    "#8c564b",
    "#7f7f7f",
    "#aec7e8",
    "#98df8a",
    "#c5b0d5",
]

PENALTY_COLORS = [
    "#d62728",
    "#ff7f0e",
    "#e377c2",
    "#a55194",
    "#b15928",
    "#fb9a99",
    "#fdbf6f",
    "#cab2d6",
]

STATE_COLORS = [
    "#4c78a8",
    "#72b7b2",
    "#54a24b",
    "#b279a2",
    "#9d755d",
]

MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p")

DUPLICATE_WHEN_CANONICAL_EXISTS = {
    "positive_reward": "positive_reward",
    "reward_positive_total": "positive_reward",
    "penalties": "penalties",
    "penalty_total": "penalties",
    "supervivencia_reward_ponderado": "supervivencia_reward_ponderado",
    "reward_supervivencia_gate": "supervivencia_reward_ponderado",
    "reward_supervivencia_reward": "supervivencia_reward_ponderado",
    "supervivencia_reward_gate": "supervivencia_reward_ponderado",
    "vector_gravedad_reward_ponderado": "vector_gravedad_reward_ponderado",
    "reward_vector_gravedad": "vector_gravedad_reward_ponderado",
    "poligono_CoG_reward_ponderado": "poligono_CoG_reward_ponderado",
    "reward_poligono_CoG": "poligono_CoG_reward_ponderado",
    "altura_reward_ponderado": "altura_reward_ponderado",
    "reward_altura": "altura_reward_ponderado",
    "plano_CoG_arana_reward_ponderado": "plano_CoG_arana_reward_ponderado",
    "plano_CoG_ara?a_reward": "plano_CoG_arana_reward_ponderado",
    "reward_plano_CoG_ara?a": "plano_CoG_arana_reward_ponderado",
    "contactos_suelo_reward_ponderado": "contactos_suelo_reward_ponderado",
    "reward_contactos_suelo": "contactos_suelo_reward_ponderado",
    "soporte_estatico_reward_ponderado": "soporte_estatico_reward_ponderado",
    "reward_soporte_estatico": "soporte_estatico_reward_ponderado",
    "reward/soporte_estatico": "soporte_estatico_reward_ponderado",
    "apertura_efectores_q3_cerca_reward_ponderado": "apertura_efectores_q3_cerca_reward_ponderado",
    "q1_centrados_reward_ponderado": "q1_centrados_reward_ponderado",
    "q3_separados_centro_reward_ponderado": "q3_separados_centro_reward_ponderado",
    "reward_q3_separados_centro": "q3_separados_centro_reward_ponderado",
    "simetria_patas_reward_ponderado": "simetria_patas_reward_ponderado",
    "reward_simetria_patas": "simetria_patas_reward_ponderado",
    "contacto_invalido_penalty_ponderado": "contacto_invalido_penalty_ponderado",
    "invalid_contact_penalty": "contacto_invalido_penalty_ponderado",
    "penalty_invalid_contact": "contacto_invalido_penalty_ponderado",
    "penalty_invalid_ground_contact": "contacto_invalido_penalty_ponderado",
    "contacto_suelo_invalido_penalty": "contacto_invalido_penalty_ponderado",
    "contacto_invalido_legacy_penalty": "contacto_invalido_penalty_ponderado",
    "contacto_invalido_persistente_penalty_ponderado": "contacto_invalido_persistente_penalty_ponderado",
    "invalid_contact_persistence_penalty": "contacto_invalido_persistente_penalty_ponderado",
    "efector_encima_q3_penalty_ponderado": "efector_encima_q3_penalty_ponderado",
    "penalty_efector_encima_q3": "efector_encima_q3_penalty_ponderado",
    "velocidad_vertical_cuerpo_penalty_ponderado": "velocidad_vertical_cuerpo_penalty_ponderado",
    "penalty_velocidad_vertical_cuerpo": "velocidad_vertical_cuerpo_penalty_ponderado",
    "velocidad_angular_cuerpo_penalty_ponderado": "velocidad_angular_cuerpo_penalty_ponderado",
    "penalty_velocidad_angular_cuerpo": "velocidad_angular_cuerpo_penalty_ponderado",
    "control_penalty_ponderado": "control_penalty_ponderado",
    "penalty_control": "control_penalty_ponderado",
    "action_rate_penalty": "cambio_accion_penalty_ponderado",
    "penalty_action_rate": "cambio_accion_penalty_ponderado",
    "joint_limit_penalty": "limite_articular_penalty_ponderado",
    "penalty_joint_limit": "limite_articular_penalty_ponderado",
}

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


def _run_sort_key(run_dir: Path) -> float:
  candidates = [run_dir.stat().st_mtime]
  for name in ("recompensas.csv", "progreso.csv"):
    path = run_dir / name
    if path.exists():
      candidates.append(path.stat().st_mtime)
  return max(candidates)


def _latest_run(logs_dir: Path) -> Path:
  pointed_run = _last_run_from_pointer(logs_dir)
  if pointed_run is not None:
    return pointed_run
  runs = [path for path in logs_dir.iterdir() if path.is_dir()]
  if not runs:
    raise FileNotFoundError(f"No hay runs en {logs_dir}")
  return max(runs, key=_run_sort_key)


def _csvs_recompensa_de_run(run_dir: Path) -> list[Path]:
  csv_paths = [
      path
      for path in run_dir.glob("recompensas*.csv")
      if path.is_file() and path.stat().st_size > 0
  ]
  if not csv_paths:
    return []

  def csv_sort_key(path: Path) -> tuple[int, float, str]:
    active_rank = 0 if path.name == "recompensas.csv" else 1
    return (active_rank, -path.stat().st_mtime, path.name)

  return sorted(csv_paths, key=csv_sort_key)


def _read_csv(path: Path) -> list[dict[str, str]]:
  with path.open(newline="", encoding="utf-8") as f:
    return list(csv.DictReader(f))


def _to_float(value: str) -> float | None:
  if value is None or value == "":
    return None
  try:
    return float(value)
  except ValueError:
    return None


def _legacy_metric_column(metric_name: str) -> str:
  return metric_name.replace("/", "_")


def _real_metric_name(column: str) -> str:
  for real_name, aliases in BASE_COLUMN_ALIASES.items():
    if column == real_name or column in aliases:
      return real_name
  if "/" in column:
    return column
  for prefix in ("reward", "penalty", "done", "state", "final", "debug"):
    marker = f"{prefix}_"
    if column.startswith(marker):
      return f"{prefix}/{column[len(marker):]}"
  return column


def _display_metric_name(column: str) -> str:
  for prefix in ("reward", "penalty", "done", "state", "final", "debug"):
    marker = f"{prefix}_"
    if column.startswith(marker):
      return f"{prefix}/{column[len(marker):]}"
  return column


def _column_candidates(column: str) -> tuple[str, ...]:
  candidates = [column]
  candidates.extend(COLUMN_ALIASES.get(column, ()))
  aliases = BASE_COLUMN_ALIASES.get(column)
  if aliases:
    candidates.extend(aliases)
  real_name = _real_metric_name(column)
  if real_name != column:
    candidates.append(real_name)
  if "/" in real_name:
    candidates.append(_legacy_metric_column(real_name))
  return tuple(dict.fromkeys(candidates))


def _matched_column_name(row: dict[str, str], column: str) -> str | None:
  for candidate in _column_candidates(column):
    if candidate in row:
      return candidate
  return None


def _legend_label(row: dict[str, str], column: str) -> str:
  del row
  if column == "reward_total":
    return "reward_total = positive_reward - penalties"
  if column == "positive_reward":
    return "+ positive_reward"
  if column == "penalties":
    return "- penalties"
  if column.endswith("_reward_ponderado"):
    return f"+ {column}"
  if column.endswith("_penalty_ponderado"):
    return f"- {column}"
  return column


def _value_for_column(row: dict[str, str], column: str) -> str:
  for candidate in _column_candidates(column):
    if candidate in row:
      return row.get(candidate, "")
  return ""


def _series(
    rows: list[dict[str, str]],
    column: str,
    unidades_recompensa: str = "raw",
) -> list[float | None]:
  if column == "reward_total":
    direct_values = [_to_float(_value_for_column(row, column)) for row in rows]
    if _has_data(direct_values):
      return _quizas_escalar_serie_recompensa(rows, column, direct_values, unidades_recompensa)
    valores_recompensa = _first_available_series(rows, ("positive_reward", "positive_reward"))
    valores_penalizacion = _first_available_series(rows, ("penalties", "penalties"))
    values = [
        None
        if valor_recompensa is None or valor_penalizacion is None
        else valor_recompensa - valor_penalizacion
        for valor_recompensa, valor_penalizacion in zip(valores_recompensa, valores_penalizacion)
    ]
    return _quizas_escalar_serie_recompensa(rows, column, values, unidades_recompensa)
  values = [_to_float(_value_for_column(row, column)) for row in rows]
  if _should_negate_column(column):
    values = [None if value is None else -value for value in values]
  return _quizas_escalar_serie_recompensa(rows, column, values, unidades_recompensa)


def _has_data(values: list[float | None]) -> bool:
  return any(value is not None for value in values)


def _has_nonzero_data(values: list[float | None]) -> bool:
  return any(value is not None and abs(value) > 1e-12 for value in values)


def _first_available_series(
    rows: list[dict[str, str]], columns: tuple[str, ...]
) -> list[float | None]:
  for column in columns:
    values = [_to_float(_value_for_column(row, column)) for row in rows]
    if _has_data(values):
      return values
  return [None for _ in rows]


def _median(values: list[float]) -> float:
  ordered = sorted(values)
  middle = len(ordered) // 2
  if len(ordered) % 2:
    return ordered[middle]
  return 0.5 * (ordered[middle - 1] + ordered[middle])


def _reward_scale_factor(rows: list[dict[str, str]]) -> float:
  explicit_values = []
  for row in rows:
    value = _to_float(row.get("debug_reward_scale", ""))
    if value is not None and value > 0.0:
      explicit_values.append(value)
  if explicit_values:
    return _median(explicit_values)

  ratios = []
  for row in rows:
    reward_scaled = _to_float(_value_for_column(row, "eval_episode_reward"))
    reward_raw = _to_float(_value_for_column(row, "reward_total"))
    if (
        reward_scaled is not None
        and reward_raw is not None
        and abs(reward_raw) > 1e-9
    ):
      ratio = reward_scaled / reward_raw
      if 0.0 < abs(ratio) <= 1.0:
        ratios.append(abs(ratio))
  if ratios:
    return _median(ratios)
  return 1.0


def _quizas_escalar_serie_recompensa(
    rows: list[dict[str, str]],
    column: str,
    values: list[float | None],
    unidades_recompensa: str,
) -> list[float | None]:
  if unidades_recompensa != "scaled" or not _is_default_plot_column(column):
    return values
  scale = _reward_scale_factor(rows)
  return [None if value is None else value * scale for value in values]


def _es_columna_recompensa(column: str) -> bool:
  if column in PLOT_SKIP_COLUMNS:
    return False
  if column in {"reward_total", "positive_reward"}:
    return True
  return column.endswith("_reward_ponderado")


def _es_columna_penalizacion(column: str) -> bool:
  if column in PLOT_SKIP_COLUMNS:
    return False
  if column == "penalties":
    return True
  return column.endswith("_penalty_ponderado")


def _should_negate_column(column: str) -> bool:
  return column in NEGATED_PLOT_COLUMNS or _es_columna_penalizacion(column)


def _is_default_plot_column(column: str) -> bool:
  return column == "reward_total" or _es_columna_recompensa(column) or _es_columna_penalizacion(column)


def _stable_index(text: str, modulo: int) -> int:
  return sum((index + 1) * ord(char) for index, char in enumerate(text)) % modulo


def _plot_style(column: str) -> dict[str, object]:
  if column == "reward_total":
    return {
        "color": "#111111",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 3.0,
        "markersize": 4.8,
        "alpha": 1.0,
        "zorder": 5,
    }
  if column == "positive_reward":
    return {
        "color": "#178c4f",
        "linestyle": "-",
        "marker": "s",
        "linewidth": 2.4,
        "markersize": 4.4,
        "alpha": 0.95,
        "zorder": 4,
    }
  if column == "penalties":
    return {
        "color": "#b2182b",
        "linestyle": "--",
        "marker": "X",
        "linewidth": 2.4,
        "markersize": 4.4,
        "alpha": 0.95,
        "zorder": 4,
    }
  if _es_columna_penalizacion(column):
    index = _stable_index(column, len(PENALTY_COLORS))
    return {
        "color": PENALTY_COLORS[index],
        "linestyle": "--",
        "marker": MARKERS[index % len(MARKERS)],
        "linewidth": 1.8,
        "markersize": 4.0,
        "alpha": 0.92,
        "zorder": 3,
    }
  if _es_columna_recompensa(column):
    index = _stable_index(column, len(REWARD_COLORS))
    return {
        "color": REWARD_COLORS[index],
        "linestyle": "-",
        "marker": MARKERS[index % len(MARKERS)],
        "linewidth": 1.9,
        "markersize": 4.0,
        "alpha": 0.92,
        "zorder": 3,
    }
  index = _stable_index(column, len(STATE_COLORS))
  return {
      "color": STATE_COLORS[index],
      "linestyle": ":",
      "marker": MARKERS[index % len(MARKERS)],
      "linewidth": 1.4,
      "markersize": 3.4,
      "alpha": 0.82,
      "zorder": 2,
  }


def _resolve_columns(
    rows: list[dict[str, str]],
    columns_arg: str,
) -> list[str]:
  if columns_arg in {"default", "recompensas", "rewards", "reward"}:
    return _resolve_default_columns(rows)

  if columns_arg != "all":
    return [col.strip() for col in columns_arg.split(",") if col.strip()]

  return [
      col
      for col in rows[0].keys()
      if _is_default_plot_column(col) and _has_data(_series(rows, col))
  ]


def _resolve_default_columns(rows: list[dict[str, str]]) -> list[str]:
  if not rows:
    return []

  header = list(rows[0].keys())
  columns: list[str] = []
  for preferred in DEFAULT_COLUMNS:
    matched = _matched_column_name(rows[0], preferred)
    if matched is None or preferred in columns:
      continue
    values = _series(rows, preferred)
    if preferred in {"reward_total", "positive_reward", "penalties"}:
      columns.append(preferred)
      continue
    if _has_nonzero_data(values):
      columns.append(preferred)

  for column in header:
    if not _is_default_plot_column(column) or column in columns:
      continue
    canonical = DUPLICATE_WHEN_CANONICAL_EXISTS.get(column)
    if canonical is not None:
      canonical_match = _matched_column_name(rows[0], canonical)
      if canonical_match is not None and canonical in columns:
        continue
    values = _series(rows, column)
    if _has_nonzero_data(values):
      columns.append(column)

  return columns


def _pick_rows_for_plot(
    rows: list[dict[str, str]],
    source: str,
    columns_arg: str,
) -> tuple[list[dict[str, str]], str]:
  filtered_rows = _filtrar_origen(rows, source)
  columns = _resolve_columns(filtered_rows, columns_arg)
  if any(_has_data(_series(filtered_rows, column)) for column in columns):
    return filtered_rows, source

  if source == "all":
    return filtered_rows, source

  fallback_source = "eval" if source == "train" else "all"
  fallback_rows = _filtrar_origen(rows, fallback_source)
  fallback_columns = _resolve_columns(fallback_rows, columns_arg)
  if any(_has_data(_series(fallback_rows, column)) for column in fallback_columns):
    print(
        f"No hay columnas con datos para source={source!r}; uso source={fallback_source!r}.",
        file=sys.stderr,
    )
    return fallback_rows, fallback_source

  return filtered_rows, source


def _format_wall_time(timestamp: float) -> str:
  return dt.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def _run_start_from_name(run_dir: Path) -> float | None:
  match = re.search(r"(\d{8})-(\d{6})", run_dir.name)
  if not match:
    return None
  try:
    parsed = dt.datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
  except ValueError:
    return None
  return parsed.timestamp()


def _time_window_text(csv_path: Path, run_dir: Path, rows: list[dict[str, str]]) -> str:
  generated_ts = dt.datetime.now().timestamp()
  csv_mtime = csv_path.stat().st_mtime
  last_row = rows[-1]
  elapsed = _to_float(last_row.get("elapsed_seconds", ""))
  start_ts = None
  if elapsed is not None:
    start_ts = csv_mtime - elapsed
  if start_ts is None:
    start_ts = _run_start_from_name(run_dir)
  if start_ts is None:
    start_ts = run_dir.stat().st_mtime

  active = generated_ts - csv_mtime <= 300.0
  end_label = "en curso" if active else "finalizada"
  parts = [
      f"inicio: {_format_wall_time(start_ts)}",
      f"estado: {end_label}",
      f"ultimo CSV: {_format_wall_time(csv_mtime)}",
      f"grafica: {_format_wall_time(generated_ts)}",
  ]
  num_steps = last_row.get("num_steps", "")
  debug_version = last_row.get("debug_version", "")
  if num_steps:
    parts.append(f"steps: {num_steps}")
  if debug_version:
    parts.append(f"version_recompensa: {debug_version}")
  return " | ".join(parts)


def _filtrar_origen(rows: list[dict[str, str]], source: str) -> list[dict[str, str]]:
  if source == "all" or "source" not in rows[0]:
    return rows
  filtered = [row for row in rows if row.get("source") == source]
  if filtered:
    return filtered
  print(
      f"No hay filas source={source!r}; uso todas las filas disponibles.",
      file=sys.stderr,
  )
  return rows


def _filtrar_ultimo_segmento(rows: list[dict[str, str]]) -> list[dict[str, str]]:
  """Devuelve las filas desde el ultimo reinicio de num_steps/tiempo."""

  if not rows:
    return rows
  start = 0
  prev_steps = _to_float(rows[0].get("num_steps", ""))
  prev_elapsed = _to_float(rows[0].get("elapsed_seconds", ""))
  for index, row in enumerate(rows[1:], start=1):
    steps = _to_float(row.get("num_steps", ""))
    elapsed = _to_float(row.get("elapsed_seconds", ""))
    steps_reset = (
        steps is not None and prev_steps is not None and steps < prev_steps
    )
    elapsed_reset = (
        elapsed is not None and prev_elapsed is not None and elapsed < prev_elapsed
    )
    if steps_reset or elapsed_reset:
      start = index
    if steps is not None:
      prev_steps = steps
    if elapsed is not None:
      prev_elapsed = elapsed
  if start > 0:
    print(
        f"CSV con varias sesiones detectadas; uso el ultimo segmento "
        f"({len(rows) - start} de {len(rows)} filas).",
        file=sys.stderr,
    )
  return rows[start:]


def _smooth(values: list[float | None], window: int) -> list[float | None]:
  if window <= 1:
    return values
  smoothed: list[float | None] = []
  for index in range(len(values)):
    chunk = [
        value
        for value in values[max(0, index - window + 1) : index + 1]
        if value is not None
    ]
    smoothed.append(sum(chunk) / len(chunk) if chunk else None)
  return smoothed


def _safe_stem(path: Path) -> str:
  chars = [
      char if char.isalnum() or char in ("-", "_") else "_"
      for char in path.stem
  ]
  return "".join(chars).strip("_") or "recompensas"


def _output_path_for_csv(
    csv_path: Path,
    run_dir: Path,
    output_arg: str | None,
    multiple_outputs: bool,
) -> Path:
  default_name = "recompensas.png"
  if multiple_outputs:
    default_name = f"{_safe_stem(csv_path)}.png"

  if not output_arg:
    return run_dir / default_name

  output = Path(output_arg)
  if output.suffix.lower() == ".png" and not multiple_outputs:
    return output
  if output.suffix.lower() == ".png":
    return output.with_name(f"{output.stem}_{_safe_stem(csv_path)}{output.suffix}")
  return output / default_name


def _plot_csv(
    csv_path: Path,
    run_dir: Path,
    output_path: Path,
    x_axis: str,
    columns_arg: str,
    source: str,
    segment: str,
    smooth_window: int,
    unidades_recompensa: str,
    show: bool,
) -> None:
  rows = _read_csv(csv_path)
  if not rows:
    raise RuntimeError(f"{csv_path} esta vacio.")
  if segment == "last":
    rows = _filtrar_ultimo_segmento(rows)
  rows, source = _pick_rows_for_plot(rows, source, columns_arg)

  x_values = _series(rows, x_axis)
  if not _has_data(x_values):
    raise RuntimeError(f"No hay datos validos para eje X: {x_axis}")
  x_clean = [0.0 if value is None else value for value in x_values]

  columns = _resolve_columns(rows, columns_arg)

  fig, ax = plt.subplots(figsize=(13, 7))
  plotted = 0
  sample_row = rows[0]
  for column in columns:
    values = _smooth(_series(rows, column, unidades_recompensa=unidades_recompensa), smooth_window)
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
    raise RuntimeError(f"No hay columnas con datos para graficar en {csv_path}.")

  time_text = _time_window_text(csv_path, run_dir, rows)
  ax.set_title(
      f"Evolucion de recompensas - {run_dir.name} / {csv_path.name} "
      f"(origen={source}, tramo={segment}, suavizado={smooth_window}, "
      f"unidades={unidades_recompensa})\n"
      f"{time_text}"
  )
  ax.set_xlabel(x_axis)
  unidades_y = (
      "recompensa PPO escalada"
      if unidades_recompensa == "scaled"
      else "recompensa raw"
  )
  ax.set_ylabel(f"reward ponderada ({unidades_y}); penalties dibujadas en negativo")
  ax.grid(True, alpha=0.3)
  ax.legend(loc="best", fontsize="small", ncols=3)
  fig.text(
      0.01,
      0.01,
      f"CSV: {csv_path.resolve()}",
      ha="left",
      va="bottom",
      fontsize="x-small",
  )
  fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))

  output_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output_path, dpi=160)
  print(f"CSV usado: {csv_path}")
  print(f"Intervalo: {time_text}")
  print(f"Grafica guardada en: {output_path}")

  if show:
    plt.show()
  plt.close(fig)


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Grafica la evolucion temporal de los componentes de recompensa."
  )
  parser.add_argument("--logs_dir", default="logs_tarantulin_mjx")
  parser.add_argument("--run_dir", default=None)
  parser.add_argument("--csv_path", default=None)
  parser.add_argument(
      "--csv_mode",
      choices=["all", "active"],
      default="all",
      help=(
          "all genera una grafica por cada recompensas*.csv de la run; "
          "active usa solo recompensas.csv."
      ),
  )
  parser.add_argument("--output", default=None)
  parser.add_argument(
      "--x",
      choices=["num_steps", "elapsed_seconds", "elapsed_hours", "percent"],
      default="num_steps",
  )
  parser.add_argument(
      "--columns",
      default=DEFAULT_COLUMNS_ARG,
      help=(
          "Columnas separadas por coma. Usa 'default' para reward_total, "
          "recompensas y penalties ponderadas presentes; 'all' para todos los componentes finales disponibles."
      ),
  )
  parser.add_argument(
      "--source",
      choices=["train", "eval", "all"],
      default="all",
      help="Origen de metricas a graficar. all usa todo lo disponible y evita quedarse esperando si la ultima fila no es eval.",
  )
  parser.add_argument(
      "--segment",
      choices=["last", "all"],
      default="last",
      help="Usa solo la ultima sesion dentro del CSV o todas las filas.",
  )
  parser.add_argument(
      "--smooth",
      type=int,
      default=5,
      help="Media movil causal en numero de puntos. Usa 1 para desactivar.",
  )
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
  parser.add_argument("--show", action="store_true")
  args = parser.parse_args()

  run_dir = Path(args.run_dir) if args.run_dir else _latest_run(Path(args.logs_dir))
  if args.csv_path:
    csv_paths = [Path(args.csv_path)]
  elif args.csv_mode == "active":
    csv_paths = [run_dir / "recompensas.csv"]
  else:
    csv_paths = _csvs_recompensa_de_run(run_dir)

  if not csv_paths:
    raise FileNotFoundError(
        f"No hay recompensas*.csv en {run_dir}. Lanza una run nueva o usa --csv_path."
    )

  multiple_outputs = len(csv_paths) > 1
  for csv_path in csv_paths:
    if not csv_path.exists():
      raise FileNotFoundError(f"No existe {csv_path}.")
    output_path = _output_path_for_csv(
        csv_path=csv_path,
        run_dir=run_dir,
        output_arg=args.output,
        multiple_outputs=multiple_outputs,
    )
    _plot_csv(
        csv_path=csv_path,
        run_dir=run_dir,
        output_path=output_path,
        x_axis=args.x,
        columns_arg=args.columns,
        source=args.source,
        segment=args.segment,
        smooth_window=args.smooth,
        unidades_recompensa=args.unidades_recompensa,
        show=args.show,
    )


if __name__ == "__main__":
  main()
