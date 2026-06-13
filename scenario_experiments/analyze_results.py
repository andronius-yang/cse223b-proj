#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_ROOT / "results"
MANIFEST_PATH = RESULTS_DIR / "experiment_manifest.jsonl"
STATS_CSV = RESULTS_DIR / "stats.csv"
FIGURE_DIR = RESULTS_DIR / "figures"
FULL_NODE_FIGURE = FIGURE_DIR / "full_node_recovery_2x2.png"

STATS_FIELDS = (
    "experiment_id",
    "scenario_id",
    "scenario_label",
    "strategy_id",
    "strategy_label",
    "strategy_group",
    "replication_strategy",
    "capacity_per_rank_per_layer",
    "average_replicas_per_expert",
    "initial_replication_us",
    "migration_network_us",
    "cold_start_storage_us",
    "all2allv_us",
    "total_steady_state_us",
    "total_including_initialization_us",
    "initial_replication_bytes",
    "network_repair_bytes",
    "all2allv_bytes",
    "cold_start_bytes",
    "speedup_vs_control_steady",
    "speedup_vs_control_total_including_init",
    "steady_state_delta_vs_control_us",
    "cold_start_bytes_avoided_vs_control",
    "ft_tax_steady_ratio",
    "ft_tax_total_including_init_ratio",
    "steps",
    "warnings",
)

FULL_NODE_SCENARIO_IDS = (
    "armageddon_one_node_low_holdout",
    "armageddon_one_node_high_holdout",
    "armageddon_rotating_single_node",
)
PLOT_STRATEGY_IDS = (
    "control_single_owner",
    "uniform_fixed_2",
    "adaptive_mro_avg3",
    "adaptive_mro_avg4",
    "adaptive_mro_avg6",
    "adaptive_mro_avg8",
)
STRATEGY_SHORT_LABELS = {
    "control_single_owner": "1x",
    "uniform_fixed_2": "U2",
    "adaptive_mro_avg3": "A3",
    "adaptive_mro_avg4": "A4",
    "adaptive_mro_avg6": "A6",
    "adaptive_mro_avg8": "A8",
}
FULL_NODE_LABELS = {
    "armageddon_one_node_low_holdout": "Full node low holdout",
    "armageddon_one_node_high_holdout": "Full node high holdout",
    "armageddon_rotating_single_node": "Full node rotating",
}
SCENARIO_SHORT_LABELS = {
    "armageddon_one_node_low_holdout": "1-node low",
    "armageddon_one_node_high_holdout": "1-node high",
    "armageddon_rotating_single_node": "Rotating",
}
PAPER_SQUARE_FIGSIZE = (3.35, 3.35)
PAPER_DPI = 300
FIGURE_SPECS = (
    ("Total time", "total_steady_state_us", 1000.0),
    ("Cold start", "cold_start_storage_us", 1000.0),
    ("MoE AllToAllV", "all2allv_us", 1000.0),
    ("Expert migration", "migration_network_us", 1000.0),
)


def fail(message: str) -> None:
    raise SystemExit(f"analysis error: {message}")


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        fail(f"malformed JSON in {path}: {exc}")
    except OSError as exc:
        fail(f"could not read {path}: {exc}")


def load_manifest(path: Path = MANIFEST_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        fail(f"missing manifest: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"{path}:{line_number}: invalid JSON: {exc}")
            if not isinstance(row, dict):
                fail(f"{path}:{line_number}: row must be a JSON object")
            rows.append(row)
    if not rows:
        fail(f"{path} contains no experiment rows")
    return rows


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ffloat(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def build_records(manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in manifest_rows:
        json_path = project_path(str(row["topsim_json"]))
        payload = load_json(json_path)
        totals = payload.get("totals")
        if not isinstance(totals, dict):
            fail(f"{json_path}: missing totals object")
        warnings = payload.get("warnings", [])
        if not isinstance(warnings, list):
            warnings = [str(warnings)]
        record = dict(row)
        if str(record.get("scenario_id")) in FULL_NODE_LABELS:
            record["scenario_label"] = FULL_NODE_LABELS[str(record["scenario_id"])]
        record.update(
            {
                "initial_replication_us": ffloat(totals.get("initial_replication_us")),
                "migration_network_us": ffloat(totals.get("migration_network_us")),
                "cold_start_storage_us": ffloat(totals.get("cold_start_storage_us")),
                "all2allv_us": ffloat(totals.get("all2allv_us")),
                "total_steady_state_us": ffloat(totals.get("total_steady_state_us")),
                "total_including_initialization_us": ffloat(
                    totals.get("total_including_initialization_us")
                ),
                "initial_replication_bytes": ffloat(totals.get("initial_replication_bytes")),
                "network_repair_bytes": ffloat(totals.get("network_repair_bytes")),
                "all2allv_bytes": ffloat(totals.get("all2allv_bytes")),
                "cold_start_bytes": ffloat(totals.get("cold_start_bytes")),
                "steps": int(totals.get("steps", 0)),
                "warnings": "; ".join(str(warning) for warning in warnings),
            }
        )
        records.append(record)
    add_comparisons(records)
    return records


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return math.nan
    return numerator / denominator


def add_comparisons(records: list[dict[str, Any]]) -> None:
    by_scenario_strategy = {
        (record["scenario_id"], record["strategy_id"]): record for record in records
    }
    for record in records:
        control = by_scenario_strategy.get((record["scenario_id"], "control_single_owner"))
        healthy = by_scenario_strategy.get(("no_fail", record["strategy_id"]))
        if control is None:
            fail(f"missing single-owner control for scenario {record['scenario_id']}")
        if healthy is None:
            fail(f"missing no_fail baseline for strategy {record['strategy_id']}")

        record["speedup_vs_control_steady"] = safe_ratio(
            control["total_steady_state_us"], record["total_steady_state_us"]
        )
        record["speedup_vs_control_total_including_init"] = safe_ratio(
            control["total_including_initialization_us"],
            record["total_including_initialization_us"],
        )
        record["steady_state_delta_vs_control_us"] = (
            record["total_steady_state_us"] - control["total_steady_state_us"]
        )
        record["cold_start_bytes_avoided_vs_control"] = (
            control["cold_start_bytes"] - record["cold_start_bytes"]
        )
        record["ft_tax_steady_ratio"] = safe_ratio(
            record["total_steady_state_us"], healthy["total_steady_state_us"]
        )
        record["ft_tax_total_including_init_ratio"] = safe_ratio(
            record["total_including_initialization_us"],
            healthy["total_including_initialization_us"],
        )


def csv_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def write_stats_csv(records: list[dict[str, Any]]) -> None:
    STATS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with STATS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({field: csv_value(record.get(field, "")) for field in STATS_FIELDS})


def find_record(
    records: list[dict[str, Any]],
    scenario_id: str,
    strategy_id: str,
) -> dict[str, Any] | None:
    for record in records:
        if record["scenario_id"] == scenario_id and record["strategy_id"] == strategy_id:
            return record
    return None


def cividis_colors(count: int) -> list[Any]:
    cmap = plt.get_cmap("cividis")
    if count <= 1:
        return [cmap(0.55)]
    return [cmap(0.12 + 0.76 * index / (count - 1)) for index in range(count)]


def compact_tick(value: float, _position: int) -> str:
    value = float(value)
    if abs(value) >= 1000.0:
        return f"{value / 1000.0:.0f}k"
    if value == int(value):
        return str(int(value))
    return f"{value:.1f}"


def draw_metric_panel(
    ax: Any,
    records: list[dict[str, Any]],
    *,
    scenario_ids: tuple[str, ...],
    value_key: str,
    scale: float,
    title: str,
) -> None:
    x_positions = list(range(len(PLOT_STRATEGY_IDS)))
    x_labels = [STRATEGY_SHORT_LABELS[strategy] for strategy in PLOT_STRATEGY_IDS]
    colors = cividis_colors(len(scenario_ids))
    markers = ("o", "s", "^", "D", "v", "P")

    for index, scenario_id in enumerate(scenario_ids):
        values: list[float] = []
        for strategy_id in PLOT_STRATEGY_IDS:
            record = find_record(records, scenario_id, strategy_id)
            values.append(math.nan if record is None else record[value_key] / scale)
        ax.plot(
            x_positions,
            values,
            color=colors[index],
            marker=markers[index % len(markers)],
            linewidth=0.95,
            markersize=2.5,
            label=SCENARIO_SHORT_LABELS.get(scenario_id, scenario_id),
        )

    ax.set_title(title, fontsize=6.6, pad=2)
    ax.set_ylabel("ms", fontsize=5.9, labelpad=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=5.8)
    ax.tick_params(axis="y", labelsize=5.8, pad=1)
    ax.tick_params(axis="x", labelsize=5.8, pad=1)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FuncFormatter(compact_tick))
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.35, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def write_full_node_figure(records: list[dict[str, Any]]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for figure_path in FIGURE_DIR.glob("*.png"):
        figure_path.unlink()

    fig, axes = plt.subplots(2, 2, figsize=PAPER_SQUARE_FIGSIZE)
    for ax, (title, value_key, scale) in zip(axes.flat, FIGURE_SPECS):
        draw_metric_panel(
            ax,
            records,
            scenario_ids=FULL_NODE_SCENARIO_IDS,
            value_key=value_key,
            scale=scale,
            title=title,
        )

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        fontsize=5.8,
        frameon=False,
        handlelength=1.35,
        columnspacing=0.75,
    )
    fig.tight_layout(pad=0.18, h_pad=0.55, w_pad=0.55, rect=(0.0, 0.105, 1.0, 1.0))
    fig.savefig(FULL_NODE_FIGURE, dpi=PAPER_DPI)
    plt.close(fig)


def analyze(*, quiet: bool) -> None:
    records = build_records(load_manifest())
    write_stats_csv(records)
    write_full_node_figure(records)
    if not quiet:
        print(f"wrote {STATS_CSV.relative_to(PROJECT_ROOT)}")
        print(f"wrote {FIGURE_DIR.relative_to(PROJECT_ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate scenario experiment stats and figure from TopSim JSON outputs."
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output paths.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze(quiet=args.quiet)


if __name__ == "__main__":
    main()
