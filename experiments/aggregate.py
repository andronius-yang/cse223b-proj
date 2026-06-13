#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


EPS = 1e-9


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"[skip] {path}: {e}")
        return None


def norm_subjects(x: Any) -> str:
    if isinstance(x, list):
        return ",".join(sorted(map(str, x)))
    if isinstance(x, str):
        # handle either "a,b,c" or "['a', 'b']" style strings conservatively
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                v = json.loads(s.replace("'", '"'))
                if isinstance(v, list):
                    return ",".join(sorted(map(str, v)))
            except Exception:
                pass
        return ",".join(sorted([p.strip() for p in s.split(",") if p.strip()]))
    return ""


def to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def first_nonzero(*vals: float) -> float:
    for v in vals:
        if pd.notna(v) and float(v) > 0:
            return float(v)
    return 0.0


def safe_div(a: float, b: float) -> float:
    if b is None or pd.isna(b) or abs(float(b)) < EPS:
        return np.nan
    return float(a) / float(b)


def collect_summaries(run_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for root in run_dirs:
        for p in sorted(root.rglob("summary.json")):
            row = load_json(p)
            if row is None:
                continue
            row["_summary_path"] = str(p)
            row["_point_dir"] = str(p.parent)
            row["_run_root"] = root.name
            rows.append(row)

    if not rows:
        raise SystemExit("No summary.json files found.")

    df = pd.DataFrame(rows)

    # Normalize common schema.
    if "placement" not in df.columns and "replication_strategy" in df.columns:
        df["placement"] = df["replication_strategy"]

    if "scenario" not in df.columns and "scenario_family" in df.columns:
        df["scenario"] = df["scenario_family"]

    if "decode_steps_effective" not in df.columns:
        df["decode_steps_effective"] = df.get("decode_steps_requested", np.nan)

    if "request_rerun_us" not in df.columns:
        df["request_rerun_us"] = df.get("request_rerun_penalty_us", 0)

    if "stall_step_us" not in df.columns:
        df["stall_step_us"] = df.get("kv_penalty_per_stalled_request_us", 0)

    df["subjects_key"] = df.get("subjects", "").apply(norm_subjects)
    df["workload_key"] = (
        df.get("benchmark", "").astype(str)
        + "::"
        + df["subjects_key"].astype(str)
    )

    # Boolean serviceability.
    if "serviceable" not in df.columns:
        df["serviceable"] = True
    df["serviceable"] = df["serviceable"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )

    if "terminal_failure_reason" not in df.columns:
        df["terminal_failure_reason"] = ""

    numeric = [
        "capacity",
        "decode_steps_requested",
        "decode_steps_effective",
        "ranks_per_node",
        "scaleup_gbps",
        "scaleout_gbps",
        "storage_read_gbps",
        "all2allv_us",
        "migration_network_us",
        "cold_start_storage_us",
        "initial_replication_us",
        "all2allv_bytes",
        "network_repair_bytes",
        "cold_start_bytes",
        "lost_expert_bytes",
        "lost_expert_count",
        "paused_stream_step_area",
        "max_paused_request_streams",
        "mean_paused_request_streams",
        "stalled_request_pct_max",
        "stalled_request_pct_mean",
        "data_lake_reload_us",
        "request_rerun_us",
        "stall_step_us",
        "T_healthy",
        "ft_tax",
    ]
    df = to_num(df, numeric)

    # Fill missing values.
    for c in [
        "all2allv_us",
        "migration_network_us",
        "cold_start_storage_us",
        "network_repair_bytes",
        "cold_start_bytes",
        "lost_expert_bytes",
        "lost_expert_count",
        "paused_stream_step_area",
        "max_paused_request_streams",
        "request_rerun_us",
        "stall_step_us",
    ]:
        df[c] = df[c].fillna(0)

    # Story-first derived metrics.
    recovery_bytes = []
    for _, r in df.iterrows():
        recovery_bytes.append(
            first_nonzero(
                r.get("cold_start_bytes", 0),
                r.get("lost_expert_bytes", 0),
                r.get("network_repair_bytes", 0),
            )
        )
    df["recovery_bytes"] = recovery_bytes

    df["network_repair_us"] = (
        df["migration_network_us"] + df["cold_start_storage_us"]
    )

    df["request_stall_penalty_us"] = (
        df["paused_stream_step_area"] * df["stall_step_us"]
    )

    df["request_rerun_penalty_us"] = (
        df["max_paused_request_streams"] * df["request_rerun_us"]
    )

    # Recompute data-lake reload if missing or zero.
    computed_lake = df.apply(
        lambda r: safe_div(
            r["recovery_bytes"],
            r["storage_read_gbps"] * 125.0,
        ),
        axis=1,
    )
    df["data_lake_reload_us"] = df["data_lake_reload_us"].fillna(0)
    df.loc[df["data_lake_reload_us"] <= 0, "data_lake_reload_us"] = computed_lake

    df["T_network_repair_path"] = (
        df["all2allv_us"]
        + df["network_repair_us"]
        + df["request_stall_penalty_us"]
    )

    df["T_data_lake_path"] = (
        df["all2allv_us"]
        + df["data_lake_reload_us"]
        + df["request_stall_penalty_us"]
    )

    df["T_request_rerun_path"] = (
        df["all2allv_us"]
        + df["request_rerun_penalty_us"]
        + df["request_stall_penalty_us"]
    )

    df["benefit_network_vs_lake"] = df.apply(
        lambda r: safe_div(r["T_data_lake_path"], r["T_network_repair_path"]),
        axis=1,
    )
    df["benefit_network_vs_rerun"] = df.apply(
        lambda r: safe_div(r["T_request_rerun_path"], r["T_network_repair_path"]),
        axis=1,
    )
    df["repair_source_speedup_vs_lake"] = df.apply(
        lambda r: safe_div(r["data_lake_reload_us"], r["network_repair_us"]),
        axis=1,
    )
    df["break_even_storage_gbps"] = (
        df["storage_read_gbps"] * df["repair_source_speedup_vs_lake"]
    )
    df["replica_repair_fraction"] = df.apply(
        lambda r: safe_div(r["network_repair_us"], r["T_network_repair_path"]),
        axis=1,
    )
    df["lake_repair_fraction"] = df.apply(
        lambda r: safe_div(r["data_lake_reload_us"], r["T_data_lake_path"]),
        axis=1,
    )

    df["network_repair_wins"] = (
        df["T_network_repair_path"]
        < df[["T_data_lake_path", "T_request_rerun_path"]].min(axis=1)
    )

    # Attach healthy baseline from no_events rows when available.
    healthy = df[df["scenario"].astype(str).eq("no_events")].copy()
    if not healthy.empty:
        key = [
            "workload_key",
            "placement",
            "capacity",
            "decode_steps_effective",
            "scaleout_gbps",
            "ranks_per_node",
        ]
        h = (
            healthy.groupby(key, dropna=False)["all2allv_us"]
            .mean()
            .reset_index()
            .rename(columns={"all2allv_us": "T_healthy_from_no_events"})
        )
        df = df.merge(h, on=key, how="left")
        df["T_healthy"] = df["T_healthy"].fillna(df["T_healthy_from_no_events"])

    df["ft_tax"] = df.apply(
        lambda r: safe_div(r["T_network_repair_path"], r["T_healthy"]),
        axis=1,
    ).combine_first(df["ft_tax"])

    return df


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    # Keep one row per semantic point. Prefer serviceable rows, then newer/smaller point dirs.
    keys = [
        "workload_key",
        "scenario",
        "placement",
        "capacity",
        "decode_steps_effective",
        "scaleout_gbps",
        "storage_read_gbps",
        "request_rerun_us",
        "stall_step_us",
        "ranks_per_node",
    ]
    for k in keys:
        if k not in df.columns:
            df[k] = ""

    df = df.sort_values(
        ["serviceable", "_run_root", "_point_dir"],
        ascending=[False, True, True],
    )
    return df.drop_duplicates(keys, keep="first")


def matched_pairs(df: pd.DataFrame) -> pd.DataFrame:
    # Only retain cells with both lazarus and uniform so placement comparisons are fair.
    axes = [
        "workload_key",
        "scenario",
        "capacity",
        "decode_steps_effective",
        "scaleout_gbps",
        "storage_read_gbps",
        "request_rerun_us",
        "stall_step_us",
        "ranks_per_node",
    ]

    kept = []
    for _, g in df.groupby(axes, dropna=False):
        placements = set(g["placement"].astype(str))
        if {"lazarus", "uniform"}.issubset(placements):
            kept.append(g[g["placement"].isin(["lazarus", "uniform"])])
    if not kept:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(kept, ignore_index=True)


def agg(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    metrics = {
        "serviceable": "mean",
        "benefit_network_vs_lake": "mean",
        "benefit_network_vs_rerun": "mean",
        "repair_source_speedup_vs_lake": "mean",
        "break_even_storage_gbps": "mean",
        "ft_tax": "mean",
        "stalled_request_pct_max": "mean",
        "T_network_repair_path": "mean",
        "T_data_lake_path": "mean",
        "T_request_rerun_path": "mean",
        "network_repair_us": "mean",
        "data_lake_reload_us": "mean",
        "recovery_bytes": "mean",
        "network_repair_wins": "mean",
    }

    out = (
        df.groupby(keys, dropna=False)
        .agg(n=("scenario", "size"), **{f"mean_{k}": (k, v) for k, v in metrics.items()})
        .reset_index()
    )
    out = out.rename(
        columns={
            "mean_serviceable": "serviceable_rate",
            "mean_network_repair_wins": "network_repair_win_rate",
        }
    )
    return out


def write_outputs(df: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    all_csv = out / "all_results.csv"
    df.to_csv(all_csv, index=False)
    df.to_json(out / "all_results.jsonl", orient="records", lines=True)

    deduped = dedupe(df)
    deduped.to_csv(out / "deduped_results.csv", index=False)

    paired = matched_pairs(deduped)
    paired.to_csv(out / "paired_results.csv", index=False)

    # Story tables.
    tables = {
        "story_by_scenario.csv": agg(deduped, ["scenario", "placement"]),
        "story_by_capacity.csv": agg(deduped, ["scenario", "placement", "capacity"]),
        "story_by_storage.csv": agg(deduped, ["scenario", "placement", "storage_read_gbps"]),
        "story_by_scaleout.csv": agg(deduped, ["scenario", "placement", "scaleout_gbps"]),
        "story_by_decode.csv": agg(deduped, ["scenario", "placement", "decode_steps_effective"]),
        "story_by_workload.csv": agg(deduped, ["workload_key", "scenario", "placement"]),
    }

    if not paired.empty:
        tables["paired_by_capacity.csv"] = agg(
            paired, ["scenario", "placement", "capacity"]
        )
        tables["paired_by_storage.csv"] = agg(
            paired, ["scenario", "placement", "storage_read_gbps"]
        )

    for name, table in tables.items():
        table.to_csv(out / name, index=False)

    coverage = (
        deduped.groupby(["scenario", "placement", "capacity"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    coverage.to_csv(out / "coverage.csv", index=False)

    summary = {
        "raw_rows": int(len(df)),
        "deduped_rows": int(len(deduped)),
        "paired_rows": int(len(paired)),
        "serviceable_rate": float(deduped["serviceable"].mean()) if len(deduped) else None,
        "scenarios": sorted(map(str, deduped["scenario"].dropna().unique())),
        "placements": sorted(map(str, deduped["placement"].dropna().unique())),
        "capacities": sorted(map(float, deduped["capacity"].dropna().unique())),
        "decode_steps_effective": sorted(map(float, deduped["decode_steps_effective"].dropna().unique())),
        "outputs": sorted(p.name for p in out.glob("*.csv")),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def make_plots(out: Path) -> None:
    plot_dir = out / "plots"
    plot_dir.mkdir(exist_ok=True)

    deduped = pd.read_csv(out / "deduped_results.csv")
    paired_path = out / "paired_results.csv"
    paired = pd.read_csv(paired_path) if paired_path.exists() else pd.DataFrame()

    def savefig(name: str):
        plt.tight_layout()
        plt.savefig(plot_dir / f"{name}.png", dpi=240)
        plt.savefig(plot_dir / f"{name}.pdf")
        plt.close()

    # 1. Benefit over data-lake by scenario.
    t = pd.read_csv(out / "story_by_scenario.csv")
    plt.figure(figsize=(7.2, 4.0))
    for placement, g in t.groupby("placement"):
        plt.plot(
            g["scenario"],
            g["mean_benefit_network_vs_lake"],
            marker="o",
            label=placement,
        )
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.ylabel("Data-lake path / network-repair path")
    plt.xlabel("Failure scenario")
    plt.title("Replica repair benefit over data-lake recovery")
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    savefig("benefit_vs_lake_by_scenario")

    # 2. Break-even storage by capacity.
    t = pd.read_csv(out / "story_by_capacity.csv")
    plt.figure(figsize=(7.2, 4.0))
    for (scenario, placement), g in t.groupby(["scenario", "placement"]):
        if "node1" not in str(scenario) and "rank4" not in str(scenario):
            continue
        plt.plot(
            g["capacity"],
            g["mean_break_even_storage_gbps"] / 1000.0,
            marker="o",
            label=f"{scenario}/{placement}",
        )
    plt.ylabel("Break-even storage bandwidth (Tbps)")
    plt.xlabel("Replica capacity per rank/layer")
    plt.title("Storage bandwidth needed to match in-cluster repair")
    plt.legend(fontsize=8)
    savefig("break_even_storage_by_capacity")

    # 3. Absolute path cost by scenario, log scale.
    t = pd.read_csv(out / "story_by_scenario.csv")
    x = np.arange(len(t))
    width = 0.35
    plt.figure(figsize=(8.4, 4.2))
    labels = [f"{r.scenario}\n{r.placement}" for r in t.itertuples()]
    plt.bar(x - width / 2, t["mean_T_network_repair_path"] / 1000.0, width, label="Network repair")
    plt.bar(x + width / 2, t["mean_T_data_lake_path"] / 1000.0, width, label="Data-lake reload")
    plt.yscale("log")
    plt.xticks(x, labels, rotation=25, ha="right", fontsize=8)
    plt.ylabel("Mean path time (ms, log scale)")
    plt.title("Failure recovery path cost")
    plt.legend()
    savefig("absolute_recovery_path_cost")

    # 4. Placement comparison only on matched pairs.
    if not paired.empty:
        t = agg(paired, ["scenario", "placement", "capacity"])
        plt.figure(figsize=(7.2, 4.0))
        for (scenario, placement), g in t.groupby(["scenario", "placement"]):
            plt.plot(
                g["capacity"],
                g["mean_benefit_network_vs_lake"],
                marker="o",
                label=f"{scenario}/{placement}",
            )
        plt.axhline(1.0, linestyle="--", linewidth=1)
        plt.xlabel("Replica capacity per rank/layer")
        plt.ylabel("Benefit vs data-lake reload")
        plt.title("Matched Lazarus/uniform comparison")
        plt.legend(fontsize=8)
        savefig("matched_placement_benefit_by_capacity")


def maybe_delete_raw_txt(run_dirs: list[Path]) -> None:
    removed = 0
    for root in run_dirs:
        for p in root.rglob("*.txt"):
            try:
                p.unlink()
                removed += 1
            except Exception:
                pass
    print(f"deleted {removed} .txt files")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--delete-raw-txt", action="store_true")
    args = ap.parse_args()

    run_dirs = [Path(p) for p in args.run_dir]
    df = collect_summaries(run_dirs)
    write_outputs(df, Path(args.out))

    if args.plots:
        make_plots(Path(args.out))

    if args.delete_raw_txt:
        maybe_delete_raw_txt(run_dirs)

    print(f"wrote analysis to {args.out}")


if __name__ == "__main__":
    main()