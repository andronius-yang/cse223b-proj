#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAFFIC_GEN = REPO_ROOT / "traffic-gen" / "generate_scenario.py"
DATASET_REPO = "core12345/MoE_expert_selection_trace"
DEFAULT_MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
NUM_RANKS = 16
REQUESTS_PER_RANK = 16
BATCH_SIZE = NUM_RANKS * REQUESTS_PER_RANK
EXPERT_SIZE_BYTES = 251_658_240
GBPS_TO_BYTES_PER_US = 125.0
EPSILON = 1e-9

SUMMARY_FIELDS = [
    "benchmark",
    "subjects",
    "scenario",
    "placement",
    "capacity",
    "decode_steps_requested",
    "decode_steps_effective",
    "ranks_per_node",
    "scaleup_gbps",
    "scaleout_gbps",
    "storage_read_gbps",
    "serviceable",
    "terminal_failure_reason",
    "all2allv_us",
    "migration_network_us",
    "cold_start_storage_us",
    "initial_replication_us",
    "all2allv_bytes",
    "network_repair_bytes",
    "cold_start_bytes",
    "lost_expert_bytes",
    "lost_expert_count",
    "max_paused_request_streams",
    "mean_paused_request_streams",
    "paused_stream_step_area",
    "stalled_request_pct_max",
    "stalled_request_pct_mean",
    "data_lake_reload_us",
    "stalled_request_penalty_us",
    "T_healthy",
    "T_lake",
    "T_replica_repair",
    "ft_tax",
    "repair_source_speedup",
    "system_benefit_vs_lake",
    "repair_fraction_of_total",
]


def fail(message: str, code: int = 1) -> None:
    raise SystemExit(f"eval-cli error: {message}") from None


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_int_csv(value: str) -> list[int]:
    try:
        return [int(part) for part in parse_csv(value)]
    except ValueError as exc:
        fail(f"expected comma-separated integers, got {value!r}: {exc}")


def parse_float_csv(value: str) -> list[float]:
    try:
        return [float(part) for part in parse_csv(value)]
    except ValueError as exc:
        fail(f"expected comma-separated numbers, got {value!r}: {exc}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                fail(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) <= EPSILON:
        return 0.0
    return numerator / denominator


def placement_to_strategy(placement: str) -> str:
    if placement == "lazarus":
        return "adaptive"
    if placement == "uniform":
        return "uniform"
    fail("--placement must be lazarus or uniform")


def node1_events(ranks_per_node: int) -> list[dict[str, Any]]:
    start = ranks_per_node
    ranks = list(range(start, start + ranks_per_node))
    return [
        {"step": 100, "type": "fail", "ranks": ranks},
        {"step": 200, "type": "join", "ranks": ranks},
    ]


def scenario_events(scenario: str, ranks_per_node: int) -> list[dict[str, Any]]:
    if scenario == "no_events":
        return []
    if scenario == "node1_fail_join":
        return node1_events(ranks_per_node)
    fail("supported scenarios are no_events and node1_fail_join")


def trace_matches(path: Path, benchmark: str, subject: str) -> bool:
    text = path.as_posix().lower()
    return path.suffix == ".json" and benchmark.lower() in text and subject.lower() in text


def discover_trace_files(trace_root: Path, benchmark: str, subjects: list[str]) -> list[Path]:
    if not trace_root.exists():
        fail(f"trace root does not exist: {trace_root}")
    selected: list[Path] = []
    for subject in subjects:
        subject_files: list[Path] = []
        for root, _, files in os.walk(trace_root):
            root_path = Path(root)
            for name in files:
                path = root_path / name
                if trace_matches(path, benchmark, subject):
                    subject_files.append(path)
        if not subject_files:
            fail(f"found no JSON traces for benchmark={benchmark!r} subject={subject!r} under {trace_root}")
        selected.extend(subject_files)
    if len(selected) < BATCH_SIZE:
        fail(f"found {len(selected)} matching trace files, need at least {BATCH_SIZE}")
    return selected[:BATCH_SIZE]


def effective_decode_steps(trace_paths: list[Path], requested: int) -> int:
    effective = requested
    for path in trace_paths:
        try:
            payload = read_json(path)
        except json.JSONDecodeError as exc:
            fail(f"malformed JSON in selected trace {path}: {exc}")
        if not isinstance(payload, list):
            fail(f"selected trace {path} must contain a JSON list")
        effective = min(effective, max(0, len(payload) - 1))
    return effective


def prepare_workload(run_dir: Path, benchmark: str, subjects: list[str], trace_paths: list[Path]) -> Path:
    workload = run_dir / "workload"
    workload.mkdir(parents=True, exist_ok=True)
    subject_cycle = itertools.cycle(subjects)
    for index, source in enumerate(trace_paths):
        subject = next(subject_cycle)
        for candidate in subjects:
            if candidate.lower() in source.as_posix().lower():
                subject = candidate
                break
        subject_dir = workload / f"llama4-{benchmark}-{subject}"
        subject_dir.mkdir(parents=True, exist_ok=True)
        target = subject_dir / f"{index:06d}_{source.name}"
        try:
            target.symlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, target)
    return workload


def run_command(command: list[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(command, cwd=cwd, stdout=out, stderr=err, text=True)
    return proc.returncode


def scenario_config(
    *,
    run_dir: Path,
    scenario: str,
    placement: str,
    capacity: int,
    decode_steps: int,
    ranks_per_node: int,
    workload: Path,
    output_dir: Path,
) -> Path:
    payload = {
        "scenario_id": scenario,
        "ranks_per_node": ranks_per_node,
        "capacity_per_rank_per_layer": capacity,
        "replication_strategy": placement_to_strategy(placement),
        "decode_steps": decode_steps,
        "input_glob": str(workload / "llama4-*"),
        "output_dir": str(output_dir),
        "events": scenario_events(scenario, ranks_per_node),
    }
    path = run_dir / f"{scenario}_config.json"
    write_json(path, payload)
    return path


def run_toposim(
    *,
    run_dir: Path,
    timeline: Path,
    result: Path,
    ranks_per_node: int,
    scaleup_gbps: float,
    scaleout_gbps: float,
    storage_read_gbps: float,
    label: str,
) -> tuple[list[str], int]:
    command = [
        "uv",
        "run",
        "--project",
        "topsim",
        "toposim-timeline",
        str(timeline),
        "--policy",
        "direct",
        "--gpus-per-server",
        str(ranks_per_node),
        "--scaleup-gbps",
        str(scaleup_gbps),
        "--scaleout-gbps",
        str(scaleout_gbps),
        "--storage-read-gbps",
        str(storage_read_gbps),
        "--expert-size-bytes",
        str(EXPERT_SIZE_BYTES),
        "--json",
        str(result),
    ]
    code = run_command(
        command,
        cwd=REPO_ROOT,
        stdout_path=run_dir / "logs" / f"toposim_{label}.out",
        stderr_path=run_dir / "logs" / f"toposim_{label}.err",
    )
    return command, code


def run_one(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    subjects = parse_csv(args.subjects)
    selected_traces = discover_trace_files(Path(args.trace_root), args.benchmark, subjects)
    decode_effective = effective_decode_steps(selected_traces, args.decode_steps)
    workload = prepare_workload(out_dir, args.benchmark, subjects, selected_traces)

    commands: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "parameters": {
            "benchmark": args.benchmark,
            "subjects": subjects,
            "scenario": args.scenario,
            "placement": args.placement,
            "capacity": args.capacity,
            "decode_steps_requested": args.decode_steps,
            "decode_steps_effective": decode_effective,
            "ranks_per_node": args.ranks_per_node,
            "scaleup_gbps": args.scaleup_gbps,
            "scaleout_gbps": args.scaleout_gbps,
            "storage_read_gbps": args.storage_read_gbps,
            "data_lake_fixed_us": args.data_lake_fixed_us,
            "data_lake_per_expert_us": args.data_lake_per_expert_us,
            "kv_penalty_per_stalled_request_us": args.kv_penalty_per_stalled_request_us,
            "expert_size_bytes": EXPERT_SIZE_BYTES,
        },
        "selected_traces": [str(path) for path in selected_traces],
        "outputs": {
            "run_dir": str(out_dir),
            "scenario_timeline": str(out_dir / "scenario_timeline.jsonl"),
            "toposim_result": str(out_dir / "result.json"),
            "healthy_timeline": str(out_dir / "healthy_no_events" / "scenario_timeline.jsonl"),
            "healthy_result": str(out_dir / "healthy_result.json"),
        },
        "commands": commands,
    }

    main_cfg = scenario_config(
        run_dir=out_dir,
        scenario=args.scenario,
        placement=args.placement,
        capacity=args.capacity,
        decode_steps=args.decode_steps,
        ranks_per_node=args.ranks_per_node,
        workload=workload,
        output_dir=out_dir,
    )
    traffic_command = ["python3", str(TRAFFIC_GEN), str(main_cfg)]
    traffic_code = run_command(
        traffic_command,
        cwd=REPO_ROOT,
        stdout_path=out_dir / "logs" / "traffic_gen.out",
        stderr_path=out_dir / "logs" / "traffic_gen.err",
    )
    commands.append({"label": "traffic-gen", "argv": traffic_command, "returncode": traffic_code})
    if traffic_code != 0:
        write_json(out_dir / "run_manifest.json", manifest)
        fail(f"traffic-gen failed with exit code {traffic_code}; see {out_dir / 'logs'}")

    toposim_command, toposim_code = run_toposim(
        run_dir=out_dir,
        timeline=out_dir / "scenario_timeline.jsonl",
        result=out_dir / "result.json",
        ranks_per_node=args.ranks_per_node,
        scaleup_gbps=args.scaleup_gbps,
        scaleout_gbps=args.scaleout_gbps,
        storage_read_gbps=args.storage_read_gbps,
        label="main",
    )
    commands.append({"label": "toposim-timeline", "argv": toposim_command, "returncode": toposim_code})
    if toposim_code != 0:
        write_json(out_dir / "run_manifest.json", manifest)
        fail(f"toposim-timeline failed with exit code {toposim_code}; see {out_dir / 'logs'}")

    healthy_dir = out_dir if args.scenario == "no_events" else out_dir / "healthy_no_events"
    if args.scenario != "no_events":
        healthy_cfg = scenario_config(
            run_dir=out_dir,
            scenario="no_events",
            placement=args.placement,
            capacity=args.capacity,
            decode_steps=args.decode_steps,
            ranks_per_node=args.ranks_per_node,
            workload=workload,
            output_dir=healthy_dir,
        )
        healthy_command = ["python3", str(TRAFFIC_GEN), str(healthy_cfg)]
        healthy_code = run_command(
            healthy_command,
            cwd=REPO_ROOT,
            stdout_path=out_dir / "logs" / "traffic_gen_healthy.out",
            stderr_path=out_dir / "logs" / "traffic_gen_healthy.err",
        )
        commands.append({"label": "traffic-gen-healthy", "argv": healthy_command, "returncode": healthy_code})
        if healthy_code != 0:
            write_json(out_dir / "run_manifest.json", manifest)
            fail(f"healthy traffic-gen failed with exit code {healthy_code}; see {out_dir / 'logs'}")

    healthy_toposim_command, healthy_toposim_code = run_toposim(
        run_dir=out_dir,
        timeline=healthy_dir / "scenario_timeline.jsonl",
        result=out_dir / "healthy_result.json",
        ranks_per_node=args.ranks_per_node,
        scaleup_gbps=args.scaleup_gbps,
        scaleout_gbps=args.scaleout_gbps,
        storage_read_gbps=args.storage_read_gbps,
        label="healthy",
    )
    commands.append(
        {"label": "toposim-timeline-healthy", "argv": healthy_toposim_command, "returncode": healthy_toposim_code}
    )
    if healthy_toposim_code != 0:
        write_json(out_dir / "run_manifest.json", manifest)
        fail(f"healthy toposim-timeline failed with exit code {healthy_toposim_code}; see {out_dir / 'logs'}")

    write_json(out_dir / "run_manifest.json", manifest)
    row = summarize_run(out_dir)
    return row


def numeric_total(totals: dict[str, Any], key: str) -> float:
    value = totals.get(key, 0.0)
    return float(value or 0.0)


def timeline_metrics(rows: list[dict[str, Any]], batch_size: int, expert_size_bytes: float) -> dict[str, Any]:
    lost_expert_bytes = 0.0
    terminal_failure_reason = ""
    paused_by_step: dict[int, int] = {}
    for row in rows:
        if row.get("lost_expert_bytes") is not None:
            lost_expert_bytes += float(row["lost_expert_bytes"])
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if metadata.get("lost_expert_bytes") is not None:
            lost_expert_bytes += float(metadata["lost_expert_bytes"])
        if row.get("kind") == "terminal_failure":
            terminal_failure_reason = str(metadata.get("reason", "terminal_failure"))
        step = row.get("step")
        paused = row.get("paused_request_streams")
        if isinstance(step, int) and step >= 0 and isinstance(paused, int):
            paused_by_step[step] = max(paused_by_step.get(step, 0), paused)

    paused_values = list(paused_by_step.values())
    max_paused = max(paused_values, default=0)
    mean_paused = safe_div(sum(paused_values), len(paused_values)) if paused_values else 0.0
    paused_area = sum(paused_values)
    return {
        "lost_expert_bytes": lost_expert_bytes,
        "lost_expert_count": safe_div(lost_expert_bytes, expert_size_bytes),
        "terminal_failure_reason": terminal_failure_reason,
        "max_paused_request_streams": max_paused,
        "mean_paused_request_streams": mean_paused,
        "paused_stream_step_area": paused_area,
        "stalled_request_pct_max": 100.0 * safe_div(max_paused, batch_size),
        "stalled_request_pct_mean": 100.0 * safe_div(mean_paused, batch_size),
    }


def summarize_run(run_dir: Path) -> dict[str, Any]:
    manifest = read_json(run_dir / "run_manifest.json")
    params = manifest["parameters"]
    result = read_json(run_dir / "result.json")
    healthy_result_path = Path(manifest["outputs"].get("healthy_result", run_dir / "healthy_result.json"))
    healthy_result = read_json(healthy_result_path)
    rows = read_jsonl(run_dir / "scenario_timeline.jsonl")

    totals = result.get("totals", {})
    healthy_totals = healthy_result.get("totals", {})
    timeline = timeline_metrics(
        rows,
        batch_size=int(result.get("header", {}).get("batch_size", BATCH_SIZE)),
        expert_size_bytes=float(params.get("expert_size_bytes", EXPERT_SIZE_BYTES)),
    )

    all2allv_us = numeric_total(totals, "all2allv_us")
    migration_network_us = numeric_total(totals, "migration_network_us")
    cold_start_storage_us = numeric_total(totals, "cold_start_storage_us")
    initial_replication_us = numeric_total(totals, "initial_replication_us")
    all2allv_bytes = numeric_total(totals, "all2allv_bytes")
    network_repair_bytes = numeric_total(totals, "network_repair_bytes")
    cold_start_bytes = numeric_total(totals, "cold_start_bytes")
    lost_expert_bytes = float(timeline["lost_expert_bytes"])
    lost_or_repair_bytes = cold_start_bytes or lost_expert_bytes or network_repair_bytes
    lost_count = float(timeline["lost_expert_count"]) or safe_div(
        lost_or_repair_bytes,
        float(params.get("expert_size_bytes", EXPERT_SIZE_BYTES)),
    )

    storage_read_gbps = float(params["storage_read_gbps"])
    data_lake_reload_us = (
        float(params.get("data_lake_fixed_us", 0.0))
        + safe_div(lost_or_repair_bytes, storage_read_gbps * GBPS_TO_BYTES_PER_US)
        + lost_count * float(params.get("data_lake_per_expert_us", 0.0))
    )
    stalled_request_penalty_us = (
        float(timeline["paused_stream_step_area"]) * float(params.get("kv_penalty_per_stalled_request_us", 0.0))
    )
    t_healthy = numeric_total(healthy_totals, "all2allv_us")
    t_lake = all2allv_us + data_lake_reload_us + stalled_request_penalty_us
    t_replica = all2allv_us + migration_network_us + cold_start_storage_us + stalled_request_penalty_us

    serviceable = not timeline["terminal_failure_reason"] and all(
        command.get("returncode") == 0 for command in manifest.get("commands", [])
    )
    row: dict[str, Any] = {
        "benchmark": params["benchmark"],
        "subjects": ",".join(params["subjects"]),
        "scenario": params["scenario"],
        "placement": params["placement"],
        "capacity": params["capacity"],
        "decode_steps_requested": params["decode_steps_requested"],
        "decode_steps_effective": params["decode_steps_effective"],
        "ranks_per_node": params["ranks_per_node"],
        "scaleup_gbps": params["scaleup_gbps"],
        "scaleout_gbps": params["scaleout_gbps"],
        "storage_read_gbps": params["storage_read_gbps"],
        "serviceable": serviceable,
        "terminal_failure_reason": timeline["terminal_failure_reason"],
        "all2allv_us": all2allv_us,
        "migration_network_us": migration_network_us,
        "cold_start_storage_us": cold_start_storage_us,
        "initial_replication_us": initial_replication_us,
        "all2allv_bytes": all2allv_bytes,
        "network_repair_bytes": network_repair_bytes,
        "cold_start_bytes": cold_start_bytes,
        "lost_expert_bytes": lost_expert_bytes,
        "lost_expert_count": lost_count,
        "max_paused_request_streams": timeline["max_paused_request_streams"],
        "mean_paused_request_streams": timeline["mean_paused_request_streams"],
        "paused_stream_step_area": timeline["paused_stream_step_area"],
        "stalled_request_pct_max": timeline["stalled_request_pct_max"],
        "stalled_request_pct_mean": timeline["stalled_request_pct_mean"],
        "data_lake_reload_us": data_lake_reload_us,
        "stalled_request_penalty_us": stalled_request_penalty_us,
        "T_healthy": t_healthy,
        "T_lake": t_lake,
        "T_replica_repair": t_replica,
        "ft_tax": safe_div(t_replica, t_healthy),
        "repair_source_speedup": safe_div(data_lake_reload_us, migration_network_us),
        "system_benefit_vs_lake": safe_div(t_lake, t_replica),
        "repair_fraction_of_total": safe_div(
            migration_network_us + cold_start_storage_us + data_lake_reload_us,
            max(t_replica, EPSILON),
        ),
    }
    write_json(run_dir / "summary.json", row)
    write_csv(run_dir / "summary.csv", [row])
    write_csv(run_dir / "plot_points.csv", [row])
    return row


def cmd_fetch_traces(args: argparse.Namespace) -> None:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        fail("fetch-traces requires huggingface_hub; install it in the uv environment")

    subjects = parse_csv(args.subjects)
    token = args.token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    files = api.list_repo_files(DATASET_REPO, repo_type="dataset")
    patterns: list[str] = []
    for subject in subjects:
        matches = [
            path
            for path in files
            if path.endswith(".json")
            and args.benchmark.lower() in path.lower()
            and subject.lower() in path.lower()
            and (args.model.split("/")[-1].lower() in path.lower() or "llama4" in path.lower())
        ]
        if not matches:
            fail(f"no remote JSON files matched benchmark={args.benchmark!r} subject={subject!r}")
        patterns.extend(matches[: args.max_files_per_subject])

    if args.dry_run:
        print(json.dumps({"repo_id": DATASET_REPO, "allow_patterns": patterns}, indent=2, sort_keys=True))
        return

    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        token=token,
        local_dir=args.out_dir,
        allow_patterns=patterns,
    )
    print(f"downloaded {len(patterns)} selected trace files under {args.out_dir}")


def cmd_run(args: argparse.Namespace) -> None:
    if args.sweep:
        run_sweep(args)
        return
    row = run_one(args, Path(args.out_dir))
    print(f"wrote summary to {Path(args.out_dir) / 'summary.json'}")
    print(f"T_replica_repair={row['T_replica_repair']:.6f} us, T_lake={row['T_lake']:.6f} us")


def combo_slug(params: dict[str, Any]) -> str:
    return (
        f"{params['scenario']}_{params['placement']}_cap{params['capacity']}"
        f"_steps{params['decode_steps']}_so{params['scaleout_gbps']}"
        f"_store{params['storage_read_gbps']}"
    ).replace(".", "p")


def run_sweep(args: argparse.Namespace) -> None:
    root = Path(args.out_dir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    placements = parse_csv(args.placement)
    capacities = parse_int_csv(str(args.capacity))
    decode_steps = parse_int_csv(str(args.decode_steps))
    scaleout_values = parse_float_csv(str(args.scaleout_gbps))
    storage_values = parse_float_csv(str(args.storage_read_gbps))
    for placement, capacity, steps, scaleout, storage in itertools.product(
        placements, capacities, decode_steps, scaleout_values, storage_values
    ):
        point = argparse.Namespace(**vars(args))
        point.sweep = False
        point.placement = placement
        point.capacity = capacity
        point.decode_steps = steps
        point.scaleout_gbps = scaleout
        point.storage_read_gbps = storage
        params = {
            "scenario": point.scenario,
            "placement": placement,
            "capacity": capacity,
            "decode_steps": steps,
            "scaleout_gbps": scaleout,
            "storage_read_gbps": storage,
        }
        run_dir = root / combo_slug(params)
        print(f"running {run_dir.name}")
        rows.append(run_one(point, run_dir))
    write_csv(root / "sweep_results.csv", rows)
    print(f"wrote {len(rows)} rows to {root / 'sweep_results.csv'}")


def cmd_summarize(args: argparse.Namespace) -> None:
    row = summarize_run(Path(args.run_dir))
    print(json.dumps(row, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight MoE fault-tolerance evaluation CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch-traces")
    fetch.add_argument("--benchmark", choices=["mmlu", "mmlu_ZH_CN"], required=True)
    fetch.add_argument("--subjects", required=True)
    fetch.add_argument("--max-files-per-subject", type=int, default=256)
    fetch.add_argument("--out-dir", required=True)
    fetch.add_argument("--model", default=DEFAULT_MODEL)
    fetch.add_argument("--token")
    fetch.add_argument("--dry-run", action="store_true")
    fetch.set_defaults(func=cmd_fetch_traces)

    run = sub.add_parser("run")
    run.add_argument("--trace-root", required=True)
    run.add_argument("--benchmark", choices=["mmlu", "mmlu_ZH_CN"], required=True)
    run.add_argument("--subjects", required=True)
    run.add_argument("--scenario", choices=["no_events", "node1_fail_join"], required=True)
    run.add_argument("--placement", default="lazarus")
    run.add_argument("--capacity", default=16)
    run.add_argument("--decode-steps", default=32)
    run.add_argument("--ranks-per-node", type=int, default=4)
    run.add_argument("--scaleup-gbps", type=float, default=3600)
    run.add_argument("--scaleout-gbps", default=400)
    run.add_argument("--storage-read-gbps", default=25)
    run.add_argument("--data-lake-fixed-us", type=float, default=0)
    run.add_argument("--data-lake-per-expert-us", type=float, default=0)
    run.add_argument("--kv-penalty-per-stalled-request-us", type=float, default=0)
    run.add_argument("--out-dir", required=True)
    run.add_argument("--sweep", action="store_true")
    run.set_defaults(func=cmd_run)

    summarize = sub.add_parser("summarize")
    summarize.add_argument("run_dir")
    summarize.set_defaults(func=cmd_summarize)
    return parser


def normalize_run_args(args: argparse.Namespace) -> None:
    if args.command != "run" or args.sweep:
        return
    args.capacity = int(args.capacity)
    args.decode_steps = int(args.decode_steps)
    args.scaleout_gbps = float(args.scaleout_gbps)
    args.storage_read_gbps = float(args.storage_read_gbps)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    normalize_run_args(args)
    args.func(args)


if __name__ == "__main__":
    main()
