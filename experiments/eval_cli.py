#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import itertools
import json
import os
import random
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - keeps the CLI usable in minimal envs.
    def tqdm(iterable: Iterable[Any], **kwargs: Any) -> Iterable[Any]:
        total = int(kwargs.get("total") or 0)
        desc = str(kwargs.get("desc") or "progress")
        completed = 0
        for item in iterable:
            completed += 1
            if total:
                width = 24
                filled = int(width * completed / total)
                bar = "#" * filled + "-" * (width - filled)
                print(f"\r{desc}: |{bar}| {completed}/{total}", end="", file=sys.stderr, flush=True)
            yield item
        if total:
            print(file=sys.stderr)


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
    "request_rerun_us",
    "stall_step_us",
    "serviceable",
    "error",
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
    "recovery_bytes",
    "max_paused_request_streams",
    "mean_paused_request_streams",
    "paused_stream_step_area",
    "stalled_request_pct_max",
    "stalled_request_pct_mean",
    "network_repair_us",
    "data_lake_reload_us",
    "request_rerun_penalty_us",
    "request_stall_penalty_us",
    "stalled_request_penalty_us",
    "T_healthy",
    "T_lake",
    "T_replica_repair",
    "T_network_repair_path",
    "T_data_lake_path",
    "T_request_rerun_path",
    "ft_tax",
    "repair_source_speedup",
    "repair_source_speedup_vs_lake",
    "system_benefit_vs_lake",
    "benefit_network_vs_lake",
    "benefit_network_vs_rerun",
    "break_even_storage_gbps",
    "replica_repair_fraction",
    "lake_repair_fraction",
    "network_repair_wins",
    "data_lake_wins",
    "rerun_wins",
]

SCENARIOS = [
    "no_events",
    "rank4_fail_join",
    "rank4_fail_no_join",
    "node1_fail_join",
    "node1_fail_no_join",
    "two_node_fail_join",
    "two_node_fail_no_join",
    "probabilistic",
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


def write_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


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
    return fail_join_events(ranks)


def fail_join_events(ranks: list[int], *, join: bool = True) -> list[dict[str, Any]]:
    events = [{"step": 100, "type": "fail", "ranks": ranks}]
    if join:
        events.append({"step": 200, "type": "join", "ranks": ranks})
    return events


def node_ranks(node_id: int, ranks_per_node: int) -> list[int]:
    start = node_id * ranks_per_node
    return list(range(start, start + ranks_per_node))


def validate_scenario_name(scenario: str) -> None:
    if scenario not in SCENARIOS:
        fail(f"supported scenarios are {', '.join(SCENARIOS)}")


def validate_scenario_list(value: str) -> None:
    for scenario in parse_csv(value):
        validate_scenario_name(scenario)


def ensure_single_value(name: str, value: str) -> None:
    if len(parse_csv(str(value))) != 1:
        fail(f"{name} accepts comma-separated values only with --sweep")


def node_failure_events(node_ids: list[int], ranks_per_node: int, *, join: bool) -> list[dict[str, Any]]:
    ranks: list[int] = []
    for node_id in node_ids:
        ranks.extend(node_ranks(node_id, ranks_per_node))
    return fail_join_events(ranks, join=join)


def rank4_events(*, join: bool) -> list[dict[str, Any]]:
    return fail_join_events([4], join=join)


def scenario_events(scenario: str, ranks_per_node: int) -> list[dict[str, Any]]:
    validate_scenario_name(scenario)
    if scenario == "probabilistic":
        fail("--scenario probabilistic requires --sweep and --probabilistic-scenarios > 0")
    if scenario == "no_events":
        return []
    if scenario == "rank4_fail_join":
        return rank4_events(join=True)
    if scenario == "rank4_fail_no_join":
        return rank4_events(join=False)
    if scenario == "node1_fail_join":
        return node_failure_events([1], ranks_per_node, join=True)
    if scenario == "node1_fail_no_join":
        return node_failure_events([1], ranks_per_node, join=False)
    if scenario == "two_node_fail_join":
        return node_failure_events([1, 2], ranks_per_node, join=True)
    if scenario == "two_node_fail_no_join":
        return node_failure_events([1, 2], ranks_per_node, join=False)
    fail(f"unsupported scenario: {scenario}")


def sampled_failure_scenario(args: argparse.Namespace, index: int) -> tuple[str, list[dict[str, Any]]]:
    rng = random.Random(int(args.failure_seed) + index)
    fail_step = rng.randint(int(args.failure_step_min), int(args.failure_step_max))
    join = rng.random() >= float(args.no_join_prob)
    join_step = fail_step + rng.randint(int(args.join_delay_min), int(args.join_delay_max))
    num_nodes = NUM_RANKS // int(args.ranks_per_node)

    roll = rng.random()
    if roll < float(args.two_node_failure_prob):
        node_ids = sorted(rng.sample(range(num_nodes), 2))
        ranks: list[int] = []
        for node_id in node_ids:
            ranks.extend(node_ranks(node_id, int(args.ranks_per_node)))
        scope = f"twonode{node_ids[0]}_{node_ids[1]}"
    elif roll < float(args.two_node_failure_prob) + float(args.node_failure_prob):
        node_id = rng.randrange(num_nodes)
        ranks = node_ranks(node_id, int(args.ranks_per_node))
        scope = f"node{node_id}"
    else:
        rank = rng.randrange(NUM_RANKS)
        ranks = [rank]
        scope = f"rank{rank}"

    events = [{"step": fail_step, "type": "fail", "ranks": ranks}]
    suffix = "nojoin"
    if join:
        events.append({"step": join_step, "type": "join", "ranks": ranks})
        suffix = f"join{join_step}"
    scenario_id = f"prob_{index:03d}_{scope}_fail{fail_step}_{suffix}"
    return scenario_id, events


def path_component_matches(path: Path, token: str) -> bool:
    token = token.lower()
    for part in path.parts:
        normalized = part.lower()
        pieces = normalized.replace(".", "-").split("-")
        if token in pieces or normalized == token:
            return True
    return False


def trace_matches(path: Path, benchmark: str, subject: str) -> bool:
    return (
        path.suffix == ".json"
        and path_component_matches(path, benchmark)
        and path_component_matches(path, subject)
    )


def trace_has_requested_decode(path: Path, requested: int) -> bool:
    try:
        payload = read_json(path)
    except json.JSONDecodeError as exc:
        fail(f"malformed JSON in candidate trace {path}: {exc}")
    if not isinstance(payload, list):
        fail(f"candidate trace {path} must contain a JSON list")
    return len(payload) - 1 >= requested


def balanced_quotas(total: int, count: int) -> list[int]:
    base = total // count
    remainder = total % count
    return [base + (1 if index < remainder else 0) for index in range(count)]


def discover_trace_files(
    trace_root: Path,
    benchmark: str,
    subjects: list[str],
    *,
    decode_steps: int,
    seed: int,
) -> tuple[list[Path], dict[str, int]]:
    if not trace_root.exists():
        fail(f"trace root does not exist: {trace_root}")
    if not subjects:
        fail("at least one subject is required")

    rng = random.Random(seed)
    selected: list[Path] = []
    selected_counts: dict[str, int] = {}
    quotas = balanced_quotas(BATCH_SIZE, len(subjects))
    for subject, quota in zip(subjects, quotas, strict=True):
        subject_files: list[Path] = []
        for root, _, files in os.walk(trace_root):
            root_path = Path(root)
            for name in files:
                path = root_path / name
                if trace_matches(path, benchmark, subject):
                    subject_files.append(path)
        if not subject_files:
            fail(f"found no JSON traces for benchmark={benchmark!r} subject={subject!r} under {trace_root}")

        subject_files = sorted(subject_files)
        rng.shuffle(subject_files)
        long_enough = [path for path in subject_files if trace_has_requested_decode(path, decode_steps)]
        long_enough_set = set(long_enough)
        candidates = long_enough if len(long_enough) >= quota else long_enough + [
            path for path in subject_files if path not in long_enough_set
        ]
        if len(candidates) < quota:
            fail(
                f"found {len(candidates)} matching trace files for benchmark={benchmark!r} "
                f"subject={subject!r}, need {quota}"
            )
        chosen = candidates[:quota]
        selected.extend(chosen)
        selected_counts[subject] = len(chosen)

    if len(selected) != BATCH_SIZE:
        fail(f"selected {len(selected)} trace files, expected exactly {BATCH_SIZE}")
    return selected, selected_counts


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
    events: list[dict[str, Any]] | None = None,
) -> Path:
    payload = {
        "scenario_id": scenario,
        "ranks_per_node": ranks_per_node,
        "capacity_per_rank_per_layer": capacity,
        "replication_strategy": placement_to_strategy(placement),
        "decode_steps": decode_steps,
        "input_glob": str(workload / "llama4-*"),
        "output_dir": str(output_dir),
        "events": events if events is not None else scenario_events(scenario, ranks_per_node),
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


def cleanup_point(run_dir: Path, keep_raw: bool, keep_logs: bool, succeeded: bool) -> None:
    if not keep_raw:
        for path in run_dir.rglob("*.txt"):
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
        workload = run_dir / "workload"
        if workload.exists() and workload.is_dir():
            shutil.rmtree(workload)

    if succeeded and not keep_logs:
        logs = run_dir / "logs"
        if logs.exists():
            for pattern in ("*.out", "*.err"):
                for path in logs.glob(pattern):
                    path.unlink(missing_ok=True)


def slim_point_artifacts(run_dir: Path) -> None:
    for relative in (
        "result.json",
        "healthy_result.json",
        "scenario_timeline.jsonl",
        "topsim_matrix_manifest.jsonl",
        "healthy_no_events/scenario_timeline.jsonl",
        "healthy_no_events/topsim_matrix_manifest.jsonl",
    ):
        path = run_dir / relative
        if path.exists() and path.is_file():
            path.unlink()


def run_one(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    subjects = parse_csv(args.subjects)
    selected_traces, selected_counts = discover_trace_files(
        Path(args.trace_root),
        args.benchmark,
        subjects,
        decode_steps=int(args.decode_steps),
        seed=int(args.seed),
    )
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
            "request_rerun_us": args.request_rerun_us,
            "stall_step_us": args.stall_step_us,
            "kv_penalty_per_stalled_request_us": args.kv_penalty_per_stalled_request_us,
            "events_override": getattr(args, "events_override", None),
            "failure_seed": args.failure_seed,
            "expert_size_bytes": EXPERT_SIZE_BYTES,
            "seed": args.seed,
        },
        "selected_traces": [str(path) for path in selected_traces],
        "selected_counts_by_subject": selected_counts,
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
        events=getattr(args, "events_override", None),
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
    cleanup_point(out_dir, keep_raw=args.keep_raw, keep_logs=args.keep_logs, succeeded=True)
    if args.slim_artifacts:
        slim_point_artifacts(out_dir)
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
    recovery_bytes = cold_start_bytes or lost_expert_bytes or network_repair_bytes
    lost_count = float(timeline["lost_expert_count"]) or safe_div(
        recovery_bytes,
        float(params.get("expert_size_bytes", EXPERT_SIZE_BYTES)),
    )

    storage_read_gbps = float(params["storage_read_gbps"])
    network_repair_us = migration_network_us + cold_start_storage_us
    data_lake_reload_us = (
        float(params.get("data_lake_fixed_us", 0.0))
        + safe_div(recovery_bytes, storage_read_gbps * GBPS_TO_BYTES_PER_US)
        + lost_count * float(params.get("data_lake_per_expert_us", 0.0))
    )
    stall_step_us = float(params.get("stall_step_us", params.get("kv_penalty_per_stalled_request_us", 0.0)))
    request_stall_penalty_us = float(timeline["paused_stream_step_area"]) * stall_step_us
    affected_requests = float(timeline["max_paused_request_streams"])
    request_rerun_penalty_us = affected_requests * float(params.get("request_rerun_us", 0.0))
    t_healthy = numeric_total(healthy_totals, "all2allv_us")
    t_data_lake_path = all2allv_us + data_lake_reload_us + request_stall_penalty_us
    t_network_repair_path = all2allv_us + network_repair_us + request_stall_penalty_us
    t_request_rerun_path = all2allv_us + request_rerun_penalty_us
    repair_source_speedup_vs_lake = safe_div(data_lake_reload_us, max(network_repair_us, EPSILON))

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
        "request_rerun_us": params.get("request_rerun_us", 0.0),
        "stall_step_us": stall_step_us,
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
        "recovery_bytes": recovery_bytes,
        "max_paused_request_streams": timeline["max_paused_request_streams"],
        "mean_paused_request_streams": timeline["mean_paused_request_streams"],
        "paused_stream_step_area": timeline["paused_stream_step_area"],
        "stalled_request_pct_max": timeline["stalled_request_pct_max"],
        "stalled_request_pct_mean": timeline["stalled_request_pct_mean"],
        "network_repair_us": network_repair_us,
        "data_lake_reload_us": data_lake_reload_us,
        "request_rerun_penalty_us": request_rerun_penalty_us,
        "request_stall_penalty_us": request_stall_penalty_us,
        "stalled_request_penalty_us": request_stall_penalty_us,
        "T_healthy": t_healthy,
        "T_lake": t_data_lake_path,
        "T_replica_repair": t_network_repair_path,
        "T_network_repair_path": t_network_repair_path,
        "T_data_lake_path": t_data_lake_path,
        "T_request_rerun_path": t_request_rerun_path,
        "ft_tax": safe_div(t_network_repair_path, t_healthy),
        "repair_source_speedup": repair_source_speedup_vs_lake,
        "repair_source_speedup_vs_lake": repair_source_speedup_vs_lake,
        "system_benefit_vs_lake": safe_div(t_data_lake_path, t_network_repair_path),
        "benefit_network_vs_lake": safe_div(t_data_lake_path, t_network_repair_path),
        "benefit_network_vs_rerun": safe_div(t_request_rerun_path, t_network_repair_path),
        "break_even_storage_gbps": storage_read_gbps * repair_source_speedup_vs_lake,
        "replica_repair_fraction": safe_div(network_repair_us, max(t_network_repair_path, EPSILON)),
        "lake_repair_fraction": safe_div(data_lake_reload_us, max(t_data_lake_path, EPSILON)),
        "network_repair_wins": t_network_repair_path < min(t_data_lake_path, t_request_rerun_path),
        "data_lake_wins": t_data_lake_path < min(t_network_repair_path, t_request_rerun_path),
        "rerun_wins": t_request_rerun_path < min(t_network_repair_path, t_data_lake_path),
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
            and path_component_matches(Path(path), args.benchmark)
            and path_component_matches(Path(path), subject)
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
    print(
        f"T_network_repair_path={row['T_network_repair_path']:.6f} us, "
        f"T_data_lake_path={row['T_data_lake_path']:.6f} us, "
        f"T_request_rerun_path={row['T_request_rerun_path']:.6f} us"
    )


def combo_slug(params: dict[str, Any]) -> str:
    return (
        f"{params['scenario']}_{params['placement']}_cap{params['capacity']}"
        f"_steps{params['decode_steps']}_so{params['scaleout_gbps']}"
        f"_store{params['storage_read_gbps']}_rerun{params['request_rerun_us']}"
        f"_stall{params['stall_step_us']}"
    ).replace(".", "p")


def error_row(point: argparse.Namespace, run_dir: Path, error: BaseException) -> dict[str, Any]:
    return {
        "benchmark": point.benchmark,
        "subjects": point.subjects,
        "scenario": point.scenario,
        "placement": point.placement,
        "capacity": point.capacity,
        "decode_steps_requested": point.decode_steps,
        "decode_steps_effective": "",
        "ranks_per_node": point.ranks_per_node,
        "scaleup_gbps": point.scaleup_gbps,
        "scaleout_gbps": point.scaleout_gbps,
        "storage_read_gbps": point.storage_read_gbps,
        "request_rerun_us": point.request_rerun_us,
        "stall_step_us": point.stall_step_us,
        "serviceable": False,
        "error": str(error),
        "terminal_failure_reason": str(error),
    }


def run_or_resume_point(point: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    if point.resume and (run_dir / "summary.json").exists():
        row = read_json(run_dir / "summary.json")
        cleanup_point(run_dir, keep_raw=point.keep_raw, keep_logs=point.keep_logs, succeeded=True)
        if point.slim_artifacts:
            slim_point_artifacts(run_dir)
        return row
    return run_one(point, run_dir)


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value in ("", None):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = numeric_values(rows, key)
    return safe_div(sum(values), len(values)) if values else 0.0


def serviceable_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if str(row.get("serviceable")).lower() == "true") / len(rows)


def aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    group_by: list[str],
    metrics: dict[str, str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in group_by)
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        group_rows = groups[key]
        aggregate = {field: value for field, value in zip(group_by, key, strict=True)}
        aggregate["n"] = len(group_rows)
        for output_name, source_name in metrics.items():
            if output_name == "serviceable_rate":
                aggregate[output_name] = serviceable_rate(group_rows)
            else:
                aggregate[output_name] = mean_metric(group_rows, source_name)
        out.append(aggregate)
    return out


def write_story_csvs(root: Path, rows: list[dict[str, Any]]) -> None:
    specs = {
        "story_by_scenario.csv": (
            ["scenario", "placement"],
            {
                "serviceable_rate": "serviceable",
                "mean_benefit_network_vs_lake": "benefit_network_vs_lake",
                "mean_benefit_network_vs_rerun": "benefit_network_vs_rerun",
                "mean_ft_tax": "ft_tax",
                "mean_stalled_request_pct_max": "stalled_request_pct_max",
                "mean_break_even_storage_gbps": "break_even_storage_gbps",
            },
        ),
        "story_by_capacity.csv": (
            ["scenario", "placement", "capacity"],
            {
                "mean_benefit_network_vs_lake": "benefit_network_vs_lake",
                "mean_ft_tax": "ft_tax",
                "mean_break_even_storage_gbps": "break_even_storage_gbps",
                "serviceable_rate": "serviceable",
            },
        ),
        "story_by_storage.csv": (
            ["scenario", "placement", "storage_read_gbps"],
            {
                "mean_benefit_network_vs_lake": "benefit_network_vs_lake",
                "mean_repair_source_speedup_vs_lake": "repair_source_speedup_vs_lake",
                "mean_break_even_storage_gbps": "break_even_storage_gbps",
            },
        ),
        "story_by_scaleout.csv": (
            ["scenario", "placement", "scaleout_gbps"],
            {
                "mean_benefit_network_vs_lake": "benefit_network_vs_lake",
                "mean_network_repair_us": "network_repair_us",
                "mean_data_lake_reload_us": "data_lake_reload_us",
            },
        ),
        "story_by_decode.csv": (
            ["scenario", "placement", "decode_steps_effective"],
            {
                "mean_T_network_repair_path": "T_network_repair_path",
                "mean_T_data_lake_path": "T_data_lake_path",
                "mean_benefit_network_vs_lake": "benefit_network_vs_lake",
            },
        ),
        "story_heatmap_storage_scaleout.csv": (
            ["scenario", "placement", "capacity", "storage_read_gbps", "scaleout_gbps"],
            {
                "mean_benefit_network_vs_lake": "benefit_network_vs_lake",
                "mean_break_even_storage_gbps": "break_even_storage_gbps",
            },
        ),
    }
    for filename, (group_by, metrics) in specs.items():
        aggregates = aggregate_rows(rows, group_by=group_by, metrics=metrics)
        write_rows_csv(root / filename, aggregates, group_by + ["n"] + list(metrics.keys()))


def run_sweep(args: argparse.Namespace) -> None:
    root = Path(args.out_dir)
    if root.exists() and not args.resume:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    scenarios = parse_csv(args.scenario)
    placements = parse_csv(args.placement)
    capacities = parse_int_csv(str(args.capacity))
    decode_steps = parse_int_csv(str(args.decode_steps))
    scaleout_values = parse_float_csv(str(args.scaleout_gbps))
    storage_values = parse_float_csv(str(args.storage_read_gbps))
    request_rerun_values = parse_float_csv(str(args.request_rerun_us))
    stall_step_values = parse_float_csv(str(args.stall_step_us))
    points: list[tuple[argparse.Namespace, Path]] = []
    expanded_scenarios: list[tuple[str, list[dict[str, Any]] | None]] = []
    for scenario in scenarios:
        if scenario == "probabilistic":
            if int(args.probabilistic_scenarios) <= 0:
                fail("--scenario probabilistic requires --probabilistic-scenarios > 0")
            for index in range(int(args.probabilistic_scenarios)):
                expanded_scenarios.append(sampled_failure_scenario(args, index))
        else:
            expanded_scenarios.append((scenario, None))

    for scenario_item, placement, capacity, steps, scaleout, storage, request_rerun_us, stall_step_us in itertools.product(
        expanded_scenarios,
        placements,
        capacities,
        decode_steps,
        scaleout_values,
        storage_values,
        request_rerun_values,
        stall_step_values,
    ):
        scenario, events_override = scenario_item
        if events_override is None:
            validate_scenario_name(scenario)
        point = argparse.Namespace(**vars(args))
        point.sweep = False
        point.scenario = scenario
        point.events_override = events_override
        point.placement = placement
        point.capacity = capacity
        point.decode_steps = steps
        point.scaleout_gbps = scaleout
        point.storage_read_gbps = storage
        point.request_rerun_us = request_rerun_us
        point.stall_step_us = stall_step_us
        if args.kv_penalty_per_stalled_request_us and not args.stall_step_us:
            point.stall_step_us = args.kv_penalty_per_stalled_request_us
        params = {
            "scenario": scenario,
            "placement": placement,
            "capacity": capacity,
            "decode_steps": steps,
            "scaleout_gbps": scaleout,
            "storage_read_gbps": storage,
            "request_rerun_us": request_rerun_us,
            "stall_step_us": stall_step_us,
        }
        points.append((point, root / combo_slug(params)))

    partial_path = root / "partial_sweep_results.csv"
    final_path = root / "sweep_results.csv"
    max_workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_or_resume_point, point, run_dir): (point, run_dir) for point, run_dir in points}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="sweep"):
            point, run_dir = futures[fut]
            try:
                row = fut.result()
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    raise
                if args.fail_fast:
                    raise
                row = error_row(point, run_dir, exc)
                cleanup_point(run_dir, keep_raw=point.keep_raw, keep_logs=True, succeeded=False)
            rows.append(row)
            write_csv(partial_path, rows)
    write_csv(root / "sweep_results.csv", rows)
    write_story_csvs(root, rows)
    print(f"wrote {len(rows)} rows to {final_path}")


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
    run.add_argument("--scenario", required=True)
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
    run.add_argument("--request-rerun-us", default=0)
    run.add_argument("--stall-step-us", default=0)
    run.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    run.add_argument("--resume", action="store_true")
    run.add_argument("--keep-raw", action="store_true")
    run.add_argument("--keep-logs", action="store_true")
    run.add_argument("--slim-artifacts", action="store_true")
    run.add_argument("--fail-fast", action="store_true")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--probabilistic-scenarios", type=int, default=0)
    run.add_argument("--failure-seed", type=int, default=0)
    run.add_argument("--failure-step-min", type=int, default=80)
    run.add_argument("--failure-step-max", type=int, default=220)
    run.add_argument("--join-delay-min", type=int, default=40)
    run.add_argument("--join-delay-max", type=int, default=140)
    run.add_argument("--no-join-prob", type=float, default=0.2)
    run.add_argument("--node-failure-prob", type=float, default=0.55)
    run.add_argument("--two-node-failure-prob", type=float, default=0.15)
    run.add_argument("--out-dir", required=True)
    run.add_argument("--sweep", action="store_true")
    run.set_defaults(func=cmd_run)

    summarize = sub.add_parser("summarize")
    summarize.add_argument("run_dir")
    summarize.set_defaults(func=cmd_summarize)
    return parser


def normalize_run_args(args: argparse.Namespace) -> None:
    if args.command != "run":
        return
    validate_scenario_list(args.scenario)
    if args.kv_penalty_per_stalled_request_us and str(args.stall_step_us) in {"0", "0.0"}:
        args.stall_step_us = args.kv_penalty_per_stalled_request_us
    if args.failure_step_min > args.failure_step_max:
        fail("--failure-step-min must be <= --failure-step-max")
    if args.join_delay_min > args.join_delay_max:
        fail("--join-delay-min must be <= --join-delay-max")
    if not (0.0 <= args.no_join_prob <= 1.0):
        fail("--no-join-prob must be between 0 and 1")
    if args.node_failure_prob < 0.0 or args.two_node_failure_prob < 0.0:
        fail("--node-failure-prob and --two-node-failure-prob must be non-negative")
    if args.node_failure_prob + args.two_node_failure_prob > 1.0:
        fail("--node-failure-prob + --two-node-failure-prob must be <= 1")
    if args.sweep:
        return
    if args.scenario == "probabilistic":
        scenario_id, events = sampled_failure_scenario(args, 0)
        args.scenario = scenario_id
        args.events_override = events
    for name in (
        "--scenario",
        "--placement",
        "--capacity",
        "--decode-steps",
        "--scaleout-gbps",
        "--storage-read-gbps",
        "--request-rerun-us",
        "--stall-step-us",
    ):
        attr = name.removeprefix("--").replace("-", "_")
        ensure_single_value(name, getattr(args, attr))
    args.capacity = int(args.capacity)
    args.decode_steps = int(args.decode_steps)
    args.scaleout_gbps = float(args.scaleout_gbps)
    args.storage_read_gbps = float(args.storage_read_gbps)
    args.request_rerun_us = float(args.request_rerun_us)
    args.stall_step_us = float(args.stall_step_us)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    normalize_run_args(args)
    args.func(args)


if __name__ == "__main__":
    main()
