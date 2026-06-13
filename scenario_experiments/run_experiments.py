#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
TRAFFIC_GEN_ROOT = PROJECT_ROOT / "traffic-gen"
TOPSIM_PROJECT = PROJECT_ROOT / "topsim"

SCENARIO_CONFIG_DIR = EXPERIMENT_ROOT / "configs" / "scenarios"
STRATEGY_CONFIG_DIR = EXPERIMENT_ROOT / "configs" / "strategies"
RESULTS_DIR = EXPERIMENT_ROOT / "results"
SCENARIO_OUTPUT_DIR = RESULTS_DIR / "scenarios"
TOPSIM_OUTPUT_DIR = RESULTS_DIR / "topsim"
GENERATED_CONFIG_DIR = RESULTS_DIR / "generated_configs"
MANIFEST_PATH = RESULTS_DIR / "experiment_manifest.jsonl"

SCENARIO_ORDER = (
    "no_fail",
    "round_robin_rank_failure",
    "round_robin_node_failure",
    "hotspot_rank_repeated_failure",
    "armageddon_two_node_low_band",
    "armageddon_two_node_high_band",
    "armageddon_one_node_low_holdout",
    "armageddon_one_node_high_holdout",
    "armageddon_rotating_single_node",
)
STRATEGY_ORDER = (
    "control_single_owner",
    "uniform_fixed_2",
    "adaptive_mro_avg3",
    "adaptive_mro_avg4",
    "adaptive_mro_avg6",
    "adaptive_mro_avg8",
)

DEFAULT_TIMELINE_POLICY = "direct"


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    label: str
    description: str
    ranks_per_node: int
    events: list[dict[str, Any]]
    path: Path


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    label: str
    group: str
    description: str
    control_single_owner: bool
    capacity_per_rank_per_layer: int
    average_replicas_per_expert: float
    replication_strategy: str | None
    path: Path


def fail(message: str) -> None:
    raise SystemExit(f"experiment error: {message}")


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        fail(f"malformed JSON in {path}: {exc}")
    except OSError as exc:
        fail(f"could not read {path}: {exc}")


def require_str(data: dict[str, Any], key: str, path: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        fail(f"{path}: {key} must be a non-empty string")
    return value


def require_int(data: dict[str, Any], key: str, path: Path) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        fail(f"{path}: {key} must be an integer")
    return value


def require_number(data: dict[str, Any], key: str, path: Path) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        fail(f"{path}: {key} must be numeric")
    return float(value)


def load_scenario_spec(name: str) -> ScenarioSpec:
    path = SCENARIO_CONFIG_DIR / f"{name}.json"
    data = load_json(path)
    if not isinstance(data, dict):
        fail(f"{path} must contain a JSON object")
    events = data.get("events")
    if not isinstance(events, list):
        fail(f"{path}: events must be a list")
    scenario_id = require_str(data, "scenario_id", path)
    if scenario_id != name:
        fail(f"{path}: scenario_id must match filename stem {name!r}")
    return ScenarioSpec(
        scenario_id=scenario_id,
        label=require_str(data, "label", path),
        description=require_str(data, "description", path),
        ranks_per_node=require_int(data, "ranks_per_node", path),
        events=events,
        path=path,
    )


def load_strategy_spec(name: str) -> StrategySpec:
    path = STRATEGY_CONFIG_DIR / f"{name}.json"
    data = load_json(path)
    if not isinstance(data, dict):
        fail(f"{path} must contain a JSON object")
    strategy_id = require_str(data, "strategy_id", path)
    if strategy_id != name:
        fail(f"{path}: strategy_id must match filename stem {name!r}")
    control = data.get("control_single_owner")
    if not isinstance(control, bool):
        fail(f"{path}: control_single_owner must be boolean")
    replication_strategy = data.get("replication_strategy")
    if replication_strategy is not None and not isinstance(replication_strategy, str):
        fail(f"{path}: replication_strategy must be a string when present")
    if not control and not replication_strategy:
        fail(f"{path}: non-control strategies must set replication_strategy")
    return StrategySpec(
        strategy_id=strategy_id,
        label=require_str(data, "label", path),
        group=require_str(data, "group", path),
        description=require_str(data, "description", path),
        control_single_owner=control,
        capacity_per_rank_per_layer=require_int(data, "capacity_per_rank_per_layer", path),
        average_replicas_per_expert=require_number(data, "average_replicas_per_expert", path),
        replication_strategy=replication_strategy,
        path=path,
    )


def import_generate_scenario():
    if str(TRAFFIC_GEN_ROOT) not in sys.path:
        sys.path.insert(0, str(TRAFFIC_GEN_ROOT))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import generate_scenario as gs  # type: ignore

    return gs


@contextmanager
def pushd(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def write_generated_config(
    gs: Any,
    scenario: ScenarioSpec,
    strategy: StrategySpec,
) -> Any:
    experiment_id = f"{scenario.scenario_id}__{strategy.strategy_id}"
    config_path = GENERATED_CONFIG_DIR / f"{experiment_id}.json"
    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # The single-owner control is an experiment-only wrapper around run_scenario,
    # so validate the shared event schedule with an ordinary adaptive config and
    # then replace the placement strategy before generation.
    validation_strategy = strategy.replication_strategy or "adaptive"
    validation_capacity = max(strategy.capacity_per_rank_per_layer, gs.DEFAULT_CAPACITY_PER_RANK_PER_LAYER)
    payload = {
        "scenario_id": experiment_id,
        "ranks_per_node": scenario.ranks_per_node,
        "capacity_per_rank_per_layer": validation_capacity,
        "replication_strategy": validation_strategy,
        "events": scenario.events,
    }
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    loaded = gs.load_config(config_path)
    if not strategy.control_single_owner:
        return loaded
    return gs.ScenarioConfig(
        scenario_id=loaded.scenario_id,
        ranks_per_node=loaded.ranks_per_node,
        capacity_per_rank_per_layer=strategy.capacity_per_rank_per_layer,
        replication_strategy=strategy.strategy_id,
        events=loaded.events,
    )


def clone_streams(gs: Any, streams: list[Any]) -> list[Any]:
    return [
        gs.RequestStream(
            source_rank=stream.source_rank,
            local_request_index=stream.local_request_index,
            path=stream.path,
            work=stream.work,
            cursor=0,
        )
        for stream in streams
    ]


def build_single_owner_plans(
    gs: Any,
    layer_loads: dict[int, list[float]],
    ranks_per_node: int,
) -> dict[int, Any]:
    from allocator import Placement, Slot

    plans: dict[int, Any] = {}
    for layer_id in sorted(layer_loads):
        expert_to_slots: dict[int, list[Any]] = {}
        slot_to_expert: dict[Any, int] = {}
        rank_slot_counts = [0 for _ in range(gs.generate.NUM_RANKS)]
        for expert_id in range(gs.generate.NUM_EXPERTS):
            owner_rank = gs.baseline_owner_rank(layer_id, expert_id)
            slot = Slot(
                node=owner_rank // ranks_per_node,
                local_rank=owner_rank % ranks_per_node,
                slot=rank_slot_counts[owner_rank],
            )
            rank_slot_counts[owner_rank] += 1
            expert_to_slots[expert_id] = [slot]
            slot_to_expert[slot] = expert_id
        plans[layer_id] = gs.LayerPlan(
            layer_id=layer_id,
            placement=Placement(
                expert_to_slots=expert_to_slots,
                slot_to_expert=slot_to_expert,
            ),
        )
    return plans


def build_plans(
    gs: Any,
    layer_loads: dict[int, list[float]],
    config: Any,
    strategy: StrategySpec,
) -> dict[int, Any]:
    if strategy.control_single_owner:
        return build_single_owner_plans(gs, layer_loads, config.ranks_per_node)
    return gs.build_layer_plans(layer_loads, config)


def run_topsim_timeline(
    *,
    timeline_path: Path,
    json_path: Path,
    ranks_per_node: int,
    expert_state_bytes: int,
    policy: str,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "--project",
        str(TOPSIM_PROJECT),
        "toposim-timeline",
        str(timeline_path),
        "--policy",
        policy,
        "--gpus-per-server",
        str(ranks_per_node),
        "--expert-size-bytes",
        str(expert_state_bytes),
        "--json",
        str(json_path),
    ]
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout, file=sys.stderr)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        fail(f"toposim-timeline failed for {timeline_path}")


def run_analyzer() -> None:
    cmd = [
        "uv",
        "run",
        "--project",
        str(TOPSIM_PROJECT),
        "python",
        str(EXPERIMENT_ROOT / "analyze_results.py"),
        "--quiet",
    ]
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout, file=sys.stderr)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        fail("experiment analysis failed")
    if proc.stdout:
        print(proc.stdout, end="")


def write_manifest(rows: list[dict[str, Any]]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def run_experiments(*, skip_topsim: bool, skip_analysis: bool) -> None:
    scenarios = [load_scenario_spec(name) for name in SCENARIO_ORDER]
    strategies = [load_strategy_spec(name) for name in STRATEGY_ORDER]
    gs = import_generate_scenario()

    gs.generate.validate_constants()
    SCENARIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TOPSIM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("building trace-derived workload once")
    with pushd(TRAFFIC_GEN_ROOT):
        trace_paths = gs.generate.discover_trace_paths()
        if len(trace_paths) < gs.generate.BATCH_SIZE:
            fail(
                f"found {len(trace_paths)} trace files, need at least {gs.generate.BATCH_SIZE}"
            )
        base_streams, layer_loads = gs.build_workload(trace_paths)
    print(f"loaded {len(base_streams)} request streams and {len(layer_loads)} MoE layers")

    manifest_rows: list[dict[str, Any]] = []
    old_root = gs.SCENARIO_ROOT
    gs.SCENARIO_ROOT = SCENARIO_OUTPUT_DIR
    try:
        total = len(scenarios) * len(strategies)
        index = 0
        for scenario in scenarios:
            for strategy in strategies:
                index += 1
                config = write_generated_config(gs, scenario, strategy)
                experiment_id = config.scenario_id
                streams = clone_streams(gs, base_streams)
                gs.validate_event_completion(config, streams)
                plans = build_plans(gs, layer_loads, config, strategy)

                print(f"[{index}/{total}] generating {experiment_id}")
                code = gs.run_scenario(config, streams, plans)
                if code != 0:
                    fail(f"scenario generation returned {code} for {experiment_id}")

                scenario_dir = SCENARIO_OUTPUT_DIR / experiment_id
                timeline_path = scenario_dir / "scenario_timeline.jsonl"
                topsim_json_path = TOPSIM_OUTPUT_DIR / f"{experiment_id}.json"
                if not skip_topsim:
                    print(f"[{index}/{total}] measuring {experiment_id} with topsim-timeline")
                    run_topsim_timeline(
                        timeline_path=timeline_path,
                        json_path=topsim_json_path,
                        ranks_per_node=config.ranks_per_node,
                        expert_state_bytes=gs.EXPERT_STATE_BYTES,
                        policy=DEFAULT_TIMELINE_POLICY,
                    )

                manifest_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "scenario_id": scenario.scenario_id,
                        "scenario_label": scenario.label,
                        "scenario_description": scenario.description,
                        "strategy_id": strategy.strategy_id,
                        "strategy_label": strategy.label,
                        "strategy_group": strategy.group,
                        "strategy_description": strategy.description,
                        "control_single_owner": strategy.control_single_owner,
                        "replication_strategy": config.replication_strategy,
                        "capacity_per_rank_per_layer": config.capacity_per_rank_per_layer,
                        "average_replicas_per_expert": strategy.average_replicas_per_expert,
                        "ranks_per_node": config.ranks_per_node,
                        "policy": DEFAULT_TIMELINE_POLICY,
                        "config": relative(GENERATED_CONFIG_DIR / f"{experiment_id}.json"),
                        "scenario_timeline": relative(timeline_path),
                        "topsim_json": relative(topsim_json_path),
                    }
                )
    finally:
        gs.SCENARIO_ROOT = old_root

    write_manifest(manifest_rows)
    print(f"wrote {relative(MANIFEST_PATH)}")

    if not skip_topsim and not skip_analysis:
        run_analyzer()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic replication/failure experiments under scenario_experiments/results."
    )
    parser.add_argument(
        "--skip-topsim",
        action="store_true",
        help="Generate timelines only; do not run topsim-timeline or analysis.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Run generation and topsim-timeline, but leave stats and figure generation to analyze_results.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiments(skip_topsim=args.skip_topsim, skip_analysis=args.skip_analysis)


if __name__ == "__main__":
    main()
