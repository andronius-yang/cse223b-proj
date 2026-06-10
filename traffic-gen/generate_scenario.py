#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import generate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from allocator import (  # noqa: E402
    REPLICATION_STRATEGIES,
    Placement,
    Slot,
    plan_layer,
)


DEFAULT_RANKS_PER_NODE = 4
DEFAULT_CAPACITY_PER_RANK_PER_LAYER = 16
DEFAULT_REPLICATION_STRATEGY = "adaptive"
K_MIN = 2
EXPERT_STATE_BYTES = 251_658_240
SCENARIO_ROOT = generate.OUTPUT_DIR / "scenarios"

SLUG_RE = re.compile(r"^[A-Za-z0-9_-]+$")

Matrix = generate.Matrix
ExpertKey = tuple[int, int]


@dataclass(frozen=True)
class ScenarioConfig:
    scenario_id: str
    ranks_per_node: int
    capacity_per_rank_per_layer: int
    replication_strategy: str
    events: list["NodeEvent"]


@dataclass(frozen=True)
class NodeEvent:
    step: int
    event_type: str
    ranks: tuple[int, ...]


@dataclass(frozen=True)
class WorkItem:
    token_index: int
    layer_id: int
    expert_ids: tuple[int, ...]


@dataclass
class RequestStream:
    source_rank: int
    local_request_index: int
    path: Path
    work: list[WorkItem]
    cursor: int = 0

    def completed(self) -> bool:
        return self.cursor >= len(self.work)


@dataclass(frozen=True)
class LayerPlan:
    layer_id: int
    placement: Placement


def fail(message: str) -> None:
    raise SystemExit(f"traffic-gen error: {message}")


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        fail(f"malformed JSON in {path}: {exc}")
    except OSError as exc:
        fail(f"could not read {path}: {exc}")


def require_int(value: Any, name: str) -> int:
    if not isinstance(value, int):
        fail(f"{name} must be an integer")
    return value


def require_rank_list(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        fail(f"{name} must be a list of rank ids")
    if not value:
        fail(f"{name} must not be empty")

    ranks: list[int] = []
    seen: set[int] = set()
    for index, rank in enumerate(value):
        if not isinstance(rank, int) or isinstance(rank, bool):
            fail(f"{name}[{index}] must be an integer")
        if rank < 0 or rank >= generate.NUM_RANKS:
            fail(
                f"{name}[{index}] {rank} is outside "
                f"0..{generate.NUM_RANKS - 1}"
            )
        if rank in seen:
            fail(f"{name} contains duplicate rank {rank}")
        seen.add(rank)
        ranks.append(rank)

    return tuple(ranks)


def load_config(path: Path) -> ScenarioConfig:
    data = load_json(path)
    if not isinstance(data, dict):
        fail(f"{path} must contain a JSON object")

    scenario_id = data.get("scenario_id")
    if not isinstance(scenario_id, str) or not SLUG_RE.fullmatch(scenario_id):
        fail("scenario_id must be a slug containing only letters, numbers, '_' and '-'")

    ranks_per_node = require_int(
        data.get("ranks_per_node", DEFAULT_RANKS_PER_NODE), "ranks_per_node"
    )
    capacity = require_int(
        data.get(
            "capacity_per_rank_per_layer", DEFAULT_CAPACITY_PER_RANK_PER_LAYER
        ),
        "capacity_per_rank_per_layer",
    )
    replication_strategy = data.get(
        "replication_strategy", DEFAULT_REPLICATION_STRATEGY
    )
    if replication_strategy not in REPLICATION_STRATEGIES:
        fail(
            f"replication_strategy must be one of "
            f"{sorted(REPLICATION_STRATEGIES)}, got {replication_strategy!r}"
        )
    if ranks_per_node <= 0:
        fail("ranks_per_node must be positive")
    if capacity <= 0:
        fail("capacity_per_rank_per_layer must be positive")
    if generate.NUM_RANKS % ranks_per_node != 0:
        fail(
            f"NUM_RANKS={generate.NUM_RANKS} must be divisible by "
            f"ranks_per_node={ranks_per_node}"
        )

    if generate.NUM_RANKS * capacity < generate.NUM_EXPERTS * K_MIN:
        fail(
            "capacity_per_rank_per_layer is too small for "
            f"{generate.NUM_EXPERTS} experts with k_min={K_MIN}"
        )

    raw_events = data.get("events", [])
    if not isinstance(raw_events, list):
        fail("events must be a list")

    events: list[NodeEvent] = []
    seen_steps: set[int] = set()
    previous_step = -1
    failed_ranks_set: set[int] = set()
    for index, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, dict):
            fail(f"events[{index}] must be a JSON object")

        step = require_int(raw_event.get("step"), f"events[{index}].step")
        event_type = raw_event.get("type")
        ranks = require_rank_list(raw_event.get("ranks"), f"events[{index}].ranks")

        if step < 0:
            fail(f"events[{index}].step must be non-negative")
        if step <= previous_step:
            fail("events must be ordered by strictly increasing step")
        if step in seen_steps:
            fail(f"at most one rank event is allowed at step {step}")
        if event_type not in {"fail", "join"}:
            fail(f"events[{index}].type must be 'fail' or 'join'")

        if event_type == "fail":
            already_failed = sorted(set(ranks) & failed_ranks_set)
            if already_failed:
                fail(
                    f"events[{index}] tries to fail already-failed ranks "
                    f"{already_failed}"
                )
            failed_ranks_set.update(ranks)
        else:
            already_live = sorted(rank for rank in ranks if rank not in failed_ranks_set)
            if already_live:
                fail(
                    f"events[{index}] tries to join already-live ranks "
                    f"{already_live}"
                )
            failed_ranks_set.difference_update(ranks)

        seen_steps.add(step)
        previous_step = step
        events.append(NodeEvent(step=step, event_type=event_type, ranks=ranks))

    return ScenarioConfig(
        scenario_id=scenario_id,
        ranks_per_node=ranks_per_node,
        capacity_per_rank_per_layer=capacity,
        replication_strategy=replication_strategy,
        events=events,
    )


def flatten_selected_experts(
    path: Path,
    token_index: int,
    layer_id: int,
    selected_experts: Any,
) -> tuple[int, ...]:
    if not isinstance(selected_experts, list):
        fail(f"{path} token {token_index} layer {layer_id} selected_experts is not a list")
    if not selected_experts:
        fail(f"{path} token {token_index} layer {layer_id} selected_experts is empty")

    expert_ids: list[int] = []
    for row_index, expert_row in enumerate(selected_experts):
        if not isinstance(expert_row, list):
            fail(
                f"{path} token {token_index} layer {layer_id} row {row_index} "
                "is not a list"
            )
        if not expert_row:
            fail(
                f"{path} token {token_index} layer {layer_id} row {row_index} "
                "has no expert ids"
            )
        for expert_id in expert_row:
            if not isinstance(expert_id, int):
                fail(
                    f"{path} token {token_index} layer {layer_id} row {row_index} "
                    f"has non-integer expert id {expert_id!r}"
                )
            generate.owner_ranks(layer_id, expert_id)
            expert_ids.append(expert_id)
    return tuple(expert_ids)


def build_workload(trace_paths: list[Path]) -> tuple[list[RequestStream], dict[int, list[float]]]:
    streams: list[RequestStream] = []
    layer_loads: dict[int, list[float]] = {}

    for request_index, path in enumerate(trace_paths[: generate.BATCH_SIZE]):
        source_rank = request_index // generate.REQUESTS_PER_RANK
        local_request_index = request_index % generate.REQUESTS_PER_RANK
        trace = generate.load_trace(path)
        last_token_index = min(generate.DECODE_STEPS, len(trace) - 1)

        work: list[WorkItem] = []
        for token_index in range(1, last_token_index + 1):
            token_entry = trace[token_index]
            if not isinstance(token_entry, dict):
                fail(f"{path} token {token_index} must be a JSON object")

            for layer_key, selected_experts in token_entry.items():
                layer_id = generate.parse_layer_id(path, token_index, layer_key)
                if selected_experts is None:
                    continue

                expert_ids = flatten_selected_experts(
                    path=path,
                    token_index=token_index,
                    layer_id=layer_id,
                    selected_experts=selected_experts,
                )
                loads = layer_loads.setdefault(
                    layer_id, [0.0 for _ in range(generate.NUM_EXPERTS)]
                )
                for expert_id in expert_ids:
                    loads[expert_id] += 1.0
                work.append(
                    WorkItem(
                        token_index=token_index,
                        layer_id=layer_id,
                        expert_ids=expert_ids,
                    )
                )

        streams.append(
            RequestStream(
                source_rank=source_rank,
                local_request_index=local_request_index,
                path=path,
                work=work,
            )
        )

    if not layer_loads:
        fail("no non-null MoE layer selections found in selected traces")
    return streams, layer_loads


def rank_to_node(rank: int, ranks_per_node: int) -> int:
    return rank // ranks_per_node


def slot_rank(slot: Slot, ranks_per_node: int) -> int:
    return slot.node * ranks_per_node + slot.local_rank


def baseline_owner_rank(layer_id: int, expert_id: int) -> int:
    return generate.choose_owner_rank(generate.owner_ranks(layer_id, expert_id))


def build_layer_plans(
    layer_loads: dict[int, list[float]],
    config: ScenarioConfig,
) -> dict[int, LayerPlan]:
    plans: dict[int, LayerPlan] = {}
    num_nodes = generate.NUM_RANKS // config.ranks_per_node
    for layer_id, loads in layer_loads.items():
        _, placement = plan_layer(
            loads,
            num_nodes=num_nodes,
            gpus_per_node=config.ranks_per_node,
            capacity=config.capacity_per_rank_per_layer,
            k_min=K_MIN,
            strategy=config.replication_strategy,
        )
        plans[layer_id] = LayerPlan(
            layer_id=layer_id,
            placement=placement,
        )
    return plans


def validate_event_completion(config: ScenarioConfig, streams: list[RequestStream]) -> None:
    completion_step = max((len(stream.work) for stream in streams), default=0)
    for event in config.events:
        if event.step >= completion_step:
            fail(
                f"event at step {event.step} is after request completion "
                f"at step {completion_step - 1}"
            )


def total_bytes(matrix: Matrix) -> int:
    return sum(sum(row) for row in matrix)


def node_rank_ids(node: int, ranks_per_node: int) -> range:
    start = node * ranks_per_node
    return range(start, start + ranks_per_node)


def live_node_ids(failed_ranks_set: set[int], ranks_per_node: int) -> list[int]:
    nodes: list[int] = []
    for node in range(generate.NUM_RANKS // ranks_per_node):
        if any(rank not in failed_ranks_set for rank in node_rank_ids(node, ranks_per_node)):
            nodes.append(node)
    return nodes


def failed_node_ids(failed_ranks_set: set[int], ranks_per_node: int) -> list[int]:
    nodes: list[int] = []
    for node in range(generate.NUM_RANKS // ranks_per_node):
        if all(rank in failed_ranks_set for rank in node_rank_ids(node, ranks_per_node)):
            nodes.append(node)
    return nodes


def failed_ranks(failed_ranks_set: set[int]) -> list[int]:
    return sorted(failed_ranks_set)


def request_counts(
    streams: list[RequestStream],
    failed_ranks_set: set[int],
) -> dict[str, int]:
    live = 0
    paused = 0
    completed = 0
    for stream in streams:
        if stream.completed():
            completed += 1
        elif stream.source_rank in failed_ranks_set:
            paused += 1
        else:
            live += 1
    return {
        "live_request_streams": live,
        "paused_request_streams": paused,
        "completed_request_streams": completed,
    }


def state_fields(
    streams: list[RequestStream],
    failed_ranks_set: set[int],
    ranks_per_node: int,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "live_nodes": live_node_ids(failed_ranks_set, ranks_per_node),
        "failed_nodes": failed_node_ids(failed_ranks_set, ranks_per_node),
        "failed_ranks": failed_ranks(failed_ranks_set),
    }
    fields.update(request_counts(streams, failed_ranks_set))
    return fields


def write_jsonl(handle: TextIO, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
    handle.write("\n")
    handle.flush()


def topsim_row_from_timeline(
    row: dict[str, Any],
    scenario_id: str,
    ranks_per_node: int,
) -> dict[str, Any]:
    metadata = {
        key: value
        for key, value in row.items()
        if key not in {"matrix"}
    }
    row_id = str(row["matrix"]).removesuffix(".txt")
    return {
        "id": f"{scenario_id}_{row_id}",
        "matrix": row["matrix"],
        "gpus_per_server": ranks_per_node,
        "metadata": metadata,
    }


def emit_matrix_row(
    timeline_handle: TextIO,
    topsim_handle: TextIO,
    scenario_id: str,
    ranks_per_node: int,
    row: dict[str, Any],
) -> None:
    write_jsonl(timeline_handle, row)
    write_jsonl(
        topsim_handle,
        topsim_row_from_timeline(row, scenario_id, ranks_per_node),
    )


def prepare_output_dir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        for owned in (
            "scenario_timeline.jsonl",
            "topsim_matrix_manifest.jsonl",
            "initial_expert_replication.txt",
        ):
            target = path / owned
            if target.exists():
                target.unlink()
        for target in path.glob("step_*_all2allv.txt"):
            target.unlink()
        for target in path.glob("step_*_expert_migration.txt"):
            target.unlink()
    except OSError as exc:
        fail(f"could not prepare scenario output directory {path}: {exc}")


def initial_current_slots(plans: dict[int, LayerPlan]) -> dict[ExpertKey, set[Slot]]:
    current: dict[ExpertKey, set[Slot]] = {}
    for layer_id, plan in plans.items():
        for expert_id, slots in plan.placement.expert_to_slots.items():
            current[(layer_id, expert_id)] = set(slots)
    return current


def build_initial_replication_matrix(
    plans: dict[int, LayerPlan],
    ranks_per_node: int,
) -> Matrix:
    matrix = generate.new_matrix()
    for layer_id, plan in plans.items():
        for expert_id, slots in plan.placement.expert_to_slots.items():
            src_rank = baseline_owner_rank(layer_id, expert_id)
            for slot in slots:
                dst_rank = slot_rank(slot, ranks_per_node)
                if src_rank != dst_rank:
                    matrix[src_rank][dst_rank] += EXPERT_STATE_BYTES
    return matrix


def remove_failed_rank_state(
    current_slots: dict[ExpertKey, set[Slot]],
    failed_rank_ids: set[int],
    ranks_per_node: int,
) -> None:
    for key, slots in list(current_slots.items()):
        current_slots[key] = {
            slot
            for slot in slots
            if slot_rank(slot, ranks_per_node) not in failed_rank_ids
        }


def choose_repair_source(
    current_slots: dict[ExpertKey, set[Slot]],
    key: ExpertKey,
    dst_slot: Slot,
    failed_ranks_set: set[int],
    ranks_per_node: int,
) -> Slot | None:
    candidates = [
        slot
        for slot in current_slots.get(key, set())
        if slot_rank(slot, ranks_per_node) not in failed_ranks_set
    ]
    if not candidates:
        return None
    same_node = [slot for slot in candidates if slot.node == dst_slot.node]
    if same_node:
        return min(same_node, key=lambda slot: (slot_rank(slot, ranks_per_node), slot.slot))

    dst_rank = slot_rank(dst_slot, ranks_per_node)
    candidate_ranks = {slot_rank(slot, ranks_per_node) for slot in candidates}
    source_rank = circular_rank_search(dst_rank, candidate_ranks)
    if source_rank is None:
        return None
    return min(
        (slot for slot in candidates if slot_rank(slot, ranks_per_node) == source_rank),
        key=lambda slot: slot.slot,
    )


def build_join_repair_matrix(
    plans: dict[int, LayerPlan],
    current_slots: dict[ExpertKey, set[Slot]],
    joined_ranks: set[int],
    failed_ranks_set: set[int],
    ranks_per_node: int,
) -> tuple[Matrix, int]:
    matrix = generate.new_matrix()
    disk_bytes = 0
    for layer_id in sorted(plans):
        plan = plans[layer_id]
        for expert_id in range(generate.NUM_EXPERTS):
            key = (layer_id, expert_id)
            planned_slots = [
                slot
                for slot in plan.placement.expert_to_slots[expert_id]
                if slot_rank(slot, ranks_per_node) in joined_ranks
            ]
            for dst_slot in planned_slots:
                if dst_slot in current_slots.get(key, set()):
                    continue
                src_slot = choose_repair_source(
                    current_slots=current_slots,
                    key=key,
                    dst_slot=dst_slot,
                    failed_ranks_set=failed_ranks_set,
                    ranks_per_node=ranks_per_node,
                )
                if src_slot is None:
                    disk_bytes += EXPERT_STATE_BYTES
                else:
                    src_rank = slot_rank(src_slot, ranks_per_node)
                    dst_rank = slot_rank(dst_slot, ranks_per_node)
                    if src_rank != dst_rank:
                        matrix[src_rank][dst_rank] += EXPERT_STATE_BYTES
                current_slots.setdefault(key, set()).add(dst_slot)
    return matrix, disk_bytes


def live_replica_ranks(
    current_slots: dict[ExpertKey, set[Slot]],
    key: ExpertKey,
    failed_ranks_set: set[int],
    ranks_per_node: int,
) -> list[int]:
    ranks = {
        slot_rank(slot, ranks_per_node)
        for slot in current_slots.get(key, set())
        if slot_rank(slot, ranks_per_node) not in failed_ranks_set
    }
    return sorted(ranks)


def circular_rank_search(start_rank: int, candidate_ranks: set[int]) -> int | None:
    for offset in range(generate.NUM_RANKS):
        rank = (start_rank + offset) % generate.NUM_RANKS
        if rank in candidate_ranks:
            return rank
    return None


def choose_route_destination(
    src_rank: int,
    replica_ranks: list[int],
    ranks_per_node: int,
) -> int | None:
    if not replica_ranks:
        return None
    src_node = rank_to_node(src_rank, ranks_per_node)
    same_node = [
        rank for rank in replica_ranks if rank_to_node(rank, ranks_per_node) == src_node
    ]
    if same_node:
        return min(same_node)
    return circular_rank_search(src_rank, set(replica_ranks))


def build_all2allv_matrix(
    streams: list[RequestStream],
    current_slots: dict[ExpertKey, set[Slot]],
    failed_ranks_set: set[int],
    ranks_per_node: int,
) -> tuple[Matrix, dict[str, int], list[RequestStream]]:
    matrix = generate.new_matrix()
    histogram: dict[str, int] = {}
    advancing: list[RequestStream] = []

    for stream in streams:
        if stream.completed():
            continue
        if stream.source_rank in failed_ranks_set:
            continue

        item = stream.work[stream.cursor]
        destinations: list[int] = []

        for expert_id in item.expert_ids:
            key = (item.layer_id, expert_id)
            replicas = live_replica_ranks(
                current_slots=current_slots,
                key=key,
                failed_ranks_set=failed_ranks_set,
                ranks_per_node=ranks_per_node,
            )
            dst_rank = choose_route_destination(
                src_rank=stream.source_rank,
                replica_ranks=replicas,
                ranks_per_node=ranks_per_node,
            )
            if dst_rank is None:
                destinations = []
                break
            destinations.append(dst_rank)

        if not destinations:
            continue

        cursor_key = f"tok{item.token_index}_layer{item.layer_id}"
        histogram[cursor_key] = histogram.get(cursor_key, 0) + 1
        advancing.append(stream)
        for dst_rank in destinations:
            if stream.source_rank != dst_rank:
                matrix[stream.source_rank][dst_rank] += generate.PAYLOAD_BYTES

    return matrix, histogram, advancing


def all_completed(streams: list[RequestStream]) -> bool:
    return all(stream.completed() for stream in streams)


def write_terminal_failure(
    timeline_handle: TextIO,
    step: int,
    reason: dict[str, Any],
    streams: list[RequestStream],
    failed_ranks_set: set[int],
    ranks_per_node: int,
) -> None:
    row = {
        "step": step,
        "kind": "terminal_failure",
        **state_fields(streams, failed_ranks_set, ranks_per_node),
        "metadata": reason,
    }
    write_jsonl(timeline_handle, row)


def run_scenario(config: ScenarioConfig, streams: list[RequestStream], plans: dict[int, LayerPlan]) -> int:
    output_dir = SCENARIO_ROOT / config.scenario_id
    prepare_output_dir(output_dir)

    failed_ranks_set: set[int] = set()
    current_slots = initial_current_slots(plans)
    events = config.events
    event_index = 0
    step = 0

    timeline_path = output_dir / "scenario_timeline.jsonl"
    topsim_path = output_dir / "topsim_matrix_manifest.jsonl"
    try:
        with timeline_path.open("w", encoding="utf-8") as timeline_handle, topsim_path.open(
            "w", encoding="utf-8"
        ) as topsim_handle:
            header = {
                "kind": "scenario_header",
                "metadata": {
                    "scenario_id": config.scenario_id,
                    "num_ranks": generate.NUM_RANKS,
                    "ranks_per_node": config.ranks_per_node,
                    "num_nodes": generate.NUM_RANKS // config.ranks_per_node,
                    "rank_blocks": [
                        {
                            "node": node,
                            "ranks": list(
                                range(
                                    node * config.ranks_per_node,
                                    (node + 1) * config.ranks_per_node,
                                )
                            ),
                        }
                        for node in range(generate.NUM_RANKS // config.ranks_per_node)
                    ],
                    "capacity_per_rank_per_layer": config.capacity_per_rank_per_layer,
                    "replication_strategy": config.replication_strategy,
                    "k_min": K_MIN,
                    "expert_state_bytes": EXPERT_STATE_BYTES,
                    "payload_bytes": generate.PAYLOAD_BYTES,
                    "decode_steps": generate.DECODE_STEPS,
                    "requests_per_rank": generate.REQUESTS_PER_RANK,
                    "batch_size": generate.BATCH_SIZE,
                    "input_glob": generate.INPUT_GLOB,
                },
            }
            write_jsonl(timeline_handle, header)

            initial_matrix = build_initial_replication_matrix(plans, config.ranks_per_node)
            initial_matrix_path = output_dir / "initial_expert_replication.txt"
            generate.write_matrix(initial_matrix_path, initial_matrix)
            emit_matrix_row(
                timeline_handle=timeline_handle,
                topsim_handle=topsim_handle,
                scenario_id=config.scenario_id,
                ranks_per_node=config.ranks_per_node,
                row={
                    "step": -1,
                    "kind": "initial_expert_replication",
                    "matrix": initial_matrix_path.name,
                    "total_bytes": total_bytes(initial_matrix),
                    **state_fields(
                        streams, failed_ranks_set, config.ranks_per_node
                    ),
                },
            )

            while not all_completed(streams):
                if not request_counts(streams, failed_ranks_set)["live_request_streams"]:
                    if event_index < len(events):
                        step = max(step, events[event_index].step)
                    else:
                        write_terminal_failure(
                            timeline_handle=timeline_handle,
                            step=step,
                            reason={"reason": "deadlock_all_incomplete_streams_paused"},
                            streams=streams,
                            failed_ranks_set=failed_ranks_set,
                            ranks_per_node=config.ranks_per_node,
                        )
                        return 1

                if event_index < len(events) and events[event_index].step == step:
                    event = events[event_index]
                    migration_matrix = None
                    disk_bytes = 0
                    if event.event_type == "fail":
                        event_ranks = set(event.ranks)
                        failed_ranks_set.update(event_ranks)
                        remove_failed_rank_state(
                            current_slots, event_ranks, config.ranks_per_node
                        )
                    else:
                        event_ranks = set(event.ranks)
                        failed_ranks_set.difference_update(event_ranks)
                        migration_matrix, disk_bytes = build_join_repair_matrix(
                            plans=plans,
                            current_slots=current_slots,
                            joined_ranks=event_ranks,
                            failed_ranks_set=failed_ranks_set,
                            ranks_per_node=config.ranks_per_node,
                        )

                    node_event_row = {
                        "step": step,
                        "kind": "node_event",
                        **state_fields(
                            streams,
                            failed_ranks_set,
                            config.ranks_per_node,
                        ),
                        "metadata": {
                            "event_type": event.event_type,
                            "ranks": list(event.ranks),
                        },
                    }
                    if disk_bytes:
                        node_event_row["lost_expert_bytes"] = disk_bytes
                    write_jsonl(timeline_handle, node_event_row)
                    event_index += 1

                    if migration_matrix is not None:
                        migration_bytes = total_bytes(migration_matrix)
                        if migration_bytes:
                            migration_path = (
                                output_dir / f"step_{step:06d}_expert_migration.txt"
                            )
                            generate.write_matrix(migration_path, migration_matrix)
                            emit_matrix_row(
                                timeline_handle=timeline_handle,
                                topsim_handle=topsim_handle,
                                scenario_id=config.scenario_id,
                                ranks_per_node=config.ranks_per_node,
                                row={
                                    "step": step,
                                    "kind": "expert_migration",
                                    "matrix": migration_path.name,
                                    "total_bytes": migration_bytes,
                                    **state_fields(
                                        streams,
                                        failed_ranks_set,
                                        config.ranks_per_node,
                                    ),
                                },
                            )

                counts_before = request_counts(streams, failed_ranks_set)
                if counts_before["live_request_streams"]:
                    matrix, histogram, advancing = build_all2allv_matrix(
                        streams=streams,
                        current_slots=current_slots,
                        failed_ranks_set=failed_ranks_set,
                        ranks_per_node=config.ranks_per_node,
                    )
                    if not advancing:
                        if event_index < len(events):
                            step += 1
                            continue
                        write_terminal_failure(
                            timeline_handle=timeline_handle,
                            step=step,
                            reason={"reason": "deadlock_all_live_streams_blocked"},
                            streams=streams,
                            failed_ranks_set=failed_ranks_set,
                            ranks_per_node=config.ranks_per_node,
                        )
                        return 1

                    matrix_path = output_dir / f"step_{step:06d}_all2allv.txt"
                    generate.write_matrix(matrix_path, matrix)
                    emit_matrix_row(
                        timeline_handle=timeline_handle,
                        topsim_handle=topsim_handle,
                        scenario_id=config.scenario_id,
                        ranks_per_node=config.ranks_per_node,
                        row={
                            "step": step,
                            "kind": "all2allv",
                            "matrix": matrix_path.name,
                            "total_bytes": total_bytes(matrix),
                            **state_fields(
                                streams,
                                failed_ranks_set,
                                config.ranks_per_node,
                            ),
                            **counts_before,
                            "metadata": {"cursor_histogram": histogram},
                        },
                    )

                    for stream in advancing:
                        stream.cursor += 1

                step += 1

    except OSError as exc:
        fail(f"could not write scenario outputs under {output_dir}: {exc}")

    if event_index < len(events):
        fail(f"scenario completed before event at step {events[event_index].step}")

    print(f"wrote scenario outputs to {output_dir}")
    return 0


def main() -> None:
    if len(sys.argv) != 2:
        fail("usage: python3 generate_scenario.py SCENARIO_CONFIG_JSON")

    generate.validate_constants()
    config = load_config(Path(sys.argv[1]))

    trace_paths = generate.discover_trace_paths()
    if len(trace_paths) < generate.BATCH_SIZE:
        fail(
            f"found {len(trace_paths)} trace files, need at least {generate.BATCH_SIZE} "
            f"for {generate.NUM_RANKS} ranks * {generate.REQUESTS_PER_RANK} requests per rank"
        )

    streams, layer_loads = build_workload(trace_paths)
    validate_event_completion(config, streams)
    plans = build_layer_plans(layer_loads, config)
    raise SystemExit(run_scenario(config, streams, plans))


if __name__ == "__main__":
    main()
