"""Tests for generate_scenario.py (the team-canonical failure-aware generator).
Stdlib only; run from traffic-gen/:

    python3 scenario_tests.py

Unit tests use synthetic data and never touch traces or disk. Integration tests
run the real example scenarios into a temp dir; they SKIP if traces aren't
discoverable from the current working directory.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import generate_scenario as gs
from generate_scenario import (
    EXPERT_STATE_BYTES,
    LayerPlan,
    NodeEvent,
    ScenarioConfig,
)
from allocator import Placement, Slot


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _write_config(tmp: Path, data: dict) -> Path:
    path = tmp / "cfg.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _expect_fail(fn, *args) -> None:
    try:
        fn(*args)
    except SystemExit:
        return
    raise AssertionError("expected SystemExit (fail)")


def _traces_available() -> bool:
    return len(gs.generate.discover_trace_paths()) >= gs.generate.BATCH_SIZE


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #
def test_config_valid() -> None:
    with tempfile.TemporaryDirectory() as d:
        cfg = gs.load_config(_write_config(Path(d), {
            "scenario_id": "ok", "ranks_per_node": 4,
            "capacity_per_rank_per_layer": 16,
            "events": [{"step": 100, "type": "fail", "node": 1},
                       {"step": 200, "type": "join", "node": 1}],
        }))
    assert cfg.ranks_per_node == 4
    assert [e.event_type for e in cfg.events] == ["fail", "join"]


def test_config_rejects_double_fail() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = _write_config(Path(d), {"scenario_id": "x", "events": [
            {"step": 1, "type": "fail", "node": 0},
            {"step": 2, "type": "fail", "node": 0}]})
        _expect_fail(gs.load_config, p)


def test_config_rejects_join_live_node() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = _write_config(Path(d), {"scenario_id": "x", "events": [
            {"step": 1, "type": "join", "node": 0}]})
        _expect_fail(gs.load_config, p)


def test_config_rejects_unordered_events() -> None:
    # group requires strictly increasing step (covers two-events-at-one-step too).
    with tempfile.TemporaryDirectory() as d:
        p = _write_config(Path(d), {"scenario_id": "x", "events": [
            {"step": 5, "type": "fail", "node": 0},
            {"step": 5, "type": "fail", "node": 1}]})
        _expect_fail(gs.load_config, p)


def test_config_rejects_indivisible_ranks() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = _write_config(Path(d), {"scenario_id": "x", "ranks_per_node": 5})
        _expect_fail(gs.load_config, p)


def test_config_rejects_small_capacity() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = _write_config(Path(d), {"scenario_id": "x",
                                    "capacity_per_rank_per_layer": 8})  # 16*8 < 128*2
        _expect_fail(gs.load_config, p)


def test_config_rejects_bad_node_and_slug() -> None:
    with tempfile.TemporaryDirectory() as d:
        _expect_fail(gs.load_config, _write_config(Path(d), {
            "scenario_id": "x", "events": [{"step": 1, "type": "fail", "node": 9}]}))
        _expect_fail(gs.load_config, _write_config(Path(d), {"scenario_id": "bad id!"}))


# --------------------------------------------------------------------------- #
# topology helpers + locality-first routing
# --------------------------------------------------------------------------- #
def test_topology_helpers() -> None:
    assert gs.rank_to_node(0, 4) == 0 and gs.rank_to_node(7, 4) == 1
    assert gs.slot_rank(Slot(node=1, local_rank=2, slot=0), 4) == 6
    assert gs.failed_ranks({1}, 4) == [4, 5, 6, 7]


def test_choose_route_destination_locality() -> None:
    assert gs.choose_route_destination(2, [2, 8, 9], 4) == 2          # local hit
    assert gs.choose_route_destination(0, [2, 3, 8], 4) == 2          # same node, lowest
    assert gs.choose_route_destination(0, [8, 9, 13], 4) == 8         # lowest remote


# --------------------------------------------------------------------------- #
# baseline-pinned placement
# --------------------------------------------------------------------------- #
def test_baseline_pinned_place() -> None:
    loads = [1.0] * gs.generate.NUM_EXPERTS  # uniform -> every expert exactly k_min=2
    placement = gs.baseline_pinned_place(layer_id=7, expert_loads=loads,
                                         ranks_per_node=4, capacity_per_rank=16)
    for expert in range(gs.generate.NUM_EXPERTS):
        slots = placement.expert_to_slots[expert]
        ranks = {gs.slot_rank(s, 4) for s in slots}
        nodes = {s.node for s in slots}
        assert gs.baseline_owner_rank(7, expert) in ranks, f"expert {expert} owner not pinned"
        assert len(slots) == 2 and len(nodes) == 2, f"expert {expert} not on 2 distinct nodes"
    # every single-node failure leaves a live replica for every expert
    for node in range(4):
        for expert in range(gs.generate.NUM_EXPERTS):
            nodes = {s.node for s in placement.expert_to_slots[expert]}
            assert nodes - {node}, f"node {node} failure unservable for expert {expert}"


# --------------------------------------------------------------------------- #
# traffic-kind matrices
# --------------------------------------------------------------------------- #
def _single_expert_plan() -> dict[int, LayerPlan]:
    # expert 5 in layer 7: baseline owner is rank 0; replicas on node 0 (rank 0) and
    # node 1 (rank 4). All other experts present but empty (the generator iterates the
    # full expert range).
    slots = [Slot(node=0, local_rank=0, slot=0), Slot(node=1, local_rank=0, slot=0)]
    expert_to_slots = {e: [] for e in range(gs.generate.NUM_EXPERTS)}
    expert_to_slots[5] = slots
    placement = Placement(expert_to_slots=expert_to_slots,
                          slot_to_expert={s: 5 for s in slots})
    return {7: LayerPlan(layer_id=7, placement=placement)}


def test_initial_replication_sources_from_owner() -> None:
    m = gs.build_initial_replication_matrix(_single_expert_plan(), ranks_per_node=4)
    assert m[0][4] == EXPERT_STATE_BYTES          # owner(0) -> remote replica(4)
    assert gs.total_bytes(m) == EXPERT_STATE_BYTES  # owner self-copy free


def test_join_repair_restores_node() -> None:
    plans = _single_expert_plan()
    current = gs.initial_current_slots(plans)
    gs.remove_failed_node_state(current, node=1)   # node 1 fails -> rank-4 replica lost
    m, reason = gs.build_join_repair_matrix(
        plans=plans, current_slots=current, joined_node=1,
        live_nodes={0, 1, 2, 3}, ranks_per_node=4)
    assert reason is None
    assert m[0][4] == EXPERT_STATE_BYTES           # restored from live source rank 0
    assert gs.total_bytes(m) == EXPERT_STATE_BYTES


# --------------------------------------------------------------------------- #
# integration: run real example scenarios into a temp dir (need traces)
# --------------------------------------------------------------------------- #
def _run(config: ScenarioConfig) -> tuple[int, list[dict]]:
    streams, layer_loads = gs.build_workload(gs.generate.discover_trace_paths())
    plans = gs.build_layer_plans(layer_loads, config)
    with tempfile.TemporaryDirectory() as d:
        gs.SCENARIO_ROOT = Path(d)               # redirect output away from out/
        code = gs.run_scenario(config, streams, plans)
        rows = [json.loads(l) for l in
                (gs.SCENARIO_ROOT / config.scenario_id / "scenario_timeline.jsonl").open()]
    return code, rows


def test_integration_no_events() -> None:
    if not _traces_available():
        print("  SKIP test_integration_no_events (no traces in cwd)")
        return
    code, rows = _run(ScenarioConfig("no_events", 4, 16, []))
    kinds = [r["kind"] for r in rows]
    assert code == 0
    assert kinds[0] == "scenario_header"
    assert "terminal_failure" not in kinds
    assert kinds.count("all2allv") == 768  # 32 tokens * 24 MoE layers, lockstep


def test_integration_node_fail_join_survives() -> None:
    if not _traces_available():
        print("  SKIP test_integration_node_fail_join_survives (no traces in cwd)")
        return
    code, rows = _run(ScenarioConfig("node1_fail_join", 4, 16,
                                     [NodeEvent(100, "fail", 1), NodeEvent(200, "join", 1)]))
    kinds = [r["kind"] for r in rows]
    assert code == 0
    assert "terminal_failure" not in kinds
    assert kinds.count("node_event") == 2
    assert kinds.count("expert_migration") == 1
    assert kinds.count("all2allv") == 868  # 64 streams pause 100 steps -> 768 + 100
    during = [r for r in rows if r["kind"] == "all2allv" and r["step"] == 150][0]
    assert during["failed_nodes"] == [1] and during["paused_request_streams"] == 64


def test_integration_terminal_failure() -> None:
    if not _traces_available():
        print("  SKIP test_integration_terminal_failure (no traces in cwd)")
        return
    # Fail two nodes with no rejoin: some expert had both replicas on {0,1}.
    code, rows = _run(ScenarioConfig("two_fail", 4, 16,
                                     [NodeEvent(5, "fail", 0), NodeEvent(10, "fail", 1)]))
    assert code == 1
    assert rows[-1]["kind"] == "terminal_failure"


def main() -> None:
    tests = [
        test_config_valid,
        test_config_rejects_double_fail,
        test_config_rejects_join_live_node,
        test_config_rejects_unordered_events,
        test_config_rejects_indivisible_ranks,
        test_config_rejects_small_capacity,
        test_config_rejects_bad_node_and_slug,
        test_topology_helpers,
        test_choose_route_destination_locality,
        test_baseline_pinned_place,
        test_initial_replication_sources_from_owner,
        test_join_repair_restores_node,
        test_integration_no_events,
        test_integration_node_fail_join_survives,
        test_integration_terminal_failure,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nall {len(tests)} tests passed")


if __name__ == "__main__":
    main()
