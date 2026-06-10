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
SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"


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


def _load_fixture(name: str) -> ScenarioConfig:
    return gs.load_config(SCENARIO_DIR / name)


def _slot_for_rank(rank: int, ranks_per_node: int = 4) -> Slot:
    return Slot(node=rank // ranks_per_node, local_rank=rank % ranks_per_node, slot=0)


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #
def test_config_accepts_single_rank_event() -> None:
    with tempfile.TemporaryDirectory() as d:
        cfg = gs.load_config(_write_config(Path(d), {
            "scenario_id": "ok", "ranks_per_node": 4,
            "capacity_per_rank_per_layer": 16,
            "events": [{"step": 100, "type": "fail", "ranks": [5]},
                       {"step": 200, "type": "join", "ranks": [5]}],
        }))
    assert cfg.ranks_per_node == 4
    assert [e.event_type for e in cfg.events] == ["fail", "join"]
    assert [e.ranks for e in cfg.events] == [(5,), (5,)]


def test_config_accepts_full_node_rank_event() -> None:
    with tempfile.TemporaryDirectory() as d:
        cfg = gs.load_config(_write_config(Path(d), {
            "scenario_id": "ok", "ranks_per_node": 4,
            "capacity_per_rank_per_layer": 16,
            "events": [{"step": 100, "type": "fail", "ranks": [4, 5, 6, 7]},
                       {"step": 200, "type": "join", "ranks": [4, 5, 6, 7]}],
        }))
    assert [e.ranks for e in cfg.events] == [(4, 5, 6, 7), (4, 5, 6, 7)]


def test_config_rejects_double_fail() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = _write_config(Path(d), {"scenario_id": "x", "events": [
            {"step": 1, "type": "fail", "ranks": [5]},
            {"step": 2, "type": "fail", "ranks": [5]}]})
        _expect_fail(gs.load_config, p)


def test_config_rejects_join_live_rank() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = _write_config(Path(d), {"scenario_id": "x", "events": [
            {"step": 1, "type": "join", "ranks": [5]}]})
        _expect_fail(gs.load_config, p)


def test_config_rejects_unordered_events() -> None:
    # group requires strictly increasing step (covers two-events-at-one-step too).
    with tempfile.TemporaryDirectory() as d:
        p = _write_config(Path(d), {"scenario_id": "x", "events": [
            {"step": 5, "type": "fail", "ranks": [0]},
            {"step": 5, "type": "fail", "ranks": [1]}]})
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


def test_config_rejects_bad_rank_list() -> None:
    with tempfile.TemporaryDirectory() as d:
        _expect_fail(gs.load_config, _write_config(Path(d), {
            "scenario_id": "x", "events": [{"step": 1, "type": "fail", "ranks": []}]}))
        _expect_fail(gs.load_config, _write_config(Path(d), {
            "scenario_id": "x", "events": [{"step": 1, "type": "fail", "ranks": [5, 5]}]}))
        _expect_fail(gs.load_config, _write_config(Path(d), {
            "scenario_id": "x", "events": [{"step": 1, "type": "fail", "ranks": [16]}]}))
        _expect_fail(gs.load_config, _write_config(Path(d), {
            "scenario_id": "x", "events": [{"step": 1, "type": "fail", "ranks": ["5"]}]}))


def test_config_rejects_bad_slug() -> None:
    with tempfile.TemporaryDirectory() as d:
        _expect_fail(gs.load_config, _write_config(Path(d), {"scenario_id": "bad id!"}))


# --------------------------------------------------------------------------- #
# topology helpers + locality-first routing
# --------------------------------------------------------------------------- #
def test_topology_helpers() -> None:
    assert gs.rank_to_node(0, 4) == 0 and gs.rank_to_node(7, 4) == 1
    assert gs.slot_rank(Slot(node=1, local_rank=2, slot=0), 4) == 6
    assert gs.failed_ranks({4, 5, 6, 7}) == [4, 5, 6, 7]
    assert gs.failed_node_ids({4, 5, 6, 7}, 4) == [1]
    assert gs.live_node_ids({5}, 4) == [0, 1, 2, 3]
    assert gs.failed_node_ids({5}, 4) == []


def test_choose_route_destination_locality() -> None:
    assert gs.choose_route_destination(2, [2, 8, 9], 4) == 2          # local hit
    assert gs.choose_route_destination(0, [2, 3, 8], 4) == 2          # same node, lowest
    assert gs.choose_route_destination(5, [0, 2, 9], 4) == 9          # circular remote
    assert gs.choose_route_destination(14, [1, 5, 9], 4) == 1         # circular wrap
    assert gs.choose_route_destination(0, [], 4) is None


# --------------------------------------------------------------------------- #
# Lazarus placement
# --------------------------------------------------------------------------- #
def test_build_layer_plans_uses_lazarus_placement() -> None:
    loads = [1.0] * gs.generate.NUM_EXPERTS  # uniform -> every expert exactly k_min=2
    config = ScenarioConfig("placement", 4, 16, "adaptive", [])
    placement = gs.build_layer_plans({7: loads}, config)[7].placement
    _, expected = gs.plan_layer(loads, 4, 4, capacity=16, k_min=gs.K_MIN, strategy="adaptive")
    assert placement.expert_to_slots == expected.expert_to_slots
    assert placement.slot_to_expert == expected.slot_to_expert
    for expert in range(gs.generate.NUM_EXPERTS):
        slots = placement.expert_to_slots[expert]
        nodes = {s.node for s in slots}
        assert len(slots) == 2 and len(nodes) == 2, f"expert {expert} not on 2 distinct nodes"
    ranks = {gs.slot_rank(s, 4) for s in placement.expert_to_slots[8]}
    assert gs.baseline_owner_rank(7, 8) not in ranks
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


def _join_only_expert_plan() -> dict[int, LayerPlan]:
    slot = Slot(node=1, local_rank=0, slot=0)
    expert_to_slots = {e: [] for e in range(gs.generate.NUM_EXPERTS)}
    expert_to_slots[5] = [slot]
    placement = Placement(expert_to_slots=expert_to_slots, slot_to_expert={slot: 5})
    return {7: LayerPlan(layer_id=7, placement=placement)}


def test_initial_replication_sources_from_owner() -> None:
    m = gs.build_initial_replication_matrix(_single_expert_plan(), ranks_per_node=4)
    assert m[0][4] == EXPERT_STATE_BYTES          # owner(0) -> remote replica(4)
    assert gs.total_bytes(m) == EXPERT_STATE_BYTES  # owner self-copy free


def test_join_repair_restores_rank() -> None:
    plans = _single_expert_plan()
    current = gs.initial_current_slots(plans)
    gs.remove_failed_rank_state(current, {4}, 4)   # rank 4 fails -> rank-4 replica lost
    m, disk_bytes = gs.build_join_repair_matrix(
        plans=plans, current_slots=current, joined_ranks={4},
        failed_ranks_set=set(), ranks_per_node=4)
    assert disk_bytes == 0
    assert m[0][4] == EXPERT_STATE_BYTES           # restored from live source rank 0
    assert gs.total_bytes(m) == EXPERT_STATE_BYTES


def test_join_repair_prefers_same_node_source() -> None:
    src = Slot(node=1, local_rank=2, slot=0)        # rank 6
    dst = Slot(node=1, local_rank=0, slot=0)        # rank 4
    expert_to_slots = {e: [] for e in range(gs.generate.NUM_EXPERTS)}
    expert_to_slots[5] = [dst, src]
    plans = {7: LayerPlan(7, Placement(expert_to_slots, {dst: 5, src: 5}))}
    current = {(7, 5): {src}}
    m, disk_bytes = gs.build_join_repair_matrix(
        plans=plans, current_slots=current, joined_ranks={4},
        failed_ranks_set=set(), ranks_per_node=4)
    assert disk_bytes == 0
    assert m[6][4] == EXPERT_STATE_BYTES
    assert dst in current[(7, 5)]


def test_join_repair_uses_disk_when_no_live_source() -> None:
    plans = _single_expert_plan()
    current: dict[tuple[int, int], set[Slot]] = {}
    m, disk_bytes = gs.build_join_repair_matrix(
        plans=plans, current_slots=current, joined_ranks={4},
        failed_ranks_set=set(), ranks_per_node=4)
    assert disk_bytes == EXPERT_STATE_BYTES
    assert gs.total_bytes(m) == 0
    assert Slot(node=1, local_rank=0, slot=0) in current[(7, 5)]
    assert gs.live_replica_ranks(current, (7, 5), set(), 4) == [4]

    m2, disk_bytes2 = gs.build_join_repair_matrix(
        plans=plans, current_slots=current, joined_ranks={4},
        failed_ranks_set=set(), ranks_per_node=4)
    assert disk_bytes2 == 0
    assert gs.total_bytes(m2) == 0
    assert gs.live_replica_ranks(current, (7, 5), set(), 4) == [4]


def test_join_node_event_reports_lost_expert_bytes() -> None:
    stream = gs.RequestStream(
        source_rank=4,
        local_request_index=0,
        path=Path("trace.json"),
        work=[gs.WorkItem(token_index=1, layer_id=7, expert_ids=(5,))],
    )
    config = ScenarioConfig("disk_join", 4, 16, "adaptive",
                            [NodeEvent(0, "fail", (4,)), NodeEvent(1, "join", (4,))])
    old_root = gs.SCENARIO_ROOT
    with tempfile.TemporaryDirectory() as d:
        gs.SCENARIO_ROOT = Path(d)
        try:
            code = gs.run_scenario(config, [stream], _join_only_expert_plan())
            rows = [json.loads(l) for l in
                    (gs.SCENARIO_ROOT / config.scenario_id / "scenario_timeline.jsonl").open()]
        finally:
            gs.SCENARIO_ROOT = old_root

    join_events = [r for r in rows
                   if r["kind"] == "node_event" and r["metadata"]["event_type"] == "join"]
    assert code == 0
    assert len(join_events) == 1
    assert join_events[0]["lost_expert_bytes"] == EXPERT_STATE_BYTES
    assert "expert_disk_io" not in [r["kind"] for r in rows]


def test_all2allv_blocks_without_partial_traffic() -> None:
    stream = gs.RequestStream(
        source_rank=0,
        local_request_index=0,
        path=Path("trace.json"),
        work=[gs.WorkItem(token_index=1, layer_id=7, expert_ids=(5, 6))],
    )
    current = {(7, 5): {Slot(node=1, local_rank=0, slot=0)}, (7, 6): set()}
    m, histogram, advancing = gs.build_all2allv_matrix(
        streams=[stream], current_slots=current, failed_ranks_set=set(), ranks_per_node=4)
    assert gs.total_bytes(m) == 0
    assert histogram == {}
    assert advancing == []


def test_one_rank_fail_pauses_only_that_rank_and_removes_only_that_rank_slots() -> None:
    streams = [
        gs.RequestStream(rank, 0, Path(f"rank{rank}.json"),
                         [gs.WorkItem(token_index=1, layer_id=7, expert_ids=(5,))])
        for rank in (4, 5, 6)
    ]
    current = {(7, 5): {_slot_for_rank(rank) for rank in (4, 5, 6)}}

    gs.remove_failed_rank_state(current, {5}, 4)
    remaining_ranks = {gs.slot_rank(slot, 4) for slot in current[(7, 5)]}
    assert remaining_ranks == {4, 6}

    counts = gs.request_counts(streams, {5})
    assert counts["live_request_streams"] == 2
    assert counts["paused_request_streams"] == 1

    _, histogram, advancing = gs.build_all2allv_matrix(
        streams=streams, current_slots=current, failed_ranks_set={5}, ranks_per_node=4)
    assert histogram == {"tok1_layer7": 2}
    assert [stream.source_rank for stream in advancing] == [4, 6]


def test_each_rank_fails_then_joins_one_at_a_time() -> None:
    slots = [_slot_for_rank(rank) for rank in range(gs.generate.NUM_RANKS)]
    expert_to_slots = {e: [] for e in range(gs.generate.NUM_EXPERTS)}
    expert_to_slots[5] = slots
    plans = {7: LayerPlan(7, Placement(expert_to_slots, {slot: 5 for slot in slots}))}
    work = [
        gs.WorkItem(token_index=index + 1, layer_id=7, expert_ids=(5,))
        for index in range(64)
    ]
    streams = [gs.RequestStream(0, 0, Path("trace.json"), work)]
    events: list[NodeEvent] = []
    for rank in range(gs.generate.NUM_RANKS):
        events.append(NodeEvent(rank * 2, "fail", (rank,)))
        events.append(NodeEvent(rank * 2 + 1, "join", (rank,)))
    config = ScenarioConfig("rank_cycle", 4, 16, "adaptive", events)

    old_root = gs.SCENARIO_ROOT
    with tempfile.TemporaryDirectory() as d:
        gs.SCENARIO_ROOT = Path(d)
        try:
            code = gs.run_scenario(config, streams, plans)
            rows = [json.loads(l) for l in
                    (gs.SCENARIO_ROOT / config.scenario_id / "scenario_timeline.jsonl").open()]
        finally:
            gs.SCENARIO_ROOT = old_root

    node_events = [row for row in rows if row["kind"] == "node_event"]
    migrations = [row for row in rows if row["kind"] == "expert_migration"]
    assert code == 0
    assert len(node_events) == 32
    assert len(migrations) == 16
    assert all("ranks" in row["metadata"] and "node" not in row["metadata"]
               for row in node_events)
    assert "terminal_failure" not in [row["kind"] for row in rows]


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
    code, rows = _run(ScenarioConfig("no_events", 4, 16, "adaptive", []))
    kinds = [r["kind"] for r in rows]
    assert code == 0
    assert kinds[0] == "scenario_header"
    assert "terminal_failure" not in kinds
    assert kinds.count("all2allv") == 768  # 32 tokens * 24 MoE layers, lockstep


def test_integration_rank_block_fail_join_survives() -> None:
    if not _traces_available():
        print("  SKIP test_integration_rank_block_fail_join_survives (no traces in cwd)")
        return
    code, rows = _run(ScenarioConfig("node1_fail_join", 4, 16, "adaptive",
                                     [NodeEvent(100, "fail", (4, 5, 6, 7)),
                                      NodeEvent(200, "join", (4, 5, 6, 7))]))
    kinds = [r["kind"] for r in rows]
    assert code == 0
    assert "terminal_failure" not in kinds
    assert kinds.count("node_event") == 2
    assert kinds.count("expert_migration") == 1
    assert kinds.count("all2allv") == 868  # 64 streams pause 100 steps -> 768 + 100
    during = [r for r in rows if r["kind"] == "all2allv" and r["step"] == 150][0]
    assert during["failed_nodes"] == [1]
    assert during["failed_ranks"] == [4, 5, 6, 7]
    assert during["paused_request_streams"] == 64
    event_metadata = [r["metadata"] for r in rows if r["kind"] == "node_event"]
    assert event_metadata == [
        {"event_type": "fail", "ranks": [4, 5, 6, 7]},
        {"event_type": "join", "ranks": [4, 5, 6, 7]},
    ]


def test_integration_rank_cycle_fixture_survives() -> None:
    if not _traces_available():
        print("  SKIP test_integration_rank_cycle_fixture_survives (no traces in cwd)")
        return
    config = _load_fixture("rank_cycle_fail_join.json")
    code, rows = _run(config)
    kinds = [r["kind"] for r in rows]
    assert code == 0
    assert "terminal_failure" not in kinds
    assert kinds.count("node_event") == 32
    assert kinds.count("expert_migration") == 16
    assert kinds.count("all2allv") > 768

    node_events = [r for r in rows if r["kind"] == "node_event"]
    assert [r["metadata"]["ranks"] for r in node_events[:4]] == [[0], [0], [1], [1]]
    assert all("node" not in r["metadata"] for r in node_events)

    during_rank0_failure = [
        r for r in rows if r["kind"] == "all2allv" and r["step"] == 104
    ][0]
    assert during_rank0_failure["failed_ranks"] == [0]
    assert during_rank0_failure["failed_nodes"] == []
    assert during_rank0_failure["live_nodes"] == [0, 1, 2, 3]
    assert during_rank0_failure["paused_request_streams"] == 16


def test_integration_terminal_failure() -> None:
    if not _traces_available():
        print("  SKIP test_integration_terminal_failure (no traces in cwd)")
        return
    # Fail two nodes with no rejoin: some expert had both replicas on {0,1}.
    code, rows = _run(ScenarioConfig("two_fail", 4, 16, "adaptive",
                                     [NodeEvent(5, "fail", (0, 1, 2, 3)),
                                      NodeEvent(10, "fail", (4, 5, 6, 7))]))
    assert code == 1
    assert rows[-1]["kind"] == "terminal_failure"


def main() -> None:
    tests = [
        test_config_accepts_single_rank_event,
        test_config_accepts_full_node_rank_event,
        test_config_rejects_double_fail,
        test_config_rejects_join_live_rank,
        test_config_rejects_unordered_events,
        test_config_rejects_indivisible_ranks,
        test_config_rejects_small_capacity,
        test_config_rejects_bad_rank_list,
        test_config_rejects_bad_slug,
        test_topology_helpers,
        test_choose_route_destination_locality,
        test_build_layer_plans_uses_lazarus_placement,
        test_initial_replication_sources_from_owner,
        test_join_repair_restores_rank,
        test_join_repair_prefers_same_node_source,
        test_join_repair_uses_disk_when_no_live_source,
        test_join_node_event_reports_lost_expert_bytes,
        test_all2allv_blocks_without_partial_traffic,
        test_one_rank_fail_pauses_only_that_rank_and_removes_only_that_rank_slots,
        test_each_rank_fails_then_joins_one_at_a_time,
        test_integration_no_events,
        test_integration_rank_block_fail_join_survives,
        test_integration_rank_cycle_fixture_survives,
        test_integration_terminal_failure,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nall {len(tests)} tests passed")


if __name__ == "__main__":
    main()
