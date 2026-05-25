"""Tests for control_plane.py. Stdlib only; run with `python3 control_plane_tests.py`."""

from __future__ import annotations

from allocator import plan_layer
from control_plane import Policy, expert_ranks, repair, route_demand, total_bytes


def _demand(num_ranks: int, num_experts: int) -> list[list[float]]:
    # Skewed: every src sends most of its bytes to expert 0, a little to others.
    demand = [[0.0] * num_experts for _ in range(num_ranks)]
    for src in range(num_ranks):
        demand[src][0] = 1000.0
        for e in range(1, num_experts):
            demand[src][e] = 10.0
    return demand


# Server = node: 4 servers x 4 GPUs = 16 ranks. A server failure kills its 4 ranks.
NUM_NODES, GPUS_PER_NODE = 4, 4
NUM_RANKS, NUM_EXPERTS = NUM_NODES * GPUS_PER_NODE, 128


def _server_ranks(server: int) -> list[int]:
    return list(range(server * GPUS_PER_NODE, (server + 1) * GPUS_PER_NODE))


def test_replicated_preserves_total_bytes() -> None:
    demand = _demand(NUM_RANKS, NUM_EXPERTS)
    loads = [sum(demand[s][e] for s in range(NUM_RANKS)) for e in range(NUM_EXPERTS)]
    _, placement = plan_layer(loads, NUM_NODES, GPUS_PER_NODE, capacity=32, k_min=2)
    ranks = expert_ranks(placement, gpus_per_node=GPUS_PER_NODE)

    matrix = route_demand(demand, ranks, Policy.REPLICATED, gpus_per_server=GPUS_PER_NODE)
    expected = sum(sum(row) for row in demand)
    assert abs(total_bytes(matrix) - expected) < 1e-6, (total_bytes(matrix), expected)


def test_repair_no_collapse_single_server_failure() -> None:
    # Every expert has >= 2 replicas on distinct servers, so any single server
    # failure (4 ranks at once) is survivable.
    loads = [NUM_EXPERTS - e for e in range(NUM_EXPERTS)]  # skewed, all present
    _, placement = plan_layer(loads, NUM_NODES, GPUS_PER_NODE, capacity=32, k_min=2)
    ranks = expert_ranks(placement, gpus_per_node=GPUS_PER_NODE)

    for server in range(NUM_NODES):
        _, collapsed = repair(ranks, _server_ranks(server))
        assert collapsed == [], f"server{server} failure collapsed experts {collapsed}"


def test_failed_server_has_zero_traffic() -> None:
    # A failed server's ranks must send and receive nothing (toposim strict guardrail).
    demand = _demand(NUM_RANKS, NUM_EXPERTS)
    loads = [sum(demand[s][e] for s in range(NUM_RANKS)) for e in range(NUM_EXPERTS)]
    _, placement = plan_layer(loads, NUM_NODES, GPUS_PER_NODE, capacity=32, k_min=2)
    ranks = expert_ranks(placement, gpus_per_node=GPUS_PER_NODE)
    failed = _server_ranks(1)
    survivors, _ = repair(ranks, failed)

    matrix = route_demand(
        demand, survivors, Policy.REPLICATED, gpus_per_server=GPUS_PER_NODE, failed_ranks=failed
    )
    for rank in failed:
        assert all(matrix[rank][c] == 0.0 for c in range(NUM_RANKS)), "failed rank still sends"
        assert all(matrix[r][rank] == 0.0 for r in range(NUM_RANKS)), "failed rank still receives"


def main() -> None:
    tests = [
        test_replicated_preserves_total_bytes,
        test_repair_no_collapse_single_server_failure,
        test_failed_server_has_zero_traffic,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nall {len(tests)} tests passed")


if __name__ == "__main__":
    main()
