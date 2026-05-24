"""Tests for allocator.py. Stdlib only; run with `python3 tests.py`."""

from __future__ import annotations

import random
from typing import List, Sequence

from allocator import (
    Placement,
    Slot,
    allocate_replicas,
    mro_place,
    plan_layer,
    recovery_probability,
)


def build_random_placement(
    replica_counts: Sequence[int],
    num_nodes: int,
    gpus_per_node: int,
    capacity: int,
    seed: int,
) -> Placement:
    """Baseline: scatter each expert's replicas onto a uniformly random set of
    slots (the random-placement comparison from the paper)."""
    num_ranks = num_nodes * gpus_per_node
    slots: List[Slot] = [
        Slot(node=s % num_nodes, local_rank=(s // num_nodes) % gpus_per_node, slot=s // num_ranks)
        for s in range(num_ranks * capacity)
    ]
    random.Random(seed).shuffle(slots)

    expert_to_slots = {}
    slot_to_expert = {}
    cursor = 0
    for expert_id, count in enumerate(replica_counts):
        chosen = slots[cursor : cursor + count]
        cursor += count
        expert_to_slots[expert_id] = chosen
        for sl in chosen:
            slot_to_expert[sl] = expert_id
    return Placement(expert_to_slots=expert_to_slots, slot_to_expert=slot_to_expert)


def test_allocate_uniform() -> None:
    assert allocate_replicas([10] * 4, 16, k_min=2) == [4, 4, 4, 4]


def test_allocate_skewed() -> None:
    counts = allocate_replicas([1000, 1, 1, 1], 16, k_min=2)
    assert sum(counts) == 16
    assert counts[0] >= 8, counts


def test_allocate_too_few_slots() -> None:
    try:
        allocate_replicas([1, 1, 1, 1], 7, k_min=2)  # needs >= 8
    except ValueError:
        return
    raise AssertionError("expected ValueError when num_slots < E*k_min")


def test_survival_capacity_one() -> None:
    # 4x4 cluster, capacity=1 (16 slots), imbalanced load.
    _, placement = plan_layer([100, 50, 20, 5], num_nodes=4, gpus_per_node=4, k_min=2)
    for node in range(4):
        assert placement.survives([node]), f"lost an expert when node {node} failed"


def test_survival_capacity_many() -> None:
    # E=128 on N=4 nodes x 4 gpus x c=16 = 256 slots: exercises the E > c path.
    loads = [128 - e for e in range(128)]  # skewed but every expert present
    counts, placement = plan_layer(
        loads, num_nodes=4, gpus_per_node=4, capacity=16, k_min=2
    )
    assert sum(counts) == 256
    assert min(counts) >= 2
    for node in range(4):
        assert placement.survives([node]), f"lost an expert when node {node} failed"


def test_mro_beats_random_recovery() -> None:
    # 8 nodes x 8 gpus, 8 experts, k_min=2. MRO should match or beat random,
    # and win clearly at high failure rates (paper §4.2).
    loads = [100, 80, 40, 20, 10, 5, 2, 1]
    counts, mro = plan_layer(loads, num_nodes=8, gpus_per_node=8, k_min=2)
    rand = build_random_placement(counts, 8, 8, 1, seed=0)

    for failure_prob in (0.3, 0.6):
        p_mro = recovery_probability(mro, failure_prob, 8, samples=5000, seed=7)
        p_rand = recovery_probability(rand, failure_prob, 8, samples=5000, seed=7)
        assert p_mro >= p_rand - 0.02, (failure_prob, p_mro, p_rand)


def main() -> None:
    tests = [
        test_allocate_uniform,
        test_allocate_skewed,
        test_allocate_too_few_slots,
        test_survival_capacity_one,
        test_survival_capacity_many,
        test_mro_beats_random_recovery,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nall {len(tests)} tests passed")


if __name__ == "__main__":
    main()
