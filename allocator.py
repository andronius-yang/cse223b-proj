from __future__ import annotations

import heapq
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class Slot:
    """A physical expert hosting location.

    ``slot`` indexes the per-GPU capacity dimension ``0..c-1``; ``slot=0`` is the
    ``c=1`` special case.
    """

    node: int
    local_rank: int
    slot: int = 0


@dataclass
class Placement:
    expert_to_slots: Dict[int, List[Slot]]
    slot_to_expert: Dict[Slot, int]

    def replicas(self, expert_id: int) -> int:
        return len(self.expert_to_slots.get(expert_id, ()))

    def survives(self, failed_nodes: Sequence[int]) -> bool:
        """True iff every expert keeps at least one replica on a non-failed node."""
        failed = set(failed_nodes)
        for slots in self.expert_to_slots.values():
            if all(s.node in failed for s in slots):
                return False
        return True


def allocate_replicas(
    expert_loads: Sequence[float], num_slots: int, k_min: int = 2
) -> List[int]:
    """Greedy makespan-minimising replica allocation (paper §4.1).

    Start every expert at ``k_min`` replicas, then hand each remaining slot to
    whichever expert currently has the largest ``c_i / r_i``. Deterministic:
    ties break on the smaller expert id.
    """
    num_experts = len(expert_loads)
    if k_min < 0:
        raise ValueError("k_min must be non-negative")
    if num_slots < num_experts * k_min:
        raise ValueError(
            f"num_slots={num_slots} < E*k_min={num_experts * k_min}: "
            "not enough slots to give every expert its minimum replicas"
        )

    counts = [k_min] * num_experts

    def priority(expert_id: int) -> float:
        r = counts[expert_id]
        load = expert_loads[expert_id]
        return load / r if r > 0 else float("inf")

    # Min-heap of (-priority, expert_id): largest c_i/r_i pops first, ties by id.
    heap: List[Tuple[float, int]] = [
        (-priority(e), e) for e in range(num_experts)
    ]
    heapq.heapify(heap)

    for _ in range(num_slots - num_experts * k_min):
        _, expert_id = heapq.heappop(heap)
        counts[expert_id] += 1
        heapq.heappush(heap, (-priority(expert_id), expert_id))

    return counts


def mro_place(
    replica_counts: Sequence[int],
    num_nodes: int,
    gpus_per_node: int,
    capacity: int = 1,
) -> Placement:
    """Maximum Rank Overlap placement (paper §4.2), generalised to ``capacity = c``
    slots per GPU.

    Failure is per **node** (a node failure kills all its GPUs), so what governs
    recovery is the *set of nodes* each expert occupies, not which GPU within a
    node. Each node holds ``slots_per_node = gpus_per_node * capacity`` replicas.

    Experts are placed in **ascending** replica order; each expert greedily takes
    the lowest-indexed nodes that still have spare capacity, preferring nodes it
    does not already occupy (so its replicas land on distinct nodes — the
    single-node-failure survival guarantee). Placing the *coldest* experts first
    concentrates them onto a shared low-index prefix of nodes, maximising node-set
    overlap: a given failure set tends to knock out the same few experts together
    rather than independently endangering many scattered ones. That overlap is the
    recovery-maximising property; hot experts then spread across the remaining
    capacity on all nodes and survive regardless.
    """
    num_experts = len(replica_counts)
    slots_per_node = gpus_per_node * capacity
    total_slots = num_nodes * slots_per_node
    if sum(replica_counts) != total_slots:
        raise ValueError(
            f"sum(replica_counts)={sum(replica_counts)} != "
            f"num_nodes*gpus_per_node*capacity={total_slots}"
        )

    node_load = [0] * num_nodes  # replicas placed on each node so far

    def take_node(used: set[int]) -> int:
        # Lowest-index node with free capacity not yet used by this expert; fall
        # back to lowest-index node with any free capacity (only needed when an
        # expert has more replicas than there are distinct free nodes).
        fallback = -1
        for node in range(num_nodes):
            if node_load[node] >= slots_per_node:
                continue
            if fallback < 0:
                fallback = node
            if node not in used:
                return node
        return fallback

    # Place coldest experts first so they share the low-index prefix of nodes.
    order = sorted(range(num_experts), key=lambda e: (replica_counts[e], e))

    expert_to_slots: Dict[int, List[Slot]] = {e: [] for e in range(num_experts)}
    slot_to_expert: Dict[Slot, int] = {}
    for expert_id in order:
        used: set[int] = set()
        for _ in range(replica_counts[expert_id]):
            node = take_node(used)
            used.add(node)
            t = node_load[node]  # this node's next free slot index
            node_load[node] += 1
            sl = Slot(node=node, local_rank=t % gpus_per_node, slot=t // gpus_per_node)
            expert_to_slots[expert_id].append(sl)
            slot_to_expert[sl] = expert_id

    return Placement(expert_to_slots=expert_to_slots, slot_to_expert=slot_to_expert)


def plan_layer(
    expert_loads: Sequence[float],
    num_nodes: int,
    gpus_per_node: int,
    capacity: int = 1,
    k_min: int = 2,
) -> Tuple[List[int], Placement]:
    """Allocate replicas then place them. Convenience wrapper over both."""
    num_slots = num_nodes * gpus_per_node * capacity
    replica_counts = allocate_replicas(expert_loads, num_slots, k_min=k_min)
    placement = mro_place(replica_counts, num_nodes, gpus_per_node, capacity=capacity)
    return replica_counts, placement


def recovery_probability(
    placement: Placement,
    failure_prob: float,
    num_nodes: int,
    samples: int,
    seed: int,
) -> float:
    """Monte Carlo estimate of P(placement survives an iid node-failure draw).

    Each node fails independently with probability ``failure_prob``. Used only in
    tests / experiments.
    """
    rng = random.Random(seed)
    survived = 0
    for _ in range(samples):
        failed = [n for n in range(num_nodes) if rng.random() < failure_prob]
        if placement.survives(failed):
            survived += 1
    return survived / samples
