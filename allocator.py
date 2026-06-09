from __future__ import annotations

import heapq
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple


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


def uniform_replicas(
    expert_loads: Sequence[float], num_slots: int, k_min: int = 2
) -> List[int]:
    """Fixed, **load-agnostic** replication — the non-adaptive baseline.

    Every expert gets the same replica count, as evenly as the slot budget
    allows: ``base = num_slots // E`` each, with the ``num_slots % E`` leftover
    slots handed to the lowest expert ids (deterministic), so counts differ by at
    most one. Unlike :func:`allocate_replicas`, hot and cold experts are
    replicated identically; ``expert_loads`` is ignored and accepted only so this
    is a drop-in replacement (same signature, same ``num_slots`` invariant).

    When ``num_slots`` is a multiple of ``E`` this gives *exactly* ``num_slots/E``
    replicas per expert — e.g. ``E=128`` with ``capacity=16`` on 16 ranks
    (``num_slots=256``) is a flat 2 replicas each. Because ``base >= k_min``
    whenever ``num_slots >= E * k_min`` (the guard below), the per-expert minimum
    is always satisfied.
    """
    num_experts = len(expert_loads)
    if k_min < 0:
        raise ValueError("k_min must be non-negative")
    if num_experts == 0:
        if num_slots != 0:
            raise ValueError("num_slots must be 0 when there are no experts")
        return []
    if num_slots < num_experts * k_min:
        raise ValueError(
            f"num_slots={num_slots} < E*k_min={num_experts * k_min}: "
            "not enough slots to give every expert its minimum replicas"
        )

    base, remainder = divmod(num_slots, num_experts)
    return [base + 1 if e < remainder else base for e in range(num_experts)]


# Replication strategies: each maps (expert_loads, num_slots, k_min) -> per-expert
# replica counts summing to num_slots. "adaptive" is Lazarus (load-proportional,
# paper §4.1); "uniform" is the fixed load-agnostic baseline to compare against.
ReplicationStrategy = Callable[[Sequence[float], int, int], List[int]]

REPLICATION_STRATEGIES: Dict[str, ReplicationStrategy] = {
    "adaptive": allocate_replicas,
    "uniform": uniform_replicas,
}


def allocate_replica_counts(
    expert_loads: Sequence[float],
    num_slots: int,
    k_min: int = 2,
    strategy: str = "adaptive",
) -> List[int]:
    """Dispatch to a named replication strategy (see ``REPLICATION_STRATEGIES``).

    ``strategy="adaptive"`` is Lazarus's load-proportional allocator;
    ``strategy="uniform"`` is the fixed, load-agnostic baseline. Both return
    counts summing to ``num_slots``, so either can feed the same placement step.
    """
    try:
        allocate = REPLICATION_STRATEGIES[strategy]
    except KeyError:
        raise ValueError(
            f"unknown replication strategy {strategy!r}; "
            f"choose from {sorted(REPLICATION_STRATEGIES)}"
        ) from None
    return allocate(expert_loads, num_slots, k_min)


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

    Experts are placed in **ascending** replica order (coldest first). Each
    expert's replicas land on *distinct* nodes wherever capacity allows — the
    single-node-failure survival guarantee when `k_min >= 2`. The first `band`
    replicas (where `band = min(replica_counts)`, the survival fan-out shared by
    every expert) go to the lowest-indexed available nodes, so the most vulnerable
    experts pile onto a common low-index band of nodes. That deliberate overlap is
    the recovery-maximising property: a failure set tends to knock out the same
    experts together rather than independently endangering many scattered ones.
    Surplus replicas (beyond `band`, held only by hotter experts) go to the
    *least-loaded* node instead, which spreads hot experts across the cluster and —
    crucially — keeps capacity balanced so late-placed experts never get stranded
    on a single remaining node (which would silently violate survival).
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
    band = max(1, min(replica_counts)) if replica_counts else 1

    def candidates(used: set[int]) -> list[int]:
        free = [n for n in range(num_nodes) if node_load[n] < slots_per_node]
        distinct = [n for n in free if n not in used]
        return distinct or free  # reuse a node only when no distinct one is free

    # Coldest experts first so their survival sets share the low-index band.
    order = sorted(range(num_experts), key=lambda e: (replica_counts[e], e))

    expert_to_slots: Dict[int, List[Slot]] = {e: [] for e in range(num_experts)}
    slot_to_expert: Dict[Slot, int] = {}
    for expert_id in order:
        used: set[int] = set()
        for k in range(replica_counts[expert_id]):
            pool = candidates(used)
            if not pool:
                break
            if k < band:
                node = min(pool)  # concentrate the survival set on the shared band
            else:
                node = min(pool, key=lambda n: (node_load[n], n))  # balance surplus
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
    strategy: str = "adaptive",
) -> Tuple[List[int], Placement]:
    """Allocate replicas then place them. Convenience wrapper over both.

    ``strategy`` selects the replication policy: ``"adaptive"`` (Lazarus,
    load-proportional) or ``"uniform"`` (fixed replicas/expert). MRO placement is
    unchanged either way — only the per-expert replica *counts* differ.
    """
    num_slots = num_nodes * gpus_per_node * capacity
    replica_counts = allocate_replica_counts(
        expert_loads, num_slots, k_min=k_min, strategy=strategy
    )
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
