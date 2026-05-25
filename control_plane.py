from __future__ import annotations

from enum import Enum
from typing import Dict, List, Sequence

from allocator import Placement

Matrix = List[List[float]]
DemandMatrix = Sequence[Sequence[float]]
ExpertRanks = Dict[int, List[int]]


class Policy(Enum):
    BASELINE = "baseline"        # send to the expert's single (first) owner rank
    REPLICATED = "replicated"    # split evenly across all replica ranks
    LOCALITY = "locality"        # prefer same-server replicas, else split evenly


def expert_ranks(placement: Placement, gpus_per_node: int) -> ExpertRanks:
    """Map each expert to the sorted list of distinct ranks hosting its replicas.

    rank = node * gpus_per_node + local_rank. With `gpus_per_node=1` (GPU = node),
    rank == node. Multiple capacity slots on one GPU collapse to one rank.
    """
    out: ExpertRanks = {}
    for expert_id, slots in placement.expert_to_slots.items():
        ranks = {s.node * gpus_per_node + s.local_rank for s in slots}
        out[expert_id] = sorted(ranks)
    return out


def repair(
    expert_to_ranks: ExpertRanks, failed_ranks: Sequence[int]
) -> tuple[ExpertRanks, List[int]]:
    """Drop failed ranks from every expert's replica set.

    Returns the surviving map and the list of *collapsed* experts (no replica
    left on any live rank — the model-collapse signal).
    """
    failed = set(failed_ranks)
    survivors: ExpertRanks = {
        e: [r for r in ranks if r not in failed] for e, ranks in expert_to_ranks.items()
    }
    collapsed = [e for e, ranks in survivors.items() if not ranks]
    return survivors, collapsed


def route_demand(
    demand: DemandMatrix,
    expert_to_ranks: ExpertRanks,
    policy: Policy,
    gpus_per_server: int,
    failed_ranks: Sequence[int] = (),
) -> Matrix:
    """Render the rank-to-rank traffic matrix.

    For each (src, expert) demand, pick destination rank(s) among the expert's
    *live* replicas per `policy` and split the bytes across them. Total bytes are
    preserved (a token is dispatched once) unless a replica set is empty
    (collapse) or the source rank is dead (its requests are lost).
    """
    num_ranks = len(demand)
    num_experts = len(demand[0]) if num_ranks else 0
    failed = set(failed_ranks)
    matrix: Matrix = [[0.0] * num_ranks for _ in range(num_ranks)]

    def server_of(rank: int) -> int:
        return rank // gpus_per_server

    for src in range(num_ranks):
        if src in failed:
            continue  # attention worker on a dead GPU: its requests are lost
        src_server = server_of(src)
        for expert in range(num_experts):
            payload = demand[src][expert]
            if payload <= 0:
                continue
            ranks = [r for r in expert_to_ranks.get(expert, ()) if r not in failed]
            if not ranks:
                continue  # collapsed expert: nowhere live to route (counted via repair)

            if policy is Policy.BASELINE:
                dests = ranks[:1]
            elif policy is Policy.LOCALITY:
                locals_ = [r for r in ranks if server_of(r) == src_server]
                dests = locals_ if locals_ else ranks
            else:  # REPLICATED
                dests = ranks

            share = payload / len(dests)
            for r in dests:
                matrix[src][r] += share

    return matrix


def total_bytes(matrix: Matrix) -> float:
    return sum(sum(row) for row in matrix)
