from __future__ import annotations

import numpy as np

from toposim.engine.stage_model import stage_time_us
from toposim.metrics import SimulationResult, compute_lower_bounds, identify_matrix_bottlenecks, server_matrix
from toposim.schedule import decompose_matchings
from toposim.topology import Topology


def _local_intra_time_us(matrix: np.ndarray, topology: Topology) -> float:
    loads: list[float] = []
    for server in range(topology.num_servers):
        gpus = list(topology.gpus_in_server(server))
        tile = matrix[np.ix_(gpus, gpus)]
        server_bound = float(np.sum(tile)) / topology.server_scaleup_bytes_per_us
        gpu_send_bound = float(np.max(np.sum(tile, axis=1))) / topology.scaleup_bytes_per_us
        gpu_recv_bound = float(np.max(np.sum(tile, axis=0))) / topology.scaleup_bytes_per_us
        loads.append(max(server_bound, gpu_send_bound, gpu_recv_bound))
    return max(loads) if loads else 0.0


def _balance_and_redistribute(matrix: np.ndarray, topology: Topology) -> tuple[float, float]:
    m = topology.gpus_per_server
    balance_by_server = np.zeros(topology.num_servers, dtype=float)
    redistribute_by_server = np.zeros(topology.num_servers, dtype=float)

    for src_server in range(topology.num_servers):
        src_gpus = list(topology.gpus_in_server(src_server))
        for dst_server in range(topology.num_servers):
            if src_server == dst_server:
                continue
            dst_gpus = list(topology.gpus_in_server(dst_server))
            tile = matrix[np.ix_(src_gpus, dst_gpus)]
            total = float(np.sum(tile))
            if total <= 0:
                continue
            row_target = total / m
            col_target = total / m
            balance_by_server[src_server] += 0.5 * float(np.sum(np.abs(np.sum(tile, axis=1) - row_target)))
            redistribute_by_server[dst_server] += 0.5 * float(np.sum(np.abs(np.sum(tile, axis=0) - col_target)))

    balance_us = float(np.max(balance_by_server) / topology.server_scaleup_bytes_per_us)
    redistribute_us = float(np.max(redistribute_by_server) / topology.server_scaleup_bytes_per_us)
    return balance_us, redistribute_us


def simulate(matrix: np.ndarray, topology: Topology, *, metadata: dict | None = None) -> SimulationResult:
    a = server_matrix(matrix, topology)
    stages = decompose_matchings(a)
    balance_us, redistribute_us = _balance_and_redistribute(matrix, topology)
    scaleout_us = stage_time_us(stages, topology)
    local_intra_us = _local_intra_time_us(matrix, topology)
    scheduling_us = 0.2 * len(stages)
    total_us = balance_us + scaleout_us + redistribute_us + local_intra_us + scheduling_us

    bottlenecks = identify_matrix_bottlenecks(matrix, topology)
    bottlenecks["critical_resource"] = bottlenecks.get("worst_server_pair") or "scale-up local"
    bottlenecks["stages"] = len(stages)

    return SimulationResult(
        policy="fast",
        engine="stage",
        completion_time_us=total_us,
        total_bytes=float(np.sum(matrix)),
        phase_breakdown_us={
            "scheduling": scheduling_us,
            "balance": balance_us,
            "scale_out": scaleout_us,
            "redistribute_unhidden": redistribute_us,
            "local_intra": local_intra_us,
        },
        lower_bounds_us=compute_lower_bounds(matrix, topology),
        bottlenecks=bottlenecks,
        metadata=metadata or {},
    ).with_sanity_warnings()
