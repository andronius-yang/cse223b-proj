from __future__ import annotations

import numpy as np

from toposim.metrics import SimulationResult, compute_lower_bounds, identify_matrix_bottlenecks, server_matrix
from toposim.policies.fast import _local_intra_time_us
from toposim.topology import Topology


def simulate(matrix: np.ndarray, topology: Topology, *, metadata: dict | None = None) -> SimulationResult:
    a = server_matrix(matrix, topology)
    scaleout_us = 0.0
    stages = 0
    bandwidth = topology.server_scaleout_bytes_per_us
    for shift in range(1, topology.num_servers):
        loads = [float(a[src, (src + shift) % topology.num_servers]) for src in range(topology.num_servers)]
        stage_load = max(loads) if loads else 0.0
        if stage_load > 0:
            scaleout_us += 5.0 + stage_load / bandwidth
            stages += 1

    local_intra_us = _local_intra_time_us(matrix, topology)
    scheduling_us = 0.2 * stages
    total_us = scaleout_us + local_intra_us + scheduling_us
    bottlenecks = identify_matrix_bottlenecks(matrix, topology)
    bottlenecks["critical_resource"] = bottlenecks.get("worst_server_pair") or "scale-up local"
    bottlenecks["stages"] = stages

    return SimulationResult(
        policy="spreadout",
        engine="stage",
        completion_time_us=total_us,
        total_bytes=float(np.sum(matrix)),
        phase_breakdown_us={
            "scheduling": scheduling_us,
            "scale_out": scaleout_us,
            "local_intra": local_intra_us,
        },
        lower_bounds_us=compute_lower_bounds(matrix, topology),
        bottlenecks=bottlenecks,
        metadata=metadata or {},
    ).with_sanity_warnings()
