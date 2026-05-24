from __future__ import annotations

import numpy as np

from toposim.engine.fluid_model import simulate_fluid
from toposim.metrics import SimulationResult, compute_lower_bounds, identify_matrix_bottlenecks
from toposim.routing import build_edge_capacities, flows_from_matrix
from toposim.topology import Topology

NOTE = (
    "FuseLink-inspired heuristic only: what-if relay model, not a faithful "
    "NCCL/RDMA implementation."
)


def simulate(
    matrix: np.ndarray,
    topology: Topology,
    *,
    metadata: dict | None = None,
    max_indirect_nics: int = 3,
    remap_overhead_us: float = 150.0,
) -> SimulationResult:
    flows = flows_from_matrix(matrix, topology, fuselink=True)
    capacities = build_edge_capacities(
        topology,
        fuselink=True,
        max_indirect_nics=max_indirect_nics,
    )
    overhead = remap_overhead_us if flows else 0.0
    time_us, utilization, pressure = simulate_fluid(flows, capacities, fixed_overhead_us=overhead)
    critical_edges = sorted(pressure, key=pressure.get, reverse=True)[:5]
    bottlenecks = identify_matrix_bottlenecks(matrix, topology)
    bottlenecks["critical_edges"] = critical_edges
    bottlenecks["critical_resource"] = critical_edges[0] if critical_edges else "none"
    return SimulationResult(
        policy="fuselink-heuristic",
        engine="fluid",
        completion_time_us=time_us,
        total_bytes=float(np.sum(matrix)),
        phase_breakdown_us={"relay_setup": overhead, "fluid_contention": max(0.0, time_us - overhead)},
        lower_bounds_us=compute_lower_bounds(matrix, topology),
        bottlenecks=bottlenecks,
        metadata=metadata or {},
        utilization=utilization,
        notes=[NOTE],
    ).with_sanity_warnings()
