from __future__ import annotations

import numpy as np

from toposim.engine.fluid_model import simulate_fluid
from toposim.metrics import SimulationResult, compute_lower_bounds, identify_matrix_bottlenecks
from toposim.routing import build_edge_capacities, flows_from_matrix
from toposim.topology import Topology


def simulate(matrix: np.ndarray, topology: Topology, *, metadata: dict | None = None) -> SimulationResult:
    flows = flows_from_matrix(matrix, topology, fuselink=False)
    capacities = build_edge_capacities(topology, fuselink=False)
    time_us, utilization, pressure = simulate_fluid(flows, capacities)
    critical_edges = sorted(pressure, key=pressure.get, reverse=True)[:5]
    bottlenecks = identify_matrix_bottlenecks(matrix, topology)
    bottlenecks["critical_edges"] = critical_edges
    bottlenecks["critical_resource"] = critical_edges[0] if critical_edges else "none"
    return SimulationResult(
        policy="direct",
        engine="fluid",
        completion_time_us=time_us,
        total_bytes=float(np.sum(matrix)),
        phase_breakdown_us={"fluid_contention": time_us},
        lower_bounds_us=compute_lower_bounds(matrix, topology),
        bottlenecks=bottlenecks,
        metadata=metadata or {},
        utilization=utilization,
    ).with_sanity_warnings()
