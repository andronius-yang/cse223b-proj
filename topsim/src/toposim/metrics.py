from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from toposim.topology import Topology
from toposim.units import bytes_and_us_to_gb_per_s


@dataclass(slots=True)
class SimulationResult:
    policy: str
    engine: str
    completion_time_us: float
    total_bytes: float
    phase_breakdown_us: dict[str, float]
    lower_bounds_us: dict[str, float]
    bottlenecks: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    utilization: dict[str, float] = field(default_factory=dict)

    @property
    def completion_time_ms(self) -> float:
        return self.completion_time_us / 1000.0

    @property
    def algorithmic_bandwidth_gb_s(self) -> float:
        return bytes_and_us_to_gb_per_s(self.total_bytes, self.completion_time_us)

    @property
    def best_lower_bound_us(self) -> float:
        if not self.lower_bounds_us:
            return 0.0
        return max(float(value) for value in self.lower_bounds_us.values())

    @property
    def slowdown_vs_lb(self) -> float:
        lb = self.best_lower_bound_us
        if lb <= 0:
            return 1.0
        return self.completion_time_us / lb

    def with_sanity_warnings(self) -> "SimulationResult":
        if self.best_lower_bound_us > 0 and self.slowdown_vs_lb < 0.999:
            self.warnings.append(
                "slowdown_vs_lb is below 1.0; this indicates a unit mismatch, "
                "invalid lower bound, or simulator bug."
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "engine": self.engine,
            "completion_time_us": self.completion_time_us,
            "completion_time_ms": self.completion_time_ms,
            "algorithmic_bandwidth_GBps": self.algorithmic_bandwidth_gb_s,
            "total_bytes": self.total_bytes,
            "phase_breakdown_us": self.phase_breakdown_us,
            "lower_bounds_us": self.lower_bounds_us,
            "best_lower_bound_us": self.best_lower_bound_us,
            "slowdown_vs_lb": self.slowdown_vs_lb,
            "bottlenecks": self.bottlenecks,
            "utilization": self.utilization,
            "metadata": self.metadata,
            "warnings": self.warnings,
            "notes": self.notes,
        }


def server_matrix(matrix: np.ndarray, topology: Topology) -> np.ndarray:
    servers = topology.num_servers
    m = topology.gpus_per_server
    out = np.zeros((servers, servers), dtype=float)
    for s in range(servers):
        src = slice(s * m, (s + 1) * m)
        for t in range(servers):
            dst = slice(t * m, (t + 1) * m)
            out[s, t] = float(np.sum(matrix[src, dst]))
    np.fill_diagonal(out, 0.0)
    return out


def compute_lower_bounds(matrix: np.ndarray, topology: Topology) -> dict[str, float]:
    matrix = np.asarray(matrix, dtype=float)
    inter = matrix.copy()
    intra = np.zeros_like(inter)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if topology.server_of_gpu(i) == topology.server_of_gpu(j):
                intra[i, j] = inter[i, j]
                inter[i, j] = 0.0

    scaleout = topology.server_scaleout_bytes_per_us
    scaleup = topology.server_scaleup_bytes_per_us
    per_gpu_scaleup = topology.scaleup_bytes_per_us

    a = server_matrix(matrix, topology)
    server_out = float(np.max(np.sum(a, axis=1)) / scaleout) if a.size else 0.0
    server_in = float(np.max(np.sum(a, axis=0)) / scaleout) if a.size else 0.0
    total_inter = float(np.sum(inter) / (max(1, topology.num_servers) * scaleout))
    local_gpu = float(max(np.max(np.sum(intra, axis=1)), np.max(np.sum(intra, axis=0))) / per_gpu_scaleup)
    local_server = float(np.max([np.sum(intra[list(topology.gpus_in_server(s)), :]) for s in range(topology.num_servers)]) / scaleup)

    return {
        "server_send": server_out,
        "server_recv": server_in,
        "cluster_scaleout": total_inter,
        "local_gpu_scaleup": local_gpu,
        "local_server_scaleup": local_server,
    }


def identify_matrix_bottlenecks(matrix: np.ndarray, topology: Topology) -> dict[str, Any]:
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)
    a = server_matrix(matrix, topology)
    worst_pair = None
    if a.size and float(np.max(a)) > 0:
        s, t = np.unravel_index(np.argmax(a), a.shape)
        worst_pair = f"server{s}->server{t}"
    return {
        "worst_sender_gpu": int(np.argmax(row_sums)) if row_sums.size else None,
        "worst_receiver_gpu": int(np.argmax(col_sums)) if col_sums.size else None,
        "worst_sender_server": int(np.argmax(np.sum(a, axis=1))) if a.size else None,
        "worst_receiver_server": int(np.argmax(np.sum(a, axis=0))) if a.size else None,
        "worst_server_pair": worst_pair,
    }
