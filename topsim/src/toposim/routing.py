from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from toposim.topology import Topology


@dataclass(slots=True)
class Flow:
    src_gpu: int
    dst_gpu: int
    bytes_remaining: float
    path_edges: list[str]


def build_edge_capacities(topology: Topology, *, fuselink: bool = False, max_indirect_nics: int = 3) -> dict[str, float]:
    capacities: dict[str, float] = {}
    for gpu in range(topology.num_gpus):
        multiplier = 1 + max_indirect_nics if fuselink else 1
        capacities[f"gpu{gpu}:nic_tx"] = min(
            multiplier * topology.scaleout_bytes_per_us,
            topology.server_scaleout_bytes_per_us,
        )
        capacities[f"gpu{gpu}:nic_rx"] = min(
            multiplier * topology.scaleout_bytes_per_us,
            topology.server_scaleout_bytes_per_us,
        )
        capacities[f"gpu{gpu}:scaleup_tx"] = topology.scaleup_bytes_per_us
        capacities[f"gpu{gpu}:scaleup_rx"] = topology.scaleup_bytes_per_us
        capacities[f"gpu{gpu}:relay_scaleup"] = topology.scaleup_bytes_per_us

    for server in range(topology.num_servers):
        capacities[f"server{server}:scaleup"] = topology.server_scaleup_bytes_per_us
        capacities[f"server{server}:scaleout_tx"] = topology.server_scaleout_bytes_per_us
        capacities[f"server{server}:scaleout_rx"] = topology.server_scaleout_bytes_per_us

    for src in range(topology.num_servers):
        for dst in range(topology.num_servers):
            if src != dst:
                capacities[f"server{src}->server{dst}:scaleout"] = topology.server_scaleout_bytes_per_us
    return capacities


def route_direct(src_gpu: int, dst_gpu: int, topology: Topology) -> list[str]:
    src_server = topology.server_of_gpu(src_gpu)
    dst_server = topology.server_of_gpu(dst_gpu)
    if src_server == dst_server:
        return [
            f"gpu{src_gpu}:scaleup_tx",
            f"server{src_server}:scaleup",
            f"gpu{dst_gpu}:scaleup_rx",
        ]
    return [
        f"gpu{src_gpu}:nic_tx",
        f"server{src_server}:scaleout_tx",
        f"server{src_server}->server{dst_server}:scaleout",
        f"server{dst_server}:scaleout_rx",
        f"gpu{dst_gpu}:nic_rx",
    ]


def route_fuselink_heuristic(src_gpu: int, dst_gpu: int, topology: Topology) -> list[str]:
    src_server = topology.server_of_gpu(src_gpu)
    dst_server = topology.server_of_gpu(dst_gpu)
    if src_server == dst_server:
        return [f"server{src_server}:scaleup"]
    return [
        f"gpu{src_gpu}:relay_scaleup",
        f"gpu{src_gpu}:nic_tx",
        f"server{src_server}:scaleout_tx",
        f"server{src_server}->server{dst_server}:scaleout",
        f"server{dst_server}:scaleout_rx",
        f"gpu{dst_gpu}:nic_rx",
        f"gpu{dst_gpu}:relay_scaleup",
    ]


def flows_from_matrix(
    matrix: np.ndarray,
    topology: Topology,
    *,
    fuselink: bool = False,
) -> list[Flow]:
    flows: list[Flow] = []
    for src in range(topology.num_gpus):
        for dst in range(topology.num_gpus):
            amount = float(matrix[src, dst])
            if amount <= 0 or src == dst:
                continue
            path = (
                route_fuselink_heuristic(src, dst, topology)
                if fuselink
                else route_direct(src, dst, topology)
            )
            flows.append(Flow(src_gpu=src, dst_gpu=dst, bytes_remaining=amount, path_edges=path))
    return flows
