from __future__ import annotations

from toposim.schedule import Stage
from toposim.topology import Topology


def stage_time_us(
    stages: list[Stage],
    topology: Topology,
    *,
    alpha_us: float = 5.0,
) -> float:
    total = 0.0
    bandwidth = topology.server_scaleout_bytes_per_us
    for stage in stages:
        if not stage.matching or stage.bytes_total <= 0:
            continue
        total += alpha_us + stage.bytes_total / bandwidth
    return total
