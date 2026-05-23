from __future__ import annotations

from collections import defaultdict

from toposim.routing import Flow


def simulate_fluid(
    flows: list[Flow],
    edge_capacities: dict[str, float],
    *,
    fixed_overhead_us: float = 0.0,
) -> tuple[float, dict[str, float], dict[str, float]]:
    active = [
        Flow(
            src_gpu=flow.src_gpu,
            dst_gpu=flow.dst_gpu,
            bytes_remaining=float(flow.bytes_remaining),
            path_edges=list(flow.path_edges),
        )
        for flow in flows
        if flow.bytes_remaining > 0
    ]
    if not active:
        return fixed_overhead_us, {}, {}

    time_us = fixed_overhead_us
    edge_busy_us = defaultdict(float)
    edge_byte_pressure = defaultdict(float)

    while active:
        users: dict[str, int] = defaultdict(int)
        for flow in active:
            for edge in flow.path_edges:
                users[edge] += 1

        rates: list[float] = []
        for flow in active:
            if not flow.path_edges:
                rates.append(float("inf"))
                continue
            rate = min(edge_capacities[edge] / users[edge] for edge in flow.path_edges)
            rates.append(rate)

        delta = min(
            flow.bytes_remaining / rate
            for flow, rate in zip(active, rates, strict=True)
            if rate > 0 and rate != float("inf")
        )
        if delta <= 0:
            break

        for edge, count in users.items():
            edge_busy_us[edge] += delta
            edge_byte_pressure[edge] += delta * edge_capacities[edge] * min(1.0, count)

        remaining: list[Flow] = []
        for flow, rate in zip(active, rates, strict=True):
            flow.bytes_remaining -= delta * rate
            if flow.bytes_remaining > 1e-6:
                remaining.append(flow)
        active = remaining
        time_us += delta

    utilization = {
        edge: min(1.0, busy / max(time_us - fixed_overhead_us, 1e-9))
        for edge, busy in edge_busy_us.items()
    }
    return time_us, dict(utilization), dict(edge_byte_pressure)
