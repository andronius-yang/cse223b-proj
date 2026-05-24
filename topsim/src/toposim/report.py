from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from toposim.metrics import SimulationResult
from toposim.topology import Topology


def fmt_ms(us: float) -> str:
    return f"{us / 1000.0:.2f} ms"


def fmt_gb_s(value: float) -> str:
    return f"{value:.2f} GB/s"


def fmt_slowdown(value: float) -> str:
    return f"{value:.2f}x"


def result_to_dict(result: SimulationResult) -> dict[str, Any]:
    return result.to_dict()


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def print_single_run(
    *,
    console: Console,
    matrix_path: Path,
    original_num_gpus: int,
    total_bytes: float,
    nonzero_flows: int,
    topology: Topology,
    results: list[SimulationResult],
) -> None:
    console.print("Topology-aware all-to-allv estimate", style="bold")
    console.print("")
    console.print("Matrix:", style="bold")
    console.print(f"  Path: {matrix_path}")
    console.print(f"  GPUs: {original_num_gpus}")
    console.print(f"  Total bytes: {total_bytes:.0f}")
    console.print(f"  Nonzero flows: {nonzero_flows}")
    console.print("")
    console.print("Topology:", style="bold")
    console.print(f"  Name: {topology.name}")
    console.print(f"  Servers: {topology.num_servers}")
    console.print(f"  GPUs/server: {topology.gpus_per_server}")
    console.print(f"  Scale-up bandwidth: {topology.scaleup_gbps:g} Gbps")
    console.print(f"  Scale-out bandwidth: {topology.scaleout_gbps:g} Gbps")
    if topology.virtual_gpus:
        console.print(f"  Virtual padded GPUs: {len(topology.virtual_gpus)}")
    console.print("")
    print_policy_table(console, results)
    print_notes_and_warnings(console, results)
    print_phase_breakdown(console, results)


def print_policy_table(console: Console, results: list[SimulationResult], *, scenario: bool = False) -> None:
    table = Table(title="Policy comparison" if not scenario else None)
    if scenario:
        table.add_column("Scenario")
    table.add_column("Policy")
    table.add_column("Completion")
    table.add_column("Algo BW")
    table.add_column("Slowdown vs LB")
    table.add_column("Bottleneck")

    for result in results:
        row = [
            result.policy,
            fmt_ms(result.completion_time_us),
            fmt_gb_s(result.algorithmic_bandwidth_gb_s),
            fmt_slowdown(max(result.slowdown_vs_lb, 1.0) if result.best_lower_bound_us > 0 else 1.0),
            str(result.bottlenecks.get("critical_resource", "unknown")),
        ]
        if scenario:
            row.insert(0, str(result.metadata.get("scenario_id", result.metadata.get("id", ""))))
        table.add_row(*row)

    console.print(table)
    if results:
        best = min(results, key=lambda item: item.completion_time_us)
        console.print(f"Best policy: {best.policy}")


def print_phase_breakdown(console: Console, results: list[SimulationResult]) -> None:
    for result in results:
        if not result.phase_breakdown_us:
            continue
        table = Table(title=f"{result.policy} phase breakdown")
        table.add_column("Phase")
        table.add_column("Time")
        for phase, time_us in result.phase_breakdown_us.items():
            table.add_row(phase, fmt_ms(time_us))
        console.print(table)


def print_notes_and_warnings(console: Console, results: list[SimulationResult]) -> None:
    notes: list[str] = []
    warnings: list[str] = []
    for result in results:
        notes.extend(note for note in result.notes if note not in notes)
        warnings.extend(warning for warning in result.warnings if warning not in warnings)
    for note in notes:
        console.print(f"Note: {note}", style="cyan")
    for warning in warnings:
        console.print(f"Warning: {warning}", style="yellow")


def print_batch_summary(console: Console, scenario_results: list[tuple[str, list[SimulationResult]]]) -> None:
    table = Table(title="Scenario comparison")
    table.add_column("Scenario")
    table.add_column("Policy")
    table.add_column("Completion")
    table.add_column("Slowdown vs LB")
    table.add_column("Bottleneck")
    for scenario_id, results in scenario_results:
        for result in results:
            table.add_row(
                scenario_id,
                result.policy,
                fmt_ms(result.completion_time_us),
                fmt_slowdown(max(result.slowdown_vs_lb, 1.0) if result.best_lower_bound_us > 0 else 1.0),
                str(result.bottlenecks.get("critical_resource", "unknown")),
            )
    console.print(table)
    flat = [result for _, results in scenario_results for result in results]
    print_notes_and_warnings(console, flat)
