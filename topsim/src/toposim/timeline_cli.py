from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated

from toposim.report import write_json
from toposim.timeline import TimelineSimulationConfig, simulate_timeline

console = Console()


def main(
    timeline: Annotated[Path, typer.Argument(help="scenario_timeline.jsonl path")],
    policy: Annotated[str, typer.Option("--policy", help="Timeline mode currently supports direct only")] = "direct",
    gpus_per_server: Annotated[int, typer.Option("--gpus-per-server", help="Ranks/GPUs per server")] = 8,
    storage_read_gbps: Annotated[float, typer.Option("--storage-read-gbps", help="Storage read bandwidth in Gbps")] = 25,
    expert_size_bytes: Annotated[
        float,
        typer.Option("--expert-size-bytes", help="Uniform expert size in bytes"),
    ] = 123456789,
    scaleup_gbps: Annotated[float, typer.Option("--scaleup-gbps", help="Scale-up bandwidth in Gbps")] = 3600,
    scaleout_gbps: Annotated[float, typer.Option("--scaleout-gbps", help="Scale-out bandwidth in Gbps")] = 400,
    topology: Annotated[Optional[Path], typer.Option("--topology", help="Advanced topology YAML path")] = None,
    allow_partial_server: Annotated[
        bool,
        typer.Option("--allow-partial-server", help="Pad virtual zero-traffic GPUs for partial final server"),
    ] = False,
    json_output: Annotated[Optional[Path], typer.Option("--json", help="Write detailed JSON output")] = None,
) -> None:
    try:
        payload = simulate_timeline(
            timeline,
            TimelineSimulationConfig(
                policy=policy,
                gpus_per_server=gpus_per_server,
                scaleup_gbps=scaleup_gbps,
                scaleout_gbps=scaleout_gbps,
                topology_path=topology,
                allow_partial_server=allow_partial_server,
                storage_read_gbps=storage_read_gbps,
                expert_size_bytes=expert_size_bytes,
            ),
        )
    except Exception as exc:
        console.print(f"Error: {exc}", style="red")
        raise typer.Exit(1) from exc

    print_timeline_summary(payload)
    if json_output is not None:
        write_json(json_output, payload)


def print_timeline_summary(payload: dict) -> None:
    console.print("Timeline scenario estimate", style="bold")
    console.print(f"Timeline: {payload['timeline']}")
    console.print(f"Policy: {payload['policy']}")
    topo = payload["topology"]
    console.print(
        f"Topology: {topo.get('num_gpus')} ranks, {topo.get('gpus_per_server')} ranks/node, "
        f"{topo.get('num_servers')} nodes"
    )
    console.print("")

    totals = payload["totals"]
    table = Table(title="Steady-state summary")
    table.add_column("Total Network Migration Time (us)")
    table.add_column("Total Cold Start Disk Time (us)")
    table.add_column("Total All2Allv Time (us)")
    table.add_column("End-to-End Completion Time (us)")
    table.add_row(
        f"{totals['migration_network_us']:.6f}",
        f"{totals['cold_start_storage_us']:.6f}",
        f"{totals['all2allv_us']:.6f}",
        f"{totals['total_steady_state_us']:.6f}",
    )
    console.print(table)

    console.print("Aggregate totals:", style="bold")
    console.print(f"  Initial replication: {totals['initial_replication_us']:.6f} us")
    console.print(f"  Migration network: {totals['migration_network_us']:.6f} us")
    console.print(f"  AllToAllV: {totals['all2allv_us']:.6f} us")
    console.print(f"  Cold-start storage: {totals['cold_start_storage_us']:.6f} us")
    console.print(f"  Steady-state total: {totals['total_steady_state_us']:.6f} us")
    console.print(f"  Total including init: {totals['total_including_initialization_us']:.6f} us")
    console.print(f"  Cold-start bytes: {totals['cold_start_bytes']:.0f}")
    console.print(f"  Network repair bytes: {totals['network_repair_bytes']:.0f}")

    for warning in payload["warnings"]:
        console.print(f"Warning: {warning}", style="yellow")

def app() -> None:
    typer.run(main)


if __name__ == "__main__":
    app()
