from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from typing_extensions import Annotated

from toposim.analysis import analyze_matrix
from toposim.manifest import load_manifest
from toposim.report import print_batch_summary, result_to_dict, write_json
from toposim.traffic import load_matrix

console = Console()


def main(
    manifest: Annotated[Path, typer.Argument(help="JSONL scenario manifest")],
    gpus_per_server: Annotated[int, typer.Option("--gpus-per-server", help="Default GPUs per server")] = 8,
    scaleup_gbps: Annotated[float, typer.Option("--scaleup-gbps", help="Default scale-up bandwidth in Gbps")] = 3600,
    scaleout_gbps: Annotated[float, typer.Option("--scaleout-gbps", help="Default scale-out bandwidth in Gbps")] = 400,
    policy: Annotated[Optional[str], typer.Option("--policy", help="Default policy override")] = None,
    engine: Annotated[str, typer.Option("--engine", help="Default engine override")] = "auto",
    json_output: Annotated[Optional[Path], typer.Option("--json", help="Write detailed JSON output")] = None,
    summary_table: Annotated[bool, typer.Option("--summary-table", help="Print scenario summary table")] = False,
    allow_partial_server: Annotated[bool, typer.Option("--allow-partial-server", help="Default partial server padding")] = False,
    failed_gpu_mode: Annotated[str, typer.Option("--failed-gpu-mode", help="strict or warn")] = "strict",
) -> None:
    try:
        rows = load_manifest(manifest)
        scenario_results = []
        json_rows = []
        for row in rows:
            matrix = load_matrix(row.matrix)
            metadata = dict(row.metadata)
            metadata["scenario_id"] = row.id
            results, topo, warnings = analyze_matrix(
                matrix,
                gpus_per_server=row.gpus_per_server or gpus_per_server,
                scaleup_gbps=row.scaleup_gbps or scaleup_gbps,
                scaleout_gbps=row.scaleout_gbps or scaleout_gbps,
                policy=policy or row.policy,
                engine=row.engine or engine,
                topology_path=row.topology,
                allow_partial_server=(
                    row.allow_partial_server
                    if row.allow_partial_server is not None
                    else allow_partial_server
                ),
                failed_gpus=row.failed_gpus,
                failed_gpu_mode=failed_gpu_mode,
                metadata=metadata,
            )
            scenario_results.append((row.id, results))
            json_rows.append(
                {
                    "id": row.id,
                    "matrix": str(row.matrix),
                    "metadata": metadata,
                    "warnings": warnings,
                    "topology": {
                        "name": topo.name,
                        "num_gpus": topo.num_gpus,
                        "original_num_gpus": topo.original_num_gpus,
                        "num_servers": topo.num_servers,
                        "gpus_per_server": topo.gpus_per_server,
                        "scaleup_gbps": topo.scaleup_gbps,
                        "scaleout_gbps": topo.scaleout_gbps,
                        "virtual_gpus": sorted(topo.virtual_gpus),
                    },
                    "results": [result_to_dict(result) for result in results],
                }
            )
    except Exception as exc:
        console.print(f"Error: {exc}", style="red")
        raise typer.Exit(1) from exc

    # Human-readable output remains the default; --summary-table is accepted for clarity.
    print_batch_summary(console, scenario_results)

    if json_output is not None:
        write_json(json_output, {"manifest": str(manifest), "scenarios": json_rows})


def app() -> None:
    typer.run(main)


if __name__ == "__main__":
    app()
