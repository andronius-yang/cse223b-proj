from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from typing_extensions import Annotated

from toposim.analysis import analyze_matrix, parse_failed_gpus
from toposim.report import print_single_run, result_to_dict, write_json
from toposim.traffic import load_matrix, parse_metadata_items

console = Console()


def main(
    matrix: Annotated[Path, typer.Argument(help="Whitespace-delimited NxN byte matrix")],
    gpus_per_server: Annotated[int, typer.Option("--gpus-per-server", help="GPUs per server")] = 8,
    scaleup_gbps: Annotated[float, typer.Option("--scaleup-gbps", help="Scale-up bandwidth in Gbps")] = 3600,
    scaleout_gbps: Annotated[float, typer.Option("--scaleout-gbps", help="Scale-out bandwidth in Gbps")] = 400,
    policy: Annotated[Optional[str], typer.Option("--policy", help="direct, spreadout, fast, fuselink-heuristic, or all")] = None,
    engine: Annotated[str, typer.Option("--engine", help="auto, stage, or fluid")] = "auto",
    topology: Annotated[Optional[Path], typer.Option("--topology", help="Advanced topology YAML path")] = None,
    json_output: Annotated[Optional[Path], typer.Option("--json", help="Write detailed JSON output")] = None,
    allow_partial_server: Annotated[bool, typer.Option("--allow-partial-server", help="Pad virtual zero-traffic GPUs for a partial final server")] = False,
    failed_gpus: Annotated[Optional[str], typer.Option("--failed-gpus", help="Comma-separated failed GPU guardrail list")] = None,
    failed_gpu_mode: Annotated[str, typer.Option("--failed-gpu-mode", help="strict or warn")] = "strict",
    metadata: Annotated[Optional[List[str]], typer.Option("--metadata", help="Scenario metadata key=value")] = None,
) -> None:
    try:
        raw_matrix = load_matrix(matrix)
        meta = parse_metadata_items(metadata)
        results, topo, _warnings = analyze_matrix(
            raw_matrix,
            gpus_per_server=gpus_per_server,
            scaleup_gbps=scaleup_gbps,
            scaleout_gbps=scaleout_gbps,
            policy=policy,
            engine=engine,
            topology_path=topology,
            allow_partial_server=allow_partial_server,
            failed_gpus=parse_failed_gpus(failed_gpus),
            failed_gpu_mode=failed_gpu_mode,
            metadata=meta,
        )
    except Exception as exc:
        console.print(f"Error: {exc}", style="red")
        raise typer.Exit(1) from exc

    print_single_run(
        console=console,
        matrix_path=matrix,
        original_num_gpus=int(raw_matrix.shape[0]),
        total_bytes=float(raw_matrix.sum()),
        nonzero_flows=int((raw_matrix > 0).sum()),
        topology=topo,
        results=results,
    )
    if json_output is not None:
        write_json(
            json_output,
            {
                "matrix": str(matrix),
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
            },
        )


def app() -> None:
    typer.run(main)


if __name__ == "__main__":
    app()
