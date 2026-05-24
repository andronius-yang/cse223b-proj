from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from toposim.metrics import SimulationResult
from toposim.policies import direct, fast, fuselink_heuristic, spreadout
from toposim.topology import Topology, load_topology, synthesize_topology
from toposim.traffic import pad_matrix_to_multiple, validate_failed_gpus, validate_matrix

DEFAULT_POLICIES = ["direct", "spreadout", "fast"]
ALL_POLICIES = ["direct", "spreadout", "fast", "fuselink-heuristic"]
VALID_POLICIES = set(ALL_POLICIES)
VALID_ENGINES = {"auto", "stage", "fluid"}


def parse_failed_gpus(value: str | None) -> list[int]:
    if value is None or value.strip() == "":
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def expand_policies(policy: str | None) -> list[str]:
    if policy is None or policy in {"", "default"}:
        return DEFAULT_POLICIES
    if policy == "all":
        return ALL_POLICIES
    if policy not in VALID_POLICIES:
        raise ValueError(f"Unknown policy {policy!r}; expected one of {sorted(VALID_POLICIES | {'all'})}")
    return [policy]


def prepare_matrix_and_topology(
    matrix: np.ndarray,
    *,
    topology_path: str | Path | None = None,
    gpus_per_server: int = 8,
    scaleup_gbps: float = 3600,
    scaleout_gbps: float = 400,
    allow_partial_server: bool = False,
) -> tuple[np.ndarray, Topology, list[str]]:
    warnings: list[str] = []
    matrix = validate_matrix(matrix, zero_diagonal=True)
    original_n = int(matrix.shape[0])

    if topology_path is not None:
        topology = load_topology(topology_path, matrix_num_gpus=original_n)
        if topology.num_gpus > original_n:
            padded = np.zeros((topology.num_gpus, topology.num_gpus), dtype=float)
            padded[:original_n, :original_n] = matrix
            matrix = padded
            warnings.append(
                f"Matrix padded with {topology.num_gpus - original_n} virtual zero-traffic GPUs "
                f"to match topology {topology.name}."
            )
        return matrix, topology, warnings

    topology, padded_count = synthesize_topology(
        num_gpus=original_n,
        gpus_per_server=gpus_per_server,
        scaleup_gbps=scaleup_gbps,
        scaleout_gbps=scaleout_gbps,
        allow_partial_server=allow_partial_server,
    )
    if padded_count:
        matrix, _ = pad_matrix_to_multiple(matrix, gpus_per_server)
        warnings.append(
            f"Using --allow-partial-server: padded {padded_count} virtual zero-traffic GPUs."
        )
    return matrix, topology, warnings


def analyze_matrix(
    matrix: np.ndarray,
    *,
    gpus_per_server: int = 8,
    scaleup_gbps: float = 3600,
    scaleout_gbps: float = 400,
    policy: str | None = None,
    engine: str = "auto",
    topology_path: str | Path | None = None,
    allow_partial_server: bool = False,
    failed_gpus: list[int] | None = None,
    failed_gpu_mode: str = "strict",
    metadata: dict[str, Any] | None = None,
) -> tuple[list[SimulationResult], Topology, list[str]]:
    if engine not in VALID_ENGINES:
        raise ValueError(f"Unknown engine {engine!r}; expected auto, stage, or fluid")

    original_matrix = validate_matrix(matrix, zero_diagonal=True)
    warnings = validate_failed_gpus(
        original_matrix,
        failed_gpus or [],
        mode=failed_gpu_mode,
    )
    prepared_matrix, topology, topology_warnings = prepare_matrix_and_topology(
        original_matrix,
        topology_path=topology_path,
        gpus_per_server=gpus_per_server,
        scaleup_gbps=scaleup_gbps,
        scaleout_gbps=scaleout_gbps,
        allow_partial_server=allow_partial_server,
    )
    warnings.extend(topology_warnings)

    results: list[SimulationResult] = []
    for selected_policy in expand_policies(policy):
        if selected_policy == "direct":
            if engine == "stage":
                raise ValueError("direct policy supports fluid engine only")
            result = direct.simulate(prepared_matrix, topology, metadata=metadata)
        elif selected_policy == "spreadout":
            if engine == "fluid":
                raise ValueError("spreadout policy supports stage engine only")
            result = spreadout.simulate(prepared_matrix, topology, metadata=metadata)
        elif selected_policy == "fast":
            if engine == "fluid":
                raise ValueError("fast policy supports stage engine only")
            result = fast.simulate(prepared_matrix, topology, metadata=metadata)
        elif selected_policy == "fuselink-heuristic":
            if engine == "stage":
                raise ValueError("fuselink-heuristic supports fluid engine only")
            result = fuselink_heuristic.simulate(prepared_matrix, topology, metadata=metadata)
        else:
            raise ValueError(f"Unhandled policy {selected_policy}")
        result.warnings[:0] = warnings
        results.append(result.with_sanity_warnings())
    return results, topology, warnings
