from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from toposim.analysis import analyze_matrix
from toposim.report import result_to_dict
from toposim.topology import Topology
from toposim.traffic import load_matrix
from toposim.units import gbps_to_bytes_per_us

PHASE_ORDER = {
    "scenario_header": -100,
    "initial_expert_replication": -10,
    "node_event": 0,
    "expert_migration": 1,
    "all2allv": 2,
}

MISSING_COLD_START_WARNING = (
    "No cold-start byte annotations found in timeline; Toposim will account for "
    "network migration matrices only. This is expected if Traffic Gen has not "
    "annotated lost-replica experts yet."
)

PHASE_FIELDS = (
    "initial_replication_us",
    "migration_network_us",
    "all2allv_us",
    "cold_start_storage_us",
)


@dataclass(slots=True)
class TimelineRow:
    raw: dict[str, Any]
    line_number: int
    timeline_dir: Path
    step: int | None
    kind: str
    matrix: str | None = None
    matrix_path: Path | None = None
    total_bytes: float | None = None
    live_nodes: list[int] | None = None
    failed_nodes: list[int] | None = None
    failed_ranks: list[int] | None = None
    live_request_streams: int | None = None
    paused_request_streams: int | None = None
    completed_request_streams: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, raw: dict[str, Any], *, line_number: int, timeline_dir: Path) -> "TimelineRow":
        kind = raw.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"line {line_number}: timeline row must contain a non-empty string kind")
        if kind not in PHASE_ORDER:
            raise ValueError(
                f"line {line_number}: unknown timeline kind {kind!r}; "
                f"expected one of {sorted(PHASE_ORDER)}"
            )

        step = raw.get("step")
        if step is not None:
            try:
                step = int(step)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"line {line_number}: step must be an integer") from exc

        matrix_value = raw.get("matrix")
        matrix_path = None
        if matrix_value is not None:
            if not isinstance(matrix_value, str) or not matrix_value:
                raise ValueError(f"line {line_number}: matrix must be a non-empty string when present")
            candidate = Path(matrix_value)
            matrix_path = candidate if candidate.is_absolute() else timeline_dir / candidate

        metadata = raw.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError(f"line {line_number}: metadata must be an object when present")

        if kind != "scenario_header" and step is None:
            raise ValueError(f"line {line_number}: {kind} rows must include step")

        return cls(
            raw=dict(raw),
            line_number=line_number,
            timeline_dir=timeline_dir,
            step=step,
            kind=kind,
            matrix=matrix_value,
            matrix_path=matrix_path,
            total_bytes=_optional_float(raw.get("total_bytes"), field_name="total_bytes", line_number=line_number),
            live_nodes=_optional_int_list(raw.get("live_nodes"), field_name="live_nodes", line_number=line_number),
            failed_nodes=_optional_int_list(raw.get("failed_nodes"), field_name="failed_nodes", line_number=line_number),
            failed_ranks=_optional_int_list(raw.get("failed_ranks"), field_name="failed_ranks", line_number=line_number),
            live_request_streams=_optional_int(
                raw.get("live_request_streams"), field_name="live_request_streams", line_number=line_number
            ),
            paused_request_streams=_optional_int(
                raw.get("paused_request_streams"), field_name="paused_request_streams", line_number=line_number
            ),
            completed_request_streams=_optional_int(
                raw.get("completed_request_streams"), field_name="completed_request_streams", line_number=line_number
            ),
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_number": self.line_number,
            "step": self.step,
            "kind": self.kind,
            "matrix": self.matrix,
            "matrix_path": str(self.matrix_path) if self.matrix_path is not None else None,
            "total_bytes": self.total_bytes,
            "live_nodes": self.live_nodes,
            "failed_nodes": self.failed_nodes,
            "failed_ranks": self.failed_ranks,
            "live_request_streams": self.live_request_streams,
            "paused_request_streams": self.paused_request_streams,
            "completed_request_streams": self.completed_request_streams,
            "metadata": self.metadata,
            "raw": self.raw,
        }


@dataclass(slots=True)
class TimelineStep:
    step: int
    rows: list[TimelineRow]


@dataclass(slots=True)
class ScenarioTimeline:
    path: Path
    header_row: TimelineRow | None
    rows: list[TimelineRow]
    steps: list[TimelineStep]
    warnings: list[str] = field(default_factory=list)

    @property
    def header(self) -> dict[str, Any]:
        if self.header_row is None:
            return {}
        return _header_payload(self.header_row)


@dataclass(slots=True)
class TimelineSimulationConfig:
    policy: str = "direct"
    gpus_per_server: int = 8
    scaleup_gbps: float = 3600
    scaleout_gbps: float = 400
    topology_path: str | Path | None = None
    allow_partial_server: bool = False
    storage_read_gbps: float = 25
    storage_fixed_overhead_us: float = 0
    expert_size_bytes: float = 123456789
    cold_start_field: str = "auto"


def parse_timeline(path: str | Path) -> ScenarioTimeline:
    path = Path(path)
    timeline_dir = path.parent
    rows: list[TimelineRow] = []
    header_row: TimelineRow | None = None

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            row = TimelineRow.from_json(payload, line_number=line_number, timeline_dir=timeline_dir)
            if row.kind == "scenario_header":
                if header_row is not None:
                    raise ValueError(f"{path}:{line_number}: duplicate scenario_header row")
                header_row = row
            else:
                rows.append(row)

    grouped: dict[int, list[TimelineRow]] = {}
    for row in rows:
        assert row.step is not None
        grouped.setdefault(row.step, []).append(row)

    steps = [
        TimelineStep(
            step=step,
            rows=sorted(step_rows, key=lambda row: (PHASE_ORDER[row.kind], row.line_number)),
        )
        for step, step_rows in sorted(grouped.items())
    ]

    warnings: list[str] = []
    if header_row is None:
        warnings.append(
            "Timeline has no scenario_header; using CLI topology defaults where explicit settings are absent."
        )

    return ScenarioTimeline(path=path, header_row=header_row, rows=rows, steps=steps, warnings=warnings)


def simulate_timeline(path: str | Path, config: TimelineSimulationConfig) -> dict[str, Any]:
    if config.policy != "direct":
        raise ValueError("timeline mode currently supports the direct policy only")

    timeline = parse_timeline(path)
    warnings = list(timeline.warnings)
    header = timeline.header
    gpus_per_server = _resolve_gpus_per_server(config.gpus_per_server)
    storage_bytes_per_us = gbps_to_bytes_per_us(config.storage_read_gbps)
    saw_cold_annotation = any(
        cold_start_bytes_for_row(row, expert_size_bytes=config.expert_size_bytes, field=config.cold_start_field)
        is not None
        for row in timeline.rows
    )
    if not saw_cold_annotation:
        _append_once(warnings, MISSING_COLD_START_WARNING)

    totals: dict[str, float] = {
        "initial_replication_us": 0.0,
        "migration_network_us": 0.0,
        "all2allv_us": 0.0,
        "cold_start_storage_us": 0.0,
        "network_repair_bytes": 0.0,
        "all2allv_bytes": 0.0,
        "cold_start_bytes": 0.0,
        "initial_replication_bytes": 0.0,
    }
    output_steps: list[dict[str, Any]] = []
    topology: Topology | None = None
    live_nodes: list[int] | None = None
    failed_nodes: list[int] | None = None
    failed_ranks: list[int] | None = None

    for step in timeline.steps:
        phase_breakdown = {field: 0.0 for field in PHASE_FIELDS}
        row_outputs: list[dict[str, Any]] = []
        kinds: list[str] = []

        for row in step.rows:
            kinds.append(row.kind)
            if row.live_nodes is not None:
                live_nodes = row.live_nodes
            if row.failed_nodes is not None:
                failed_nodes = row.failed_nodes
            if row.failed_ranks is not None:
                failed_ranks = row.failed_ranks
            _validate_state(row, header, warnings)

            row_output = row.to_dict()
            cold_start_bytes = cold_start_bytes_for_row(
                row,
                expert_size_bytes=config.expert_size_bytes,
                field=config.cold_start_field,
            )
            cold_start_bytes = 0.0 if cold_start_bytes is None else cold_start_bytes
            cold_start_us = 0.0
            if cold_start_bytes > 0:
                cold_start_us = config.storage_fixed_overhead_us + cold_start_bytes / storage_bytes_per_us
                phase_breakdown["cold_start_storage_us"] += cold_start_us
                totals["cold_start_storage_us"] += cold_start_us
                totals["cold_start_bytes"] += cold_start_bytes
            row_output["cold_start_bytes"] = cold_start_bytes
            row_output["cold_start_storage_us"] = cold_start_us

            if row.matrix_path is not None:
                matrix = load_matrix(row.matrix_path)
                matrix_total_bytes = float(np.sum(matrix))
                metadata = _simulation_metadata(row)
                metadata["active_ranks"] = ranks_for_nodes(row.live_nodes or [], gpus_per_server)
                failed_gpus = [rank_to_gpu(rank) for rank in (row.failed_ranks or [])]
                row_output["active_ranks"] = metadata["active_ranks"]
                row_output["failed_gpus"] = failed_gpus
                results, row_topology, matrix_warnings = analyze_matrix(
                    matrix,
                    gpus_per_server=gpus_per_server,
                    scaleup_gbps=config.scaleup_gbps,
                    scaleout_gbps=config.scaleout_gbps,
                    policy=config.policy,
                    engine="fluid",
                    topology_path=config.topology_path,
                    allow_partial_server=config.allow_partial_server,
                    failed_gpus=failed_gpus,
                    metadata=metadata,
                )
                if len(results) != 1:
                    raise ValueError(
                        "timeline mode requires exactly one policy; pass a concrete policy such as --policy direct"
                    )
                result = results[0]
                topology = row_topology
                for warning in [*matrix_warnings, *result.warnings]:
                    _append_once(warnings, warning)
                row_output["matrix_total_bytes"] = matrix_total_bytes
                row_output["simulation"] = result_to_dict(result)

                phase_field = _phase_field_for_row(row, warnings)
                if phase_field == "initial_replication_us":
                    phase_breakdown[phase_field] += result.completion_time_us
                    totals["initial_replication_us"] += result.completion_time_us
                    totals["initial_replication_bytes"] += matrix_total_bytes
                else:
                    phase_breakdown[phase_field] += result.completion_time_us
                    totals[phase_field] += result.completion_time_us
                    if row.kind == "expert_migration":
                        totals["network_repair_bytes"] += matrix_total_bytes
                    elif row.kind == "all2allv":
                        totals["all2allv_bytes"] += matrix_total_bytes

            row_outputs.append(row_output)

        total_step_us = sum(phase_breakdown.values())
        output_steps.append(
            {
                "step": step.step,
                "kinds": kinds,
                "rows": row_outputs,
                "phase_breakdown_us": phase_breakdown,
                "total_step_us": total_step_us,
                "live_nodes": list(live_nodes) if live_nodes is not None else [],
                "failed_nodes": list(failed_nodes) if failed_nodes is not None else [],
                "failed_ranks": list(failed_ranks) if failed_ranks is not None else [],
                "failed_gpus": [rank_to_gpu(rank) for rank in (failed_ranks or [])],
                "active_ranks": ranks_for_nodes(live_nodes or [], gpus_per_server),
            }
        )

    totals["steps"] = len(output_steps)
    totals["total_steady_state_us"] = (
        totals["migration_network_us"] + totals["all2allv_us"] + totals["cold_start_storage_us"]
    )
    totals["total_including_initialization_us"] = (
        totals["initial_replication_us"] + totals["total_steady_state_us"]
    )

    return {
        "timeline": str(timeline.path),
        "header": header,
        "policy": config.policy,
        "topology": _topology_to_dict(topology, header, gpus_per_server, config),
        "storage": {
            "storage_read_gbps": config.storage_read_gbps,
            "storage_fixed_overhead_us": config.storage_fixed_overhead_us,
            "expert_size_bytes": config.expert_size_bytes,
            "cold_start_field": config.cold_start_field,
        },
        "totals": totals,
        "steps": output_steps,
        "warnings": warnings,
    }


def rank_to_gpu(rank: int) -> int:
    if int(rank) < 0:
        raise ValueError(f"rank must be nonnegative, got {rank}")
    return int(rank)


def ranks_for_nodes(nodes: list[int], gpus_per_server: int) -> list[int]:
    if gpus_per_server <= 0:
        raise ValueError("gpus_per_server must be positive")
    ranks: list[int] = []
    for node in nodes:
        node = int(node)
        if node < 0:
            raise ValueError(f"node must be nonnegative, got {node}")
        start = node * gpus_per_server
        ranks.extend(range(start, start + gpus_per_server))
    return ranks


def cold_start_bytes_for_row(
    row: TimelineRow,
    *,
    expert_size_bytes: float | None,
    field: str = "auto",
) -> float | None:
    if field != "auto":
        value = _extract_named_field(row.raw, field)
        return None if value is None else _coerce_nonnegative_float(value, field)

    for key in ("cold_start_bytes", "lost_expert_bytes"):
        value = row.raw.get(key)
        if value is not None:
            return _coerce_nonnegative_float(value, key)

    for key in ("cold_start_bytes", "lost_expert_bytes"):
        value = row.metadata.get(key)
        if value is not None:
            return _coerce_nonnegative_float(value, f"metadata.{key}")

    for key in ("cold_start_experts", "lost_experts"):
        value = row.metadata.get(key)
        if value is not None:
            if expert_size_bytes is None:
                raise ValueError(
                    f"{key} is present on timeline line {row.line_number}, but --expert-size-bytes was not provided"
                )
            return _coerce_nonnegative_float(value, f"metadata.{key}") * float(expert_size_bytes)

    return None


def _phase_field_for_row(row: TimelineRow, warnings: list[str]) -> str:
    if row.kind == "initial_expert_replication":
        return "initial_replication_us"
    if row.kind == "expert_migration":
        return "migration_network_us"
    if row.kind == "all2allv":
        return "all2allv_us"
    if row.kind == "node_event":
        _append_once(
            warnings,
            "Encountered node_event row with a matrix; attributing its network time to migration_network_us.",
        )
        return "migration_network_us"
    raise ValueError(f"Matrix-bearing timeline kind {row.kind!r} is not supported")


def _simulation_metadata(row: TimelineRow) -> dict[str, Any]:
    metadata = dict(row.metadata)
    metadata.update(
        {
            "timeline_step": row.step,
            "timeline_kind": row.kind,
            "matrix": row.matrix,
            "live_nodes": row.live_nodes or [],
            "failed_nodes": row.failed_nodes or [],
            "failed_ranks": row.failed_ranks or [],
            "failed_gpus": [rank_to_gpu(rank) for rank in (row.failed_ranks or [])],
        }
    )
    return metadata


def _resolve_gpus_per_server(cli_gpus_per_server: int) -> int:
    if int(cli_gpus_per_server) <= 0:
        raise ValueError("gpus_per_server must be positive")
    return int(cli_gpus_per_server)


def _header_payload(row: TimelineRow) -> dict[str, Any]:
    payload = dict(row.metadata)
    for key, value in row.raw.items():
        if key not in {"kind", "metadata"}:
            payload[key] = value
    return payload


def _topology_to_dict(
    topology: Topology | None,
    header: dict[str, Any],
    gpus_per_server: int,
    config: TimelineSimulationConfig,
) -> dict[str, Any]:
    if topology is not None:
        return {
            "name": topology.name,
            "num_gpus": topology.num_gpus,
            "original_num_gpus": topology.original_num_gpus,
            "num_servers": topology.num_servers,
            "gpus_per_server": topology.gpus_per_server,
            "scaleup_gbps": topology.scaleup_gbps,
            "scaleout_gbps": topology.scaleout_gbps,
            "virtual_gpus": sorted(topology.virtual_gpus),
        }
    return {
        "name": "timeline-header",
        "num_gpus": header.get("num_ranks"),
        "original_num_gpus": header.get("num_ranks"),
        "num_servers": header.get("num_nodes"),
        "gpus_per_server": gpus_per_server,
        "scaleup_gbps": config.scaleup_gbps,
        "scaleout_gbps": config.scaleout_gbps,
        "virtual_gpus": [],
    }


def _validate_state(row: TimelineRow, header: dict[str, Any], warnings: list[str]) -> None:
    if row.live_nodes is not None and row.failed_nodes is not None:
        overlap = sorted(set(row.live_nodes) & set(row.failed_nodes))
        if overlap:
            _append_once(warnings, f"Step {row.step} has nodes marked both live and failed: {overlap}")
    num_nodes = header.get("num_nodes")
    if num_nodes is not None:
        node_values = list(row.live_nodes or []) + list(row.failed_nodes or [])
        out_of_range = sorted({node for node in node_values if node < 0 or node >= int(num_nodes)})
        if out_of_range:
            _append_once(
                warnings,
                f"Step {row.step} references node IDs outside header range 0..{int(num_nodes) - 1}: {out_of_range}",
            )


def _extract_named_field(raw: dict[str, Any], field: str) -> Any:
    if field.startswith("metadata."):
        metadata = raw.get("metadata") or {}
        if not isinstance(metadata, dict):
            return None
        return metadata.get(field.removeprefix("metadata."))
    return raw.get(field)


def _optional_int(value: Any, *, field_name: str, line_number: int) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"line {line_number}: {field_name} must be an integer") from exc


def _optional_float(value: Any, *, field_name: str, line_number: int) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"line {line_number}: {field_name} must be numeric") from exc


def _optional_int_list(value: Any, *, field_name: str, line_number: int) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"line {line_number}: {field_name} must be a list")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"line {line_number}: {field_name} must contain integers") from exc


def _coerce_nonnegative_float(value: Any, field_name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if out < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return out


def _append_once(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)
