from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ManifestRow:
    id: str
    matrix: Path
    policy: str | None = None
    engine: str | None = None
    topology: Path | None = None
    gpus_per_server: int | None = None
    scaleup_gbps: float | None = None
    scaleout_gbps: float | None = None
    allow_partial_server: bool | None = None
    failed_gpus: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_manifest(path: str | Path) -> list[ManifestRow]:
    path = Path(path)
    rows: list[ManifestRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc
            if "matrix" not in data:
                raise ValueError(f"{path}:{line_number}: manifest row requires a matrix path")
            matrix_path = Path(data["matrix"])
            if not matrix_path.is_absolute():
                matrix_path = path.parent / matrix_path
            topology_path = data.get("topology")
            topology = None
            if topology_path:
                topology = Path(topology_path)
                if not topology.is_absolute():
                    topology = path.parent / topology
            rows.append(
                ManifestRow(
                    id=str(data.get("id", matrix_path.stem)),
                    matrix=matrix_path,
                    policy=data.get("policy"),
                    engine=data.get("engine"),
                    topology=topology,
                    gpus_per_server=data.get("gpus_per_server"),
                    scaleup_gbps=data.get("scaleup_gbps"),
                    scaleout_gbps=data.get("scaleout_gbps"),
                    allow_partial_server=data.get("allow_partial_server"),
                    failed_gpus=list(data.get("failed_gpus", [])),
                    metadata=dict(data.get("metadata", {})),
                )
            )
    return rows
