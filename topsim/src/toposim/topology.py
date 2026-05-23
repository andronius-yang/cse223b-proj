from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from toposim.units import gbps_to_bytes_per_us


@dataclass(slots=True)
class Topology:
    num_gpus: int
    gpus_per_server: int
    scaleup_gbps: float
    scaleout_gbps: float
    original_num_gpus: int | None = None
    virtual_gpus: set[int] = field(default_factory=set)
    name: str = "synthetic-two-tier"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_gpus < 1:
            raise ValueError("num_gpus must be at least 1")
        if self.gpus_per_server < 1:
            raise ValueError("gpus_per_server must be at least 1")
        if self.num_gpus % self.gpus_per_server != 0:
            raise ValueError(
                "num_gpus must be divisible by gpus_per_server for a full two-tier topology"
            )
        if self.scaleup_gbps <= 0 or self.scaleout_gbps <= 0:
            raise ValueError("scaleup_gbps and scaleout_gbps must be positive")
        if self.original_num_gpus is None:
            self.original_num_gpus = self.num_gpus

    @property
    def num_servers(self) -> int:
        return self.num_gpus // self.gpus_per_server

    @property
    def scaleup_bytes_per_us(self) -> float:
        return gbps_to_bytes_per_us(self.scaleup_gbps)

    @property
    def scaleout_bytes_per_us(self) -> float:
        return gbps_to_bytes_per_us(self.scaleout_gbps)

    @property
    def server_scaleup_bytes_per_us(self) -> float:
        return self.gpus_per_server * self.scaleup_bytes_per_us

    @property
    def server_scaleout_bytes_per_us(self) -> float:
        return self.gpus_per_server * self.scaleout_bytes_per_us

    def server_of_gpu(self, gpu: int) -> int:
        return gpu // self.gpus_per_server

    def gpus_in_server(self, server: int) -> range:
        start = server * self.gpus_per_server
        return range(start, start + self.gpus_per_server)


def synthesize_topology(
    *,
    num_gpus: int,
    gpus_per_server: int,
    scaleup_gbps: float,
    scaleout_gbps: float,
    allow_partial_server: bool = False,
) -> tuple[Topology, int]:
    if num_gpus % gpus_per_server != 0:
        if not allow_partial_server:
            raise ValueError(
                f"Matrix has {num_gpus} GPUs, which is not divisible by "
                f"--gpus-per-server={gpus_per_server}. Change --gpus-per-server, "
                "provide --topology, or use --allow-partial-server to pad virtual "
                "zero-traffic GPUs."
            )
        padded = num_gpus + (gpus_per_server - (num_gpus % gpus_per_server))
    else:
        padded = num_gpus

    virtual_gpus = set(range(num_gpus, padded))
    return (
        Topology(
            num_gpus=padded,
            original_num_gpus=num_gpus,
            gpus_per_server=gpus_per_server,
            scaleup_gbps=scaleup_gbps,
            scaleout_gbps=scaleout_gbps,
            virtual_gpus=virtual_gpus,
        ),
        padded - num_gpus,
    )


def load_topology(path: str | Path, *, matrix_num_gpus: int | None = None) -> Topology:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    gpus_per_server = int(data.get("gpus_per_server", 8))
    servers = data.get("servers")
    num_gpus = data.get("num_gpus")
    if num_gpus is None:
        if servers is None:
            if matrix_num_gpus is None:
                raise ValueError("Topology YAML must define num_gpus or servers")
            num_gpus = matrix_num_gpus
        else:
            num_gpus = int(servers) * gpus_per_server

    if matrix_num_gpus is not None and int(num_gpus) < matrix_num_gpus:
        raise ValueError(
            f"Topology {path} has {num_gpus} GPUs but matrix requires {matrix_num_gpus}"
        )

    links = data.get("links", {})
    scaleup_gbps = data.get("scaleup_gbps")
    scaleout_gbps = data.get("scaleout_gbps")
    if scaleup_gbps is None:
        scaleup_gbps = links.get("intra_gpu", {}).get("bandwidth_gbps", 3600)
    if scaleout_gbps is None:
        scaleout_gbps = links.get("inter_server", {}).get("bandwidth_gbps", 400)

    virtual_gpus = set()
    if matrix_num_gpus is not None and int(num_gpus) > matrix_num_gpus:
        virtual_gpus = set(range(matrix_num_gpus, int(num_gpus)))

    return Topology(
        num_gpus=int(num_gpus),
        original_num_gpus=matrix_num_gpus or int(num_gpus),
        gpus_per_server=gpus_per_server,
        scaleup_gbps=float(scaleup_gbps),
        scaleout_gbps=float(scaleout_gbps),
        virtual_gpus=virtual_gpus,
        name=str(data.get("name", path.stem)),
        metadata=data,
    )
