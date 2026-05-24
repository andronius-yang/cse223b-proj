from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class TrafficMatrix:
    matrix: np.ndarray
    phase: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    original_num_gpus: int | None = None

    def __post_init__(self) -> None:
        self.matrix = validate_matrix(self.matrix, zero_diagonal=True)
        if self.original_num_gpus is None:
            self.original_num_gpus = int(self.matrix.shape[0])

    @property
    def num_gpus(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def total_bytes(self) -> float:
        return float(np.sum(self.matrix))

    @property
    def nonzero_flows(self) -> int:
        return int(np.count_nonzero(self.matrix))


def load_whitespace_matrix(path: str | Path, *, zero_diagonal: bool = True) -> np.ndarray:
    rows: list[list[float]] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            try:
                rows.append([float(value) for value in line.split()])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: matrix entries must be numeric") from exc

    if not rows:
        raise ValueError(f"{path}: expected a square NxN matrix, got an empty file")

    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError(f"{path}: expected rectangular rows, got row widths {sorted(widths)}")

    return validate_matrix(np.array(rows, dtype=float), zero_diagonal=zero_diagonal)


def load_matrix(path: str | Path, *, zero_diagonal: bool = True) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return validate_matrix(np.load(path), zero_diagonal=zero_diagonal)
    if path.suffix == ".csv":
        return validate_matrix(np.loadtxt(path, delimiter=","), zero_diagonal=zero_diagonal)
    return load_whitespace_matrix(path, zero_diagonal=zero_diagonal)


def validate_matrix(matrix: np.ndarray, *, zero_diagonal: bool = True) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square NxN matrix, got shape {matrix.shape}")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("Traffic matrix cannot contain NaN or infinite values")
    if np.any(matrix < 0):
        raise ValueError("Traffic matrix cannot contain negative byte counts")
    matrix = matrix.copy()
    if zero_diagonal:
        np.fill_diagonal(matrix, 0.0)
    return matrix


def pad_matrix_to_multiple(matrix: np.ndarray, multiple: int) -> tuple[np.ndarray, int]:
    if multiple < 1:
        raise ValueError("multiple must be at least 1")
    matrix = validate_matrix(matrix, zero_diagonal=True)
    n = int(matrix.shape[0])
    remainder = n % multiple
    if remainder == 0:
        return matrix, 0
    padded_n = n + (multiple - remainder)
    padded = np.zeros((padded_n, padded_n), dtype=float)
    padded[:n, :n] = matrix
    return padded, padded_n - n


def parse_metadata_items(items: list[str] | None) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Metadata item must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Metadata key cannot be empty in {item!r}")
        metadata[key] = value.strip()
    return metadata


def validate_failed_gpus(
    matrix: np.ndarray,
    failed_gpus: list[int],
    *,
    mode: str = "strict",
) -> list[str]:
    if mode not in {"strict", "warn"}:
        raise ValueError("failed GPU mode must be 'strict' or 'warn'")
    matrix = validate_matrix(matrix, zero_diagonal=True)
    n = matrix.shape[0]
    warnings: list[str] = []
    for gpu in failed_gpus:
        if gpu < 0 or gpu >= n:
            raise ValueError(f"Failed GPU index {gpu} is outside matrix range 0..{n - 1}")
        traffic = float(np.sum(matrix[gpu, :]) + np.sum(matrix[:, gpu]))
        if traffic > 0:
            message = (
                f"Failed GPU {gpu} has {traffic:.0f} bytes of incoming/outgoing traffic. "
                "Fault recovery is outside the simulator; pass a repaired matrix instead."
            )
            if mode == "strict":
                raise ValueError(message)
            warnings.append(message)
    return warnings
