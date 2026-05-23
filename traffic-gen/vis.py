#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


DEFAULT_DIR = Path("out/mmlu_english_partial")
DEFAULT_PATTERN = "*_network.txt"

Matrix = list[list[int]]


def fail(message: str) -> None:
    raise SystemExit(f"traffic-vis error: {message}")


def import_plotting():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/traffic-gen-matplotlib")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LogNorm
    except ModuleNotFoundError as exc:
        if exc.name in {"matplotlib", "numpy"}:
            fail(
                "matplotlib is required. Install it with: "
                "python3 -m pip install -r requirements.txt"
            )
        raise

    return plt, np, LogNorm


def discover_inputs(args: list[str]) -> list[Path]:
    if args:
        paths = [Path(arg) for arg in args]
    else:
        paths = sorted(DEFAULT_DIR.glob(DEFAULT_PATTERN))

    if not paths:
        fail(f"no matrix files found; expected {DEFAULT_DIR / DEFAULT_PATTERN}")

    for path in paths:
        if not path.is_file():
            fail(f"{path} is not a file")

    return paths


def read_matrix(path: Path) -> Matrix:
    rows: Matrix = []

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        fail(f"could not read {path}: {exc}")

    if not lines:
        fail(f"{path} is empty")

    expected_cols: int | None = None
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if not fields:
            fail(f"{path}:{line_number} is blank")

        row: list[int] = []
        for field in fields:
            try:
                value = int(field)
            except ValueError:
                fail(f"{path}:{line_number} has non-integer value {field!r}")
            if value < 0:
                fail(f"{path}:{line_number} has negative byte value {value}")
            row.append(value)

        if expected_cols is None:
            expected_cols = len(row)
        elif len(row) != expected_cols:
            fail(
                f"{path}:{line_number} has {len(row)} columns, "
                f"expected {expected_cols}"
            )

        rows.append(row)

    if expected_cols != len(rows):
        fail(f"{path} is {len(rows)}x{expected_cols}, expected a square matrix")

    return rows


def render_heatmap(path: Path, matrix: Matrix) -> Path:
    plt, np, LogNorm = import_plotting()

    data = np.array(matrix, dtype=float)
    positives = data[data > 0]
    if positives.size == 0:
        fail(f"{path} has no positive entries for log-scale rendering")

    masked = np.ma.masked_where(data <= 0, data)
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    image = ax.imshow(
        masked,
        cmap=cmap,
        norm=LogNorm(vmin=float(positives.min()), vmax=float(positives.max())),
        interpolation="nearest",
        aspect="equal",
    )

    rank_count = len(matrix)
    ax.set_title(path.stem.replace("_", " "))
    ax.set_xlabel("Destination rank")
    ax.set_ylabel("Source rank")
    ax.set_xticks(range(rank_count))
    ax.set_yticks(range(rank_count))
    ax.tick_params(axis="both", labelsize=7)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Bytes (log scale)")

    output_path = path.with_suffix(".png")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main(argv: list[str]) -> None:
    paths = discover_inputs(argv)
    written: list[Path] = []

    for path in paths:
        matrix = read_matrix(path)
        written.append(render_heatmap(path, matrix))

    print(f"wrote {len(written)} heatmaps")
    for path in written:
        print(path)


if __name__ == "__main__":
    main(sys.argv[1:])
