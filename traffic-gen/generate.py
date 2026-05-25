#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


NUM_RANKS = 16
NUM_EXPERTS = 128
REQUESTS_PER_RANK = 16
DECODE_STEPS = 32
HIDDEN_SIZE = 5120
DTYPE_BYTES = 2

INPUT_GLOB = "llama4-mmlu-*"
OUTPUT_DIR = Path("out/mmlu_english_partial")
PAYLOAD_BYTES = HIDDEN_SIZE * DTYPE_BYTES
BATCH_SIZE = NUM_RANKS * REQUESTS_PER_RANK

Matrix = list[list[int]]
DemandMatrix = list[list[int]]


def fail(message: str) -> None:
    raise SystemExit(f"traffic-gen error: {message}")


def new_matrix() -> Matrix:
    return [[0 for _ in range(NUM_RANKS)] for _ in range(NUM_RANKS)]


def validate_constants() -> None:
    if NUM_RANKS <= 0:
        fail("NUM_RANKS must be positive")
    if NUM_EXPERTS <= 0:
        fail("NUM_EXPERTS must be positive")
    if REQUESTS_PER_RANK <= 0:
        fail("REQUESTS_PER_RANK must be positive")
    if DECODE_STEPS <= 0:
        fail("DECODE_STEPS must be positive")
    if HIDDEN_SIZE <= 0:
        fail("HIDDEN_SIZE must be positive")
    if DTYPE_BYTES <= 0:
        fail("DTYPE_BYTES must be positive")


def discover_trace_paths() -> list[Path]:
    subject_dirs = [path for path in Path(".").glob(INPUT_GLOB) if path.is_dir()]
    trace_paths: list[Path] = []
    for subject_dir in subject_dirs:
        trace_paths.extend(path for path in subject_dir.glob("*.json") if path.is_file())
    return trace_paths


def owner_ranks(layer_id: int, expert_id: int, num_ranks: int = NUM_RANKS) -> list[int]:
    if expert_id < 0 or expert_id >= NUM_EXPERTS:
        fail(
            f"expert id {expert_id} in layer {layer_id} is outside "
            f"0..{NUM_EXPERTS - 1}"
        )

    base = NUM_EXPERTS // num_ranks
    remainder = NUM_EXPERTS % num_ranks
    larger_block_size = base + 1
    larger_block_total = larger_block_size * remainder

    if expert_id < larger_block_total:
        return [expert_id // larger_block_size]

    if base == 0:
        fail(f"expert id {expert_id} has no owner rank")

    return [remainder + ((expert_id - larger_block_total) // base)]


def choose_owner_rank(owner_rank_ids: list[int]) -> int:
    if not owner_rank_ids:
        fail("owner rank list is empty")

    rank = owner_rank_ids[0]
    if rank < 0 or rank >= NUM_RANKS:
        fail(f"chosen owner rank {rank} is outside 0..{NUM_RANKS - 1}")
    return rank


def load_trace(path: Path) -> list[Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            trace = json.load(handle)
    except json.JSONDecodeError as exc:
        fail(f"malformed JSON in {path}: {exc}")
    except OSError as exc:
        fail(f"could not read {path}: {exc}")

    if not isinstance(trace, list):
        fail(f"{path} must contain a JSON list of token entries")
    return trace


def parse_layer_id(path: Path, token_index: int, layer_key: Any) -> int:
    if not isinstance(layer_key, str):
        fail(f"{path} token {token_index} has non-string layer key {layer_key!r}")
    try:
        return int(layer_key)
    except ValueError:
        fail(f"{path} token {token_index} has non-integer layer key {layer_key!r}")


def iter_selected_expert_ids(
    path: Path,
    token_index: int,
    layer_id: int,
    selected_experts: Any,
) -> Iterator[int]:
    if not isinstance(selected_experts, list):
        fail(f"{path} token {token_index} layer {layer_id} selected_experts is not a list")
    if not selected_experts:
        fail(f"{path} token {token_index} layer {layer_id} selected_experts is empty")

    for row_index, expert_row in enumerate(selected_experts):
        if not isinstance(expert_row, list):
            fail(
                f"{path} token {token_index} layer {layer_id} row {row_index} "
                "is not a list"
            )
        if not expert_row:
            fail(
                f"{path} token {token_index} layer {layer_id} row {row_index} "
                "has no expert ids"
            )

        for expert_id in expert_row:
            if not isinstance(expert_id, int):
                fail(
                    f"{path} token {token_index} layer {layer_id} row {row_index} "
                    f"has non-integer expert id {expert_id!r}"
                )
            yield expert_id


def add_selected_experts(
    matrix: Matrix,
    path: Path,
    token_index: int,
    layer_id: int,
    src_rank: int,
    selected_experts: Any,
) -> None:
    for expert_id in iter_selected_expert_ids(path, token_index, layer_id, selected_experts):
        dst_rank = choose_owner_rank(owner_ranks(layer_id, expert_id))
        matrix[src_rank][dst_rank] += PAYLOAD_BYTES


def build_layer_matrices(trace_paths: list[Path]) -> dict[int, Matrix]:
    layer_matrices: dict[int, Matrix] = {}

    for request_index, path in enumerate(trace_paths[:BATCH_SIZE]):
        src_rank = request_index // REQUESTS_PER_RANK
        trace = load_trace(path)
        last_token_index = min(DECODE_STEPS, len(trace) - 1)

        for token_index in range(1, last_token_index + 1):
            token_entry = trace[token_index]
            if not isinstance(token_entry, dict):
                fail(f"{path} token {token_index} must be a JSON object")

            for layer_key, selected_experts in token_entry.items():
                layer_id = parse_layer_id(path, token_index, layer_key)
                if selected_experts is None:
                    continue

                matrix = layer_matrices.setdefault(layer_id, new_matrix())
                add_selected_experts(
                    matrix=matrix,
                    path=path,
                    token_index=token_index,
                    layer_id=layer_id,
                    src_rank=src_rank,
                    selected_experts=selected_experts,
                )

    return layer_matrices


def iter_routes(trace_paths: list[Path]) -> Iterator[tuple[int, int, int]]:
    for request_index, path in enumerate(trace_paths[:BATCH_SIZE]):
        src_rank = request_index // REQUESTS_PER_RANK
        trace = load_trace(path)
        last_token_index = min(DECODE_STEPS, len(trace) - 1)

        for token_index in range(1, last_token_index + 1):
            token_entry = trace[token_index]
            if not isinstance(token_entry, dict):
                fail(f"{path} token {token_index} must be a JSON object")

            for layer_key, selected_experts in token_entry.items():
                layer_id = parse_layer_id(path, token_index, layer_key)
                if selected_experts is None:
                    continue
                for expert_id in iter_selected_expert_ids(
                    path, token_index, layer_id, selected_experts
                ):
                    yield layer_id, src_rank, expert_id


def new_demand_matrix() -> DemandMatrix:
    return [[0 for _ in range(NUM_EXPERTS)] for _ in range(NUM_RANKS)]


def build_demand_matrices(trace_paths: list[Path]) -> dict[int, DemandMatrix]:
    demand: dict[int, DemandMatrix] = {}
    for layer_id, src_rank, expert_id in iter_routes(trace_paths):
        if expert_id < 0 or expert_id >= NUM_EXPERTS:
            fail(f"expert id {expert_id} in layer {layer_id} is outside 0..{NUM_EXPERTS - 1}")
        layer = demand.setdefault(layer_id, new_demand_matrix())
        layer[src_rank][expert_id] += PAYLOAD_BYTES
    return demand


def aggregate_demand(demand: dict[int, DemandMatrix]) -> DemandMatrix:
    total = new_demand_matrix()
    for layer in demand.values():
        for src in range(NUM_RANKS):
            row = layer[src]
            for expert in range(NUM_EXPERTS):
                total[src][expert] += row[expert]
    return total


def network_matrix(original: Matrix) -> Matrix:
    network = [row[:] for row in original]
    for rank in range(NUM_RANKS):
        network[rank][rank] = 0
    return network


def add_matrix(dst: Matrix, src: Matrix) -> None:
    validate_matrix_shape(src)
    validate_matrix_shape(dst)
    for row_index in range(NUM_RANKS):
        for col_index in range(NUM_RANKS):
            dst[row_index][col_index] += src[row_index][col_index]


def validate_matrix_shape(matrix: Matrix) -> None:
    if len(matrix) != NUM_RANKS:
        fail(f"matrix has {len(matrix)} rows, expected {NUM_RANKS}")
    for row_index, row in enumerate(matrix):
        if len(row) != NUM_RANKS:
            fail(f"matrix row {row_index} has {len(row)} columns, expected {NUM_RANKS}")
        for value in row:
            if not isinstance(value, int):
                fail(f"matrix row {row_index} contains non-integer value {value!r}")


def write_matrix(path: Path, matrix: Matrix) -> None:
    validate_matrix_shape(matrix)
    with path.open("w", encoding="utf-8") as handle:
        for row in matrix:
            handle.write(" ".join(str(value) for value in row))
            handle.write("\n")


def write_outputs(layer_matrices: dict[int, Matrix]) -> None:
    if not layer_matrices:
        fail("no non-null MoE layer selections found in selected traces")

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        fail(f"could not create output directory {OUTPUT_DIR}: {exc}")

    aggregate_original = new_matrix()
    aggregate_network = new_matrix()

    for layer_id in sorted(layer_matrices):
        original = layer_matrices[layer_id]
        network = network_matrix(original)

        write_matrix(OUTPUT_DIR / f"layer_{layer_id}_original.txt", original)
        write_matrix(OUTPUT_DIR / f"layer_{layer_id}_network.txt", network)

        add_matrix(aggregate_original, original)
        add_matrix(aggregate_network, network)

    write_matrix(OUTPUT_DIR / "aggregate_original.txt", aggregate_original)
    write_matrix(OUTPUT_DIR / "aggregate_network.txt", aggregate_network)


def main() -> None:
    validate_constants()

    trace_paths = discover_trace_paths()
    if len(trace_paths) < BATCH_SIZE:
        fail(
            f"found {len(trace_paths)} trace files, need at least {BATCH_SIZE} "
            f"for {NUM_RANKS} ranks * {REQUESTS_PER_RANK} requests per rank"
        )

    layer_matrices = build_layer_matrices(trace_paths)
    write_outputs(layer_matrices)
    print(f"wrote {len(layer_matrices) * 2 + 2} matrix files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
