# Project Init

This project builds a minimal trace-driven rank-to-rank AllToAllV traffic matrix generator from realistic MoE expert-selection traces.

The MVP implementation is a root-level Python script named `generate.py`.

## Source Context

- Paper: "Patterns behind Chaos: Forecasting Data Movement for Efficient Large-Scale MoE LLM Inference" (`arXiv:2510.05497`).
- Dataset: `core12345/MoE_expert_selection_trace` on Hugging Face.
- Local partial dataset is already checked into this workspace as:
  - `llama4-mmlu-abstract_algebra/`
  - `llama4-mmlu-anatomy/`
  - `llama4-mmlu-astronomy/`

## MVP Choice

Use `meta-llama/Llama-4-Maverick-17B-128E-Instruct` on English `mmlu`.

Reason: the paper's strongest simple skew example is Llama4 layer 7 on MMLU, where a subset of experts is activated over 16x more frequently than average. The paper also analyzes all 57 MMLU subjects and notes that Llama selects one expert per MoE layer, avoiding top-k co-activation complexity while still exposing destination-rank skew.

The MVP workload is the union of all local Llama4 English MMLU subject folders. Treat this as `mmlu_english_partial`: enough local requests to populate a 16-rank workload with 16 requests per rank, but not a full 57-subject MMLU dataset.

Discover input subject folders by globbing local `llama4-mmlu-*` directories. When filling the global batch, use simple filesystem traversal order over the discovered trace files. Do not shuffle, round-robin across subjects, or rebalance subject representation for the MVP.

The selected batch size is exactly `NUM_RANKS * REQUESTS_PER_RANK`. If fewer JSON trace files are available, fail clearly instead of generating a smaller partial workload.

Fail clearly on unexpected selected-input conditions, including malformed JSON, malformed token/layer structure, non-list `selected_experts` where a matrix is expected, non-integer expert ids, out-of-range expert ids, or output matrix shape mismatches. Do not skip bad selected traces silently.

Assign the selected requests to source ranks in contiguous chunks: the first `REQUESTS_PER_RANK` requests belong to rank 0, the next `REQUESTS_PER_RANK` requests belong to rank 1, and so on.

## Minimal Model

Keep only these first-class parameters and payload assumptions:

- `NUM_RANKS = 16`
- `NUM_EXPERTS = 128`
- `REQUESTS_PER_RANK = 16`
- `DECODE_STEPS = 32`
- `HIDDEN_SIZE = 5120` as the assumed activation width
- `DTYPE_BYTES = 2` for BF16-sized activation elements

Hard-code these values as constants in the MVP Python script. Do not add CLI flags until explicitly requested.

Use decode tokens only. In each JSON trace, element `0` is prefill and elements `1+` are decode tokens. Prefill is intentionally out of scope for the MVP because variable prompt lengths complicate the traffic story.

For `DECODE_STEPS = 32`, consume decode tokens `1..32` from each selected trace. If a trace is shorter, consume only `1..min(32, len(trace) - 1)`; shorter traces contribute fewer routes rather than failing the run.

## Placement Contract

Use contiguous expert placement for now:

```python
def owner_ranks(layer_id: int, expert_id: int, num_ranks: int = 16) -> list[int]:
    base = NUM_EXPERTS // num_ranks
    remainder = NUM_EXPERTS % num_ranks
    # Earliest ranks receive one extra expert until the remainder is exhausted.
    ...
```

Assign experts contiguously. Each rank owns `NUM_EXPERTS // NUM_RANKS` experts, and any remainder experts are distributed evenly by giving one extra expert to the earliest ranks.

Treat expert ids as zero-based. Valid expert ids are `0..NUM_EXPERTS-1`; anything outside that range is unexpected selected input and must fail clearly.

Keep the return type as `list[int]` so later expert replication experiments can return multiple owner ranks without changing the placement contract. Route accounting must choose one destination rank through a separate policy function:

```python
def choose_owner_rank(owner_rank_ids: list[int]) -> int:
    return owner_rank_ids[0]
```

For the MVP, this policy returns the first owner. It must fail clearly if the owner list is empty. Later load-balancing, replication, or fault-tolerance experiments should change only this decision policy.

## Traffic Matrix Contract

Build per-layer route-count matrices internally:

```text
M[layer][src_rank][dst_rank] = number of token-to-expert routes
```

Because the model is fixed, the MoE-gated layer set is fixed. Generate matrices only for layers with non-null expert selections; do not emit all-zero matrices for dense/null layers.

For the current Llama4 traces, assume top-1 routing. Still parse the dataset's `selected_experts` value as a 2D matrix and count every expert id in every row. This keeps the MVP compatible with future same-model traces that contain top-k selections. In decode, the expected current shape is one row per output token, but a row with `k` expert ids contributes `k` routes.

Top-k selections scale traffic with `k`. For example, if one decode token from source rank `s` selects experts `[1, 24]`, add `HIDDEN_SIZE * DTYPE_BYTES` bytes to the owner rank of expert 1 and another `HIDDEN_SIZE * DTYPE_BYTES` bytes to the owner rank of expert 24. Do not split one activation across selected experts.

Source load is controlled by assigning exactly `REQUESTS_PER_RANK` selected requests to each source rank.

Keep diagonal entries in the canonical matrix because they represent local expert hits. If a network-only view is needed, derive it separately by zeroing the diagonal.

## Output Contract

Write generated matrix files under `out/mmlu_english_partial/` by default.

Write two whitespace/newline separated traffic matrices per layer:

Rows are source ranks, columns are destination ranks, and each entry is bytes transferred:

```text
bytes[src_rank][dst_rank] = routes[src_rank][dst_rank] * HIDDEN_SIZE * DTYPE_BYTES
```

Each `.txt` matrix file must contain exactly `NUM_RANKS` lines, each with `NUM_RANKS` decimal integer byte entries separated by single spaces and terminated by a trailing newline. Do not include headers, comments, commas, or metadata in matrix files.

Do not write a metadata sidecar for the MVP. The output directory should contain only matrix `.txt` files.

On rerun, overwrite the expected matrix `.txt` files owned by this generator: `layer_*_original.txt`, `layer_*_network.txt`, `aggregate_original.txt`, and `aggregate_network.txt`. Do not delete or clean the whole output directory.

The original matrix keeps diagonal bytes for local expert hits. The network traffic matrix is derived from the original matrix by zeroing diagonal entries.

Encode the raw dataset layer key and matrix variant in the filename, for example `layer_7_original.txt` and `layer_7_network.txt`. Do not renumber MoE layers. Also write aggregate variants summed over layers, for example `aggregate_original.txt` and `aggregate_network.txt`. Compute `aggregate_original.txt` by summing per-layer original matrices and compute `aggregate_network.txt` by summing per-layer network matrices.

Write per-layer files in numeric raw-layer-key order.

Do not add CSV, correctness checking, plots, replication policy, or communication simulation until explicitly requested.

## Design Boundary

Treat the source files as expert-selection traces, not GPU communication traces. The generator derives a communication demand matrix from routing decisions plus an explicit expert-placement function.

The MVP can be a simple Python script, but still take performance seriously: process only the selected batch, avoid loading unused trace files, keep matrices as compact integer lists, and do not add expensive dataframe or plotting dependencies.
