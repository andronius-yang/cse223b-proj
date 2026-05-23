# traffic-gen

Minimal trace-driven AllToAllV traffic matrix generator for MoE inference routing.

The tool reads local Llama 4 MMLU expert-selection traces, maps selected experts to
owner ranks with contiguous placement, and writes byte-valued rank-to-rank traffic
matrices suitable for communication benchmarking or profiling.

## Scope

- Model context: `meta-llama/Llama-4-Maverick-17B-128E-Instruct`
- Dataset context: `core12345/MoE_expert_selection_trace`
- Local workload: all checked-in `llama4-mmlu-*` subject folders
- Fixed workload size: 16 ranks, 16 requests per rank, 32 decode steps
- Payload assumption: `5120` BF16 elements per token-to-expert route

The generator uses decode tokens only. Prefill tokens are intentionally ignored.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

`generate.py` itself uses only the Python standard library. The dependency in
`requirements.txt` is for optional heatmap rendering with `vis.py`.

## Replication

From the repository root:

```bash
python3 generate.py
```

This writes matrices under:

```text
out/mmlu_english_partial/
```

Expected outputs include one original and one network-only matrix per observed MoE
layer, plus aggregate matrices:

```text
layer_<raw_layer_id>_original.txt
layer_<raw_layer_id>_network.txt
aggregate_original.txt
aggregate_network.txt
```

Original matrices keep diagonal entries for local expert hits. Network matrices
zero the diagonal. Matrix files are 16x16 whitespace-separated decimal byte counts
with no headers or metadata.

To regenerate heatmaps for network matrices:

```bash
python3 vis.py
```

To render specific matrix files:

```bash
python3 vis.py out/mmlu_english_partial/aggregate_network.txt
```

## Notes

The MVP is intentionally parameter-free at the command line. Constants, placement,
and routing policy live in `generate.py` so runs are reproducible from the checked-in
trace folders.
