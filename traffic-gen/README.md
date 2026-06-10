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

## Failure-Aware Scenarios

Scenario mode is a separate planned entrypoint:

```bash
python3 generate_scenario.py scenarios/node1_fail_join.json
```

The scenario config is JSON:

```json
{
  "scenario_id": "node1_fail_join",
  "ranks_per_node": 4,
  "capacity_per_rank_per_layer": 16,
  "events": [
    {"step": 100, "type": "fail", "ranks": [4, 5, 6, 7]},
    {"step": 200, "type": "join", "ranks": [4, 5, 6, 7]}
  ]
}
```

The MVP should include checked-in examples at `scenarios/no_events.json` and
`scenarios/node1_fail_join.json`. The no-event scenario must complete normally.
The node 1 fail-at-step-100 join-at-step-200 scenario represents node 1 by its
rank block `[4, 5, 6, 7]` and must complete end-to-end.

Events use zero-based rank ids and absolute simulation steps. A single-rank
failure is written as `{"ranks": [5]}`; a full-node failure is written as the
node's rank block. Step `0` is the first inference tick; initial expert
replication is emitted separately with `step = -1`. The MVP allows at most one
fail/join event per step.
The default topology is 16 ranks arranged as 4 contiguous nodes with 4 ranks per
node; `traffic-gen` and `topsim` must use the same ranks-per-node value.

Expected output:

```text
out/mmlu_english_partial/scenarios/node1_fail_join/
  initial_expert_replication.txt
  scenario_timeline.jsonl
  topsim_matrix_manifest.jsonl
  step_000010_all2allv.txt
  step_000025_expert_migration.txt
  step_000025_all2allv.txt
```

Scenario matrices are network-only: diagonal entries are zero. Migration and
AllToAllV traffic are separate files linked by the same `step` in the timeline.
Disk IO repair is recorded as `lost_expert_bytes` on the join `node_event`, not
as a rank-to-rank matrix. If any live replica exists for a missing expert,
repair uses network migration instead of disk IO for that replica.
The `scenario_timeline.jsonl` is the source of truth for ordering, fail/join events,
traffic kind, and request-progress metadata. The `topsim_matrix_manifest.jsonl`
contains only matrix-bearing rows for current `topsim-batch` compatibility.
Each `topsim_matrix_manifest.jsonl` row sets `gpus_per_server` to the scenario
`ranks_per_node` value.

Fail/join events appear as explicit timeline rows. The row kind remains
`node_event` for compatibility, but event metadata records `ranks`, not `node`.
A join step can therefore contain `node_event`, `expert_migration`, and
`all2allv` rows with the same `step`. Rows with the same `step` use this phase
order: `node_event`,
`expert_migration`, then `all2allv`.
`initial_expert_replication` uses `step = -1`.

Example timeline row:

```json
{
  "step": 25,
  "kind": "all2allv",
  "matrix": "step_000025_all2allv.txt",
  "total_bytes": 123456789,
  "live_nodes": [0, 1],
  "failed_nodes": [],
  "failed_ranks": [],
  "live_request_streams": 256,
  "paused_request_streams": 0,
  "completed_request_streams": 0,
  "metadata": {
    "cursor_histogram": {"tok3_layer7": 240, "tok2_layer15": 16}
  }
}
```

Matrix paths in both manifests are relative to the scenario directory.
The first `scenario_timeline.jsonl` row is a `scenario_header` row with no
`step`; it records topology mapping metadata such as `num_ranks`,
`ranks_per_node`, and `num_nodes`.
