# Toposim

Matrix-first topology-aware all-to-allv simulator for GPU communication studies.

The core contract is:

```bash
uv run toposim matrix.txt \
  --gpus-per-server 8 \
  --scaleup-gbps 3600 \
  --scaleout-gbps 400 \
  --policy all
```

`matrix.txt` is a whitespace-delimited `N x N` byte matrix. Entry `M[i][j]`
is the number of bytes GPU `i` sends to GPU `j`. Blank lines and `#` comments
are allowed. Diagonal entries are treated as local/no communication and zeroed.

## Commands

Single matrix:

```bash
uv run toposim data/matrices/toy_16gpu.txt --policy fast
```

Batch/scenario manifest:

```bash
uv run toposim-batch demo/scenario_manifest.jsonl \
  --policy fast \
  --json results/scenarios.json
```

Run all policies:

```bash
uv run toposim-batch demo/scenario_manifest.jsonl \
  --policy all \
  --summary-table
```

## Policies

- `direct`: fluid contention model over direct GPU/NIC/server paths.
- `spreadout`: shifted server-level matchings.
- `fast`: FAST-style server collapse, matching decomposition, balance, scale-out, and redistribution timing.
- `fuselink-heuristic`: FuseLink-inspired what-if relay model only, not a faithful NCCL/RDMA implementation.

Omitting `--policy` runs `direct`, `spreadout`, and `fast`. `--policy all`
also includes `fuselink-heuristic`.

## Topology

Without `--topology`, Toposim synthesizes a two-tier topology from:

- `--gpus-per-server`
- `--scaleup-gbps`
- `--scaleout-gbps`

The matrix size must be divisible by `--gpus-per-server` by default. Use
`--allow-partial-server` to pad virtual zero-traffic GPUs for an incomplete
final server.

Bandwidth flags are in Gbps. Reports label algorithmic bandwidth as GB/s.

## Batch Manifest

Each JSONL row can include `id`, `matrix`, topology overrides, `policy`,
`engine`, `failed_gpus`, and arbitrary `metadata`. Metadata is preserved in
JSON output but not interpreted by the simulator.

```jsonl
{"id":"layer0_dispatch","matrix":"matrices/layer0_dispatch.txt","metadata":{"layer":0,"phase":"dispatch"}}
{"id":"failure_gpu7_view1","matrix":"matrices/failure_gpu7.txt","metadata":{"failed_gpus":[7],"repair_policy":"nearest_replica","view_id":1}}
```

Fault recovery stays outside Toposim. Pass repaired matrices from the
fault-tolerance branch; `--failed-gpus` is only a guardrail.
