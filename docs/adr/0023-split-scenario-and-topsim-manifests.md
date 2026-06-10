# Split Scenario Timeline and Topsim Matrix Manifests

Failure-aware generation emits two JSONL manifests: a complete scenario timeline manifest and a matrix-only `topsim-batch` manifest. Current `topsim-batch` evaluates each matrix independently; it does not model ordered scenario execution or sum total inference time.

**Consequences**

The scenario timeline manifest is the source of truth for ordering, event-only rows, traffic kind, and request-progress metadata. The `topsim-batch` manifest is a compatibility artifact for running existing per-matrix performance estimates. Total scenario latency must be computed by a separate aggregation step over `topsim` results in scenario timeline order.

The scenario timeline is written as the authoritative artifact. The `topsim-batch` matrix manifest is derived or rewritten from matrix-bearing timeline rows, including partial timelines after terminal failure.

The `topsim-batch` matrix manifest includes only rows with a matrix path. Timeline-only rows such as `node_event`, `expert_disk_io`, and `terminal_failure` are excluded.

Each `topsim-batch` matrix row includes `gpus_per_server` set to the scenario `ranks_per_node` value so `topsim` uses the same contiguous node/rank grouping.

The derived `topsim-batch` manifest leaves policy absent. Users choose `topsim` policy at runtime.

The derived `topsim-batch` manifest does not set `failed_gpus`. Failed node and rank state is preserved as metadata because scenario matrices already omit traffic from unavailable ranks.

Implementation validation should prioritize `traffic-gen` scenario correctness. A small `topsim-batch` smoke test is useful only to confirm derived manifest compatibility.
