# Scenario Row Metadata

Matrix-bearing rows in the scenario timeline manifest include step, traffic kind, matrix path, total bytes, live and failed node sets, failed ranks, and request-stream counts. AllToAllV rows also include cursor histogram metadata; migration and initial replication rows use `total_bytes` as the moved byte total. A join `node_event` can include `lost_expert_bytes` when disk IO is needed because no live expert replica survived.

Rows include `live_nodes` and `failed_ranks`, but not `live_ranks`; live ranks are derivable from topology mapping and failed ranks.

**Consequences**

The timeline manifest is inspectable without opening every matrix file. The `topsim-batch` matrix manifest can preserve the same metadata for per-matrix performance evaluation, but it remains a derived compatibility artifact.

Node events are emitted as explicit timeline rows even when the same step also contains expert migration or AllToAllV rows. This keeps the event history auditable without inferring events from placement changes.

Rows with the same step use stable phase order: `initial_expert_replication` for step `-1`, then `node_event`, `expert_migration`, `all2allv`, and `terminal_failure` when applicable.

Matrix paths in manifests are relative to the manifest directory.

The scenario timeline begins with a `{"kind": "scenario_header", "metadata": {...}}` row with no `step`. It records topology mapping metadata, including `num_ranks`, `ranks_per_node`, and `num_nodes`, documenting how node ids map to contiguous rank blocks.
