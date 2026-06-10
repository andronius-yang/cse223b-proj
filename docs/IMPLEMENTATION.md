# IMPLEMENTATION.md — code reference for the Lazarus replication work

Single-file map of everything implemented so far. Read this to recover full context
without re-deriving from the source. Authoritative *design* lives in `docs/adr/0001..0031`,
`CONTEXT.md` (domain glossary), and `AGENTS.md` (baseline generator spec); this file
documents what the **code actually does** and how it maps to those decisions.

Repo conventions: Python ≥ 3.10, stdlib-only for traffic-gen / allocator / control-plane
(numpy/torch only inside `topsim/`). Tests are plain `*_tests.py` scripts with a `main()`
of `assert`s, run directly (no pytest). Every randomized routine takes a `seed`.

---

## 1. Module map & dependency graph

```
allocator.py            ← shared Lazarus core (pure stdlib). Build/test first.
  │  allocate_replicas, mro_place, plan_layer, Slot, Placement, recovery_probability
  │
  ├── control_plane.py        ← steady-state routing primitives (prototype track)
  │     Policy, expert_ranks, repair, route_demand
  │
  └── traffic-gen/
        generate.py           ← baseline trace→matrix generator + demand profiling
        lazarus_plan.py        ← PROTOTYPE driver (aggregate, steady-state) [track A]
        generate_scenario.py   ← failure-aware layer-clock scenario generator [track B]

topsim/ (package `toposim`)   ← matrix+topology → comms-cost. Consumes our matrices.
```

Two **independent evaluation tracks** share `allocator.py`:
- **Track A (prototype, `lazarus_plan.py`)**: aggregate-over-layers, steady-state load
  balancing. Diverges from the ADRs on purpose; kept for the load-balancing story.
- **Track B (ADR-conformant, `generate_scenario.py`)**: per-layer, failure-aware
  layer-clock simulation. This is the real target. ADRs 0001–0031.

### Run & test

```bash
# allocator + control-plane (from repo root)
python3 allocator_tests.py            # 6 tests
python3 control_plane_tests.py        # 3 tests

# everything traffic-gen (from traffic-gen/, traces are globbed from cwd)
cd traffic-gen
python3 generate.py                                      # baseline aggregate matrices
python3 lazarus_plan.py                                  # track A: 3 matrices + manifest
python3 generate_scenario.py scenarios/no_events.json    # track B
python3 generate_scenario.py scenarios/node1_fail_join.json
python3 scenario_tests.py             # 15 tests (integration tests SKIP w/o traces)
```

Workload constants (`generate.py`): `NUM_RANKS=16`, `NUM_EXPERTS=128`, `REQUESTS_PER_RANK=16`,
`DECODE_STEPS=32`, `PAYLOAD_BYTES=10240` (= 5120 hidden × 2 BF16 bytes, one token→expert
route). Real traces: `traffic-gen/llama4-mmlu-*/` (Llama-4-Maverick, top-1, 24 MoE layers
= odd ids 1..47, 32 decode tokens).

---

## 2. `allocator.py` — shared Lazarus core

Pure functions, no I/O, no torch. The only code shared across both tracks (and the future
training side). Implements the paper's *general* model (arXiv:2407.04656): `N` nodes, `c`
slots/GPU, `E` experts, multiple experts per GPU, `E > c` allowed.

- **`Slot(node, local_rank, slot=0)`** frozen — physical hosting location. `slot` indexes the
  per-GPU capacity dimension `0..c-1` (`slot=0` ⇒ the `c=1` special case).
- **`Placement(expert_to_slots, slot_to_expert)`** — `replicas(e)` count;
  `survives(failed_nodes)` ⇒ True iff every expert keeps ≥1 replica on a live node.
- **`allocate_replicas(expert_loads, num_slots, k_min=2) -> list[int]`** — greedy makespan
  allocator (paper §4.1), the **"adaptive"** Lazarus strategy. Start each expert at `k_min`;
  hand each remaining slot to the largest `c_i/r_i` via a min-heap keyed `(-c_i/r_i, expert_id)`.
  Sums to `num_slots`; raises `ValueError` if `num_slots < E·k_min`. `num_slots = num_nodes·gpus_per_node·c`.
- **`uniform_replicas(expert_loads, num_slots, k_min=2) -> list[int]`** — the **"uniform"**
  fixed, **load-agnostic** baseline. Every expert gets `num_slots // E` replicas; the
  `num_slots % E` leftover slots go to the lowest expert ids (counts differ by ≤1). Ignores
  `expert_loads` (same signature only so it's drop-in). At `num_slots = E·k_min` it gives a
  flat `k_min` each; with headroom it stays flat while `allocate_replicas` concentrates on hot
  experts — that contrast is the point of the baseline.
- **`REPLICATION_STRATEGIES` / `allocate_replica_counts(loads, num_slots, k_min=2, strategy="adaptive")`**
  — registry + dispatcher. `strategy ∈ {"adaptive","uniform"}`; raises `ValueError` on an
  unknown name. Both strategies feed the *same* `mro_place`, so only the per-expert counts differ.
- **`mro_place(replica_counts, num_nodes, gpus_per_node, capacity=1) -> Placement`** —
  Maximum Rank Overlap, generalized to `capacity=c`. **Failure domain = node**, so recovery
  depends on the *set of nodes* an expert occupies. Algorithm (greedy realization, not the
  paper's `⌈E/c⌉` recursion):
  - `band = min(replica_counts)` (the survival fan-out every expert shares).
  - experts placed **ascending** by replica count (coldest first).
  - replicas go to **distinct** nodes where capacity allows (reuse only if no distinct node
    is free — hot experts only).
  - first `band` replicas → **lowest-indexed** available node (concentration: vulnerable
    experts pile on a shared low-index band → correlated failures, maximizes recovery);
    replicas beyond `band` → **least-loaded** node (balance: prevents capacity stranding
    that would silently break survival — a real bug that a pure lowest-index fill had).
  - within a node, the `t`-th replica → `local_rank = t % gpus_per_node`, `slot = t // gpus_per_node`.
  Empirically MRO ties random recovery at low failure rates and pulls ahead at high rates.
- **`plan_layer(expert_loads, num_nodes, gpus_per_node, capacity=1, k_min=2, strategy="adaptive")`**
  — allocate (via `strategy`) then place; returns `(replica_counts, placement)`.
- **`recovery_probability(placement, failure_prob, num_nodes, samples, seed)`** — Monte Carlo,
  tests only.

Tests `allocator_tests.py`: uniform/skewed allocation, ValueError on too-few slots, single-node
survival at `c=1` and `c>1` (E>c path), MRO-beats-random recovery.

---

## 3. `control_plane.py` — steady-state routing primitives (track A)

Renders a rank-to-rank matrix from a placement + a placement-independent demand matrix.
Pure stdlib + `allocator.Placement`. Used by `lazarus_plan.py`.

- **`Policy`** enum: `BASELINE` (single owner), `REPLICATED` (even fractional split across
  replicas), `LOCALITY` (prefer same-server replicas, else even split).
- **`expert_ranks(placement, gpus_per_node) -> {expert: [rank,...]}`** — `rank = node·gpus_per_node
  + local_rank`, deduped+sorted. (Reused by track B too.)
- **`repair(expert_to_ranks, failed_ranks) -> (survivors, collapsed)`** — drop failed ranks;
  `collapsed` = experts with no live replica.
- **`route_demand(demand, expert_to_ranks, policy, gpus_per_server, failed_ranks=()) -> 16×16`**
  — for each (src, expert) demand, split bytes across chosen live replica ranks; preserves
  total bytes unless collapsed/failed. `total_bytes(matrix)` helper.

Tests `control_plane_tests.py` (server=node, 4×4): REPLICATED preserves total bytes;
single-server failure never collapses an expert; failed server's ranks send/receive zero.

---

## 4. `traffic-gen/generate.py` — baseline generator + demand profiling

Original behavior unchanged (emits `layer_*_{original,network}.txt`, `aggregate_*`). Key
contract: contiguous single-owner placement — `owner_ranks(layer, expert)` (each rank owns
`128//16 = 8` experts; earliest ranks get the remainder) and `choose_owner_rank(list)` (picks
`[0]`). This is the BASELINE placement both tracks compare against.

Additions (refactors that preserve existing output):
- **`iter_selected_expert_ids(...)`** — validates a token's nested `selected_experts` and
  yields expert ids (extracted from `add_selected_experts`).
- **`iter_routes(trace_paths)`** — yields `(layer_id, src_rank, expert_id)` for every decode
  token→expert route (placement-independent raw demand).
- **`build_demand_matrices(trace_paths) -> {layer: 16×128}`** — `D[layer][src][expert]` bytes,
  experts NOT collapsed to owners. `aggregate_demand` sums over layers.

`src_rank = request_index // REQUESTS_PER_RANK`; first `BATCH_SIZE=256` traces; contiguous
assignment (requests 0–15 → rank 0, …).

---

## 5. `traffic-gen/lazarus_plan.py` — PROTOTYPE driver (track A)

Aggregate-over-layers, steady-state. **server = node**: `NUM_NODES=4, GPUS_PER_NODE=4,
CAPACITY=32, K_MIN=2, GPUS_PER_SERVER=4, FAIL_NODE=1` (ranks 4–7). Pipeline: `aggregate_demand`
→ `plan_layer` → `expert_ranks` → `route_demand` for three scenarios (`baseline` single-owner,
`replicated` even split, `repaired` survivors after failing server 1) → writes
`out/lazarus/{baseline,replicated,repaired}.txt` + `manifest.jsonl`, then best-effort
`uv run toposim-batch` (skips gracefully if `uv` absent).

Result (matrix-level, since `uv`/toposim not installed): replication cuts peak rank-recv ~21%
and inter-server recv ~7% at identical total bytes; survives a full server failure with 0
collapsed experts.

**Deliberately diverges from the ADRs** (this is the prototype track, NOT to be "fixed"):
aggregates layers (ADR 0004 wants per-layer), raw plan_layer (ADR 0029 now makes Lazarus
placement authoritative),
even-split (ADR 0005 wants locality-first), drops failed-server requests (ADR 0001 wants
pause), reroute-on-fail (ADR 0008/0019 want migrate-on-join), no migration traffic
(ADR 0009/0011/0017).

---

## 6. `traffic-gen/generate_scenario.py` — failure-aware scenario generator (track B) ★

The ADR-conformant target. **This is the team-canonical version (≈946 lines) that came in
on master; my own earlier implementation was discarded during the rebase in favor of it.**
A **layer-clock discrete simulation** over the 256-stream workload. Run:
`python3 generate_scenario.py scenarios/<id>.json`. It imports the root-level Lazarus
planner (`plan_layer`, `REPLICATION_STRATEGIES`, `Placement`, `Slot`) from `allocator.py`
plus `generate.py` helpers. Scenario mode no longer has its own placement algorithm.

### Topology / identity
`ranks_per_node` (default 4) ⇒ `num_nodes = 16 // ranks_per_node`. Contiguous rank-block node
mapping: node `n` owns ranks `[n·rpn, (n+1)·rpn)` (`rank_to_node`, `slot_rank`). The Lazarus
node ≡ the toposim server (so derived manifest rows set `gpus_per_server = ranks_per_node`).
Request streams are `RequestStream(source_rank, local_request_index, path, work, cursor)`,
256 of them, 16 per rank. `WorkItem(token_index, layer_id, expert_ids)`; `LayerPlan(layer_id,
placement)`; `NodeEvent(step, event_type, node)`; `ScenarioConfig(scenario_id, ranks_per_node,
capacity_per_rank_per_layer, events)`.

### Config (`load_config`, ADR 0022)
Validates: `scenario_id` slug, `ranks_per_node` divides 16, `capacity_per_rank_per_layer`
(default 16; `16·cap ≥ 128·k_min`), `replication_strategy` (default `"adaptive"`; must be a key
of `allocator.REPLICATION_STRATEGIES`, i.e. `adaptive`/`uniform`), events **strictly increasing
by step** (so ≤1 event/step),
type ∈ {fail,join}, node bounds, fail/join state transitions (no double-fail / join-live).
`K_MIN=2`, `EXPERT_STATE_BYTES=251_658_240` are module constants. `SCENARIO_ROOT =
generate.OUTPUT_DIR / "scenarios"` (monkeypatch this in tests to redirect output).

### Workload (`build_workload`, ADRs 0013/0014/0027)
Returns `(streams, layer_loads)`. Each stream's `work` is the ordered `WorkItem` list over decode
tokens `1..min(32,len-1)`, MoE (non-null) layers only; `layer_loads[layer][expert]` accumulates
the static oracle counts simultaneously. One work item = one layer-clock tick (≈768 total).
`validate_event_completion` rejects events scheduled at/after the max stream length.

### Planning (`build_layer_plans`, ADRs 0003/0004/0029/0030)
Per layer: call `allocator.plan_layer(loads, num_nodes, ranks_per_node, capacity=cap,
k_min=2, strategy=replication_strategy)`. The returned `Placement` is the sole planned
placement truth. Baseline owners are no longer pinned into the placement; they are only the
source ranks for initial expert-state replication.

### Traffic kinds (network-only, diagonal zeroed — ADRs 0011/0025)
- **`build_initial_replication_matrix(plans, rpn)`** (ADR 0017, step −1): one aggregated matrix;
  for each `(layer, expert)` charge `EXPERT_STATE_BYTES` from the baseline owner to each *other*
  planned replica rank (self-copy free).
- **`build_join_repair_matrix(plans, current_slots, joined_node, live_nodes, rpn)`** (ADRs
  0008/0018/0019): on join, restore each planned slot on the rejoined node that isn't currently
  present, sourced via `choose_repair_source` (lowest live rank on the destination node,
  else circular rank search from the destination rank). Returns `(matrix, disk_bytes)`; no live
  source increments disk IO and still restores the planned slot.
- **all2allv** (`build_all2allv_matrix`): per step, per live stream at its cursor, route each
  expert to a live replica via **`choose_route_destination`** (same-node lowest rank, else
  circular rank search from source rank), `+PAYLOAD_BYTES`; local hits dropped (network-only).
  If any expert for a stream has no live replica, that stream blocks for the tick and emits no
  partial traffic. The function returns the cursor histogram and the streams that advanced.

### State model (key difference from a naive "node up" check)
`current_slots: {(layer,expert): set[Slot]}` tracks *actual* live expert state. `fail` calls
`remove_failed_node_state` (deletes the failed node's slots); `join` restores planned slots via
network migration or disk IO. Routing reads `live_replica_ranks` from
`current_slots ∩ live_nodes`.

### Simulation loop (`run_scenario(config, streams, plans)`, ADRs 0001/0007/0012/0013/0019/0020/0028/0031)
Per absolute step from 0: if no live streams remain but streams are incomplete, jump to the next
event step (else `terminal_failure` "deadlock", return 1). Apply the step's event (fail: drop
state + `node_event` row; join: `node_event` row, optional timeline-only `expert_disk_io` row,
then optional `expert_migration` matrix). Then if any streams are live, build the all2allv
matrix; unavailable experts block their streams. If no stream can advance and no future event
exists, emit `terminal_failure` and return 1. Emit matrices only when nonzero. Advance advancing
streams. Returns 0 on completion.

### Outputs (`out/mmlu_english_partial/scenarios/<id>/`, ADRs 0023/0026/0031)
- **`scenario_timeline.jsonl`** (authoritative, written incrementally with `flush`):
  `scenario_header` (with `rank_blocks`, constants), `initial_expert_replication` (step −1),
  `node_event`, `expert_disk_io`, `expert_migration`, `all2allv`, `terminal_failure`. State
  fields on rows: `live_nodes`, `failed_nodes`, `failed_ranks`, `live_request_streams`,
  `paused_request_streams`, `completed_request_streams`; all2allv adds
  `metadata.cursor_histogram` keyed **`tok{token}_layer{layer}`**.
- **`topsim_matrix_manifest.jsonl`** (derived, note the name — not `topsim_batch.jsonl`):
  matrix-bearing rows only, `id = "<scenario>_<matrixstem>"`, `gpus_per_server = ranks_per_node`,
  full timeline row carried under `metadata`, no policy.
- Matrix files: `initial_expert_replication.txt`, `step_{NNNNNN}_{all2allv|expert_migration}.txt`.
  Whitespace integer N×N (`generate.new_matrix`/`write_matrix`).

### Verified behavior (`scenario_tests.py`, 18 tests, rewritten against this API)
- `no_events`: exit 0, 768 all2allv matrices, lockstep.
- `node1_fail_join` (fail@100 / join@200): exit 0, **survives**, 868 all2allv steps (768 + 100
  paused), 2 node_events + 1 migration; during failure `failed_nodes == [1]` and 64 streams paused.
- two-node-fail (no rejoin): `terminal_failure`, exit 1.

---

## 7. Caveats & gotchas (read before extending)

1. **`capacity_per_rank_per_layer=16` ⇒ no adaptive replication.** `num_slots = 16·16 = 256 =
   128·k_min`, so every expert gets *exactly* 2 replicas regardless of load. This is the
   ADR-0030 default's known consequence (minimal to satisfy k_min=2). Raise capacity in the
   scenario config to get load-adaptive replica counts. **Corollary:** at cap=16 the `adaptive`
   and `uniform` strategies are *identical* (both flat 2); the two only diverge once there's
   headroom (e.g. cap=32 → adaptive concentrates on hot experts, uniform stays flat 4).
2. **Baseline pinning is intentionally gone.** The canonical scenario planner now trusts
   `allocator.plan_layer` for placement truth. Keep baseline owners only as initial replication
   sources unless a future ADR explicitly changes the placement contract again.
3. **Output volume**: a full scenario writes ~768+ matrix files. Tests redirect
   `gs.SCENARIO_ROOT` to a temp dir to avoid polluting `out/`.
4. **toposim not run here**: `uv` and some deps (networkx) aren't installed in this env.
   Track A degrades gracefully; track B emits `topsim_matrix_manifest.jsonl` to run later:
   `uv run --project ../topsim toposim-batch <id>/topsim_matrix_manifest.jsonl --policy fast`.
5. **CLAUDE.md vs the ADRs**: CLAUDE.md is a from-spec guide for a *different* (training + DES
   inference-sim) design; the scenario generator follows the ADRs, not CLAUDE.md's module list.
   CLAUDE.md's allocator §1 was corrected to the paper's general-`c` model.

## 8. ADR → code cross-reference

| ADR | Where implemented |
|-----|-------------------|
| 0001 pause requests | `build_all2allv_matrix` skips failed-node streams; cursor not advanced |
| 0003 static load | `build_workload` accumulates `layer_loads` |
| 0004 per-layer placement | `build_layer_plans` loops per layer |
| 0005 locality-first routing | `choose_route_destination` |
| 0006 contiguous node map | `rank_to_node`, `slot_rank` |
| 0007 unavailable expert blocks stream | `build_all2allv_matrix` skips non-routable streams |
| 0008/0018/0019 join repair | `build_join_repair_matrix`, `choose_repair_source`, `expert_disk_io` |
| 0009 EXPERT_STATE_BYTES | constant `EXPERT_STATE_BYTES` |
| 0011 traffic kinds | separate `*_all2allv` / `*_expert_migration` matrices |
| 0013 completion | `all_completed`; `validate_event_completion` |
| 0014 skip dense | `build_workload` skips null layers |
| 0016 one event/step | `load_config` strictly-increasing-step check |
| 0017 initial replication | `build_initial_replication_matrix`, step −1 |
| 0022 JSON config | `load_config` |
| 0023 dual manifests | `scenario_timeline.jsonl` + `topsim_matrix_manifest.jsonl` |
| 0025 network-only | diagonal dropped on emit |
| 0026 partial on terminal | no-progress deadlock uses `write_terminal_failure`, exit 1 |
| 0027 stream identity | `RequestStream(source_rank, local_request_index)` |
| 0028 cursor histogram | `metadata.cursor_histogram`, keys `tok{t}_layer{l}` |
| 0029 Lazarus placement truth | `build_layer_plans` calls `allocator.plan_layer` |
| 0030 per-layer capacity | `capacity_per_rank_per_layer`, `K_MIN=2` |
| 0031 row metadata | `state_fields`, `emit_matrix_row` |

## 9. Status & open items
- Rebased onto master (group's `generate.py` + canonical `generate_scenario.py` + README spec).
  Tests: allocator (6), control_plane (3), scenario (15, rewritten for the group's API) = 24 passing.
- Committed on `yash/replication-lazarus`: `control_plane.py` (track A), `lazarus_plan.py`,
  `scenario_tests.py`. Untracked: `docs/`, `AGENTS.md`, `CONTEXT.md`, `claude.md`.
- Not done: running toposim + per-step latency aggregation (ADR 0024); adaptive-capacity
  experiments; the training side (`dispatcher.py`/`moe_layer.py`/… from CLAUDE.md).
