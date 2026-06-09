# CLAUDE.md — Lazarus MoE training & inference simulator

This project implements [Lazarus](https://arxiv.org/abs/2407.04656) (Wu et al., 2024) — resilient and elastic MoE training via adaptive expert replication and Maximum Rank Overlap (MRO) placement — plus an inference-side simulator that applies the same principles to serving.

There is **no public reference implementation** from the paper authors. This codebase is a from-spec implementation. When the paper and this spec disagree, prefer the paper for §4–§6 algorithms; prefer this spec for the systems plumbing and the inference extension (which is not in the paper).

## Goals

1. **Training side**: a Lazarus-compatible MoE layer that adapts expert replica counts to observed load, places replicas via MRO for failure resilience, and reconfigures peer-to-peer on node failure without restarting from checkpoint.
2. **Inference side**: a discrete-event simulator that evaluates the same allocation + placement under serving workloads, with three replica selection policies and a failure injector.

The algorithmic core (`allocator.py`) is **shared** between the two sides. That's a design invariant — do not duplicate the allocator or MRO code.

## Architecture

```
                    Controller (one CPU node, async RPC)
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
    Agent          Agent          Agent             ← one per GPU node
     │ │ │ │        │ │ │ │
     ▼ ▼ ▼ ▼        ▼ ▼ ▼ ▼
     workers (1 per GPU, each holds LazarusMoELayer + Dispatcher)
```

The **inference simulator** replaces the workers with a discrete-event loop driven by a workload generator, but keeps the controller's planning logic identical.

## Module specs

Build in the order listed. Each module's public surface is the contract; implementation details are at the implementer's discretion.

### 1. `allocator.py` — algorithmic core (build first)

Two pure functions, no I/O, no torch dependency. This is the only code that's shared between training and inference paths.

**Placement model (matches the paper, arXiv:2407.04656).** `N` = number of nodes (the paper treats each GPU as a node for evaluation), `c` = per-GPU/per-node slot capacity (*"the number of replicas each node can hold"*, bounded by GPU memory), `E` = number of experts. The placement plan is the matrix `T ∈ ℕ^{c×N}`, i.e. **multiple experts per GPU is the native model, not a v2 extension**. `E > c` is the general case: the paper *"partition[s] the experts and nodes both into ⌈E/c⌉ groups"* and places each group as a nested chain of node-sets `S₁ ⊂ S₂ ⊂ … ⊂ S_E`. The repo's 128-experts-on-16-ranks workload is exactly this case (`N=16`, `E=128`, pick `c` so `N·c ≥ E` with replication headroom). Two real limitations to keep in mind: `c` is memory-bounded (need `c > E/N` for any replication slack — at `E=128, N=16`, `c=8` is full at `k_min=1`), and the inference-side toposim cost model is **comms-only**, so heterogeneous per-rank compute from uneven expert counts is not captured by a rank-to-rank byte matrix.

**`Slot`** (frozen dataclass): `(node: int, local_rank: int, slot: int = 0)`. A physical expert hosting location; `slot` indexes the per-GPU capacity dimension `0..c-1` (`slot=0` is the `c=1` special case).

**`Placement`** (dataclass):
- `expert_to_slots: Dict[int, List[Slot]]`
- `slot_to_expert: Dict[Slot, int]`
- `replicas(expert_id) -> int`
- `survives(failed_nodes: Sequence[int]) -> bool` — true iff every expert has ≥1 replica on a non-failed node.

**`allocate_replicas(expert_loads, num_slots, k_min=2) -> List[int]`**
Greedy makespan-minimising allocator (paper §4.1). Here `num_slots = num_nodes * gpus_per_node * c` (total replica slots across the cluster). Start every expert at `k_min` replicas (`k_min` is the paper's minimum replication factor `f`), then hand each remaining slot to whichever expert currently has the largest `c_i / r_i`. Use a min-heap keyed by `-c_i/r_i` with expert id as deterministic tiebreak. Output must sum to `num_slots`; raise `ValueError` if `num_slots < E * k_min`. This load-proportional rule is equivalent in spirit to the paper's closed form `r_e = max{⌊(t_e / Σ_{e'≥e} t_{e'}) · (N·c − Σ_{e'<e} r_{e'})⌋, f}`.

**`mro_place(replica_counts, num_nodes, gpus_per_node, capacity=1) -> Placement`**
Maximum Rank Overlap placement (paper §4.2), generalised to `capacity = c` slots per GPU. Total slots `= num_nodes * gpus_per_node * capacity`; `sum(replica_counts)` must equal it.

The failure domain is the **node** (a node failure takes all its GPUs), so what governs recovery is the *set of nodes* each expert occupies, not which GPU within a node. Each node holds `slots_per_node = gpus_per_node * capacity` replicas. Let `band = min(replica_counts)` — the survival fan-out every expert shares.
- Sort experts by **ascending** replica count, tiebreak on id — coldest first.
- For each expert, place replicas on **distinct** nodes wherever capacity allows (reuse a node only if no distinct one is free — happens only for hot experts with `r_e >` available nodes, which survive anyway). This gives the single-node-failure survival guarantee when `k_min ≥ 2`.
- The first `band` replicas go to the **lowest-indexed** available nodes (concentration); replicas beyond `band` go to the **least-loaded** node (balance).
- Within a node, the `t`-th replica placed maps to `local_rank = t % gpus_per_node`, `slot = t // gpus_per_node`.

Why the split: putting every expert's *survival set* (its first `band` replicas) on a shared low-index band maximises node-set overlap among the most vulnerable (low-replica) experts — a failure set tends to knock out the same experts together rather than independently endangering many scattered ones, which is what maximises recovery probability. Routing the *surplus* replicas to the least-loaded node instead spreads hot experts across the cluster and keeps capacity balanced, so late-placed experts are never stranded on a single remaining node (a pure lowest-index fill silently violates survival in the near-uniform, tight-capacity case — verified and fixed). Empirically MRO ties random at low failure rates and pulls clearly ahead as the failure count grows (paper §4.2). (Note: this is a greedy realisation of the overlap principle, not a line-by-line transcription of Algorithm 1's `⌈E/c⌉`-group recursion; it achieves the same recovery-maximising property and is what `allocator_tests.py` validates against a random baseline.)

**`plan_layer(expert_loads, num_nodes, gpus_per_node, capacity=1, k_min=2) -> Tuple[List[int], Placement]`**
Convenience wrapper that calls both (passes `num_slots = num_nodes * gpus_per_node * capacity` to the allocator).

**`recovery_probability(placement, failure_prob, num_nodes, samples, seed) -> float`**
Monte Carlo estimate. Used only in tests.

**Tests** (`allocator_tests.py`):
- `allocate_replicas([10]*4, 16, k_min=2) == [4,4,4,4]`
- `allocate_replicas([1000,1,1,1], 16, k_min=2)` — expert 0 gets ≥ 8 replicas.
- Plan a 4×4 cluster (`capacity=1`) with imbalanced loads, verify `placement.survives([n])` is true for every single-node failure when `k_min=2`.
- Plan a `capacity>1` cluster (e.g. `E=128, num_nodes=4, gpus_per_node=4, capacity=16` → 256 slots) and verify `placement.survives([n])` for every single-node failure when `k_min=2`. This exercises the `E > c` grouping path.
- MRO vs random recovery probability comparison: **use `k_min ≥ 2` for this test**. With `k_min=1` and singleton experts, both placements lose those experts on any failure and the comparison becomes statistically noisy.

### 2. `dispatcher.py` — flexible token routing (training side)

Requires torch + torch.distributed.

**`RoutingTable.from_placement(placement, gpus_per_node) -> RoutingTable`**
Holds `expert_to_ranks: Dict[int, List[int]]`, `rank_to_expert: Dict[int, int]`, and a `replicas: torch.Tensor[num_experts]` for fast on-device lookup.

**`select_replica(expert_assignment: Tensor[N], table, rng=None) -> Tensor[N]`**
For each token, pick a destination rank uniformly at random among the expert's replicas. Production code should make this a fused CUDA kernel; the reference does it host-side via tensor indexing.

**`variable_all_to_all(send_buf, send_counts, group) -> (recv_buf, recv_counts)`**
Wraps `dist.all_to_all_single` with explicit `input_split_sizes` / `output_split_sizes`. First exchanges the count vector (small symmetric all-to-all), then issues the variable-size payload all-to-all.

**`LazarusDispatcher`**
- `dispatch(tokens, expert_assignment) -> (recv_tokens, recv_counts, sort_index)`
  Sort tokens by destination rank, all-to-all, return sort permutation so combine can invert it.
- `combine(processed, recv_counts, sort_index, num_tokens) -> Tensor`
  Reverse path; `index_copy_` to undo the sort.

### 3. `moe_layer.py` — Lazarus-aware MoE block (training side)

**`ExpertMLP(hidden, ffn_hidden)`**: standard SwiGLU two-layer MLP. Orthogonal to Lazarus.

**`LazarusMoELayer(hidden, ffn_hidden, num_experts, placement, gpus_per_node, process_group=None)`**
- Holds only the experts assigned to *this* rank by the current placement.
- `forward(x)`: top-1 gate → update `load_counter` buffer → dispatcher.dispatch → expert(s) → dispatcher.combine → multiply by gate weight.
- `pop_load_counts() -> Tensor`: returns and zeroes the accumulated per-expert load. The trainer all-reduces this and reports to the controller.
- `reload_placement(new_placement)`: rebuilds the dispatcher and routing table; called after reconfiguration. Assumes new expert weights have been migrated in (see `reconfig.py`).

**Top-1 only in v1.** Top-k is a v2 extension: scatter k copies of each token into the dispatcher and weight-sum on combine.

### 4. `reconfig.py` — weight migration (training side)

**`rebuild_process_group(new_world_size, new_rank, init_method='env://', backend='nccl')`**
Destroy the old PG, init a new one. Must be called by every surviving worker with the same world_size. Use `torch.distributed.elastic` rendezvous backend in production so the rank assignment is consistent across the cluster.

**`migrate_expert(tensors, src_rank, dst_rank, group=None)`**
Point-to-point copy via `dist.send`/`dist.recv`. Spectator ranks no-op. Caller is responsible for allocating destination buffers of matching shape/dtype before calling.

**`apply_migration_plan(migration, layer, new_placement, gpus_per_node, group=None)`**
Execute a list of `(expert_id, src_rank, dst_rank)` tuples in order. All ranks must iterate over the same list in the same order for the send/recv pairs to match.

**`reconfigure_worker(layer, new_placement, new_world_size, new_rank, gpus_per_node, migration)`**
Glue: rebuild PG → reload placement → run migration → barrier.

**Critical gotcha**: optimizer state (AdamW `exp_avg`, `exp_avg_sq`) must migrate with the parameters or training diverges. Extend `migrate_expert`'s `tensors` argument to include moment buffers in production. The reference implementation only handles parameters.

### 5. `controller.py` — cluster controller (training side)

Asyncio + line-delimited JSON RPC over TCP for the reference; swap for etcd / c10d rendezvous in production so the controller itself is restartable.

**Controller responsibilities** (in order of complexity):
1. Maintain `ClusterState`: `nodes_alive`, `gpus_per_node`, per-layer `loads`, per-layer `placements`, monotonic `version`.
2. Accept agent connections; first message is `{"hello": node_id}`.
3. Send initial plan on connect (uniform load → plan_layer per layer).
4. Heartbeat watchdog: any node silent for `heartbeat_timeout` seconds is marked dead → triggers `_maybe_replan`.
5. `_maybe_replan`: re-run `plan_layer` per layer with updated `nodes_alive`. Diff old vs new placement to build a migration plan: for each new (expert, rank) not in the old placement, pick a surviving source rank holding that expert.
6. Broadcast `{"kind": "plan", "version", "placements", "migration"}` to every alive agent. Wait for `{"kind": "plan_applied"}` from all before resuming.
7. EMA load aggregation: `loads[l][e] = 0.9 * loads[l][e] + 0.1 * agent_report[l][e]`.

**Replan triggers**: failure (mandatory) and periodic every `repl_period` steps (optional, lets you adapt to drifting load).

### 6. `agent.py` — per-node bridge (training side)

One process per GPU node. Responsibilities:
1. Connect to controller, send `hello`.
2. Spawn one worker per local GPU (via `torch.multiprocessing.Process`).
3. Heartbeat the controller every few seconds with piggybacked load reports drained from workers via `/dev/shm` shared-memory tensors.
4. On plan receipt: signal workers to pause at next safe point, forward migration plan, wait for ready ack, signal `plan_applied` to controller.

Most of this is plumbing. The interesting bit is the worker IPC: use `mp.Queue` for control messages and a shared-memory tensor for load reports (zero-copy, lock-free reads).

### 7. `train_example.py` — integration glue

Not a runnable end-to-end trainer. Shows how `dist.init_process_group` → `LazarusMoELayer` → training loop → `pop_load_counts` → optional `reconfigure_worker` compose. The actual end-to-end launcher uses `torchrun` with elastic rendezvous.

---

## Inference simulator (`inference/`)

A discrete-event simulator that reuses `allocator.py` unchanged.

### `inference/workload.py`

- `Request(rid, arrival_t, expert, tokens=1)` dataclass.
- `ZipfianStream(num_experts, alpha, rate, seed)`: stationary Zipf with Poisson arrivals. `stream(duration_s)` yields Requests. Pre-compute the CDF for fast sampling.
- `ShiftingStream(...)`: extends Zipfian; every `shift_period_s` permute the expert-id-to-rank mapping. Adversarial workload for sliding-window profilers.
- `SpotTrace.synthetic(duration_s, num_nodes, mttf_s, seed)`: generate iid node failures with exponential interarrival.
- `SpotTrace(path)`: load a real CSV trace of `(timestamp, node_id)`.

### `inference/network_model.py`

**`NetworkConfig`**: `intra_node_bw_gbps=600` (NVLink 4.0), `inter_node_bw_gbps=50` (400 Gbps IB effective), `launch_latency_us=5`, `gpus_per_node`.

**`NetworkModel.transfer_time_s(bytes, rank_a, rank_b)`**: bandwidth-proportional, picks intra-node BW iff `same_node(a, b)`.

**`NetworkModel.all_to_allv_time_s(traffic: Dict[src, Dict[dst, bytes]]) -> float`**: bottleneck cost = max over pairs of their point-to-point time. This is the alpha-beta model assumption.

### `inference/router.py`

**`Policy`** enum: `BASELINE`, `LOCALITY_FIRST`, `QUEUE_AWARE`.

**`Router(placement, policy, gpus_per_node)`**:
- `route(expert, src_rank) -> int`: returns destination rank, or -1 if expert has no live replicas (model collapse signal).
- `on_dispatch(rank)` / `on_complete(rank)`: maintain per-rank queue depth.
- `update_placement(placement)`: called after reconfiguration; preserves queue depths for surviving ranks.

**Policy semantics**:
- `BASELINE`: one rank per expert, send there. No load balancing.
- `LOCALITY_FIRST`: prefer replicas on the source rank's node; among locals pick shortest queue; if no local replica, fall through to `QUEUE_AWARE`.
- `QUEUE_AWARE`: shortest queue across all replicas globally. Ignores locality.

### `inference/simulator.py`

**`run_serving_sim(requests, placement, policy, net, gpus_per_node, expert_compute_time_s, token_bytes, failures, detect_latency_s) -> (records, collapse_count, request_loss_count)`**

Event-driven loop with a single priority queue. Event kinds:
- `arrival`: route, dispatch, schedule `complete` event.
- `complete`: record latency, free the rank in `next_free_t`.
- `fail_start`: mark node dead, lose all in-flight requests on its ranks.
- `fail_detected` (at `t + detect_latency_s`): re-plan and update router.

**Critical: model the per-rank serial queue.** `next_free_t[rank]` tracks earliest time the rank is idle; new tokens compute at `max(arrive_at_dst, next_free_t[rank])`. Without this, hot-spot bottlenecks don't appear and the baseline-vs-Lazarus latency comparison shows no difference. (I hit this exact bug; symptom: identical P99 across policies regardless of skew.)

**Track `model_collapse_count` and `request_loss_count` separately.** They have different semantics: collapse = expert weights gone; request loss = in-flight token's destination rank died. The proposal document conflates these; keeping them split matters for the recovery story.

**`run_survival_trial(placement, num_nodes, num_failures, samples, seed) -> float`**
Pure placement Monte Carlo. No network, no requests. Sample `num_failures` random nodes, check `placement.survives(failed)`. Used for the MRO-vs-random-vs-compact chart.

### `inference/experiments/survival_rate.py`

Compares three placements on the same replica counts:
1. `mro_place(...)` from the allocator.
2. Random shuffle of slots.
3. Compact: fill node 0 first, then node 1, etc. (worst case baseline.)

Sweep `num_failures` from 0 to `num_nodes - 1`, run 5000 MC samples each, print a table. Expected output (8 nodes × 8 GPUs, 8 experts, k_min=2):

```
 #fail   MRO   Random  Compact
     1  1.00   1.00     0.88
     3  0.87   0.88     0.24
     5  0.49   0.43     0.00
     6  0.22   0.15     0.00
```

MRO ties Random at low failure counts and wins at high failure counts. Compact is destroyed early.

### `inference/experiments/latency_skew.py`

Sweep Zipf alpha; compare BASELINE (one rank per expert) vs LAZARUS (`plan_layer` allocation + `QUEUE_AWARE` routing). Expected output (4 experts, 4 nodes × 2 GPUs = 8 slots, 8000 req/s):

```
 alpha   baseline P50/P99    lazarus P50/P99
  1.0    223us /  1729us    205us /   399us
  1.5    597us / 10566us    205us /   362us
  2.0     74ms /   240ms    205us /   397us
```

**Critical: the slot budget must exceed `num_experts`** or Lazarus has no room to replicate and the comparison collapses. Use `num_slots ≥ 2 * num_experts` for visible effect.

---

## Build order & dependencies

```
allocator.py  ← no dependencies; build and test first
    │
    ├── dispatcher.py ─→ moe_layer.py ─→ reconfig.py
    │                                        │
    │                                        ▼
    │                                   controller.py + agent.py
    │                                        │
    │                                        ▼
    │                                   train_example.py
    │
    └── inference/
            workload.py  network_model.py
                │              │
                └──→ router.py ←┘
                       │
                       ▼
                  simulator.py
                       │
                       ▼
                  experiments/*.py
```

The inference simulator is **independent** of the training-side modules apart from `allocator.py`. Build the inference path first if you want runnable experiments before committing to the distributed training setup.

## Conventions

- **Python ≥ 3.10** (uses `X | Y` union syntax and `from __future__ import annotations`).
- **PyTorch ≥ 2.1** for `dist.all_to_all_single` with explicit split sizes.
- **No external dependencies** for the simulator: only stdlib. For the training side: torch only.
- **Determinism**: every randomised algorithm takes a `seed` parameter. Allocator and MRO are deterministic given the same `(loads, num_slots, k_min)`.
- **Type hints everywhere.** Public functions are fully typed; internal helpers can use `Any` if it keeps things readable.
- **Tests live next to the modules they test** (`tests.py`) or as runnable scripts (`experiments/*.py`).

## What's intentionally out of scope

These are real and important but not part of this implementation:
1. Fused CUDA kernel for dispatcher's replica selection.
2. Optimizer state migration (the AdamW moments). Hooks are there; production must extend `migrate_expert`'s tensor list.
3. Hierarchical (intra-node-then-inter-node) all-to-all.
4. Top-k gating (only top-1 in v1).
5. *Heterogeneous* per-GPU capacity. Multiple experts per GPU **is supported** via a uniform capacity `c` (see the placement-model note in §1); only differing `c` across GPUs is out of scope.
6. Overlapped migration for inference (current simulator pauses affected ranks during migration; production Tarragon-style overlaps with serving via dual placements + atomic ERT swap).
7. KV cache replication for inference. The simulator drops in-flight requests on AW node death.
8. Real rendezvous backend (etcd / c10d). The controller uses asyncio + line-JSON which is fine for prototyping but not production-restartable.

## References

- Wu et al. 2024. *Lazarus: Resilient and Elastic Training of MoE Models with Adaptive Expert Placement.* arXiv:2407.04656.
- Zhang et al. 2025. *Making MoE-based LLM Inference Resilient with TARRAGON.* arXiv:2601.01310. (Inference-side disaggregation pattern.)
- Zhu et al. 2025. *MegaScale-Infer.* (Original Attention-Worker / Expert-Worker split.)
- Singh et al. 2025. *ElasticMoE.* arXiv:2510.02613. (Adjacent autoscaling work.)