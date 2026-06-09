# Traffic Matrix Generation

This context defines the domain language for deriving rank-to-rank traffic matrices from MoE expert-selection traces.

## Language

**MMLU English Partial Workload**:
The MVP workload composed of all locally available English MMLU subject traces for Llama4. It is a partial MMLU workload, not a claim to cover all MMLU subjects.
_Avoid_: Single-subject workload, full MMLU workload

**Simulation Step**:
A token/layer coordinate in the decode pass at which traffic demand and failure-aware placement state are observed. Node events scheduled for a simulation step are effective before that step's inference traffic.
_Avoid_: Layer-only step, post-layer event

**Node Join Event**:
A failed node becoming available again as empty compute and storage capacity. Its previous expert replicas are not assumed to survive; experts become available on the joined node only after migration.
_Avoid_: State-preserving recovery, replica restore

**Paused Request Stream**:
A request stream whose rank is unavailable and whose token/layer progress does not advance until that rank rejoins. Its request execution state is assumed recoverable; expert state on the failed node is not.
_Avoid_: Terminated request, dropped request, globally lockstepped request

**Request Stream**:
One selected inference request assigned to a source rank and local request index. Failure-aware scenarios identify request streams by `(source_rank, local_request_index)` because pause and resume are rank-scoped.
_Avoid_: Anonymous global request, reshuffled request ownership

**Cursor Histogram**:
A compact manifest summary of how many request streams contributed traffic from each token/layer cursor during a simulation step. It records request-progress drift without listing every request stream.
_Avoid_: Full request cursor dump, hidden progress drift

**Expert State**:
The model expert weights resident on a rank for serving routed MoE tokens. Expert state on a failed node is treated as lost and must be recreated through expert migration after the node rejoins.
_Avoid_: Request state, KV cache, restored replica

**Expert State Bytes**:
The byte size used to represent moving one layer expert replica during expert migration. For the MVP, it is fixed to the Llama4 Maverick BF16 expert weight size; each layer/expert pair has distinct expert state.
_Avoid_: Activation payload bytes, token route bytes, shared expert bytes

**Layer-Clock Tick**:
A global simulation timestep in which each live request stream advances by one request-local layer. Paused request streams do not advance, so live streams may contribute traffic from different token/layer positions in the same tick.
_Avoid_: Global token/layer step, batch-synchronous layer

**MoE Communication Layer**:
A model layer position that produces expert-selection communication demand. Dense layers are skipped by traffic generation because scenario outputs contain only expert migration and collective communication matrices.
_Avoid_: Dense compute layer, all transformer layers

**Expert Migration**:
The movement of expert state from live source ranks to destination ranks chosen for repaired or rebalanced placement. In the MVP, expert migration completes before inference traffic for the same layer-clock tick.
_Avoid_: Request recovery, overlapped repair, implicit replica restore

**Repair Migration Source**:
The live replica chosen as the source for recreating a missing layer expert replica during repair. The MVP prefers a live replica on the destination node, otherwise the lowest-rank live replica.
_Avoid_: Oracle migration source, random repair source

**Initial Expert Replication**:
The startup movement of expert state from baseline expert owners to the planned placement target before inference begins. It is emitted as a special traffic matrix and is distinct from repair migration after node events.
_Avoid_: Free initial placement, failure repair migration, synthetic bootstrap source

**Baseline Expert Owner**:
The rank that owns a layer expert under the generator's original contiguous expert ownership model. Baseline owners are the sources for initial expert replication traffic.
_Avoid_: Rank-zero bootstrap owner, external model source

**Immediate Repair Migration**:
An MVP repair model where all currently missing expert replicas targeted for restoration are migrated in the same layer-clock tick. The migration is not staged by future layer use.
_Avoid_: Layer-staged repair, lazy migration, bandwidth-capped repair

**Traffic Kind**:
The semantic category of a generated traffic matrix for a simulation step. The MVP emits separate matrices for expert migration and inference AllToAllV traffic, linked by the same simulation step in the scenario manifest.
_Avoid_: Combined recovery/serving matrix, untyped matrix output

**Network-Only Scenario Matrix**:
A failure-aware scenario matrix whose diagonal entries are zero because local movement is not network traffic. Scenario mode emits only network-only matrices.
_Avoid_: Original scenario matrix, local-hit matrix

**Event-Only Step**:
A simulation step where a node event occurs but no traffic matrix is emitted. It is represented in the scenario manifest so the failure/join timeline remains auditable without creating empty matrices.
_Avoid_: Empty traffic matrix, invisible event

**Scenario Completion**:
The point at which every request stream has completed its selected decode token/layer work. Completion is based on per-request progress, not a fixed global layer count.
_Avoid_: Fixed layer-count completion, aggregate-per-layer completion

**Failure-Aware Scenario**:
A separate traffic-generation mode that emits simulation-step matrices, event metadata, and per-request progress semantics. It does not replace the baseline aggregate layer output contract.
_Avoid_: Replacement baseline output, aggregate layer scenario

**Scenario Config**:
The JSON input that names a failure-aware scenario and provides ranks-per-node plus its node event schedule. It is the MVP control surface for failure-aware generation.
_Avoid_: Event CLI flags, ad hoc schedule arguments

**Scenario Timeline Manifest**:
The complete JSONL output for a failure-aware scenario, including ordered simulation steps, node events, traffic kinds, and event-only rows. It is the source of truth for scenario ordering.
_Avoid_: Topsim batch manifest, unordered matrix list

**Topsim Matrix Manifest**:
A JSONL output containing only matrix rows compatible with current `topsim-batch`. It is derived from the scenario timeline manifest and does not itself encode full scenario execution semantics.
_Avoid_: Scenario timeline, total latency manifest

**Performance Result**:
The timing, bandwidth, and bottleneck estimate for generated matrices or ordered scenarios. Performance results belong to `topsim`, while `traffic-gen` owns traffic demand and scenario metadata.
_Avoid_: Traffic-gen latency result, generator-owned performance summary

**Planned Expert Load**:
The full-workload estimate of how often each expert is selected, computed before failure simulation begins. It is an MVP planning input, not a claim that real inference systems know future expert hotness.
_Avoid_: Live expert load, oracle runtime load, adaptive hotness

**Layer Expert**:
A specific expert within a specific MoE layer, identified by both layer id and expert id. The same expert id in two different layers refers to different expert state.
_Avoid_: Global expert id, shared cross-layer expert

**Replica Routing**:
The choice of which live replica serves a routed token for a layer expert. The MVP uses local-node visibility first and does not assume global per-step load-balancing knowledge across ranks.
_Avoid_: Oracle load balancing, globally least-loaded routing

**Rank Block Node Mapping**:
The mapping where node membership is defined by contiguous blocks of ranks. The ranks-per-node value is scenario input and must divide the total rank count; `traffic-gen` and `topsim` must use the same value for a scenario.
_Avoid_: Round-robin node mapping, inferred topology mismatch

**Node Event Schedule**:
The scenario input that identifies node fail and join events by simulation step and node id. Rank effects are derived from the rank block node mapping.
_Avoid_: Rank failure schedule, mixed node/rank event ids

**Single Node Event Step**:
An MVP schedule constraint where at most one node fail or join event occurs at a simulation step. Simultaneous node events are outside the MVP.
_Avoid_: Simultaneous node events, batched failure events

**Unservable Layer Expert**:
A layer expert needed by inference for which no live replica exists. The scenario is invalid because traffic generation cannot route to or migrate from lost expert state.
_Avoid_: Dropped expert, synthetic replica, silent token skip

**Terminal Scenario Failure**:
An explicit scenario-ending failure record emitted when generation cannot continue, such as reaching an unservable layer expert. Partial outputs before the terminal failure remain valid for inspection, but the scenario itself is invalid.
_Avoid_: Silent truncation, successful partial scenario

**Planned Placement Target**:
The precomputed intended placement of layer-expert replicas before failure simulation. Repair attempts to restore missing replicas toward this target rather than re-optimizing placement from scratch after each event.
_Avoid_: Event-local optimal placement, full reshuffle target

**Baseline-Pinned Placement**:
An MVP adaptation of Lazarus placement where each layer expert's baseline owner is pinned as one planned replica, and remaining replicas are placed around that constraint. It preserves the existing sharded source of expert state.
_Avoid_: Raw unpinned Lazarus placement, replacing baseline owner

**Per-Layer Expert Capacity**:
The number of layer-expert replica slots available on each rank for a single MoE layer. In the MVP, capacity is not a global memory budget shared across all layers.
_Avoid_: Cross-layer memory capacity, total model capacity

**Join Repair**:
Expert migration triggered by a node join event to restore missing planned replicas onto the rejoined blank node. Node failure events remove capacity and expert state but do not trigger migration in the MVP.
_Avoid_: Failure repair, proactive failover migration

**Post-Repair Resume**:
The rule that request streams on a rejoined node resume only after join repair migration for that simulation step completes. Rejoined ranks do not produce inference traffic before their repair phase.
_Avoid_: Immediate join resume, pre-repair inference

## Example Dialogue

Dev: "Should the MVP read only anatomy traces?"

Domain expert: "No. Use the MMLU English Partial Workload: all local Llama4 MMLU subject folders together."

Dev: "So the result is not full MMLU coverage?"

Domain expert: "Correct. It is enough local trace data for the MVP, but it remains a partial English MMLU workload."
