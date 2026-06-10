# Reading the expert → GPU placement mapping

How to get, interpret, and export the mapping of MoE expert replicas to physical
GPUs from a Lazarus `Placement`. This is the output side of the allocator/MRO
pipeline — once you've called `plan_layer` (or `mro_place`), everything about
*where each replica lives* is in the returned `Placement` object.

- Source: `allocator.py` (`Slot`, `Placement`, `plan_layer`, `mro_place`).
- Rank helper: `control_plane.py` (`expert_ranks`, `repair`).
- Scenario-side equivalents: `traffic-gen/generate_scenario.py`
  (`slot_rank`, `rank_to_node`).

---

## 1. The identity model: nodes, GPUs (ranks), slots

Three coordinates locate a replica. Understanding them is the whole game:

| Term | Meaning | Why it matters |
|------|---------|----------------|
| **node** | A server-sized placement domain containing a contiguous block of ranks. | Lazarus survival is checked per node; scenario configs represent a full-node failure by failing every rank in that node block. |
| **local_rank** | Which GPU *within* a node, `0 .. gpus_per_node-1`. | Identifies the physical GPU. |
| **slot** | Which capacity slot *on* that GPU, `0 .. c-1` (`c` = `capacity`). | Multiple experts share one GPU; this is the per-GPU stacking dimension. |

The **global rank** (a flat GPU id across the whole cluster) is derived:

```
rank = node * gpus_per_node + local_rank
```

and inversely `node = rank // gpus_per_node`, `local_rank = rank % gpus_per_node`.

> In this codebase the **Lazarus node ≡ the toposim server**, so `gpus_per_node`
> and `ranks_per_node` are the same quantity under two names (training side vs
> scenario side). A 4-node × 4-GPU cluster has 16 ranks: node 1 owns ranks 4–7.

---

## 2. The data structure: `Slot` and `Placement`

```python
@dataclass(frozen=True)
class Slot:
    node: int
    local_rank: int
    slot: int = 0          # slot=0 is the capacity=1 special case

@dataclass
class Placement:
    expert_to_slots: Dict[int, List[Slot]]   # the mapping you want
    slot_to_expert: Dict[Slot, int]          # the inverse
    def replicas(self, expert_id) -> int      # = len(expert_to_slots[expert_id])
    def survives(self, failed_nodes) -> bool  # ≥1 replica on a live node, every expert
```

`expert_to_slots` **is** the canonical expert→GPU mapping. Everything else in this
doc is a view, a reduction, or an export of it.

---

## 3. Getting a `Placement`

```python
from allocator import plan_layer

# loads[e] = relative load of expert e (e.g. dispatch bytes from the traces)
replica_counts, placement = plan_layer(
    loads,
    num_nodes=4,
    gpus_per_node=4,
    capacity=32,          # c: slots per GPU -> 4*4*32 = 512 total slots
    k_min=2,              # min replicas/expert (single-node-failure survival)
    strategy="adaptive",  # "adaptive" (Lazarus) or "uniform" (fixed baseline)
)
```

If you already have replica counts and only want placement, call the second stage
directly:

```python
from allocator import allocate_replica_counts, mro_place

counts = allocate_replica_counts(loads, num_slots=512, k_min=2, strategy="adaptive")
placement = mro_place(counts, num_nodes=4, gpus_per_node=4, capacity=32)
```

---

## 4. The two views of the mapping

### 4a. Full physical view — slots (use `expert_to_slots`)

Every replica with its exact `(node, GPU, slot)`:

```python
placement.expert_to_slots[0]
# [Slot(node=0, local_rank=0, slot=0),
#  Slot(node=1, local_rank=2, slot=5),
#  Slot(node=2, local_rank=0, slot=1), ...]

placement.replicas(0)        # number of replicas for expert 0
placement.slot_to_expert[Slot(0, 0, 0)]   # which expert sits in that exact slot
```

Use this when you care about per-GPU stacking (e.g. memory accounting, how many
experts share a GPU, migration of a specific replica).

### 4b. GPU view — ranks (use `control_plane.expert_ranks`)

When you only care *which GPUs* host an expert (the usual case for routing and
failure analysis), collapse the slot dimension to global ranks:

```python
from control_plane import expert_ranks

mapping = expert_ranks(placement, gpus_per_node=4)
# {expert_id: [rank, rank, ...]}   rank = node*gpus_per_node + local_rank
mapping[0]   # e.g. [0, 6, 8]  -> global GPU ids hosting expert 0, sorted, deduped
```

On the scenario side the equivalent per-slot helper is `slot_rank(slot, ranks_per_node)`:

```python
import generate_scenario as gs
ranks = sorted({gs.slot_rank(s, ranks_per_node=4) for s in placement.expert_to_slots[0]})
```

> **Dedup caveat.** `expert_ranks` deduplicates ranks: if an expert holds two
> replicas in *different capacity slots on the same GPU*, they collapse to one
> rank entry. That's correct for routing/failure (a GPU is alive or dead as a
> whole), but if you need the slot-level replica *count*, read `expert_to_slots`
> — `len(expert_to_slots[e])` can exceed `len(expert_ranks(...)[e])`. With MRO at
> `k_min ≥ 2` replicas land on distinct nodes first, so this is rare.

---

## 5. Inverse mappings

### GPU → experts (what each rank holds)

```python
from collections import defaultdict

rank_to_experts: dict[int, list[int]] = defaultdict(list)
for expert, ranks in expert_ranks(placement, gpus_per_node=4).items():
    for r in ranks:
        rank_to_experts[r].append(expert)
```

Or, at slot granularity, straight from the inverse dict:

```python
for slot, expert in placement.slot_to_expert.items():
    ...  # slot.node, slot.local_rank, slot.slot -> expert
```

### Node → experts (what a failure domain holds)

```python
node_to_experts: dict[int, set[int]] = defaultdict(set)
for expert, slots in placement.expert_to_slots.items():
    for s in slots:
        node_to_experts[s.node].add(expert)
```

---

## 6. Mapping under failure

Drop dead GPUs and find which experts lost *all* replicas (model collapse):

```python
from control_plane import repair

failed_ranks = [4, 5, 6, 7]          # node 1 died (4 GPUs)
survivors, collapsed = repair(expert_ranks(placement, gpus_per_node=4), failed_ranks)
# survivors[e] = live ranks for expert e; collapsed = [e, ...] with no live replica
```

For the allocator's node-level survival check without building the rank map,
`placement.survives` answers directly:

```python
placement.survives(failed_nodes=[1])   # True iff every expert keeps a live replica
```

---

## 7. Exporting the mapping (JSON / CSV / pretty-print)

A self-contained dumper covering all three views:

```python
import json
from control_plane import expert_ranks

def dump_mapping(placement, gpus_per_node: int) -> dict:
    """Serialise the expert->GPU mapping in slot, rank, and node views."""
    ranks = expert_ranks(placement, gpus_per_node)
    return {
        "experts": {
            str(e): {
                "replicas": placement.replicas(e),
                "ranks": ranks.get(e, []),
                "nodes": sorted({s.node for s in slots}),
                "slots": [
                    {"node": s.node, "local_rank": s.local_rank, "slot": s.slot}
                    for s in slots
                ],
            }
            for e, slots in sorted(placement.expert_to_slots.items())
        }
    }

# JSON
print(json.dumps(dump_mapping(placement, 4), indent=2))

# CSV (one row per replica)
import csv, sys
w = csv.writer(sys.stdout)
w.writerow(["expert", "rank", "node", "local_rank", "slot"])
for e, slots in sorted(placement.expert_to_slots.items()):
    for s in slots:
        w.writerow([e, s.node * 4 + s.local_rank, s.node, s.local_rank, s.slot])
```

---

## 8. End-to-end example (real MMLU load)

```python
import sys; sys.path.insert(0, "traffic-gen")
import generate
from allocator import plan_layer
from control_plane import expert_ranks

# Per-expert load = aggregate dispatch bytes from the MMLU traces.
paths  = generate.discover_trace_paths()[: generate.BATCH_SIZE]
demand = generate.aggregate_demand(generate.build_demand_matrices(paths))
loads  = [sum(demand[s][e] for s in range(generate.NUM_RANKS))
          for e in range(generate.NUM_EXPERTS)]

counts, placement = plan_layer(loads, 4, 4, capacity=32, k_min=2, strategy="adaptive")

m = expert_ranks(placement, gpus_per_node=4)
hot = max(range(128), key=lambda e: loads[e])
print(f"expert {hot}: {placement.replicas(hot)} replicas on ranks {m[hot]}")
print(f"survives any single-node failure: {all(placement.survives([n]) for n in range(4))}")
```

With `strategy="adaptive"` the hot expert gets many replicas spread across GPUs;
with `strategy="uniform"` every expert gets the same flat count. Either way MRO
keeps replicas on distinct nodes, so single-node-failure survival holds.

---

## 9. Quick reference

| You want… | Use |
|-----------|-----|
| Expert → exact physical locations | `placement.expert_to_slots[e]` (list of `Slot`) |
| Expert → GPU ids (ranks) | `expert_ranks(placement, gpus_per_node)[e]` |
| Replica count for an expert | `placement.replicas(e)` or `len(expert_to_slots[e])` |
| Which expert is in a specific slot | `placement.slot_to_expert[Slot(...)]` |
| GPU/node → experts | invert (§5) |
| Survives a node-block failure? | `placement.survives(failed_nodes)` |
| Live ranks + collapsed experts after failure | `repair(expert_ranks(...), failed_ranks)` |
| Global rank ↔ (node, local_rank) | `rank = node*gpus_per_node + local_rank` |
| Slot → global rank (scenario side) | `slot_rank(slot, ranks_per_node)` |
```
