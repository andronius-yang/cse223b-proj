# Per-Layer Expert Capacity

The MVP interprets placement capacity as expert slots per rank per MoE layer. This matches layer-specific expert placement and avoids modeling a global GPU memory budget shared across all layers.

**Consequences**

Replica allocation and placement are solved independently per MoE layer. Future work can add a cross-layer memory-capacity model if the study needs realistic whole-model residency constraints.

The scenario config exposes `capacity_per_rank_per_layer`, defaulting to `16` for the MVP. This default is chosen so 128 experts over 16 ranks can satisfy Lazarus `k_min = 2`.

Lazarus `k_min` is fixed at `2` in the MVP and is not exposed as a scenario config parameter.
