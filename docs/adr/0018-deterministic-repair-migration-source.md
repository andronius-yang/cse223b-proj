# Deterministic Repair Migration Source

For repair migration, the MVP chooses a source replica by preferring a live replica on the destination node and otherwise using the lowest-rank live replica. This is a deterministic convention for reproducible traffic generation, not an optimized migration scheduler.

**Consequences**

Repair migration may not minimize total network cost under all topologies. Future implementations can add topology-aware or bandwidth-aware source selection.
