# Immediate Repair Migration in the MVP

When repair is required, the MVP emits all expected expert migration traffic for the missing planned replicas in one layer-clock tick. This exposes the full immediate recovery cost and keeps the generated scenario simple.

**Considered Options**

- Stage migration by upcoming layer use or by bandwidth budget: likely more realistic and could reduce per-step latency, but it requires a scheduling policy for which experts to move first and how inference proceeds while repair is incomplete.

**Consequences**

Recovery steps may contain large expert migration matrices. Future implementations can add lazy, layer-aware, or bandwidth-capped migration, but the MVP does not spread repair traffic across multiple timesteps.
