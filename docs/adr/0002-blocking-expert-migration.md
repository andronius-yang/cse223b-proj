# Blocking Expert Migration in the MVP

For the MVP, expert migration is modeled as a blocking phase that completes before inference traffic in the same layer-clock tick. This gives `topsim` separate migration and inference matrices that can be evaluated sequentially, keeping the first recovery model simple and conservative.

**Considered Options**

- Overlap migration and inference: closer to a tuned runtime, but it requires scheduling semantics for partial expert availability and network contention between repair and serving traffic.

**Consequences**

Generated scenarios can sum migration and inference performance per timestep. A future implementation may add overlapped migration and inference, but that is outside the current feature boundary.
