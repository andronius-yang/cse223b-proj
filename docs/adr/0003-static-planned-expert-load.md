# Static Planned Expert Load for MVP Replica Counts

The MVP uses full selected trace data to compute planned expert load before failure simulation begins, and Lazarus-style replica counts are based on that static load. This is an intentional oracle-like simplification for studying expert placement and movement under a known workload, not a realistic assumption that an online inference system knows future expert hotness.

**Considered Options**

- Recompute load from currently live requests after failures: more reactive, but it confounds placement decisions with temporary request pauses.
- Estimate hotness from prior traces or sliding windows: more realistic, but requires an online forecasting policy outside the current traffic-generation scope.

**Consequences**

Failure and join events affect available capacity and expert migration, but they do not change the planned popularity of each expert in the MVP. Future work can replace this with history-based or online hotness estimation.
