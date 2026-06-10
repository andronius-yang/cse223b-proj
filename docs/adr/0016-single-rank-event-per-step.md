# Single Rank Event per Step in the MVP

The MVP allows at most one fail or join event at a simulation step. The event target is an explicit rank list, so a single event can model one-rank churn or a full-node rank block while preserving simple ordering, repair, and output attribution.

**Consequences**

Scenarios with simultaneous independent failures or joins should be rejected by the MVP parser. Future implementations can add batched rank events once the single-event semantics are stable.

Because at most one rank event occurs at a timestep in the MVP, step semantics are defined for either one fail event, one join event, or no event.
