# Single Node Event per Step in the MVP

The MVP allows at most one node fail or join event at a simulation step. This keeps event ordering, repair, and output attribution simple while establishing the basic failure-aware traffic generation path.

**Consequences**

Scenarios with simultaneous node failures or joins should be rejected by the MVP parser. Future implementations can add batched node events once the single-event semantics are stable.

Because at most one node event occurs at a timestep in the MVP, step semantics are defined for either one fail event, one join event, or no event.
