# Join-Only Repair Migration

The MVP performs repair migration only after rank join events. Rank failure events immediately remove rank capacity, pause request streams on failed source ranks, and make expert replicas on failed ranks unavailable, but they do not trigger migration.

**Consequences**

A live request stream blocks if it needs a layer expert with no live replica after failures. The MVP does not proactively move replicas away from failed ranks; progress resumes only when a future join repair restores the needed expert state.

For a join step `k`, the phase order is: apply join event, mark the joined ranks as blank live capacity, emit join repair migration for missing planned replicas whose destination ranks rejoined, resume request streams on joined ranks, then emit AllToAllV traffic for step `k`.
