# Join-Only Repair Migration

The MVP performs repair migration only after node join events. Node failure events immediately remove rank capacity, pause request streams on the failed node, and make expert replicas on that node unavailable, but they do not trigger migration.

**Consequences**

A live request stream blocks if it needs a layer expert with no live replica after failures. The MVP does not proactively move replicas away from failed nodes; progress resumes only when a future join repair restores the needed expert state.
