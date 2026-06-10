# Repair Toward Planned Placement

After rank join events, the MVP repairs missing replicas toward a precomputed planned placement target rather than recomputing a fresh global placement. Failure events remove capacity and expert state but do not trigger migration. For each missing planned replica whose destination rank rejoined, any surviving live replica is preferred as the network repair source; disk IO is used only when no live replica exists anywhere, and the two recovery paths are mutually exclusive for that replica.

**Consequences**

Repair is stable and interpretable, but it may miss opportunities to find a better temporary placement under the current live-node set. Adaptive re-optimization is future work.
