# Repair Toward Planned Placement

After node join events, the MVP repairs missing replicas toward a precomputed planned placement target rather than recomputing a fresh global placement. Failure events remove capacity and expert state but do not trigger migration. This makes migration traffic represent the cost of restoring the intended layout when blank capacity returns, instead of mixing recovery with broad re-optimization.

**Consequences**

Repair is stable and interpretable, but it may miss opportunities to find a better temporary placement under the current live-node set. Adaptive re-optimization is future work.
