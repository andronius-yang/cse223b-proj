# Resume Requests After Join Repair

When a node rejoins, its paused request streams resume only after join repair migration for that simulation step completes. The node's request execution state is assumed recoverable, but its expert state is blank until migration restores the planned replicas.

**Consequences**

The event order for a join step is: apply join, emit repair migration if needed, then allow the rejoined ranks to contribute inference traffic. This prevents pre-repair inference from a blank node.

For a join step `k`, the phase order is: apply join event, mark the node as blank live capacity, emit join repair migration for missing planned replicas on that node, resume request streams on that node, then emit AllToAllV traffic for step `k`.
