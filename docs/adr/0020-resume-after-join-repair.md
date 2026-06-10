# Resume Requests After Join Repair

When ranks rejoin, their paused request streams resume only after join repair migration for that simulation step completes. The ranks' request execution state is assumed recoverable, but their expert state is blank until migration restores the planned replicas.

**Consequences**

The event order for a join step is: apply join, emit repair migration if needed, then allow the rejoined ranks to contribute inference traffic. This prevents pre-repair inference from blank ranks.

For a join step `k`, the phase order is: apply join event, mark the joined ranks as blank live capacity, emit join repair migration for missing planned replicas whose destination ranks rejoined, resume request streams on those ranks, then emit AllToAllV traffic for step `k`.
