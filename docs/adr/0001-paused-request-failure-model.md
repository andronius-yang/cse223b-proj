# Paused Request Failure Model

Traffic generation models node failure as pausing request streams on unavailable ranks rather than permanently terminating them. When a node rejoins, request execution state such as request progress and any state needed to resume inference is assumed to be recovered outside the traffic model, while expert state on the failed node is assumed lost and must be restored through explicit expert migration traffic. This keeps the study focused on expert movement and recovery cost without letting total workload collapse as failures accumulate.

**Considered Options**

- Permanent fail-stop request loss: realistic for unrecovered crashes, but it makes traffic shrink over time and hides recovery behavior.
- Fully modeled request recovery traffic: more complete, but expands the scope beyond expert movement into request/KV-cache recovery.

**Consequences**

Simulation time is distinct from request progress. Each global timestep is a layer-clock tick: every live request stream advances by one request-local layer, while requests on failed ranks stop advancing and can fall out of token/layer lockstep. Scenario outputs should therefore be keyed by simulation step rather than only by global token/layer names.

Node events are effective before inference traffic for their scheduled step. A fail event pauses request streams on the failed node before that step's AllToAllV matrix is generated.

For a fail step `k`, the phase order is: apply fail event, remove the failed node's expert state from live placement, pause request streams on that node, then emit any live-request AllToAllV traffic for step `k`. Fail events do not emit repair migration in the MVP.
