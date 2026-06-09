# Request-Progress Scenario Completion

Failure-aware scenarios complete when every request stream has finished its selected decode token/layer work. This differs from the current aggregate generator, which emits matrices by existing layer ids without modeling per-request progress or pause/resume behavior.

**Consequences**

The number of simulation steps may exceed the original lockstep decode length because failed ranks pause and later resume. If incomplete request streams are paused and no future join event can make progress possible, the scenario is deadlocked and should fail explicitly.

Scenario configs should not schedule node events after all request streams have completed. Such events are outside the modeled inference and should be rejected as configuration errors.

Scenario mode inherits the baseline decode-token policy: token `0` is prefill and skipped, decode tokens `1..32` are selected when available, and shorter traces complete after their last available selected decode token. Because dense layers are skipped, a request stream is complete when it has consumed all selected MoE communication layers for those decode tokens.

Completed request streams are inert. Later node events can affect expert state on their source ranks, but completed streams do not pause, resume, or produce more traffic.
