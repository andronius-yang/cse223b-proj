# Partial Outputs on Terminal Scenario Failure

When a failure-aware scenario reaches a no-progress state with no future event that can restore progress, the MVP keeps outputs generated before the failure and records a terminal failure row in the scenario timeline manifest. It does not emit an inference matrix for the failed step.

**Consequences**

Partial outputs can be inspected to understand the lead-up to failure, but the scenario must be treated as invalid. The generator should make terminal failure explicit rather than silently truncating output.

The generator should exit nonzero after a terminal scenario failure, even when partial outputs were written.

The scenario timeline manifest should be written incrementally as generation progresses so terminal failures leave a valid partial timeline plus the terminal failure row.
