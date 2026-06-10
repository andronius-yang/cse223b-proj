# Emit Initial Expert Replication Cost

The MVP emits a special initial expert replication matrix that records the traffic required to realize the planned placement target before inference begins. Initial expert state is sourced from the baseline contiguous expert owners already used by `traffic-gen`, not from a synthetic bootstrap rank. This startup cost is separate from later expert migration caused by rank fail and join events.

**Consequences**

Initial placement is no longer treated as free. The matrix preserves the existing sharded expert ownership as the source of truth for where expert state starts, then charges traffic only when the planned placement target needs a replica on another rank. Scenario consumers can include or exclude the initial replication matrix depending on whether they want startup cost or only steady-state failure recovery cost, but the generator makes the cost explicit.

Baseline-owner self-copies generate no byte traffic. Local model-load or storage-read cost is outside the rank-to-rank traffic matrix model.

The initial replication matrix is included in the matrix-only `topsim-batch` manifest by default with `step = -1` metadata, indicating that it occurs before simulation timestep 0.

Simulation step numbering starts at `0` for the first inference tick; `step = -1` is reserved for initial expert replication.

The baseline owner is the source of initial expert state, but it is not forced into the planned placement target. Initial replication creates every planned replica that is not already on the baseline owner rank.
