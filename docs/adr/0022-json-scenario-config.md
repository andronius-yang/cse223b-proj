# JSON Scenario Config

The MVP uses a JSON scenario config as the input for failure-aware generation. The config names the scenario, records ranks per node, and lists rank-addressed fail/join events by simulation step.

The default `ranks_per_node` is `4`, giving 4 nodes for the fixed 16-rank MVP workload.

**Consequences**

Failure scenarios can be checked in and rerun. CLI flags should select the scenario config rather than encoding event schedules directly in command-line arguments.

Scenario configs should be rejected if they try to fail an already failed rank or join an already live rank. Idempotent rank events are treated as schedule mistakes, not no-ops.

Event `step` values are absolute global layer-clock ticks starting at `0`. Step `0` is the first possible inference tick after initial expert replication.

Step numbers remain absolute even when no request stream advances, such as an event-only or migration-only step. The timeline must not renumber or compress paused intervals.

If all request streams are paused and no traffic occurs before a future event, the generator may jump directly to the next scheduled event step without emitting empty rows for the idle gap. The absolute step value of the next event preserves the skipped interval.

Scenario config includes `capacity_per_rank_per_layer`, defaulting to `16` for the MVP. With 128 experts, 16 ranks, and `k_min = 2`, this provides 256 per-layer slots so every expert can have at least two replicas.

The MVP `generate_scenario.py` CLI takes the scenario config path as its only argument. Scenario-specific knobs live in JSON for reproducibility.

`scenario_id` must be a filesystem-safe slug containing only letters, numbers, `_`, and `-`.

Static config validation should happen before writing outputs where possible. This includes scenario id, ranks-per-node divisibility, event ordering, one event per step, rank list shape, rank id bounds, fail/join state transitions, and capacity feasibility. Runtime deadlocks can still occur during generation when incomplete streams cannot advance and no future event can restore progress.

The MVP includes example configs at `scenarios/no_events.json`, `scenarios/node1_fail_join.json`, and `scenarios/rank_cycle_fail_join.json`. The no-event scenario must complete normally. The node 1 scenario represents node 1 as ranks `[4, 5, 6, 7]`, and the rank-cycle scenario exercises one-rank fail/join churn across the full rank range.
