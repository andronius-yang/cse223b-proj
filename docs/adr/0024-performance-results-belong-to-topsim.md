# Performance Results Belong to Topsim

`traffic-gen` emits traffic matrices and scenario metadata, but it does not compute performance results or total scenario latency. Timing, bandwidth, bottleneck analysis, and aggregation over ordered scenario steps belong to `topsim`.

**Consequences**

Failure-aware traffic generation should include enough metadata for `topsim` or future tooling to aggregate ordered results, but it should not add a performance-result summary inside `traffic-gen`.

Scenario configs and derived matrix manifests do not set scale-up or scale-out bandwidth. Bandwidth is supplied at `topsim` runtime.
