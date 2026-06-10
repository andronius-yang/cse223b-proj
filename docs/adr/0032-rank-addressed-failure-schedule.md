# Rank-Addressed Failure Schedule

Failure-aware scenarios identify fail and join events by explicit rank list. A full-node case is represented by listing that node's contiguous rank block, while single-rank and mixed-rank churn use the same event shape.

**Consequences**

`failed_ranks` is the authoritative liveness state. `live_nodes` and `failed_nodes` are derived metadata for topology readability and TopSim compatibility, and a node is considered failed only when every rank in its block is failed. Timeline rows keep the `node_event` kind for compatibility, but event metadata records `ranks`, not `node`.
