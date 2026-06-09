# Node-Addressed Failure Schedule

Failure-aware scenarios identify fail and join events by node id, not rank id. The affected ranks are derived from the scenario's contiguous rank block node mapping.

**Consequences**

The MVP models node-level churn, not individual GPU/rank failures. Rank-level failure scenarios can be added later as a separate event type if needed.
