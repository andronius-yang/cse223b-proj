# Node-Addressed Failure Schedule

Status: superseded by ADR-0032.

Failure-aware scenarios originally identified fail and join events by node id, not rank id. The affected ranks were derived from the scenario's contiguous rank block node mapping.

**Consequences**

This has been replaced by rank-addressed fail/join events. A full-node case is now represented as a rank-block event.
