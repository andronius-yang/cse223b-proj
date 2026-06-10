# Layer-Specific Expert Placement

Replica allocation, placement, and migration operate on layer experts, not globally shared expert ids. The same expert id in different MoE layers represents different expert state, so planned load and recovery decisions are keyed by `(layer_id, expert_id)`.

**Consequences**

Expert migration traffic is layer-specific. A rank losing expert 5 for layer 7 does not imply anything about expert 5 for another layer unless that separate layer expert was also placed on a failed rank.
