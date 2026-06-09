# Baseline-Pinned Lazarus Placement

`traffic-gen` uses Lazarus-style replica counts but adapts placement by pinning each layer expert's baseline owner as one planned replica where possible. Remaining replicas are placed around that constraint, preserving the existing sharded source of expert state while still using load-aware replication.

**Consequences**

This is an MVP design choice, not a claim that raw Lazarus placement requires baseline-owner pinning. It avoids initial placement semantics where the only existing expert copy is treated as if it must be moved away before replication can begin.

The MVP imports and reuses the existing root-level `allocator.py` module for Lazarus replica allocation and placement helpers rather than copying or moving allocator logic into `traffic-gen`.
