# Lazarus Placement as Placement Truth

`traffic-gen` uses the root-level Lazarus allocator and placement policy as the sole source of planned layer-expert replica placement. It does not pin baseline owners or run a scenario-local placement adaptation.

**Consequences**

The baseline owner remains the source of initial expert state, but it is not placement truth. Initial replication charges movement from baseline owners to the planned Lazarus placement target.

The MVP imports and reuses the existing root-level `allocator.py` module for Lazarus replica allocation and placement helpers rather than copying or moving allocator logic into `traffic-gen`.
