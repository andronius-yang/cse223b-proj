# Separate Matrices by Traffic Kind

The MVP emits expert migration and inference AllToAllV traffic as separate matrix files, linked by the same simulation step in a scenario manifest. This preserves whether bytes represent expert weight movement or activation routing while keeping each matrix compatible with `topsim`'s existing byte-matrix contract.

**Consequences**

Scenario consumers should evaluate all matrices for a step in phase order rather than treating one file as the whole timestep. Combining traffic kinds into one matrix is intentionally avoided because it hides recovery cost inside serving traffic.
