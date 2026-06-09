# Network-Only Scenario Matrices

Failure-aware scenario mode emits network-only matrices with diagonal entries zeroed for both inference AllToAllV and expert movement traffic. Local copies and local expert hits do not contribute rank-to-rank network bytes.

**Consequences**

Scenario mode does not emit paired original/network variants. Baseline aggregate mode may keep its existing original and network outputs, but scenario manifests reference only network-only matrices.

Scenario matrices use the same whitespace-separated integer byte `N x N` format as baseline outputs and `topsim` inputs.
