# Skip Dense Layers in Failure-Aware Traffic Generation

Failure-aware scenarios advance request streams through MoE communication layers only. Dense layers are skipped because `traffic-gen` outputs matrices for expert migration and collective communication, not dense compute timing.

**Consequences**

Layer-clock ticks should not be read as every transformer layer. They represent communication-bearing MoE layer work, preserving the current generator's boundary around derived communication demand.
