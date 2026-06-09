# Fixed Llama4 Maverick Expert State Bytes

The MVP uses an explicit `EXPERT_STATE_BYTES` value for expert migration traffic, separate from activation payload bytes. The value is fixed to one Llama4 Maverick BF16 MoE expert weight replica: `3 * hidden_size * intermediate_size * 2 = 3 * 5120 * 8192 * 2 = 251658240` bytes. Each `(layer_id, expert_id)` has distinct expert state, so migrating the same expert id in two layers counts as two independent expert-state movements.

**Consequences**

Migration traffic reflects expert weight movement, not token activation movement. Future implementations can make this parameter model-config-derived, quantization-aware, or user-supplied, but the MVP keeps it fixed for reproducibility.
