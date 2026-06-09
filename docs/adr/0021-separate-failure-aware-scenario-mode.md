# Separate Failure-Aware Scenario Mode

Failure-aware traffic generation is a separate scenario mode rather than a replacement for the current aggregate layer outputs. The existing baseline output contract remains useful for simple per-layer matrices, while failure-aware scenarios require simulation steps, traffic kinds, event metadata, and per-request progress.

**Consequences**

The generator should preserve current aggregate files such as `layer_<id>_network.txt` and add scenario-specific outputs under a distinct scenario directory. Consumers should not assume baseline aggregate matrices and failure-aware scenario matrices have the same temporal semantics.

Failure-aware scenario mode should not emit aggregate matrices. Aggregation would erase simulation-step ordering, node-event effects, and the distinction between traffic kinds.

Failure-aware scenario mode reuses the baseline selected workload construction: local trace discovery, filesystem traversal order, the first `NUM_RANKS * REQUESTS_PER_RANK` traces, and contiguous request assignment to source ranks.

The MVP implements failure-aware scenario generation as a separate script rather than folding it into baseline `generate.py`. This preserves the parameter-free baseline generator while allowing scenario config, event state, manifests, and allocator integration in the new path.

The separate script is named `generate_scenario.py`.

Scenario outputs are written under `out/mmlu_english_partial/scenarios/<scenario_id>/`. The initial replication matrix is named `initial_expert_replication.txt` and carries `step = -1` in manifest metadata rather than using a negative step number in the filename.

Regular scenario matrix filenames use six-digit step padding and traffic kind suffixes, such as `step_000123_expert_migration.txt` and `step_000123_all2allv.txt`.

On rerun, scenario generation overwrites known outputs it owns for that scenario, such as manifests and generated matrix files, but does not blindly delete the entire scenario directory.
