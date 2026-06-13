# Scenario Experiment Harness

This directory contains the deterministic replication/failure sweep used for the
scenario experiment stats and figure. It does not modify `traffic-gen`,
`allocator`, or `topsim`.

Run the full sweep:

```bash
python3 scenario_experiments/run_experiments.py
```

The runner:

- builds the trace-derived workload once using `traffic-gen/generate_scenario.py`;
- redirects scenario outputs to `scenario_experiments/results/scenarios/`;
- runs the single-owner no-replica control with experiment-only `LayerPlan`s;
- runs uniform and adaptive strategies through the existing allocator path;
- measures every generated `scenario_timeline.jsonl` with
  `uv run --project topsim topsim-timeline --policy direct --json`;
- regenerates `scenario_experiments/results/stats.csv` and
  `scenario_experiments/results/figures/full_node_recovery_2x2.png`.

Regenerate stats and figure from existing TopSim JSON:

```bash
uv run --project topsim python scenario_experiments/analyze_results.py
```

The large matrix timelines and TopSim JSON are generated artifacts under
`scenario_experiments/results/`.
