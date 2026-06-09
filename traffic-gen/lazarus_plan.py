from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Repo root holds allocator.py / control_plane.py; traffic-gen holds generate.py.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import generate  # noqa: E402  (same directory)
from allocator import plan_layer  # noqa: E402
from control_plane import (  # noqa: E402
    Policy,
    expert_ranks,
    repair,
    route_demand,
    total_bytes,
)

NUM_NODES = 4                       # servers = failure domain
GPUS_PER_NODE = 4                   # GPUs per server (4 servers x 4 = 16 ranks)
CAPACITY = 32                       # slots/GPU: 4*4*32 = 512 >> 128*k_min, leaves adaptive room
K_MIN = 2                           # survive any single server failure
REPLICATION_STRATEGY = "adaptive"   # "adaptive" (Lazarus) or "uniform" (fixed replicas/expert)
GPUS_PER_SERVER = GPUS_PER_NODE     # toposim server == Lazarus node here
FAIL_NODE = 1                       # fail server 1 (its ranks die together)

assert NUM_NODES * GPUS_PER_NODE == generate.NUM_RANKS, "node layout must cover 16 ranks"
FAILED_RANKS = list(range(FAIL_NODE * GPUS_PER_NODE, (FAIL_NODE + 1) * GPUS_PER_NODE))

OUT_DIR = Path("out/lazarus")
TOPSIM_DIR = REPO_ROOT / "topsim"


def write_matrix(path: Path, matrix: list[list[float]]) -> None:
    lines = [" ".join(f"{value:g}" for value in row) for row in matrix]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_scenarios() -> dict:
    trace_paths = generate.discover_trace_paths()
    if len(trace_paths) < generate.BATCH_SIZE:
        generate.fail(
            f"found {len(trace_paths)} trace files, need at least {generate.BATCH_SIZE}"
        )

    demand_by_layer = generate.build_demand_matrices(trace_paths)
    demand = generate.aggregate_demand(demand_by_layer)

    # Per-expert load = total dispatch bytes to that expert (placement input).
    loads = [
        sum(demand[src][expert] for src in range(generate.NUM_RANKS))
        for expert in range(generate.NUM_EXPERTS)
    ]

    # Baseline: today's contiguous single-owner placement (one replica/expert).
    baseline_ranks = {
        expert: [generate.choose_owner_rank(generate.owner_ranks(0, expert))]
        for expert in range(generate.NUM_EXPERTS)
    }

    # Replicated: chosen strategy ("adaptive" Lazarus / "uniform" fixed) + MRO placement.
    replica_counts, placement = plan_layer(
        loads,
        NUM_NODES,
        GPUS_PER_NODE,
        capacity=CAPACITY,
        k_min=K_MIN,
        strategy=REPLICATION_STRATEGY,
    )
    repl_ranks = expert_ranks(placement, GPUS_PER_NODE)
    survivors, collapsed = repair(repl_ranks, FAILED_RANKS)

    matrices = {
        "baseline": route_demand(demand, baseline_ranks, Policy.BASELINE, GPUS_PER_SERVER),
        "replicated": route_demand(demand, repl_ranks, Policy.REPLICATED, GPUS_PER_SERVER),
        "repaired": route_demand(
            demand, survivors, Policy.REPLICATED, GPUS_PER_SERVER, failed_ranks=FAILED_RANKS
        ),
    }
    return {
        "matrices": matrices,
        "replica_counts": replica_counts,
        "collapsed": collapsed,
        "layers": sorted(demand_by_layer),
        "loads": loads,
    }


def write_artifacts(scenarios: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    matrices = scenarios["matrices"]
    for name, matrix in matrices.items():
        write_matrix(OUT_DIR / f"{name}.txt", matrix)

    rows = [
        {
            "id": "baseline",
            "matrix": "baseline.txt",
            "metadata": {"phase": "dispatch", "placement_id": "baseline"},
        },
        {
            "id": "replicated",
            "matrix": "replicated.txt",
            "metadata": {
                "phase": "dispatch",
                "placement_id": "lazarus_mro",
                "replication_strategy": REPLICATION_STRATEGY,
                "k_min": K_MIN,
                "capacity": CAPACITY,
            },
        },
        {
            "id": "repaired",
            "matrix": "repaired.txt",
            "failed_gpus": FAILED_RANKS,
            "metadata": {
                "phase": "dispatch",
                "placement_id": "lazarus_mro",
                "replication_strategy": REPLICATION_STRATEGY,
                "repair_policy": "surviving_replicas",
                "failed_server": FAIL_NODE,
                "failed_gpus": FAILED_RANKS,
            },
        },
    ]
    manifest = OUT_DIR / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return manifest


def run_toposim(manifest: Path) -> dict | None:
    results_json = OUT_DIR / "results.json"
    cmd = [
        "uv", "run", "--project", str(TOPSIM_DIR), "toposim-batch", str(manifest),
        "--policy", "fast", "--gpus-per-server", str(GPUS_PER_SERVER),
        "--json", str(results_json),
    ]
    try:
        subprocess.run(cmd, check=True, cwd=Path.cwd())
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"\n[toposim run skipped: {exc}]")
        print("Run it manually with:\n  " + " ".join(cmd))
        return None
    return json.loads(results_json.read_text(encoding="utf-8"))


def print_summary(scenarios: dict, results: dict | None) -> None:
    counts = scenarios["replica_counts"]
    print(f"\nlayers observed: {len(scenarios['layers'])}")
    print(f"replication strategy: {REPLICATION_STRATEGY}")
    print(f"replicas/expert: min={min(counts)} max={max(counts)} sum={sum(counts)} (E=128)")
    print(
        f"collapsed experts after server{FAIL_NODE} failure "
        f"(ranks {FAILED_RANKS}): {len(scenarios['collapsed'])}"
    )
    for name, matrix in scenarios["matrices"].items():
        print(f"  {name:<11} total network bytes: {total_bytes(matrix):,.0f}")

    if results is None:
        return
    print(f"\n{'scenario':<12} {'policy':<8} {'completion_us':>14} {'slowdown_vs_lb':>15}")
    for scenario in results["scenarios"]:
        for result in scenario["results"]:
            print(
                f"  {scenario['id']:<10} {result['policy']:<8} "
                f"{result['completion_time_us']:>14.2f} {result['slowdown_vs_lb']:>15.3f}"
            )


def main() -> None:
    generate.validate_constants()
    scenarios = build_scenarios()
    manifest = write_artifacts(scenarios)
    print(f"wrote 3 matrices + manifest to {OUT_DIR}")
    results = run_toposim(manifest)
    print_summary(scenarios, results)


if __name__ == "__main__":
    main()
