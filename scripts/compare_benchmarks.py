#!/usr/bin/env python3
"""
Compare benchmark results with baseline.

This script compares current benchmark results with a baseline and reports
performance regressions or improvements.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any


def load_benchmark_results(filepath: Path) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    if not filepath.exists():
        return {}
    with open(filepath) as f:
        return json.load(f)


def compare_benchmarks(current: Dict[str, Any], baseline: Dict[str, Any]) -> None:
    """Compare current benchmark results with baseline."""
    if not baseline:
        print("No baseline found. Current results will serve as baseline.")
        return

    current_benchmarks = {
        b["name"]: b for b in current.get("benchmarks", [])
    }
    baseline_benchmarks = {
        b["name"]: b for b in baseline.get("benchmarks", [])
    }

    print("| Benchmark | Current | Baseline | Change |")
    print("|-----------|---------|----------|--------|")

    for name, current_data in current_benchmarks.items():
        if name not in baseline_benchmarks:
            print(f"| {name} | {current_data['stats']['mean']:.4f}s | NEW | - |")
            continue

        baseline_data = baseline_benchmarks[name]
        current_mean = current_data["stats"]["mean"]
        baseline_mean = baseline_data["stats"]["mean"]
        change_pct = ((current_mean - baseline_mean) / baseline_mean) * 100

        emoji = "✅" if change_pct <= 0 else "⚠️" if change_pct <= 10 else "❌"
        print(
            f"| {name} | {current_mean:.4f}s | {baseline_mean:.4f}s | "
            f"{emoji} {change_pct:+.1f}% |"
        )


def main() -> int:
    """Main entry point."""
    results_file = Path("benchmark-results.json")
    baseline_file = Path(".benchmark-baseline.json")

    current = load_benchmark_results(results_file)
    baseline = load_benchmark_results(baseline_file)

    compare_benchmarks(current, baseline)

    return 0


if __name__ == "__main__":
    sys.exit(main())
