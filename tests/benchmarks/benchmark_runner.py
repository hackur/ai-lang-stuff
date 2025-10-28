"""
Unified benchmark runner for all performance tests.

This module provides:
- Unified interface to run all benchmarks
- Report generation (JSON, CSV, HTML)
- Comparison visualizations
- CI/CD integration support

Usage:
    python benchmark_runner.py --all
    python benchmark_runner.py --models
    python benchmark_runner.py --vector-stores
    python benchmark_runner.py --workflows
    python benchmark_runner.py --compare results1.json results2.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

try:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")  # Non-interactive backend
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting disabled.")

from agent_workflow_performance import AgentWorkflowBenchmark
from model_performance import ModelBenchmark
from vector_store_performance import VectorStoreBenchmark


class BenchmarkRunner:
    """Unified benchmark runner and report generator."""

    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize benchmark runner."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.all_results = {}

    def run_model_benchmarks(self, quick: bool = False) -> Dict[str, Any]:
        """
        Run model performance benchmarks.

        Args:
            quick: If True, run quick benchmark with fewer prompts

        Returns:
            Summary results
        """
        print("\n" + "=" * 70)
        print("RUNNING MODEL PERFORMANCE BENCHMARKS")
        print("=" * 70)

        benchmark = ModelBenchmark(output_dir=str(self.output_dir))

        if quick:
            prompt_types = ["simple", "short"]
        else:
            prompt_types = None  # All prompts

        summary = benchmark.benchmark_all_models(prompt_types=prompt_types)
        benchmark.save_results(format="json")
        benchmark.save_results(format="csv")

        self.all_results["model_performance"] = summary
        return summary

    def run_vector_store_benchmarks(self) -> Dict[str, Any]:
        """
        Run vector store performance benchmarks.

        Returns:
            Summary results
        """
        print("\n" + "=" * 70)
        print("RUNNING VECTOR STORE BENCHMARKS")
        print("=" * 70)

        benchmark = VectorStoreBenchmark(output_dir=str(self.output_dir))

        try:
            summary = benchmark.benchmark_all_stores()
            benchmark.save_results(format="json")
            benchmark.save_results(format="csv")

            self.all_results["vector_store_performance"] = summary
            return summary

        finally:
            benchmark.cleanup()

    def run_workflow_benchmarks(self) -> Dict[str, Any]:
        """
        Run agent workflow benchmarks.

        Returns:
            Summary results
        """
        print("\n" + "=" * 70)
        print("RUNNING AGENT WORKFLOW BENCHMARKS")
        print("=" * 70)

        benchmark = AgentWorkflowBenchmark(output_dir=str(self.output_dir))

        summary = benchmark.benchmark_all_workflows()
        benchmark.save_results(format="json")
        benchmark.save_results(format="csv")

        self.all_results["agent_workflow_performance"] = summary
        return summary

    def run_all_benchmarks(self, quick: bool = False):
        """
        Run all benchmark suites.

        Args:
            quick: If True, run quick benchmarks
        """
        start_time = time.time()

        print("\n" + "=" * 70)
        print("COMPREHENSIVE BENCHMARK SUITE")
        print("=" * 70)
        print(f"Output directory: {self.output_dir}")
        print(f"Quick mode: {quick}")
        print("=" * 70)

        # Run all benchmarks
        try:
            self.run_model_benchmarks(quick=quick)
        except Exception as e:
            print(f"\nModel benchmarks failed: {e}")

        try:
            self.run_vector_store_benchmarks()
        except Exception as e:
            print(f"\nVector store benchmarks failed: {e}")

        try:
            self.run_workflow_benchmarks()
        except Exception as e:
            print(f"\nWorkflow benchmarks failed: {e}")

        elapsed_time = time.time() - start_time

        # Generate summary report
        self.generate_summary_report(elapsed_time)

    def generate_summary_report(self, elapsed_time: float):
        """
        Generate comprehensive summary report.

        Args:
            elapsed_time: Total benchmark execution time
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"benchmark_summary_{timestamp}.json"

        report = {
            "metadata": {
                "timestamp": timestamp,
                "total_duration_seconds": elapsed_time,
                "benchmarks_run": list(self.all_results.keys()),
            },
            "results": self.all_results,
        }

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"Total duration: {elapsed_time:.1f}s")
        print(f"Benchmarks run: {len(self.all_results)}")
        print(f"\nSummary report: {report_file}")
        print("=" * 70)

        # Print key metrics
        self._print_key_metrics()

        # Generate visualizations if available
        if PLOTTING_AVAILABLE:
            try:
                self.generate_visualizations(timestamp)
            except Exception as e:
                print(f"\nVisualization generation failed: {e}")

    def _print_key_metrics(self):
        """Print key metrics from all benchmarks."""
        print("\nKey Metrics:")
        print("-" * 70)

        # Model performance
        if "model_performance" in self.all_results:
            model_data = self.all_results["model_performance"]
            if "by_model" in model_data:
                print("\nModel Performance (avg latency):")
                for model, metrics in model_data["by_model"].items():
                    latency = metrics.get("avg_latency_ms", 0)
                    tps = metrics.get("avg_tokens_per_second", 0)
                    print(f"  {model:20s}: {latency:8.0f}ms  ({tps:.1f} tok/s)")

        # Vector store performance
        if "vector_store_performance" in self.all_results:
            vs_data = self.all_results["vector_store_performance"]
            if "by_store" in vs_data:
                print("\nVector Store Performance (avg throughput):")
                for store, metrics in vs_data["by_store"].items():
                    throughput = metrics.get("avg_throughput", 0)
                    print(f"  {store:20s}: {throughput:8.1f} ops/s")

        # Workflow performance
        if "agent_workflow_performance" in self.all_results:
            wf_data = self.all_results["agent_workflow_performance"]
            if "by_workflow" in wf_data:
                print("\nAgent Workflow Performance (avg latency):")
                for workflow, metrics in wf_data["by_workflow"].items():
                    latency = metrics.get("avg_total_latency_ms", 0)
                    print(f"  {workflow:20s}: {latency:8.0f}ms")

    def generate_visualizations(self, timestamp: str):
        """
        Generate visualization charts.

        Args:
            timestamp: Timestamp for file naming
        """
        if not PLOTTING_AVAILABLE:
            print("Matplotlib not available, skipping visualizations")
            return

        print("\nGenerating visualizations...")

        # Model performance chart
        if "model_performance" in self.all_results:
            self._plot_model_performance(timestamp)

        # Vector store comparison
        if "vector_store_performance" in self.all_results:
            self._plot_vector_store_performance(timestamp)

        # Workflow timing
        if "agent_workflow_performance" in self.all_results:
            self._plot_workflow_performance(timestamp)

    def _plot_model_performance(self, timestamp: str):
        """Generate model performance visualization."""
        data = self.all_results["model_performance"]
        if "by_model" not in data:
            return

        models = []
        latencies = []
        throughputs = []

        for model, metrics in data["by_model"].items():
            models.append(model.replace(":", "\n"))  # Wrap long names
            latencies.append(metrics.get("avg_latency_ms", 0))
            throughputs.append(metrics.get("avg_tokens_per_second", 0))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Latency chart
        ax1.bar(models, latencies, color="steelblue")
        ax1.set_ylabel("Latency (ms)")
        ax1.set_title("Model Latency Comparison")
        ax1.tick_params(axis="x", rotation=45)

        # Throughput chart
        ax2.bar(models, throughputs, color="coral")
        ax2.set_ylabel("Tokens/Second")
        ax2.set_title("Model Throughput Comparison")
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        output_file = self.output_dir / f"model_performance_{timestamp}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  - Model performance chart: {output_file}")

    def _plot_vector_store_performance(self, timestamp: str):
        """Generate vector store performance visualization."""
        data = self.all_results["vector_store_performance"]
        if "by_store" not in data:
            return

        stores = []
        throughputs = []

        for store, metrics in data["by_store"].items():
            stores.append(store.upper())
            throughputs.append(metrics.get("avg_throughput", 0))

        plt.figure(figsize=(8, 5))
        plt.bar(stores, throughputs, color=["steelblue", "coral"])
        plt.ylabel("Operations/Second")
        plt.title("Vector Store Throughput Comparison")
        plt.tight_layout()

        output_file = self.output_dir / f"vector_store_performance_{timestamp}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  - Vector store chart: {output_file}")

    def _plot_workflow_performance(self, timestamp: str):
        """Generate workflow performance visualization."""
        data = self.all_results["agent_workflow_performance"]
        if "by_workflow" not in data:
            return

        workflows = []
        latencies = []

        for workflow, metrics in data["by_workflow"].items():
            workflows.append(workflow.replace("_", " ").title())
            latencies.append(metrics.get("avg_total_latency_ms", 0))

        plt.figure(figsize=(8, 5))
        plt.bar(workflows, latencies, color="mediumseagreen")
        plt.ylabel("Latency (ms)")
        plt.title("Agent Workflow Performance")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        output_file = self.output_dir / f"workflow_performance_{timestamp}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  - Workflow chart: {output_file}")

    @staticmethod
    def compare_results(file1: Path, file2: Path):
        """
        Compare two benchmark result files.

        Args:
            file1: Path to first results file
            file2: Path to second results file
        """
        with open(file1) as f:
            results1 = json.load(f)
        with open(file2) as f:
            results2 = json.load(f)

        print("\n" + "=" * 70)
        print("BENCHMARK COMPARISON")
        print("=" * 70)
        print(f"File 1: {file1.name}")
        print(f"File 2: {file2.name}")
        print("=" * 70)

        # Compare model performance
        if "model_performance" in results1 and "model_performance" in results2:
            print("\nModel Performance Changes:")
            BenchmarkRunner._compare_model_performance(
                results1["model_performance"], results2["model_performance"]
            )

        # Compare vector stores
        if "vector_store_performance" in results1 and "vector_store_performance" in results2:
            print("\nVector Store Performance Changes:")
            BenchmarkRunner._compare_vector_stores(
                results1["vector_store_performance"], results2["vector_store_performance"]
            )

    @staticmethod
    def _compare_model_performance(data1: Dict, data2: Dict):
        """Compare model performance data."""
        models1 = data1.get("by_model", {})
        models2 = data2.get("by_model", {})

        for model in set(models1.keys()) | set(models2.keys()):
            if model in models1 and model in models2:
                lat1 = models1[model].get("avg_latency_ms", 0)
                lat2 = models2[model].get("avg_latency_ms", 0)
                diff = ((lat2 - lat1) / lat1 * 100) if lat1 > 0 else 0
                direction = "slower" if diff > 0 else "faster"
                print(f"  {model:20s}: {abs(diff):6.1f}% {direction}")

    @staticmethod
    def _compare_vector_stores(data1: Dict, data2: Dict):
        """Compare vector store performance data."""
        stores1 = data1.get("by_store", {})
        stores2 = data2.get("by_store", {})

        for store in set(stores1.keys()) | set(stores2.keys()):
            if store in stores1 and store in stores2:
                tp1 = stores1[store].get("avg_throughput", 0)
                tp2 = stores2[store].get("avg_throughput", 0)
                diff = ((tp2 - tp1) / tp1 * 100) if tp1 > 0 else 0
                direction = "faster" if diff > 0 else "slower"
                print(f"  {store:20s}: {abs(diff):6.1f}% {direction}")


def main():
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Unified benchmark runner for ai-lang-stuff performance tests"
    )
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--models", action="store_true", help="Run model performance benchmarks")
    parser.add_argument("--vector-stores", action="store_true", help="Run vector store benchmarks")
    parser.add_argument("--workflows", action="store_true", help="Run agent workflow benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks (fewer tests)")
    parser.add_argument(
        "--compare", nargs=2, metavar=("FILE1", "FILE2"), help="Compare two benchmark result files"
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)",
    )

    args = parser.parse_args()

    # Handle comparison mode
    if args.compare:
        file1 = Path(args.compare[0])
        file2 = Path(args.compare[1])
        if not file1.exists() or not file2.exists():
            print("Error: One or both comparison files do not exist")
            sys.exit(1)
        BenchmarkRunner.compare_results(file1, file2)
        return

    # Create runner
    runner = BenchmarkRunner(output_dir=args.output_dir)

    # Determine what to run
    if args.all:
        runner.run_all_benchmarks(quick=args.quick)
    else:
        any_run = False

        if args.models:
            runner.run_model_benchmarks(quick=args.quick)
            any_run = True

        if args.vector_stores:
            runner.run_vector_store_benchmarks()
            any_run = True

        if args.workflows:
            runner.run_workflow_benchmarks()
            any_run = True

        if not any_run:
            print("No benchmarks specified. Use --all or specify individual benchmarks.")
            print("Run with --help for usage information.")
            sys.exit(1)

        # Generate summary if multiple benchmarks were run
        if sum([args.models, args.vector_stores, args.workflows]) > 1:
            runner.generate_summary_report(0)


if __name__ == "__main__":
    main()
