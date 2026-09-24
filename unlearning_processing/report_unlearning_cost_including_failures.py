#!/usr/bin/env python3
"""Print per-job cost with failures, optional exclusions, and critic distance."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Hard-coded critic-distance controls.
# Set ENABLE_CRITIC_DISTANCE=False to retain the original cost-only behavior.
# Missing valid caches are evaluated on D_f and saved beside each job result.
# ---------------------------------------------------------------------------
ENABLE_CRITIC_DISTANCE = True
CRITIC_DISTANCE_GPU = 0       # Use -1 for CPU.
CRITIC_DISTANCE_BATCH_SIZE = 512

ENABLE_TRAJDELETER_10K_REFERENCE = True
TRAJDELETER_REFERENCE_STEPS = 10000
TRAJDELETER_CRITIC_STATS_ROOT = (
    Path(__file__).resolve().parents[1]
    / "Offline_RL_processing"
    / "stats_results_critic"
    / "stats_critic_pointmaze"
)


try:  # Supports both python -m and direct script invocation.
    from unlearning_processing.unlearning_cost_reporting import (
        SUCCESS_STATUS,
        critic_distance_mean,
        finite_number,
        format_value,
        load_batch_runs,
        print_summary,
        resolve_batch_root,
        summarize_runs,
        write_summary_csv,
    )
    from unlearning_processing.unlearning_cost_critic_distance import (
        evaluate_or_load_critic_distances,
    )
    from unlearning_processing.trajdeleter_critic_reference import (
        all_reference_values,
        attach_reference_fields,
        load_trajdeleter_reference_summaries,
    )
except ModuleNotFoundError:
    from unlearning_cost_reporting import (
        SUCCESS_STATUS,
        critic_distance_mean,
        finite_number,
        format_value,
        load_batch_runs,
        print_summary,
        resolve_batch_root,
        summarize_runs,
        write_summary_csv,
    )
    from unlearning_cost_critic_distance import evaluate_or_load_critic_distances
    from trajdeleter_critic_reference import (
        all_reference_values,
        attach_reference_fields,
        load_trajdeleter_reference_summaries,
    )


def run_algorithm(run: Dict[str, Any]) -> Optional[str]:
    task = run.get("task")
    if isinstance(task, dict) and task.get("algo"):
        return str(task["algo"])
    value = run.get("batch_algo")
    return str(value) if value else None


def median_number(values: Sequence[Any]) -> Optional[float]:
    numbers = [
        number
        for value in values
        for number in [finite_number(value)]
        if number is not None
    ]
    return float(median(numbers)) if numbers else None


def print_critic_distance_summary(rows: Sequence[Dict[str, Any]]) -> None:
    print("\nCritic distance summary（final model vs. retrain model；D_f）")
    print("Distance:  mean |Q_final(s,a) - Q_retrained(s,a)|。")
    print(
        "{:<36} {:<18} {:>9} {:>22}".format(
            "job", "method", "samples", "mean_D_f_critic_dist"
        )
    )
    for row in rows:
        print(
            "{:<36} {:<18} {:>9} {:>22}".format(
                row["job"][:36],
                row["method"][:18],
                row["critic_distance_samples"],
                format_value(row["mean_D_f_critic_distance"]),
            )
        )


def print_trajdeleter_reference_summary(summaries: Dict[str, Any]) -> None:
    """Print 10k two-phase TrajDeleter reference medians."""
    print(
        "\nTrajDeleter 10k critic reference "
        "（phase1=8k, phase2=2k；final model vs. retrained；D_f）"
    )
    print(
        "median_D_f_mean across seeds；"
        "median_D_f_median across seeds。"
    )
    print(
        "{:<36} {:<12} {:>11} {:>20} {:>22}".format(
            "job",
            "algorithm",
            "records",
            "median_D_f_mean",
            "median_D_f_median",
        )
    )
    for job, summary in sorted(summaries.items()):
        print(
            "{:<36} {:<12} {:>11} {:>20} {:>22}".format(
                job[:36],
                summary.algorithm[:12],
                "{}/{}".format(len(summary.records), len(summary.expected_seeds)),
                format_value(summary.median_D_f_mean),
                format_value(summary.median_D_f_transition_median),
            )
        )
    values = all_reference_values(summaries)
    print(
        "overall matched records={} median_D_f_mean={}".format(
            len(values), format_value(median_number(values))
        )
    )


def print_method_summary(
    runs: Sequence[Dict[str, Any]],
    include_critic_distance: bool = False,
    reference_summaries: Optional[Dict[str, Any]] = None,
) -> None:
    """Print all-run medians, matching this script's inclusive failure policy."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[str(run.get("method") or "unknown")].append(run)

    print("\n Method summary（All jobs）")
    print(" steps_trained, iterations used.")
    if include_critic_distance:
        print(
            "{:<18} {:>7} {:>9} {:>12} {:>18} {:>18} {:>20}".format(
                "method",
                "total",
                "success",
                "success_rate",
                "median_train_steps",
                "median_wall_min",
                "median_critic_D_f",
            )
        )
    else:
        print(
            "{:<18} {:>7} {:>9} {:>12} {:>18} {:>18}".format(
                "method",
                "total",
                "success",
                "success_rate",
                "median_train_steps",
                "median_wall_min",
            )
        )

    for method, group in sorted(grouped.items()):
        successful = sum(run.get("status") == SUCCESS_STATUS for run in group)
        total = len(group)
        wall_minutes = [
            value / 60.0
            for run in group
            for value in [finite_number(run.get("training_wall_seconds"))]
            if value is not None
        ]
        values: List[Any] = [
            method[:18],
            total,
            successful,
            format_value(successful / total if total else None),
            format_value(median_number([run.get("steps_trained") for run in group])),
            format_value(median_number(wall_minutes)),
        ]
        if include_critic_distance:
            values.append(
                format_value(
                    median_number([critic_distance_mean(run) for run in group])
                )
            )
            print(
                "{:<18} {:>7} {:>9} {:>12} {:>18} {:>18} {:>20}".format(
                    *values
                )
            )
        else:
            print(
                "{:<18} {:>7} {:>9} {:>12} {:>18} {:>18}".format(*values)
            )

    if include_critic_distance and reference_summaries:
        reference_values = all_reference_values(reference_summaries)
        expected = sum(
            len(summary.expected_seeds)
            for summary in reference_summaries.values()
        )
        matched = len(reference_values)
        print(
            "{:<18} {:>7} {:>9} {:>12} {:>18} {:>18} {:>20}".format(
                "TrajDeleter-10k",
                expected,
                matched,
                format_value(matched / expected if expected else None),
                TRAJDELETER_REFERENCE_STEPS,
                "NA",
                format_value(median_number(reference_values)),
            )
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report inclusive unlearning cost, with optional algorithm exclusions."
    )
    parser.add_argument(
        "--batch-output",
        required=True,
        help="Batch output directory or unlearning_cost_batch_manifest.json.",
    )
    parser.add_argument(
        "--wo-algo",
        nargs="+",
        default=[],
        help="Algorithms to exclude, case-insensitively; for example: --wo-algo CQL.",
    )
    parser.add_argument("--output-csv", help="Optional per-job summary CSV path.")
    args = parser.parse_args(argv)
    try:
        batch_root = resolve_batch_root(args.batch_output)
        runs, warnings = load_batch_runs(batch_root)
    except ValueError as error:
        parser.error(str(error))

    excluded_algorithms = {value.upper() for value in args.wo_algo}
    excluded_count = 0
    if excluded_algorithms:
        retained = []
        for run in runs:
            algorithm = run_algorithm(run)
            if algorithm is not None and algorithm.upper() in excluded_algorithms:
                excluded_count += 1
            else:
                retained.append(run)
        runs = retained
        print(
            "已排除算法：{}（{} 条 run）。".format(
                ", ".join(sorted(excluded_algorithms)), excluded_count
            )
        )

    if ENABLE_CRITIC_DISTANCE:
        critic_report = evaluate_or_load_critic_distances(
            runs,
            gpu=CRITIC_DISTANCE_GPU,
            batch_size=CRITIC_DISTANCE_BATCH_SIZE,
        )
        print(
            "[critic] cache hits={}, newly evaluated={}, skipped={}, failed={}.".format(
                critic_report.cache_hits,
                critic_report.evaluated,
                critic_report.skipped,
                critic_report.failed,
            )
        )
        warnings.extend(critic_report.warnings)

    reference_summaries: Dict[str, Any] = {}
    if ENABLE_TRAJDELETER_10K_REFERENCE:
        reference_summaries, reference_warnings = (
            load_trajdeleter_reference_summaries(
                TRAJDELETER_CRITIC_STATS_ROOT,
                runs,
                total_steps=TRAJDELETER_REFERENCE_STEPS,
            )
        )
        warnings.extend(reference_warnings)

    rows = summarize_runs(runs, include_failures=True)
    if ENABLE_TRAJDELETER_10K_REFERENCE:
        attach_reference_fields(rows, reference_summaries)
    print_summary(rows, include_failures=True)
    if ENABLE_CRITIC_DISTANCE:
        print_critic_distance_summary(rows)
    if ENABLE_TRAJDELETER_10K_REFERENCE:
        print_trajdeleter_reference_summary(reference_summaries)
    print_method_summary(
        runs,
        include_critic_distance=ENABLE_CRITIC_DISTANCE,
        reference_summaries=reference_summaries,
    )
    output_path = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv
        else batch_root / "unlearning_cost_mean_runtime_including_failures.csv"
    )
    write_summary_csv(output_path, rows)
    print("Saved CSV to {}".format(output_path))
    for warning in warnings:
        print("[warning] {}".format(warning))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())