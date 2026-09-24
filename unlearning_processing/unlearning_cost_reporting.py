#!/usr/bin/env python3
"""Shared readers and summaries for unlearning-cost batch outputs.

The batch runner stores one ``unlearning_cost_results.json`` per job.  This
module intentionally reads that public aggregate rather than training logs, so
the reporting tools work for both completed and partially failed batches.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SUCCESS_STATUS = "achieved"
FAILURE_STATUSES = {"not_achieved", "failed", "preflight_error", "missing_result"}
SUMMARY_COLUMNS = [
    "job",
    "method",
    "total_runs",
    "achieved_runs",
    "not_achieved_runs",
    "failed_runs",
    "preflight_error_runs",
    "missing_result_runs",
    "failure_probability",
    "runtime_samples",
    "mean_training_wall_seconds",
    "mean_training_wall_minutes",
    "mean_steps_trained",
    "mean_steps_to_threshold",
    "critic_distance_samples",
    "mean_D_f_critic_distance",
    "trajdeleter_10k_reference_samples",
    "median_trajdeleter_10k_D_f_critic_distance",
    "median_trajdeleter_10k_D_f_transition_median",
]


def finite_number(value: Any) -> Optional[float]:
    """Return a finite float, or ``None`` for an absent/invalid value."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def critic_distance_mean(run: Dict[str, Any]) -> Optional[float]:
    """Return a run's cached D_f mean absolute critic difference."""
    payload = run.get("critic_distance")
    if not isinstance(payload, dict):
        return None
    statistics = payload.get("critic_diff_statistics")
    if not isinstance(statistics, dict):
        return None
    return finite_number(statistics.get("D_f_mean"))


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as input_file:
            value = json.load(input_file)
    except (OSError, ValueError) as error:
        raise ValueError("Cannot read {}: {}".format(path, error)) from error
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object.".format(path))
    return value


def resolve_batch_root(value: str) -> Path:
    """Accept a batch directory or its manifest path."""
    path = Path(value).expanduser().resolve()
    if path.is_file():
        if path.name != "unlearning_cost_batch_manifest.json":
            raise ValueError("--batch-output must be a batch directory or manifest path.")
        path = path.parent
    manifest = path / "unlearning_cost_batch_manifest.json"
    if not manifest.is_file():
        raise ValueError("Batch manifest does not exist: {}".format(manifest))
    return path


def _expected_runs(job: Dict[str, Any]) -> Iterable[Tuple[str, int]]:
    config = job.get("job") if isinstance(job.get("job"), dict) else {}
    methods = config.get("methods") or []
    seeds = config.get("seeds") or []
    for method in methods:
        for seed in seeds:
            try:
                yield str(method), int(seed)
            except (TypeError, ValueError):
                continue


def load_batch_runs(batch_root: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return runs annotated with their batch job name and read warnings.

    Missing method/seed records are materialized as failed observations.  This
    makes a failed child job visible in the reported failure probability rather
    than silently dropping it from its denominator.
    """
    manifest = load_json(batch_root / "unlearning_cost_batch_manifest.json")
    raw_jobs = manifest.get("jobs")
    if not isinstance(raw_jobs, list):
        raise ValueError("Batch manifest has no valid jobs list.")

    all_runs: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for index, job in enumerate(raw_jobs):
        if not isinstance(job, dict):
            warnings.append("Ignoring non-object manifest job {}.".format(index))
            continue
        job_name = str(job.get("name") or "job_{:03d}".format(index))
        output_value = job.get("output_dir")
        output_dir = Path(str(output_value)).expanduser() if output_value else batch_root / job_name
        if not output_dir.is_absolute():
            output_dir = (batch_root / output_dir).resolve()
        results_path = output_dir / "unlearning_cost_results.json"
        observed: Dict[Tuple[str, int], Dict[str, Any]] = {}
        if results_path.is_file():
            try:
                results = load_json(results_path)
                raw_runs = results.get("runs")
                if not isinstance(raw_runs, list):
                    raise ValueError("'runs' is not a list")
                for raw_run in raw_runs:
                    if not isinstance(raw_run, dict):
                        continue
                    run = dict(raw_run)
                    run["job"] = job_name
                    all_runs.append(run)
                    try:
                        observed[(str(run.get("method")), int(run.get("seed")))] = run
                    except (TypeError, ValueError):
                        pass
            except ValueError as error:
                warnings.append("{}: {}".format(job_name, error))
        else:
            warnings.append("{}: missing {}".format(job_name, results_path.name))

        for method, seed in _expected_runs(job):
            if (method, seed) in observed:
                continue
            all_runs.append(
                {
                    "job": job_name,
                    "method": method,
                    "seed": seed,
                    "status": "missing_result",
                    "steps_trained": 0,
                    "training_wall_seconds": 0.0,
                    "error": "No aggregate result was written for this method/seed.",
                }
            )
    return all_runs, warnings


def _average(values: Iterable[Any]) -> Optional[float]:
    valid = [number for value in values for number in [finite_number(value)] if number is not None]
    return mean(valid) if valid else None


def summarize_runs(
    runs: Sequence[Dict[str, Any]], include_failures: bool
) -> List[Dict[str, Any]]:
    """Summarize method cost by job.

    In the inclusive view, every run is in the denominator and failed child
    processes contribute the recorded zero wall-time used by the measurement
    schema.  In the successful-only view, only ``status == 'achieved'`` is
    averaged.  Both views always retain the same failure probability.
    """
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(str(run.get("job", "unknown")), str(run.get("method", "unknown")))].append(run)

    summaries = []
    for (job, method), group in sorted(grouped.items()):
        successful = [run for run in group if run.get("status") == SUCCESS_STATUS]
        selected = group if include_failures else successful
        statuses = defaultdict(int)
        for run in group:
            statuses[str(run.get("status") or "unknown")] += 1
        total = len(group)
        failed = total - len(successful)
        runtime_values = [finite_number(run.get("training_wall_seconds")) for run in selected]
        runtime_values = [value for value in runtime_values if value is not None]
        row = {
            "job": job,
            "method": method,
            "total_runs": total,
            "achieved_runs": len(successful),
            "not_achieved_runs": statuses["not_achieved"],
            "failed_runs": statuses["failed"],
            "preflight_error_runs": statuses["preflight_error"],
            "missing_result_runs": statuses["missing_result"],
            "failure_probability": failed / total if total else None,
            "runtime_samples": len(runtime_values),
            "mean_training_wall_seconds": _average(runtime_values),
            "mean_training_wall_minutes": (
                _average(runtime_values) / 60.0 if runtime_values else None
            ),
            "mean_steps_trained": _average(
                run.get("steps_trained") for run in selected
            ),
            "mean_steps_to_threshold": _average(
                run.get("steps_to_threshold") for run in successful
            ),
            "critic_distance_samples": sum(
                critic_distance_mean(run) is not None for run in selected
            ),
            "mean_D_f_critic_distance": _average(
                critic_distance_mean(run) for run in selected
            ),
        }
        summaries.append(row)
    return summaries


def format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return "{:.6g}".format(value)
    return str(value)


def print_summary(rows: Sequence[Dict[str, Any]], include_failures: bool) -> None:
    mode = "包含未成功运行" if include_failures else "仅成功运行"
    print("平均运行时间统计（{}）".format(mode))
    print("失败率定义：status != achieved。")
    print(
        "{:<36} {:<18} {:>7} {:>7} {:>9} {:>14} {:>16}".format(
            "job", "method", "total", "success", "fail_prob", "mean_time(s)", "mean_steps"
        )
    )
    for row in rows:
        print(
            "{:<36} {:<18} {:>7} {:>7} {:>9} {:>14} {:>16}".format(
                row["job"][:36], row["method"][:18], row["total_runs"],
                row["achieved_runs"], format_value(row["failure_probability"]),
                format_value(row["mean_training_wall_seconds"]),
                format_value(row["mean_steps_to_threshold"]),
            )
        )


def write_summary_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def report_main(argv: Optional[Sequence[str]], include_failures: bool) -> int:
    parser = argparse.ArgumentParser(
        description="Report per-job mean unlearning training time and failure probability."
    )
    parser.add_argument(
        "--batch-output", required=True,
        help="Batch output directory or unlearning_cost_batch_manifest.json.",
    )
    parser.add_argument("--output-csv", help="Optional summary CSV path.")
    args = parser.parse_args(argv)
    try:
        batch_root = resolve_batch_root(args.batch_output)
        runs, warnings = load_batch_runs(batch_root)
    except ValueError as error:
        parser.error(str(error))
    rows = summarize_runs(runs, include_failures)
    print_summary(rows, include_failures)
    default_name = (
        "unlearning_cost_mean_runtime_including_failures.csv"
        if include_failures else "unlearning_cost_mean_runtime_successes_only.csv"
    )
    output_path = Path(args.output_csv).expanduser().resolve() if args.output_csv else batch_root / default_name
    write_summary_csv(output_path, rows)
    print("Saved CSV to {}".format(output_path))
    for warning in warnings:
        print("[warning] {}".format(warning))
    return 0