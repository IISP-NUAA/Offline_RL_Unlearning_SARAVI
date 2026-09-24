#!/usr/bin/env python3
"""Summarize ORL-Auditor JSON outputs and calculate gaps to Retrain.

``evaluate_orl_auditor.py`` stores one ``audit_summary_seed*.json`` file per
evaluation.  Its target-level results live in
``orl_audit_statistics.primary_audit_results``.  This script flattens those
nested records and writes three CSV files:

* a detailed row-level table (one target model in one seed per row);
* a ratio-level mean/std summary over seeds;
* a final mean/std summary over ratios and seeds.

For every audit metric, ``Gap_<metric>_Audit_Positive_Rate`` is the absolute
difference between the target's audit-positive rate and its matched Retrain
baseline.  A baseline is matched by dataset, ratio, algorithm, shuffle mode,
audit split, and split seed; training/unlearning step is deliberately not a
match key because Retrain commonly uses a different step count.  Rate columns,
their references, and gaps are written as percentages.  Counts and distance
statistics remain in their native units.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from analyze_evaluation_results import (
        add_grouped_bootstrap_confidence_intervals,
        parse_model_dir, parse_step_from_model_dir, sort_with_step,
    )
except ImportError:
    # Keep the analyzer usable when copied outside the repository.
    def parse_model_dir(model_dir: str) -> Tuple[str, str, str, str]:
        return "Unknown", "N/A", "N/A", "N/A"

    def parse_step_from_model_dir(model_dir: str) -> str:
        return "N/A"

    def sort_with_step(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        valid_columns = [column for column in columns if column in df.columns]
        if not valid_columns:
            return df.reset_index(drop=True)
        return df.sort_values(
            by=valid_columns,
            key=lambda col: pd.to_numeric(col, errors="coerce")
            if col.name == "Step" else col,
        ).reset_index(drop=True)

    def add_grouped_bootstrap_confidence_intervals(
        results_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        group_cols: List[str],
        metric_prefixes: Tuple[str, ...],
        summary_mean_columns=None,
        output_prefixes=None,
    ) -> pd.DataFrame:
        return summary_df


AUDIT_METRIC_ORDER = [
    "l1_distance",
    "l2_distance",
    "cos_distance",
    "wasserstein_distance",
]

RATE_SUFFIX = "Audit_Positive_Rate"
DISTANCE_STATISTICS = (
    "Target-to-Shadow-Mean Distance",
    "Shadow Distance Distribution",
    "Standardized Distance",
)
DESCRIPTIVE_STATISTICS = ("Mean", "Std", "Median", "Min", "Max")


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _join_values(values: Iterable[Any], separator: str = "_") -> str:
    return separator.join(str(value) for value in values)


def _numeric_value(value: Any) -> Any:
    """Convert JSON numeric values while preserving non-numeric metadata."""
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _seed_from_filename(file_path: str) -> Any:
    match = re.search(r"audit_summary_seed(-?\d+)\.json$", os.path.basename(file_path))
    return int(match.group(1)) if match else "N/A"


def _metric_column_name(metric: str, field: str) -> str:
    metric_name = re.sub(r"[^A-Za-z0-9]+", "_", str(metric)).strip("_")
    field_name = re.sub(r"[^A-Za-z0-9]+", "_", str(field)).strip("_")
    return f"{metric_name}_{field_name}"


def _normalise_model_dir_for_parse(model_dir: str) -> str:
    """Make absolute model paths compatible with ``parse_model_dir``."""
    if not model_dir:
        return model_dir

    parts = [
        part for part in str(model_dir).replace("\\", "/").split("/")
        if part and part != "."
    ]
    if not parts:
        return str(model_dir)

    def last_index(token: str) -> Optional[int]:
        for index in range(len(parts) - 1, -1, -1):
            if parts[index] == token:
                return index
        return None

    offline_index = last_index("Offline_RL_processing")
    if offline_index is not None:
        return "/".join(parts[offline_index + 1:])

    unlearning_index = last_index("unlearning_processing")
    if unlearning_index is not None:
        suffix_parts = parts[unlearning_index + 1:]
        if suffix_parts and "Unlearned" in suffix_parts[0]:
            return "/".join(suffix_parts)
        return "../unlearning_processing/" + "/".join(suffix_parts)

    if parts[0] in {"Fully_trained", "Retrain"} or "Unlearned" in parts[0]:
        return "/".join(parts)
    return str(model_dir)


def _fallback_metadata(model_dir: str) -> Tuple[str, str]:
    """Best-effort method and algorithm inference for unfamiliar layouts."""
    parts = [part for part in str(model_dir).replace("\\", "/").split("/") if part]
    method = "Unknown"
    algorithm = "N/A"

    if "Retrain" in parts:
        method = "Retrain"
    elif "Fully_trained" in parts:
        method = "Fully_trained"
    else:
        for marker in ("unlearning_processing", "Unlearning"):
            if marker in parts:
                marker_index = parts.index(marker)
                if marker_index + 1 < len(parts):
                    method = parts[marker_index + 1]
                    break
        if method == "Unknown" and parts:
            method = parts[0]

    for part in parts:
        match = re.search(r"(?:Unlearning|FinetuneOnly)_([A-Za-z0-9+.-]+)", part)
        if match:
            algorithm = match.group(1).upper()
            break
    if algorithm == "N/A":
        for part in reversed(parts):
            match = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*)_\d{8}", part)
            if match:
                algorithm = match.group(1).upper()
                break
    return method, algorithm


def _lookup_by_suspect_name(data: Dict[str, Any], field: str, suspect_name: str) -> Any:
    names = _as_list(data.get("suspect_names"))
    values = _as_list(data.get(field))
    try:
        return values[names.index(suspect_name)]
    except (ValueError, IndexError):
        return None


def _extract_model_metadata(model_dir: str, file_path: str) -> Dict[str, Any]:
    parse_dir = _normalise_model_dir_for_parse(model_dir)
    method_name, _, algo_name, _ = parse_model_dir(parse_dir)
    fallback_method, fallback_algo = _fallback_metadata(parse_dir)

    if method_name in {"Unknown", "Error", "N/A", ""}:
        method_name = fallback_method
    if algo_name in {"Unknown", "Error", "N/A", ""}:
        algo_name = fallback_algo

    step = parse_step_from_model_dir(parse_dir)
    if step == "N/A":
        step = parse_step_from_model_dir(model_dir)
    if step == "N/A":
        step = parse_step_from_model_dir(file_path)

    return {
        "Method Name": method_name,
        "Algo. Name": algo_name,
        "Step": step,
    }


def _normalise_role(value: Any) -> str:
    role = str(value or "unknown").strip().lower()
    return role if role in {"original", "unlearned", "retrained", "unknown"} else "unknown"


def _normalise_membership(value: Any) -> str:
    if value is True or str(value).lower() == "member":
        return "member"
    if value is False or str(value).lower() == "nonmember":
        return "nonmember"
    return "unknown"


def _contains_nonfinite_number(value: Any) -> bool:
    """Return whether a JSON value contains a real NaN or infinity.

    JSON ``null`` is deliberately not considered invalid: ORL-Auditor uses it
    for unavailable standardized-distance statistics when too few shadows are
    present.
    """
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return not math.isfinite(float(value))
    if isinstance(value, str):
        try:
            return not math.isfinite(float(value))
        except ValueError:
            return False
    if isinstance(value, dict):
        return any(_contains_nonfinite_number(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_nonfinite_number(item) for item in value)
    return False


def _invalid_distance_metrics(metric_results: Dict[str, Any]) -> List[str]:
    """List metrics whose distance summaries contain NaN or infinity."""
    invalid_metrics = []
    for metric, metric_result in metric_results.items():
        if not isinstance(metric_result, dict):
            continue
        for statistic_name in DISTANCE_STATISTICS:
            if _contains_nonfinite_number(metric_result.get(statistic_name)):
                invalid_metrics.append(f"{metric}.{statistic_name}")
                break
    return invalid_metrics


def _add_metric_values(record: Dict[str, Any], metric: str, metric_result: Dict[str, Any]) -> None:
    scalar_fields = (
        "Audit Positive Rate",
        "Audit Positive Count",
        "Audited Trajectories",
        "Standardized Distance Valid Count",
    )
    for field in scalar_fields:
        record[_metric_column_name(metric, field)] = _numeric_value(metric_result.get(field))

    for statistic_name in DISTANCE_STATISTICS:
        statistic_values = metric_result.get(statistic_name, {})
        if not isinstance(statistic_values, dict):
            statistic_values = {}
        for statistic in DESCRIPTIVE_STATISTICS:
            record[_metric_column_name(metric, f"{statistic_name}_{statistic}")] = _numeric_value(
                statistic_values.get(statistic)
            )


def parse_audit_record(data: Dict[str, Any], file_path: str) -> List[Dict[str, Any]]:
    """Flatten one evaluator JSON into one record per audited suspect model."""
    statistics = data.get("orl_audit_statistics", {})
    primary_results = (
        statistics.get("primary_audit_results", {}) if isinstance(statistics, dict) else {}
    )
    if not isinstance(primary_results, dict) or not primary_results:
        print(f"Warning: No primary_audit_results block found in {file_path}.")
        return []

    components = _as_list(data.get("dataset_components"))
    ratios = _as_list(data.get("retained_ratios"))
    seed = data.get("seed")
    if seed is None:
        seed = _seed_from_filename(file_path)

    records: List[Dict[str, Any]] = []
    for suspect_name, suspect_result in primary_results.items():
        if not isinstance(suspect_result, dict):
            print(f"Warning: Invalid primary result for {suspect_name!r} in {file_path}.")
            continue

        metric_results = suspect_result.get("Metrics", {})
        if not isinstance(metric_results, dict) or not metric_results:
            print(f"Warning: No Metrics block for {suspect_name!r} in {file_path}.")
            continue
        invalid_metrics = _invalid_distance_metrics(metric_results)
        if invalid_metrics:
            print(
                f"Warning: Skipping suspect branch {suspect_name!r} in {file_path}; "
                f"non-finite distance statistics: {', '.join(invalid_metrics)}"
            )
            continue

        suspect_dir = str(
            suspect_result.get("Suspect Model Directory")
            or _lookup_by_suspect_name(data, "suspect_model_dirs", str(suspect_name))
            or ""
        )
        metadata = _extract_model_metadata(suspect_dir, file_path)
        role = _normalise_role(
            suspect_result.get("Suspect Role")
            or _lookup_by_suspect_name(data, "suspect_roles", str(suspect_name))
        )
        membership = _normalise_membership(
            suspect_result.get("Expected Membership")
            if "Expected Membership" in suspect_result
            else _lookup_by_suspect_name(data, "suspect_membership", str(suspect_name))
        )

        record: Dict[str, Any] = {
            **metadata,
            "Dataset Name": data.get("dataset_name", "N/A"),
            "Dataset Components": _join_values(components) or "N/A",
            "Ratios": _join_values(ratios) or "N/A",
            "Metric Type": data.get("metric_type", "ORL_Auditor_Grubbs"),
            "Shuffle Mode": "Shuffle" if data.get("shuffle", True) else "No Shuffle",
            "Audit Split": data.get("audit_split", "N/A"),
            "Audit Buffer": suspect_result.get("Audit Buffer", data.get("audit_split", "N/A")),
            "seed": seed,
            "Suspect Name": suspect_name,
            "Suspect Role": role,
            "Suspect Membership": membership,
            "Suspect Model Dir": suspect_dir or "N/A",
            "row_count": _numeric_value(data.get("row_count")),
            "Num Shadow Students": _numeric_value(data.get("num_shadow_student")),
            "Num Audited Episodes": _numeric_value(data.get("num_of_audited_episode")),
            "Trajectory Size": _numeric_value(data.get("trajectory_size")),
            "Significance Level": _numeric_value(data.get("significance_level")),
        }

        metric_results = suspect_result.get("Metrics", {})
        if not isinstance(metric_results, dict) or not metric_results:
            print(f"Warning: No Metrics block for {suspect_name!r} in {file_path}.")
            continue
        for metric, metric_result in metric_results.items():
            if isinstance(metric_result, dict):
                _add_metric_values(record, str(metric), metric_result)
        records.append(record)
    return records


def process_audit_files(root_dir: str) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    matched_files = 0
    print(f"Starting search from: {root_dir}\n")

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in sorted(filenames):
            if not (filename.startswith("audit_summary_seed") and filename.endswith(".json")):
                continue
            matched_files += 1
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as file_handle:
                    data = json.load(file_handle)
                if not isinstance(data, dict):
                    print(f"Warning: Ignoring non-object JSON in {file_path}.")
                    continue
                records.extend(parse_audit_record(data, file_path))
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                print(f"Error processing file {file_path}: {exc}")

    print(f"Matched {matched_files} audit summary file(s), produced {len(records)} row(s).")
    return pd.DataFrame(records) if records else pd.DataFrame()


def _retrain_mask(dataframe: pd.DataFrame) -> pd.Series:
    roles = dataframe["Suspect Role"].astype(str).str.lower()
    methods = dataframe["Method Name"].astype(str).str.lower()
    return roles.eq("retrained") | methods.eq("retrain")


def add_retrain_rate_gaps(results_df: pd.DataFrame) -> pd.DataFrame:
    """Attach matched Retrain rates and absolute audit-positive-rate gaps."""
    if results_df.empty:
        return results_df

    rate_columns = [column for column in results_df.columns if column.endswith(RATE_SUFFIX)]
    if not rate_columns:
        return results_df

    join_keys = [
        "Dataset Name",
        "Dataset Components",
        "Ratios",
        "Algo. Name",
        "Shuffle Mode",
        "Audit Split",
        "Audit Buffer",
        "seed",
    ]
    missing_keys = [key for key in join_keys if key not in results_df.columns]
    if missing_keys:
        print(f"Warning: Cannot calculate Retrain gaps; missing columns: {missing_keys}")
        return results_df

    retrain_rows = results_df.loc[_retrain_mask(results_df), join_keys + rate_columns]
    if retrain_rows.empty:
        print("Warning: No Retrain rows were found; Retrain reference and gap columns are unavailable.")
        return results_df

    reference_df = retrain_rows.groupby(join_keys, dropna=False)[rate_columns].mean().reset_index()
    reference_names = {column: f"Ref_{column}" for column in rate_columns}
    reference_df = reference_df.rename(columns=reference_names)
    result = results_df.merge(reference_df, on=join_keys, how="left")

    retrain_rows_after_merge = _retrain_mask(result)
    for rate_column in rate_columns:
        reference_column = reference_names[rate_column]
        gap_column = f"Gap_{rate_column}"
        result[gap_column] = (result[rate_column] - result[reference_column]).abs()
        # A Retrain row is the reference by definition.  This also avoids a
        # spurious non-zero value when multiple duplicate Retrain reports are
        # present for a single split configuration.
        result.loc[retrain_rows_after_merge, gap_column] = 0.0
    return result


def add_mean_positive_rate_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Add the mean of the four audit-positive rates and their mean gap.

    ``Gap_Mean_Audit_Positive_Rate`` is the mean of the four per-distance
    absolute gaps, not the absolute difference between already-averaged rates.
    """
    rate_columns = [
        _metric_column_name(metric, "Audit Positive Rate")
        for metric in AUDIT_METRIC_ORDER
    ]
    missing_rate_columns = [column for column in rate_columns if column not in results_df.columns]
    if missing_rate_columns:
        print(
            "Warning: Mean Audit Positive Rate was not calculated; missing columns: "
            f"{missing_rate_columns}"
        )
        return results_df

    result = results_df.copy()
    result["Mean_Audit_Positive_Rate"] = result[rate_columns].apply(
        pd.to_numeric, errors="coerce"
    ).mean(axis=1)

    reference_columns = [f"Ref_{column}" for column in rate_columns]
    if all(column in result.columns for column in reference_columns):
        result["Ref_Mean_Audit_Positive_Rate"] = result[reference_columns].apply(
            pd.to_numeric, errors="coerce"
        ).mean(axis=1)

    gap_columns = [f"Gap_{column}" for column in rate_columns]
    if all(column in result.columns for column in gap_columns):
        result["Gap_Mean_Audit_Positive_Rate"] = result[gap_columns].apply(
            pd.to_numeric, errors="coerce"
        ).mean(axis=1)
    return result

def _scale_rate_columns(results_df: pd.DataFrame) -> pd.DataFrame:
    """Convert audit rates, matched references, and gaps from [0, 1] to %."""
    results_df = results_df.copy()
    for column in results_df.columns:
        if column.endswith(RATE_SUFFIX) and pd.api.types.is_numeric_dtype(results_df[column]):
            results_df[column] = results_df[column] * 100
    return results_df


def _flatten_aggregation_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    columns: List[str] = []
    for column in dataframe.columns:
        if isinstance(column, tuple):
            name, operation = column
            if operation == "":
                columns.append(str(name))
            elif operation == "mean":
                columns.append(f"{name}_Avg")
            elif operation == "std":
                columns.append(f"{name}_Std")
            else:
                columns.append(f"{name}_{operation}")
        else:
            columns.append(str(column))
    dataframe.columns = columns
    return dataframe


def _aggregate_with_mean_std(
    results_df: pd.DataFrame,
    group_cols: Sequence[str],
    numeric_cols: Sequence[str],
) -> pd.DataFrame:
    valid_group_cols = [column for column in group_cols if column in results_df.columns]
    if not valid_group_cols:
        raise ValueError("No valid grouping columns are available.")

    grouped = results_df.groupby(valid_group_cols, dropna=False)
    aggregate_df = grouped[list(numeric_cols)].agg(["mean", "std"]).reset_index()
    aggregate_df = _flatten_aggregation_columns(aggregate_df)
    count_df = grouped.size().reset_index(name="Sample Count")
    summary_df = pd.merge(aggregate_df, count_df, on=valid_group_cols, how="left")
    summary_df = add_grouped_bootstrap_confidence_intervals(
        results_df, summary_df, valid_group_cols, ("Gap_",)
    )

    columns = list(summary_df.columns)
    columns.remove("Sample Count")
    columns.insert(len(valid_group_cols), "Sample Count")
    return _move_gap_columns_after_sample_count(summary_df[columns])


def _move_gap_columns_after_sample_count(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Place the paper-facing audit-positive-rate gaps after ``Sample Count``."""
    if "Sample Count" not in dataframe.columns:
        return dataframe

    columns = list(dataframe.columns)
    gap_columns = [column for column in columns if column.startswith("Gap_")]
    if not gap_columns:
        return dataframe

    remaining_columns = [column for column in columns if column not in gap_columns]
    insert_at = remaining_columns.index("Sample Count") + 1
    ordered_columns = (
        remaining_columns[:insert_at] + gap_columns + remaining_columns[insert_at:]
    )
    return dataframe.reindex(columns=ordered_columns)


def _sort_summary(dataframe: pd.DataFrame, include_ratios: bool) -> pd.DataFrame:
    candidates = [
        "Dataset Name",
        "Dataset Components",
        "Algo. Name",
        "Shuffle Mode",
        "Method Name",
        "Suspect Role",
        "Step",
    ]
    if include_ratios:
        candidates.append("Ratios")
    return sort_with_step(dataframe, [column for column in candidates if column in dataframe.columns])


def _output_file(output_dir: str, root_dir: str, filename: str) -> str:
    root_tag = root_dir.replace("\\", "/").strip("/").replace("/", "_") or "audit_results"
    return os.path.join(output_dir, f"{root_tag}_{filename}")


def _save_csv(dataframe: pd.DataFrame, path: str, label: str) -> None:
    dataframe.to_csv(path, index=False, encoding="utf-8")
    print(f"{label} successfully saved to {path}")


def _step_matches(value: Any, requested_step: Optional[str]) -> bool:
    if requested_step is None:
        return True
    try:
        return int(float(value)) == int(float(requested_step))
    except (TypeError, ValueError):
        return str(value) == str(requested_step)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate evaluate_orl_auditor.py audit summaries and compute Retrain gaps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default="stats_results_orl_auditor",
        help="Directory containing audit_summary_seed*.json files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="orl_auditor_summary_stats.csv",
        help="Base output CSV filename.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="summary",
        help="Directory in which detailed, ratio, and final CSV files are written.",
    )
    parser.add_argument(
        "--step",
        type=str,
        default=None,
        help="Optional reporting-step filter. Retrain rows remain available internally as references.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root_dir):
        print(f"Error: The directory '{args.root_dir}' does not exist.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    results_df = process_audit_files(args.root_dir)
    if results_df.empty:
        print("No data found.")
        return

    results_df = add_retrain_rate_gaps(results_df)
    results_df = add_mean_positive_rate_metrics(results_df)
    if args.step is not None:
        results_df = results_df.loc[
            results_df["Step"].map(lambda value: _step_matches(value, args.step))
        ].copy()
        print(f"Step filter enabled: step={args.step}; retained {len(results_df)} row(s).")
    if results_df.empty:
        print("No rows remain after applying the step filter.")
        return

    results_df = _scale_rate_columns(results_df)

    known_cols = [
        "Dataset Name",
        "Dataset Components",
        "Ratios",
        "Metric Type",
        "Shuffle Mode",
        "Audit Split",
        "Audit Buffer",
        "Algo. Name",
        "Method Name",
        "Suspect Role",
        "Step",
        "seed",
        "Suspect Name",
        "Suspect Membership",
        "Suspect Model Dir",
    ]
    run_stat_cols = [
        "row_count",
        "Num Shadow Students",
        "Num Audited Episodes",
        "Trajectory Size",
        "Significance Level",
    ]
    preferred_metric_cols: List[str] = [
        "Mean_Audit_Positive_Rate",
        "Ref_Mean_Audit_Positive_Rate",
        "Gap_Mean_Audit_Positive_Rate",
    ]
    for metric in AUDIT_METRIC_ORDER:
        preferred_metric_cols.extend(
            [
                _metric_column_name(metric, "Audit Positive Rate"),
                f"Ref_{_metric_column_name(metric, 'Audit Positive Rate')}",
                f"Gap_{_metric_column_name(metric, 'Audit Positive Rate')}",
                _metric_column_name(metric, "Audit Positive Count"),
                _metric_column_name(metric, "Audited Trajectories"),
                _metric_column_name(metric, "Standardized Distance Valid Count"),
            ]
        )
        for statistic_name in DISTANCE_STATISTICS:
            for statistic in DESCRIPTIVE_STATISTICS:
                preferred_metric_cols.append(
                    _metric_column_name(metric, f"{statistic_name}_{statistic}")
                )

    ordered_columns = [column for column in known_cols if column in results_df.columns]
    ordered_columns += [column for column in run_stat_cols if column in results_df.columns]
    ordered_columns += [column for column in preferred_metric_cols if column in results_df.columns]
    ordered_columns += sorted(set(results_df.columns) - set(ordered_columns))
    results_df = results_df.reindex(columns=ordered_columns)
    results_df = _sort_summary(results_df, include_ratios=True)

    print("\n--- ORL-Auditor Statistics Summary (Detailed) ---")
    detailed_path = _output_file(args.output_dir, args.root_dir, args.output)
    _save_csv(results_df, detailed_path, "Detailed results")

    meta_columns = set(known_cols)
    numeric_cols = [
        column
        for column in results_df.columns
        if column not in meta_columns and pd.api.types.is_numeric_dtype(results_df[column])
    ]
    if not numeric_cols:
        print("No numeric columns found to aggregate.")
        return

    base, extension = os.path.splitext(args.output)
    if not extension:
        extension = ".csv"

    ratio_group_cols = [
        "Dataset Name",
        "Dataset Components",
        "Ratios",
        "Algo. Name",
        "Method Name",
        "Suspect Role",
        "Shuffle Mode",
        "Metric Type",
        "Audit Split",
        "Audit Buffer",
        "Step",
    ]
    ratio_summary_df = _aggregate_with_mean_std(results_df, ratio_group_cols, numeric_cols)
    ratio_summary_df = _sort_summary(ratio_summary_df, include_ratios=True)
    ratio_path = _output_file(args.output_dir, args.root_dir, f"{base}_ratio_summary{extension}")
    print("\n--- Ratio Summary (Aggregated over Seed, Split by Ratio) ---")
    _save_csv(ratio_summary_df, ratio_path, "Ratio summary")

    final_group_cols = [column for column in ratio_group_cols if column != "Ratios"]
    final_summary_df = _aggregate_with_mean_std(results_df, final_group_cols, numeric_cols)
    final_summary_df = _sort_summary(final_summary_df, include_ratios=False)
    final_path = _output_file(args.output_dir, args.root_dir, f"{base}_final_summary{extension}")
    print("\n--- Final Summary (Aggregated over Ratio & Seed) ---")
    _save_csv(final_summary_df, final_path, "Final summary")


if __name__ == "__main__":
    main()
