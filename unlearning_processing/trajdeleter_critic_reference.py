#!/usr/bin/env python3
"""Read cached TrajDeleter critic references for unlearning-cost reports."""

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple


PHASE_PATTERN = re.compile(r"phase([12])_(\d+)", re.IGNORECASE)


@dataclass
class ReferenceRecord:
    seed: int
    value: float
    transition_median: Optional[float]
    source: Path


@dataclass
class ReferenceSummary:
    job: str
    algorithm: str
    total_steps: int
    expected_seeds: List[int]
    records: List[ReferenceRecord] = field(default_factory=list)

    @property
    def matched_seeds(self) -> List[int]:
        return sorted(record.seed for record in self.records)

    @property
    def median_D_f_mean(self) -> Optional[float]:
        values = [record.value for record in self.records]
        return float(median(values)) if values else None

    @property
    def median_D_f_transition_median(self) -> Optional[float]:
        values = [
            record.transition_median
            for record in self.records
            if record.transition_median is not None
        ]
        return float(median(values)) if values else None


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as input_file:
            payload = json.load(input_file)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalize_algorithm(value: Any) -> Optional[str]:
    text = str(value or "").upper().replace("_", "")
    if "TD3" in text and "BC" in text:
        return "TD3PLUSBC"
    if "PLAS" in text:
        return "PLASP"
    for name in ("IQL", "CQL", "BCQ", "BEAR", "CRR", "AWAC"):
        if name in text:
            return name
    return None


def _ratio_key(values: Sequence[Any]) -> Tuple[float, ...]:
    return tuple(round(float(value), 12) for value in values)


def _phase_steps(model_path: Any) -> Dict[str, int]:
    normalized = str(model_path or "").replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    try:
        marker = next(
            index for index, part in enumerate(parts) if part.lower() == "trajdeleter"
        )
    except StopIteration:
        return {}
    phases: Dict[str, int] = {}
    for part in parts[marker + 1 :]:
        for phase, value in PHASE_PATTERN.findall(part):
            phases[phase] = int(value)
        if phases:
            break
    return phases


def _is_target_reference(model_path: Any, total_steps: int) -> bool:
    phases = _phase_steps(model_path)
    phase1 = int(total_steps * 0.8)
    phase2 = total_steps - phase1
    return phases == {"1": phase1, "2": phase2}


def _job_specifications(
    runs: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    specifications: Dict[str, Dict[str, Any]] = {}
    all_seeds: Dict[str, set] = defaultdict(set)
    for run in runs:
        try:
            all_seeds[str(run.get("job") or "unknown")].add(int(run["seed"]))
        except (KeyError, TypeError, ValueError):
            continue

    conflicted_jobs = set()
    warnings: List[str] = []
    for run in runs:
        job = str(run.get("job") or "unknown")
        if job in conflicted_jobs:
            continue
        task = run.get("task")
        if not isinstance(task, dict):
            continue
        datasets = task.get("datasets")
        ratios = task.get("retained_ratios")
        algorithm = _normalize_algorithm(task.get("algo"))
        if (
            not task.get("dataset")
            or not isinstance(datasets, list)
            or not datasets
            or not isinstance(ratios, list)
            or len(ratios) != len(datasets)
            or algorithm is None
        ):
            continue
        try:
            seed = int(run["seed"])
            specification = {
                "dataset": str(task["dataset"]).lower(),
                "datasets": tuple(str(value) for value in datasets),
                "retained_ratios": _ratio_key(ratios),
                "algorithm": algorithm,
            }
        except (KeyError, TypeError, ValueError):
            continue

        previous = specifications.get(job)
        if previous is None:
            specification["seeds"] = {seed}
            specifications[job] = specification
        elif all(
            previous[key] == specification[key]
            for key in ("dataset", "datasets", "retained_ratios", "algorithm")
        ):
            previous["seeds"].add(seed)
        else:
            warnings.append(
                "{}: runs contain conflicting task metadata; "
                "TrajDeleter reference was skipped.".format(job)
            )
            specifications.pop(job, None)
            conflicted_jobs.add(job)

    for job, specification in specifications.items():
        specification["seeds"].update(all_seeds[job])
    return specifications, warnings


def load_trajdeleter_reference_summaries(
    stats_root: Path,
    runs: Sequence[Dict[str, Any]],
    total_steps: int = 10000,
) -> Tuple[Dict[str, ReferenceSummary], List[str]]:
    """Match two-phase TrajDeleter records and summarize per-seed D_f means."""
    if total_steps <= 0:
        raise ValueError("TrajDeleter reference steps must be positive")
    stats_root = stats_root.expanduser().resolve()
    specifications, warnings = _job_specifications(runs)
    if not specifications:
        return {}, warnings
    if not stats_root.is_dir():
        warnings.append(
            "TrajDeleter critic stats directory is missing: {}".format(stats_root)
        )
        return {}, warnings

    desired_keys: Dict[
        Tuple[str, Tuple[str, ...], Tuple[float, ...], str, int], List[str]
    ] = defaultdict(list)
    for job, specification in specifications.items():
        for seed in specification["seeds"]:
            key = (
                specification["dataset"],
                specification["datasets"],
                specification["retained_ratios"],
                specification["algorithm"],
                seed,
            )
            desired_keys[key].append(job)

    matches: Dict[
        Tuple[str, Tuple[str, ...], Tuple[float, ...], str, int],
        List[ReferenceRecord],
    ] = defaultdict(list)
    for path in stats_root.rglob("*.json"):
        payload = _read_json(path)
        if payload is None or payload.get("metric_type") != "CriticValueDiff":
            continue
        model_path = payload.get("model_2_path")
        if not _is_target_reference(model_path, total_steps):
            continue
        datasets = payload.get("dataset_components")
        ratios = payload.get("retained_ratios")
        algorithm = _normalize_algorithm(model_path)
        statistics = payload.get("critic_diff_statistics")
        if (
            not payload.get("dataset_name")
            or not isinstance(datasets, list)
            or not isinstance(ratios, list)
            or algorithm is None
            or not isinstance(statistics, dict)
        ):
            continue
        try:
            key = (
                str(payload["dataset_name"]).lower(),
                tuple(str(value) for value in datasets),
                _ratio_key(ratios),
                algorithm,
                int(payload["seed"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if key not in desired_keys:
            continue
        value = _finite_number(statistics.get("D_f_mean"))
        if value is None:
            continue
        matches[key].append(
            ReferenceRecord(
                seed=key[-1],
                value=value,
                transition_median=_finite_number(statistics.get("D_f_median")),
                source=path.resolve(),
            )
        )

    summaries: Dict[str, ReferenceSummary] = {}
    for job, specification in sorted(specifications.items()):
        expected_seeds = sorted(specification["seeds"])
        summary = ReferenceSummary(
            job=job,
            algorithm=specification["algorithm"],
            total_steps=total_steps,
            expected_seeds=expected_seeds,
        )
        for seed in expected_seeds:
            key = (
                specification["dataset"],
                specification["datasets"],
                specification["retained_ratios"],
                specification["algorithm"],
                seed,
            )
            records = matches.get(key, [])
            if len(records) == 1:
                summary.records.append(records[0])
            elif not records:
                warnings.append(
                    "{}: missing TrajDeleter {}-step critic record for "
                    "algorithm={} seed={} ratios={}.".format(
                        job,
                        total_steps,
                        specification["algorithm"],
                        seed,
                        specification["retained_ratios"],
                    )
                )
            else:
                warnings.append(
                    "{}: ambiguous TrajDeleter {}-step critic records for "
                    "algorithm={} seed={} ({} matches); seed skipped.".format(
                        job,
                        total_steps,
                        specification["algorithm"],
                        seed,
                        len(records),
                    )
                )
        summaries[job] = summary
    return summaries, warnings


def attach_reference_fields(
    rows: Sequence[Dict[str, Any]],
    summaries: Dict[str, ReferenceSummary],
) -> None:
    """Attach reference columns to each per-job/method cost summary row."""
    for row in rows:
        summary = summaries.get(str(row.get("job") or ""))
        row["trajdeleter_10k_reference_samples"] = (
            len(summary.records) if summary is not None else 0
        )
        row["median_trajdeleter_10k_D_f_critic_distance"] = (
            summary.median_D_f_mean if summary is not None else None
        )
        row["median_trajdeleter_10k_D_f_transition_median"] = (
            summary.median_D_f_transition_median
            if summary is not None
            else None
        )


def all_reference_values(
    summaries: Dict[str, ReferenceSummary],
) -> List[float]:
    return [
        record.value
        for summary in summaries.values()
        for record in summary.records
    ]