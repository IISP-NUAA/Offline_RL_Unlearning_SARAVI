#!/usr/bin/env python3
"""Evaluate and cache final-model critic distance for unlearning-cost runs."""

import importlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_FILENAME = "critic_distance.json"
CACHE_SCHEMA_VERSION = 1
METRIC_TYPE = "CriticValueDiff"
MODEL_PATTERN = re.compile(r"^model_(\d+)\.pt$")


@dataclass
class Candidate:
    run: Dict[str, Any]
    cache_path: Path
    checkpoint: Path
    retrained_checkpoint: Path
    task: Dict[str, Any]
    expected: Dict[str, Any]


@dataclass
class EvaluationReport:
    cache_hits: int = 0
    evaluated: int = 0
    skipped: int = 0
    failed: int = 0
    warnings: List[str] = field(default_factory=list)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as input_file:
            payload = json.load(input_file)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    os.replace(str(temporary), str(path))


def _file_identity(path: Path) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _resolve_latest_checkpoint(model_dir: Path) -> Path:
    model_dir = model_dir.expanduser().resolve()
    numbered: List[Tuple[int, Path]] = []
    for path in model_dir.glob("model_*.pt"):
        match = MODEL_PATTERN.match(path.name)
        if match and path.is_file():
            numbered.append((int(match.group(1)), path))
    if numbered:
        return max(numbered, key=lambda item: item[0])[1].resolve()
    fallback = model_dir / "model.pt"
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(
        "No model_*.pt or model.pt checkpoint found in {}".format(model_dir)
    )


def _normalized_task(run: Dict[str, Any]) -> Dict[str, Any]:
    task = run.get("task")
    if not isinstance(task, dict):
        raise ValueError("run has no task metadata")
    datasets = task.get("datasets")
    retained_ratios = task.get("retained_ratios")
    if not task.get("dataset") or not isinstance(datasets, list) or not datasets:
        raise ValueError("task has no valid dataset/datasets metadata")
    if not isinstance(retained_ratios, list) or len(retained_ratios) != len(datasets):
        raise ValueError("task retained_ratios do not match datasets")
    return {
        "dataset": str(task["dataset"]),
        "datasets": [str(value) for value in datasets],
        "retained_ratios": [float(value) for value in retained_ratios],
        "seed": int(run["seed"]),
        "shuffle": bool(int(task.get("shuffle", 1))),
    }


def _expected_metadata(
    run: Dict[str, Any],
    checkpoint: Path,
    retrained_checkpoint: Path,
    task: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "metric_type": METRIC_TYPE,
        "scope": "D_f",
        "method": str(run.get("method") or "unknown"),
        "seed": int(run["seed"]),
        "task": task,
        "model_checkpoint": _file_identity(checkpoint),
        "model_params": _file_identity(checkpoint.parent / "params.json"),
        "retrained_model_checkpoint": _file_identity(retrained_checkpoint),
        "retrained_model_params": _file_identity(
            retrained_checkpoint.parent / "params.json"
        ),
    }


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _valid_cached_result(
    payload: Optional[Dict[str, Any]], expected: Dict[str, Any]
) -> bool:
    if payload is None or payload.get("status") != "succeeded":
        return False
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            return False
    statistics = payload.get("critic_diff_statistics")
    if not isinstance(statistics, dict):
        return False
    count = _finite_number(statistics.get("D_f_count"))
    mean = _finite_number(statistics.get("D_f_mean"))
    return count is not None and count >= 0 and mean is not None


def _attach_result(run: Dict[str, Any], payload: Dict[str, Any], path: Path) -> None:
    run["critic_distance"] = payload
    run["critic_distance_cache"] = str(path)


def _prepare_candidate(
    run: Dict[str, Any], report: EvaluationReport
) -> Optional[Candidate]:
    label = "{} seed={} method={}".format(
        run.get("job", "unknown"), run.get("seed", "?"), run.get("method", "unknown")
    )
    checkpoint_value = run.get("retained_checkpoint")
    retrained_dir_value = run.get("retrain_model_dir")
    if not checkpoint_value or not retrained_dir_value:
        report.skipped += 1
        if run.get("status") in {"achieved", "not_achieved"}:
            report.warnings.append(
                "{}: terminal run lacks retained_checkpoint or retrain_model_dir.".format(label)
            )
        return None

    try:
        checkpoint = Path(str(checkpoint_value)).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError("final checkpoint does not exist: {}".format(checkpoint))
        params_path = checkpoint.parent / "params.json"
        if not params_path.is_file():
            raise FileNotFoundError("final-model params.json does not exist: {}".format(params_path))
        retrained_dir = Path(str(retrained_dir_value)).expanduser().resolve()
        if not (retrained_dir / "params.json").is_file():
            raise FileNotFoundError(
                "retrained params.json does not exist: {}".format(retrained_dir / "params.json")
            )
        retrained_checkpoint = _resolve_latest_checkpoint(retrained_dir)
        task = _normalized_task(run)
        expected = _expected_metadata(run, checkpoint, retrained_checkpoint, task)
    except (OSError, TypeError, ValueError) as error:
        report.skipped += 1
        report.warnings.append("{}: {}".format(label, error))
        return None

    return Candidate(
        run=run,
        cache_path=checkpoint.parent.parent / CACHE_FILENAME,
        checkpoint=checkpoint,
        retrained_checkpoint=retrained_checkpoint,
        task=task,
        expected=expected,
    )


def _stats_payload(stats: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in stats.items():
        name = "D_f_{}".format(key)
        normalized[name] = int(value) if key == "count" else float(value)
    return normalized


def _load_evaluator_module():
    root_text = str(REPO_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return importlib.import_module(
        "Offline_RL_processing.evaluate_critic_divergence"
    )


def evaluate_or_load_critic_distances(
    runs: Sequence[Dict[str, Any]],
    gpu: int = 0,
    batch_size: int = 512,
) -> EvaluationReport:
    """Attach cached/evaluated D_f critic distances to cost run dictionaries."""
    if batch_size <= 0:
        raise ValueError("critic distance batch_size must be positive")

    report = EvaluationReport()
    pending: List[Candidate] = []
    for run in runs:
        candidate = _prepare_candidate(run, report)
        if candidate is None:
            continue
        cached = _read_json(candidate.cache_path)
        if _valid_cached_result(cached, candidate.expected):
            _attach_result(run, cached, candidate.cache_path)
            report.cache_hits += 1
        else:
            pending.append(candidate)

    if not pending:
        return report

    try:
        evaluator = _load_evaluator_module()
    except BaseException as error:
        # The legacy evaluator used sys.exit on dependency import errors.
        message = "Cannot import critic evaluator: {}".format(error)
        report.failed += len(pending)
        report.warnings.extend(message for _ in pending)
        return report

    groups: Dict[str, List[Candidate]] = defaultdict(list)
    for candidate in pending:
        group_key = json.dumps(
            {
                "task": candidate.task,
                "retrained_model_checkpoint": candidate.expected[
                    "retrained_model_checkpoint"
                ],
            },
            sort_keys=True,
        )
        groups[group_key].append(candidate)

    for candidates in groups.values():
        first = candidates[0]
        task = first.task
        group_label = "{} seed={}".format(
            first.run.get("job", "unknown"), task["seed"]
        )
        print(
            "[critic] Loading D_f for {} ({} missing cache file(s)).".format(
                group_label, len(candidates)
            )
        )
        evaluator.d3rlpy.seed(task["seed"])
        evaluator.torch.manual_seed(task["seed"])
        evaluator.np.random.seed(task["seed"])

        try:
            states_f, actions_f = evaluator.load_forget_transition_arrays(
                task["dataset"],
                task["datasets"],
                task["retained_ratios"],
                task["seed"],
                shuffle=task["shuffle"],
            )
            retrained_model, _ = evaluator.load_model_from_dir(
                str(first.retrained_checkpoint.parent),
                gpu,
                checkpoint_path=str(first.retrained_checkpoint),
            )
        except Exception as error:
            report.failed += len(candidates)
            report.warnings.extend(
                "{}: critic group setup failed: {}".format(group_label, error)
                for _ in candidates
            )
            continue

        try:
            for candidate in candidates:
                label = "{} seed={} method={}".format(
                    candidate.run.get("job", "unknown"),
                    candidate.run.get("seed", "?"),
                    candidate.run.get("method", "unknown"),
                )
                print("[critic] Evaluating {}.".format(label))
                unlearned_model = None
                try:
                    unlearned_model, _ = evaluator.load_model_from_dir(
                        str(candidate.checkpoint.parent),
                        gpu,
                        checkpoint_path=str(candidate.checkpoint),
                    )
                    stats = evaluator.compute_critic_diff_stats(
                        unlearned_model,
                        retrained_model,
                        states_f,
                        actions_f,
                        batch_size,
                        desc_label="Critic Diff on D_f ({})".format(label),
                    )
                    statistics = _stats_payload(stats)
                    if (
                        _finite_number(statistics.get("D_f_mean")) is None
                        or _finite_number(statistics.get("D_f_count")) is None
                    ):
                        raise ValueError("critic evaluator returned non-finite statistics")
                    payload = dict(candidate.expected)
                    payload.update(
                        {
                            "status": "succeeded",
                            "created_at": time.strftime("%Y%m%d_%H%M%S"),
                            "batch_size": batch_size,
                            "critic_diff_statistics": statistics,
                        }
                    )
                    _atomic_write_json(candidate.cache_path, payload)
                    _attach_result(candidate.run, payload, candidate.cache_path)
                    report.evaluated += 1
                    print("[critic] Saved {}".format(candidate.cache_path))
                except Exception as error:
                    report.failed += 1
                    report.warnings.append(
                        "{}: critic evaluation failed: {}".format(label, error)
                    )
                finally:
                    if unlearned_model is not None:
                        del unlearned_model
                    if evaluator.torch.cuda.is_available():
                        evaluator.torch.cuda.empty_cache()
        finally:
            del retrained_model
            if evaluator.torch.cuda.is_available():
                evaluator.torch.cuda.empty_cache()

    return report