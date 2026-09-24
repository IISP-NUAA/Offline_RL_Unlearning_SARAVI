#!/usr/bin/env python3
"""Measure the step, wall-clock, and GPU cost of offline-RL unlearning."""

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_METHODS = (
    "SARAVI",
    "NegativeReward",
    "RandomReward",
    "Finetuning",
    "Retraining",
)

METHOD_CONFIGS: Dict[str, Dict[str, str]] = {
    "SARAVI": {
        "script": (
            "unlearning_processing/"
            "SARAVI_cache_ver.py"
        ),
        "description": "SARAVI with precomputed original-model predictions",
    },
    "NegativeReward": {
        "script": "unlearning_processing/trajectory_deleter.py",
        "description": "TrajDeleter phase-1-only negative reward baseline",
    },
    "RandomReward": {
        "script": "unlearning_processing/random_rewarding.py",
        "description": "Random reward baseline",
    },
    "Finetuning": {
        "script": "unlearning_processing/fine_tune_on_remain_set.py",
        "description": "Retain-set-only fine-tuning",
    },
    "Retraining": {
        "script": "Offline_RL_processing/fully_training.py",
        "description": "Training from scratch on the retain set",
    },
}
METHOD_ALIASES = {
    "FINETUNE": "Finetuning",
    "FINETUNING": "Finetuning",
    "NEGATIVEREWARD": "NegativeReward",
    "RANDOMREWARD": "RandomReward",
    "RETRAIN": "Retraining",
    "RETRAINING": "Retraining",
    "SARAVI": "SARAVI",

}
RATIO_PREFIXES = ("learn_ratios", "retained_ratios", "retain_ratios", "ratios")
MODEL_PATTERN = re.compile(r"^model_(\d+)\.pt$")
SEED_PATTERN = re.compile(r"(?:^|[/\\_])seed[-_]?(\d+)(?:_|[/\\]|$)", re.IGNORECASE)
TRAJDELETER_PATTERN = re.compile(
    r"^Unlearned_trajDeleter_phase2_(\d+)_phase1_(\d+)$"
)
ANALYZED_METRIC_SCALE = 100.0

CSV_FIELDS = [
    "method",
    "seed",
    "status",
    "threshold_D_f_W_unlearn_retrain",
    "steps_to_threshold",
    "steps_trained",
    "training_wall_seconds",
    "gpu_hours",
    "final_D_f_W_unlearn_retrain",
    "original_model_dir",
    "retrain_model_dir",
    "retained_checkpoint",
    "error",
]


class PreflightError(RuntimeError):
    pass


def normalize_algo_name(value: Any) -> str:
    text = str(value or "").upper().replace("_", "")
    if "PLASWITH" in text or text in {"PLAS", "PLA", "PLASP"}:
        return "PLASP"
    if "TD3" in text and "BC" in text:
        return "TD3PLUSBC"
    return text


def normalize_dataset_components(dataset: str, values: Sequence[str]) -> List[str]:
    """Normalize QuadX tokens to the Minari/CSV ``name-v0`` convention."""
    if dataset.lower() != "quadx":
        return list(values)
    normalized = []
    for value in values:
        token = str(value).split("/", 1)[-1]
        if re.search(r"-v\d+$", token) is None:
            token += "-v0"
        normalized.append(token)
    return normalized


def normalize_ratio_values(values: Sequence[Any]) -> Tuple[float, ...]:
    normalized = []
    for value in values:
        number = float(value)
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            raise ValueError(f"Invalid retained ratio: {value}")
        normalized.append(number)
    return tuple(normalized)


def ratio_tag(values: Sequence[Any]) -> str:
    return "_".join(str(float(value)) for value in values)


def parse_ratio_text(value: Any) -> Optional[Tuple[float, ...]]:
    text = str(value or "").strip()
    if not text:
        return None
    tokens = [token for token in re.split(r"[_\s]+", text) if token]
    try:
        return tuple(float(token.replace("p", ".")) for token in tokens)
    except ValueError:
        return None


def ratios_equal(left: Optional[Sequence[float]], right: Sequence[float]) -> bool:
    if left is None or len(left) != len(right):
        return False
    return all(abs(float(a) - float(b)) <= 1e-12 for a, b in zip(left, right))


def parse_ratio_from_path(path: Path) -> Optional[Tuple[float, ...]]:
    matched = None
    for part in path.parts:
        lowered = part.lower()
        for prefix in RATIO_PREFIXES:
            marker = prefix + "_"
            index = lowered.find(marker)
            if index < 0:
                continue
            parsed = parse_ratio_text(part[index + len(marker):])
            if parsed:
                matched = parsed
    return matched


def extract_seed(path: Path) -> Optional[int]:
    match = SEED_PATTERN.search(str(path))
    return int(match.group(1)) if match else None


def has_model_files(path: Path) -> bool:
    if (path / "model.pt").is_file():
        return True
    return any(MODEL_PATTERN.match(item.name) for item in path.glob("model_*.pt"))


def read_model_algorithm(path: Path) -> Optional[str]:
    params_path = path / "params.json"
    if not params_path.is_file():
        return None
    try:
        with params_path.open("r", encoding="utf-8") as params_file:
            params = json.load(params_file)
    except (OSError, ValueError):
        return None
    return normalize_algo_name(params.get("algorithm") or params.get("type"))


def discover_model_dirs(
    root: Path,
    dataset_components: Sequence[str],
    algorithm: str,
    expected_ratios: Sequence[float],
    selected_seeds: Optional[Sequence[int]] = None,
) -> Dict[int, Path]:
    root = root.resolve()
    if not root.is_dir():
        raise PreflightError(f"Model search root does not exist: {root}")

    component_tag = "_".join(dataset_components)
    component_tag_without_versions = "_".join(
        re.sub(r"-v\d+$", "", component) for component in dataset_components
    )
    component_tags = {component_tag, component_tag_without_versions}
    expected_algo = normalize_algo_name(algorithm)
    seed_filter = set(selected_seeds) if selected_seeds is not None else None
    candidates: Dict[int, List[Path]] = {}

    for dirpath, dirnames, _ in os.walk(str(root)):
        dirnames[:] = [
            name for name in dirnames
            if "shadow_models_cache" not in name
            and "unlearning_cost" not in name.lower()
        ]
        candidate = Path(dirpath)
        if component_tag and not any(tag in str(candidate) for tag in component_tags):
            continue
        if not (candidate / "params.json").is_file() or not has_model_files(candidate):
            continue
        if read_model_algorithm(candidate) != expected_algo:
            continue
        seed = extract_seed(candidate)
        if seed is None or (seed_filter is not None and seed not in seed_filter):
            continue
        parsed_ratios = parse_ratio_from_path(candidate)
        if not ratios_equal(parsed_ratios, expected_ratios):
            continue
        candidates.setdefault(seed, []).append(candidate)

    if not candidates:
        raise PreflightError(
            f"No model directories matched root={root}, algo={algorithm}, "
            f"datasets={component_tag}, ratios={ratio_tag(expected_ratios)}"
        )

    resolved: Dict[int, Path] = {}
    for seed, paths in sorted(candidates.items()):
        unique_paths = sorted(set(paths))
        if len(unique_paths) != 1:
            formatted = "\n  ".join(str(path) for path in unique_paths)
            raise PreflightError(
                f"Ambiguous model directories for seed={seed}:\n  {formatted}"
            )
        resolved[seed] = unique_paths[0]
    return resolved


def _trajdeleter_phase_name(steps: int) -> str:
    phase1 = int(steps * 0.8)
    phase2 = steps - phase1
    return f"phase2_{phase2}_phase1_{phase1}"


def _default_trajdeleter_search_root(
    original_model_dir: Path,
    retained_ratios: Sequence[float],
    reference_steps: int,
) -> Path:
    parts = list(original_model_dir.parts)
    try:
        fully_trained_index = max(
            index for index, part in enumerate(parts)
            if part.lower() == "fully_trained"
        )
    except ValueError as error:
        raise PreflightError(
            "Cannot infer the TrajDeleter model path because the original "
            "model path does not contain 'Fully_trained'. Provide "
            "--trajdeleter-dir explicitly."
        ) from error

    parts[fully_trained_index] = "Unlearned"
    return (
        Path(*parts)
        / f"unlearn_seed_{extract_seed(original_model_dir)}"
        / f"retained_ratios_{ratio_tag(retained_ratios)}"
        / "trajDeleter"
        / _trajdeleter_phase_name(reference_steps)
    )


def discover_trajdeleter_model_dir(
    original_model_dir: Path,
    trajdeleter_root: Optional[Path],
    dataset_components: Sequence[str],
    algorithm: str,
    retained_ratios: Sequence[float],
    seed: int,
    reference_steps: int,
) -> Path:
    if trajdeleter_root is None:
        search_root = _default_trajdeleter_search_root(
            original_model_dir,
            retained_ratios,
            reference_steps,
        )
    else:
        search_root = trajdeleter_root.resolve()

    if not search_root.exists():
        raise PreflightError(f"TrajDeleter model path does not exist: {search_root}")

    component_tags = {
        "_".join(dataset_components),
        "_".join(re.sub(r"-v\d+$", "", component) for component in dataset_components),
    }
    expected_algo = normalize_algo_name(algorithm)
    phase_name = _trajdeleter_phase_name(reference_steps)
    candidates: List[Path] = []

    for dirpath, dirnames, _ in os.walk(str(search_root)):
        dirnames[:] = [
            name for name in dirnames
            if "shadow_models_cache" not in name
            and "unlearning_cost" not in name.lower()
        ]
        candidate = Path(dirpath)
        if not (candidate / "params.json").is_file() or not has_model_files(candidate):
            continue
        candidate_parts_lower = {part.lower() for part in candidate.parts}
        if "trajdeleter" not in candidate_parts_lower:
            continue
        if phase_name not in candidate.parts:
            continue
        if not any(tag in str(candidate) for tag in component_tags):
            continue
        if read_model_algorithm(candidate) != expected_algo:
            continue
        if extract_seed(candidate) != seed:
            continue
        if not ratios_equal(parse_ratio_from_path(candidate), retained_ratios):
            continue
        candidates.append(candidate)

    if not candidates:
        raise PreflightError(
            f"No TrajDeleter model matched root={search_root}, "
            f"algo={algorithm}, datasets={'_'.join(dataset_components)}, "
            f"ratios={ratio_tag(retained_ratios)}, seed={seed}, "
            f"phase={phase_name}"
        )

    # If repeated exports exist for the same setting, use the newest timestamped
    # model directory deterministically.
    return max(candidates, key=lambda path: (path.name, path.stat().st_mtime_ns))


def _row_value(row: Dict[str, str], names: Sequence[str]) -> str:
    for name in names:
        if name in row:
            return row[name]
    raise PreflightError(f"CSV lacks required columns: one of {list(names)}")


def _integer_value(value: Any) -> Optional[int]:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else None


def _shuffle_matches(value: Any, shuffle: int) -> bool:
    normalized = str(value or "").strip().lower().replace("_", " ")
    if shuffle == 1:
        return normalized in {"shuffle", "shuffled", "1", "true"}
    return normalized in {"no shuffle", "noshuffle", "ordered", "0", "false"}


def find_trajdeleter_threshold(
    csv_path: Path,
    seed: int,
    algorithm: str,
    dataset_components: Sequence[str],
    retained_ratios: Sequence[float],
    shuffle: int,
    steps: int,
) -> Dict[str, Any]:
    if not csv_path.is_file():
        raise PreflightError(f"Threshold CSV does not exist: {csv_path}")

    components_tag = "_".join(dataset_components)
    expected_algo = normalize_algo_name(algorithm)
    phase1 = int(steps * 0.8)
    phase2 = steps - phase1
    exact_method = (
        f"Unlearned_trajDeleter_phase2_{phase2}_phase1_{phase1}"
    )

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    base_matches: List[Dict[str, str]] = []
    for row in rows:
        if _integer_value(_row_value(row, ["seed", "Seed"])) != seed:
            continue
        if normalize_algo_name(
            _row_value(row, ["Algo. Name", "Algo Name", "algo_name"])
        ) != expected_algo:
            continue
        if _row_value(row, ["Dataset Components"]) != components_tag:
            continue
        if not ratios_equal(
            parse_ratio_text(_row_value(row, ["Ratios", "Ratio Group", "Ratio"])),
            retained_ratios,
        ):
            continue
        if not _shuffle_matches(_row_value(row, ["Shuffle Mode"]), shuffle):
            continue
        if _row_value(row, ["KL_or_Wasserstein"]) != "critic_divergence":
            continue
        if _integer_value(_row_value(row, ["Step"])) != steps:
            continue
        base_matches.append(row)

    exact = [row for row in base_matches if row.get("Method Name") == exact_method]
    matches = exact
    match_mode = "exact_80_20"
    if not matches:
        matches = []
        for row in base_matches:
            match = TRAJDELETER_PATTERN.match(str(row.get("Method Name", "")))
            if match and int(match.group(1)) + int(match.group(2)) == steps:
                matches.append(row)
        match_mode = "phase_sum_fallback"

    if len(matches) != 1:
        methods = [row.get("Method Name") for row in matches or base_matches]
        raise PreflightError(
            f"Expected exactly one TrajDeleter threshold row for seed={seed}, "
            f"algo={algorithm}, datasets={components_tag}, "
            f"ratios={ratio_tag(retained_ratios)}, steps={steps}; "
            f"found {len(matches)}. Candidates={methods}"
        )

    row = matches[0]
    try:
        stored_metric = float(_row_value(row, ["D_f_W_unlearn_retrain"]))
        metric = stored_metric / ANALYZED_METRIC_SCALE
    except (TypeError, ValueError) as error:
        raise PreflightError(
            f"Invalid D_f_W_unlearn_retrain for seed={seed}: "
            f"{row.get('D_f_W_unlearn_retrain')}"
        ) from error
    if not math.isfinite(metric):
        raise PreflightError(
            f"Non-finite D_f_W_unlearn_retrain for seed={seed}: {metric}"
        )

    return {
        "value": metric,
        "metric": "D_f_W_unlearn_retrain",
        "source_csv": str(csv_path.resolve()),
        "match_mode": match_mode,
        "row": row,
    }


def canonicalize_methods(methods: Sequence[str]) -> List[str]:
    canonical = []
    for method in methods:
        compact = method.replace("_", "").replace("-", "").replace(" ", "").upper()
        resolved = METHOD_ALIASES.get(compact)
        if resolved is None:
            available = ", ".join(METHOD_CONFIGS)
            raise ValueError(f"Unknown method '{method}'. Available: {available}")
        if resolved not in canonical:
            canonical.append(resolved)
    return canonical


def build_method_command(
    method: str,
    args: argparse.Namespace,
    seed: int,
    original_model_dir: Path,
    cost_config_path: Path,
) -> List[str]:
    script_path = REPO_ROOT / METHOD_CONFIGS[method]["script"]
    common = [
        sys.executable,
        str(script_path),
        "--dataset",
        args.dataset,
        "--datasets",
        *args.datasets,
        "--seed",
        str(seed),
        "--gpu",
        str(args.gpu),
        "--algo",
        args.algo,
        "--shuffle",
        str(args.shuffle),
    ]
    ratios = [str(value) for value in args.retained_ratios]
    cost_option = ["--cost-measurement-config", str(cost_config_path)]

    if method in {"SARAVI"}:
        return common + [
            "--model-to-unlearn-dir", str(original_model_dir),
            "--retained-ratios", *ratios,
            "--unlearning-steps", str(args.maximum_steps),
            "--eval-for-record-interval", "0",
            "--eval-interval", str(args.maximum_steps + 1),
        ] + cost_option
    if method == "NegativeReward":
        return common + [
            "--model-to-unlearn-dir", str(original_model_dir),
            "--retained-ratios", *ratios,
            "--phase1-total-steps", str(args.maximum_steps),
            "--phase2-total-steps", "0",
        ] + cost_option
    if method == "RandomReward":
        return common + [
            "--model-to-unlearn-dir", str(original_model_dir),
            "--retained-ratios", *ratios,
            "--total-steps", str(args.maximum_steps),
            "--n-steps-per-epoch", str(args.maximum_steps),
        ] + cost_option
    if method == "Finetuning":
        return common + [
            "--model-to-unlearn-dir", str(original_model_dir),
            "--retained-ratios", *ratios,
            "--unlearning-steps", str(args.maximum_steps),
            "--eval-for-record-interval", str(args.maximum_steps + 1),
            "--eval-interval", str(args.maximum_steps + 1),
        ] + cost_option
    if method == "Retraining":
        return common + [
            "--type", "RetrainingCost",
            "--base-params", str(original_model_dir / "params.json"),
            "--ratios", *ratios,
            "--n-steps", str(args.maximum_steps),
        ] + cost_option
    raise AssertionError(f"Unhandled method: {method}")


def evaluation_schedule(max_steps: int, interval: int) -> List[int]:
    if max_steps <= 0 or interval <= 0:
        raise ValueError("maximum_steps and eval_interval must be positive.")
    steps = list(range(interval, max_steps + 1, interval))
    if not steps or steps[-1] != max_steps:
        steps.append(max_steps)
    return steps


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    os.replace(str(temporary), str(path))


def summarize_run_for_aggregate(run: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only terminal run data in the external aggregate JSON."""
    summary = dict(run)
    evaluations = summary.pop("evaluations", None)
    if evaluations:
        summary["terminal_evaluation"] = evaluations[-1]
    return summary


def write_results_csv(path: Path, runs: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for run in runs:
            threshold = run.get("threshold") or {}
            writer.writerow(
                {
                    "method": run.get("method"),
                    "seed": run.get("seed"),
                    "status": run.get("status"),
                    "threshold_D_f_W_unlearn_retrain": threshold.get("value"),
                    "steps_to_threshold": run.get("steps_to_threshold"),
                    "steps_trained": run.get("steps_trained"),
                    "training_wall_seconds": run.get("training_wall_seconds"),
                    "gpu_hours": run.get("gpu_hours"),
                    "final_D_f_W_unlearn_retrain": run.get(
                        "final_D_f_W_unlearn_retrain"
                    ),
                    "original_model_dir": run.get("original_model_dir"),
                    "retrain_model_dir": run.get("retrain_model_dir"),
                    "retained_checkpoint": run.get("retained_checkpoint"),
                    "error": run.get("error"),
                }
            )
    os.replace(str(temporary), str(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Measure the cost required for unlearning methods to match TrajDeleter."
    )
    parser.add_argument(
        "--reference-baseline--steps", "--reference-baseline-steps",
        dest="reference_baseline_steps", type=int, default=10000,
        help="TrajDeleter reference-baseline steps (default: 10000).",
    )
    parser.add_argument(
        "--Maximum-steps", "--maximum-steps",
        dest="maximum_steps", type=int, default=500000,
        help="Maximum training steps per method run (default: 500000).",
    )
    parser.add_argument(
        "--eval-interval", type=int, default=500,
        help="ORL-like evaluation interval (default: 500).",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--retained-ratios", nargs="+", type=float, required=True)
    parser.add_argument("--algo", required=True)
    parser.add_argument("--shuffle", type=int, choices=(0, 1), default=1)
    parser.add_argument("--model-to-unlearn-dir", required=True)
    parser.add_argument("--retrain-dir", required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--forget-sample-episodes", type=int, default=100,
        help="Number of D_f trajectories sampled per critic evaluation "
        "(default: 100).",
    )
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--threshold-csv",
        help="Legacy full-dataset threshold CSV; no longer used when "
        "TrajDeleter model evaluation is enabled.",
    )
    parser.add_argument(
        "--trajdeleter-dir",
        help="Optional root containing same-setting TrajDeleter models "
        "for same-sample reference evaluation.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if (
        args.reference_baseline_steps <= 0
        or args.maximum_steps <= 0
        or args.eval_interval <= 0
    ):
        raise ValueError("All step parameters must be positive.")
    if len(args.datasets) != len(args.retained_ratios):
        raise ValueError("--datasets and --retained-ratios must have equal lengths.")
    args.datasets = normalize_dataset_components(args.dataset, args.datasets)
    args.retained_ratios = normalize_ratio_values(args.retained_ratios)
    args.methods = canonicalize_methods(args.methods)
    missing_scripts = [
        REPO_ROOT / METHOD_CONFIGS[method]["script"]
        for method in args.methods
        if not (REPO_ROOT / METHOD_CONFIGS[method]["script"]).is_file()
    ]
    if missing_scripts:
        missing = ", ".join(str(path) for path in missing_scripts)
        raise ValueError(f"Training script preflight failed; missing: {missing}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    args.forget_sample_episodes = int(
        getattr(args, "forget_sample_episodes", 100)
    )
    if args.forget_sample_episodes <= 0:
        raise ValueError("--forget-sample-episodes must be positive.")


def resolve_input_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    invocation_relative = path.resolve()
    if invocation_relative.exists():
        return invocation_relative
    repository_relative = (REPO_ROOT / path).resolve()
    if repository_relative.exists():
        return repository_relative
    script_relative = (SCRIPT_DIR / path).resolve()
    if script_relative.exists():
        return script_relative
    return invocation_relative


def preflight(args: argparse.Namespace):
    original_root = resolve_input_path(args.model_to_unlearn_dir)
    retrain_root = resolve_input_path(args.retrain_dir)
    original_ratios = tuple(1.0 for _ in args.datasets)
    original_models: Dict[int, Path] = {}
    retrain_models: Dict[int, Path] = {}
    thresholds: Dict[int, Dict[str, Any]] = {}
    trajdeleter_models: Dict[int, Path] = {}
    seed_errors: Dict[int, str] = {}

    if args.seeds is None:
        original_models = discover_model_dirs(
            original_root,
            args.datasets,
            args.algo,
            original_ratios,
        )
        target_seeds = sorted(original_models)
    else:
        target_seeds = sorted(set(args.seeds))
        for seed in target_seeds:
            try:
                original_models[seed] = discover_model_dirs(
                    original_root,
                    args.datasets,
                    args.algo,
                    original_ratios,
                    [seed],
                )[seed]
            except (KeyError, PreflightError) as error:
                seed_errors[seed] = f"Original model preflight failed: {error}"

    for seed in target_seeds:
        if seed in seed_errors:
            continue
        try:
            retrain_models[seed] = discover_model_dirs(
                retrain_root,
                args.datasets,
                args.algo,
                args.retained_ratios,
                [seed],
            )[seed]
        except (KeyError, PreflightError) as error:
            seed_errors[seed] = f"Retrain model preflight failed: {error}"

    configured_trajdeleter_root = getattr(args, "trajdeleter_dir", None)
    trajdeleter_root = (
        resolve_input_path(configured_trajdeleter_root)
        if configured_trajdeleter_root
        else None
    )
    for seed in target_seeds:
        if seed in seed_errors:
            continue
        try:
            trajdeleter_models[seed] = discover_trajdeleter_model_dir(
                original_models[seed],
                trajdeleter_root,
                args.datasets,
                args.algo,
                args.retained_ratios,
                seed,
                args.reference_baseline_steps,
            )
            thresholds[seed] = {
                "value": None,
                "metric": "D_f_W_unlearn_retrain",
                "source": "same_sample_trajdeleter_model",
                "source_model_dir": str(trajdeleter_models[seed]),
            }
        except PreflightError as error:
            seed_errors[seed] = f"TrajDeleter model preflight failed: {error}"

    return (
        target_seeds,
        original_models,
        retrain_models,
        thresholds,
        seed_errors,
    )


def failed_run_from_process(
    method: str,
    seed: int,
    task: Dict[str, Any],
    threshold: Dict[str, Any],
    original_model: Path,
    retrain_model: Path,
    command: Sequence[str],
    returncode: int,
    job_result_path: Path,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "method": method,
        "seed": seed,
        "task": task,
        "threshold": threshold,
        "original_model_dir": str(original_model),
        "retrain_model_dir": str(retrain_model),
        "command": list(command),
        "status": "failed",
        "steps_to_threshold": None,
        "steps_trained": 0,
        "training_wall_seconds": 0.0,
        "gpu_hours": 0.0,
        "final_D_f_W_unlearn_retrain": None,
        "evaluations": [],
        "retained_checkpoint": None,
        "error": (
            f"Training process exited with code {returncode} and did not produce "
            f"a valid job result at {job_result_path}"
        ),
    }




def preflight_error_run(
    method: str,
    seed: int,
    task: Dict[str, Any],
    error: str,
    original_model: Optional[Path],
    retrain_model: Optional[Path],
    threshold: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "method": method,
        "seed": seed,
        "task": task,
        "threshold": threshold,
        "original_model_dir": str(original_model) if original_model else None,
        "retrain_model_dir": str(retrain_model) if retrain_model else None,
        "command": None,
        "status": "preflight_error",
        "steps_to_threshold": None,
        "steps_trained": 0,
        "training_wall_seconds": 0.0,
        "gpu_hours": 0.0,
        "final_D_f_W_unlearn_retrain": None,
        "evaluations": [],
        "retained_checkpoint": None,
        "error": error,
    }

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        seeds, original_models, retrain_models, thresholds, preflight_errors = preflight(args)
    except (ValueError, PreflightError) as error:
        parser.error(str(error))

    task = {
        "reference_baseline_steps": args.reference_baseline_steps,
        "maximum_steps": args.maximum_steps,
        "eval_interval": args.eval_interval,
        "forget_sample_episodes": args.forget_sample_episodes,
        "dataset": args.dataset,
        "datasets": list(args.datasets),
        "retained_ratios": list(args.retained_ratios),
        "algo": args.algo,
        "shuffle": args.shuffle,
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (REPO_ROOT / "unlearning_cost_outputs" / timestamp)
    )

    planned = []
    for seed in seeds:
        if seed in preflight_errors:
            continue
        for method in args.methods:
            job_dir = output_dir / "jobs" / f"{method}_seed_{seed}"
            config_path = job_dir / "cost_config.json"
            command = build_method_command(
                method, args, seed, original_models[seed], config_path
            )
            planned.append((seed, method, job_dir, config_path, command))

    if preflight_errors:
        print("Unlearning cost preflight completed with seed errors.")
    else:
        print("Unlearning cost preflight succeeded.")
    for seed in seeds:
        if seed in preflight_errors:
            print(f"[preflight_error] seed={seed}: {preflight_errors[seed]}")
        else:
            print(
                f"seed={seed} reference=pending "
                f"trajdeleter={thresholds[seed]['source_model_dir']} "
                f"original={original_models[seed]} retrain={retrain_models[seed]}"
            )
    for seed, method, _, _, command in planned:
        print(f"[plan] seed={seed} method={method}: {shlex.join(command)}")

    if args.dry_run:
        return 1 if preflight_errors else 0

    output_dir.mkdir(parents=True, exist_ok=True)
    preflight_runs = [
        preflight_error_run(
            method,
            seed,
            task,
            preflight_errors[seed],
            original_models.get(seed),
            retrain_models.get(seed),
            thresholds.get(seed),
        )
        for seed in seeds if seed in preflight_errors for method in args.methods
    ]
    aggregate = {
        "schema_version": 1,
        "created_at": timestamp,
        "task": task,
        "method_configurations": METHOD_CONFIGS,
        "thresholds_by_seed": {str(seed): value for seed, value in thresholds.items()},
        "preflight_errors_by_seed": {str(seed): error for seed, error in preflight_errors.items()},
        "runs": preflight_runs,
    }
    json_path = output_dir / "unlearning_cost_results.json"
    csv_path = output_dir / "unlearning_cost_results.csv"
    atomic_write_json(json_path, aggregate)
    write_results_csv(csv_path, aggregate["runs"])

    any_failed = bool(preflight_errors)
    for seed, method, job_dir, config_path, command in planned:
        job_dir.mkdir(parents=True, exist_ok=True)
        job_result_path = job_dir / "job_result.json"
        job_config = {
            "schema_version": 1,
            "method": method,
            "seed": seed,
            "reference_baseline_steps": args.reference_baseline_steps,
            "maximum_steps": args.maximum_steps,
            "eval_interval": args.eval_interval,
            "forget_sample_episodes": args.forget_sample_episodes,
            "gpu": args.gpu,
            "batch_size": args.batch_size,
            "task": task,
            "threshold": thresholds[seed],
            "trajdeleter_model_dir": thresholds[seed]["source_model_dir"],
            "original_model_dir": str(original_models[seed]),
            "retrain_model_dir": str(retrain_models[seed]),
            "job_result_path": str(job_result_path),
            "checkpoint_dir": str(job_dir / "checkpoints"),
            "training_log_dir": str(job_dir / "training"),
            "command": command,
        }
        atomic_write_json(config_path, job_config)

        print(f"[run] seed={seed} method={method}")
        completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
        run_result = None
        if job_result_path.is_file():
            try:
                with job_result_path.open("r", encoding="utf-8") as result_file:
                    run_result = json.load(result_file)
            except (OSError, ValueError):
                run_result = None

        if run_result is None:
            run_result = failed_run_from_process(
                method,
                seed,
                task,
                thresholds[seed],
                original_models[seed],
                retrain_models[seed],
                command,
                completed.returncode,
                job_result_path,
            )
        elif run_result.get("status") not in {"achieved", "not_achieved", "failed"}:
            run_result["status"] = "failed"
            run_result["error"] = (
                run_result.get("error")
                or "Training process ended without a terminal cost status."
            )
        elif completed.returncode != 0 and run_result.get("status") != "failed":
            run_result["status"] = "failed"
            run_result["error"] = (
                run_result.get("error")
                or f"Training process exited with code {completed.returncode}."
            )

        any_failed = any_failed or run_result.get("status") == "failed"
        aggregate["runs"].append(summarize_run_for_aggregate(run_result))
        atomic_write_json(json_path, aggregate)
        write_results_csv(csv_path, aggregate["runs"])

    print(f"Saved detailed results to {json_path}")
    print(f"Saved flat results to {csv_path}")
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())