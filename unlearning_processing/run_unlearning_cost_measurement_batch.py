#!/usr/bin/env python3
"""Preflight and execute multiple unlearning-cost task configurations."""

import argparse
import concurrent.futures
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT_SCRIPT = REPO_ROOT / "unlearning_processing" / "unlearning_cost_measurement.py"
DEFAULT_REFERENCE_BASELINE_STEPS = 10000
DEFAULT_MAXIMUM_STEPS = 500000
DEFAULT_EVAL_INTERVAL = 500
DEFAULT_FORGET_SAMPLE_EPISODES = 100
DEFAULT_METHODS = (
    "SARAVI",
    "NegativeReward",
    "RandomReward",
    "Finetuning",
    "Retraining",
)
BATCH_OUTPUTS_ROOT = REPO_ROOT / "unlearning_cost_batch_outputs"
TERMINAL_RUN_STATUSES = {"achieved", "not_achieved"}

KEY_ALIASES = {
    "reference-baseline--steps": "reference_baseline_steps",
    "reference-baseline-steps": "reference_baseline_steps",
    "Maximum-steps": "maximum_steps",
    "maximum-steps": "maximum_steps",
    "eval-interval": "eval_interval",
    "retained-ratios": "retained_ratios",
    "model-to-unlearn-dir": "model_to_unlearn_dir",
    "retrain-dir": "retrain_dir",
    "batch-size": "batch_size",
    "forget-sample-episodes": "forget_sample_episodes",
    "threshold-csv": "threshold_csv",
    "trajdeleter-dir": "trajdeleter_dir",
}
ALLOWED_KEYS = {
    "name",
    "reference_baseline_steps",
    "maximum_steps",
    "eval_interval",
    "dataset",
    "datasets",
    "retained_ratios",
    "algo",
    "shuffle",
    "model_to_unlearn_dir",
    "retrain_dir",
    "methods",
    "seeds",
    "gpu",
    "batch_size",
    "forget_sample_episodes",
    "threshold_csv",
    "trajdeleter_dir",
}
REQUIRED_KEYS = {
    "dataset",
    "datasets",
    "retained_ratios",
    "algo",
    "model_to_unlearn_dir",
    "retrain_dir",
}


class BatchConfigurationError(ValueError):
    pass


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Return a JSON object, or None when a progress file is absent/corrupt."""
    try:
        with path.open("r", encoding="utf-8") as input_file:
            payload = json.load(input_file)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _task_matches_job(task: Any, job: Dict[str, Any]) -> bool:
    """Ensure cached results were produced for exactly this measurement task."""
    if not isinstance(task, dict):
        return False
    expected = {
        "reference_baseline_steps": job["reference_baseline_steps"],
        "maximum_steps": job["maximum_steps"],
        "eval_interval": job["eval_interval"],
        "forget_sample_episodes": job.get(
            "forget_sample_episodes", DEFAULT_FORGET_SAMPLE_EPISODES
        ),
        "dataset": job["dataset"],
        "datasets": list(job["datasets"]),
        "retained_ratios": list(job["retained_ratios"]),
        "algo": job["algo"],
        "shuffle": job.get("shuffle", 1),
    }
    return all(task.get(key) == value for key, value in expected.items())


def job_output_is_complete(output_dir: Path, job: Dict[str, Any]) -> bool:
    """Return whether a job has all final results and their saved checkpoints.

    The batch manifest alone is deliberately not trusted: a process can be
    interrupted after its child has finished, leaving the manifest as
    ``running``. Every configured method/seed must instead have a terminal
    result and a checkpoint under that method/seed's output directory.
    """
    aggregate = _read_json(output_dir / "unlearning_cost_results.json")
    if aggregate is None or not _task_matches_job(aggregate.get("task"), job):
        return False
    runs = aggregate.get("runs")
    if not isinstance(runs, list):
        return False

    # With implicit seed discovery the desired seed set can change as models
    # are added. Do not silently skip such a job based on a stale aggregate.
    if job.get("seeds") is None:
        return False
    expected = {
        (str(method), int(seed))
        for method in job.get("methods", DEFAULT_METHODS)
        for seed in job["seeds"]
    }
    found: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        try:
            key = (str(run.get("method")), int(run.get("seed")))
        except (TypeError, ValueError):
            continue
        found[key] = run

    if set(found) != expected:
        return False
    for (method, seed), run in found.items():
        if run.get("status") not in TERMINAL_RUN_STATUSES:
            return False
        checkpoint = run.get("retained_checkpoint")
        if not checkpoint:
            return False
        checkpoint_path = Path(str(checkpoint)).expanduser()
        expected_dir = output_dir / "jobs" / f"{method}_seed_{seed}" / "checkpoints"
        try:
            checkpoint_path.resolve().relative_to(expected_dir.resolve())
        except ValueError:
            return False
        if not checkpoint_path.is_file():
            return False
        job_result = _read_json(
            output_dir / "jobs" / f"{method}_seed_{seed}" / "job_result.json"
        )
        if (
            job_result is None
            or job_result.get("status") not in TERMINAL_RUN_STATUSES
            or job_result.get("retained_checkpoint") != checkpoint
        ):
            return False
    return True


def find_auto_resume_root(
    config_path: Path, jobs: Sequence[Dict[str, Any]]
) -> Optional[Path]:
    """Find the prior compatible batch with the most verified completed jobs."""
    if not BATCH_OUTPUTS_ROOT.is_dir():
        return None
    candidates = []
    for manifest_path in BATCH_OUTPUTS_ROOT.glob("*/unlearning_cost_batch_manifest.json"):
        manifest = _read_json(manifest_path)
        if manifest is None or not manifest.get("config"):
            continue
        try:
            manifest_config = Path(str(manifest["config"])).expanduser().resolve()
        except OSError:
            continue
        if manifest_config != config_path:
            continue
        root = manifest_path.parent
        completed_count = sum(
            job_output_is_complete(root / job["name"], job) for job in jobs
        )
        if completed_count:
            candidates.append((completed_count, manifest_path.stat().st_mtime, root))
    if not candidates:
        return None
    # Prefer the batch that retains the most work; use recency as a tiebreaker.
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def resolve_manifest_path(resume_from: Path) -> Path:
    """Resolve either a previous batch directory or its manifest file."""
    candidate = resume_from.expanduser().resolve()
    if candidate.is_dir():
        candidate = candidate / "unlearning_cost_batch_manifest.json"
    if not candidate.is_file():
        raise BatchConfigurationError(
            f"Cannot resume: manifest does not exist at {candidate}"
        )
    return candidate


def load_previous_manifest(resume_from: Path) -> Tuple[Path, Dict[str, Any]]:
    manifest_path = resolve_manifest_path(resume_from)
    try:
        with manifest_path.open("r", encoding="utf-8") as input_file:
            manifest = json.load(input_file)
    except (OSError, ValueError) as error:
        raise BatchConfigurationError(
            f"Cannot read previous batch manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("jobs"), list):
        raise BatchConfigurationError(
            f"Previous batch manifest {manifest_path} has no valid jobs list."
        )
    return manifest_path, manifest


def _record_is_complete(record: Dict[str, Any]) -> bool:
    """Only a successfully completed child batch is safe to skip on resume."""
    return record.get("status") == "succeeded"


def records_from_previous_manifest(
    manifest_path: Path, manifest: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Convert old/new manifest records into the scheduler's internal format."""
    records = []
    for index, previous in enumerate(manifest["jobs"]):
        if not isinstance(previous, dict):
            raise BatchConfigurationError(
                f"Previous manifest job {index} is not a JSON object."
            )
        name = previous.get("name")
        output_dir = previous.get("output_dir")
        command = previous.get("command")
        if not name or not output_dir or not isinstance(command, list) or not command:
            raise BatchConfigurationError(
                f"Previous manifest job {index} is missing name, output_dir, or command."
            )
        record = dict(previous)
        record["index"] = int(record.get("index", index))
        record["name"] = str(name)
        record["output_dir"] = str(output_dir)
        record["command"] = [str(value) for value in command]
        record["gpu"] = int(record.get("gpu", 0))
        previous_status = record.get("status", "pending_preflight")
        record["previous_status"] = previous_status
        record["resume_count"] = int(record.get("resume_count", 0)) + 1
        complete = _record_is_complete(record)
        if not complete:
            record["status"] = "pending_preflight"
            record["preflight_returncode"] = None
            record["run_returncode"] = None
            record["error"] = None
        records.append(
            {
                "job": record.get("job"),
                "explicit_gpu": True,
                "record": record,
                "skip": complete,
                "manifest_path": str(manifest_path),
            }
        )
    return records


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
    os.replace(str(temporary), str(path))


def canonicalize_keys(payload: Dict[str, Any], label: str) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise BatchConfigurationError(f"{label} must be a JSON object.")
    normalized: Dict[str, Any] = {}
    for key, value in payload.items():
        canonical = KEY_ALIASES.get(key, key)
        if canonical not in ALLOWED_KEYS:
            raise BatchConfigurationError(f"Unknown key in {label}: {key}")
        if canonical in normalized:
            raise BatchConfigurationError(
                f"Duplicate aliases for '{canonical}' in {label}."
            )
        normalized[canonical] = value
    return normalized


def validate_job(job: Dict[str, Any], label: str) -> None:
    missing = sorted(REQUIRED_KEYS - set(job))
    if missing:
        raise BatchConfigurationError(f"{label} is missing required keys: {missing}")
    if not isinstance(job["datasets"], list) or not job["datasets"]:
        raise BatchConfigurationError(f"{label}.datasets must be a non-empty list.")
    if not isinstance(job["retained_ratios"], list):
        raise BatchConfigurationError(f"{label}.retained_ratios must be a list.")
    if len(job["datasets"]) != len(job["retained_ratios"]):
        raise BatchConfigurationError(
            f"{label}: datasets and retained_ratios must have equal lengths."
        )
    for key in ("reference_baseline_steps", "maximum_steps", "eval_interval"):
        value = job[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise BatchConfigurationError(f"{label}.{key} must be a positive integer.")
    if job.get("shuffle", 1) not in (0, 1):
        raise BatchConfigurationError(f"{label}.shuffle must be 0 or 1.")
    if "methods" in job and (
        not isinstance(job["methods"], list) or not job["methods"]
    ):
        raise BatchConfigurationError(f"{label}.methods must be a non-empty list.")
    if "seeds" in job and (
        not isinstance(job["seeds"], list)
        or not all(isinstance(seed, int) for seed in job["seeds"])
    ):
        raise BatchConfigurationError(f"{label}.seeds must be a list of integers.")
    if int(job.get("batch_size", 512)) <= 0:
        raise BatchConfigurationError(f"{label}.batch_size must be positive.")
    if int(job.get("forget_sample_episodes", DEFAULT_FORGET_SAMPLE_EPISODES)) <= 0:
        raise BatchConfigurationError(
            f"{label}.forget_sample_episodes must be positive."
        )


def load_batch_definition(path: Path) -> List[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as input_file:
            payload = json.load(input_file)
    except (OSError, ValueError) as error:
        raise BatchConfigurationError(f"Cannot read batch config {path}: {error}") from error
    if not isinstance(payload, dict):
        raise BatchConfigurationError("Batch config must be a JSON object.")
    defaults = canonicalize_keys(payload.get("defaults", {}), "defaults")
    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise BatchConfigurationError("Batch config must contain a non-empty jobs list.")

    step_defaults = {
        "reference_baseline_steps": DEFAULT_REFERENCE_BASELINE_STEPS,
        "maximum_steps": DEFAULT_MAXIMUM_STEPS,
        "eval_interval": DEFAULT_EVAL_INTERVAL,
        "shuffle": 1,
        "batch_size": 512,
    }
    jobs: List[Dict[str, Any]] = []
    used_names = set()
    for index, raw_job in enumerate(raw_jobs):
        merged = dict(step_defaults)
        merged.update(defaults)
        merged.update(canonicalize_keys(raw_job, f"jobs[{index}]"))
        validate_job(merged, f"jobs[{index}]")
        raw_name = str(
            merged.get("name") or f"job_{index:03d}_{merged['dataset']}_{merged['algo']}"
        )
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip("._-")
        if not name:
            raise BatchConfigurationError(f"jobs[{index}] has an empty normalized name.")
        if name in used_names:
            raise BatchConfigurationError(f"Duplicate batch job name: {name}")
        used_names.add(name)
        merged["name"] = name
        jobs.append(merged)
    return jobs


def build_measurement_command(
    job: Dict[str, Any],
    output_dir: Path,
    gpu: int,
    python_executable: str = sys.executable,
) -> List[str]:
    command = [
        python_executable,
        str(MEASUREMENT_SCRIPT),
        "--reference-baseline--steps",
        str(job["reference_baseline_steps"]),
        "--Maximum-steps",
        str(job["maximum_steps"]),
        "--eval-interval",
        str(job["eval_interval"]),
        "--dataset",
        str(job["dataset"]),
        "--datasets",
        *[str(value) for value in job["datasets"]],
        "--retained-ratios",
        *[str(value) for value in job["retained_ratios"]],
        "--algo",
        str(job["algo"]),
        "--shuffle",
        str(job.get("shuffle", 1)),
        "--model-to-unlearn-dir",
        str(job["model_to_unlearn_dir"]),
        "--retrain-dir",
        str(job["retrain_dir"]),
        "--gpu",
        str(gpu),
        "--batch-size",
        str(job.get("batch_size", 512)),
        "--forget-sample-episodes",
        str(job.get("forget_sample_episodes", DEFAULT_FORGET_SAMPLE_EPISODES)),
        "--output-dir",
        str(output_dir),
    ]
    if job.get("methods"):
        command.extend(["--methods", *[str(value) for value in job["methods"]]])
    if job.get("seeds") is not None:
        command.extend(["--seeds", *[str(value) for value in job["seeds"]]])
    if job.get("threshold_csv"):
        command.extend(["--threshold-csv", str(job["threshold_csv"])])
    if job.get("trajdeleter_dir"):
        command.extend(["--trajdeleter-dir", str(job["trajdeleter_dir"])])
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Run a JSON list of unlearning-cost configurations. Every job is "
            "preflighted before any training process is started."
        )
    )
    parser.add_argument("--config", help="Batch JSON definition.")
    parser.add_argument(
        "--resume-from",
        help=(
            "Resume a previous batch. Accepts either the previous batch output "
            "directory or unlearning_cost_batch_manifest.json. Completed jobs "
            "are skipped and unfinished jobs are retried."
        ),
    )
    parser.add_argument("--output-dir", help="Batch output root.")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--gpu-start", type=int, default=0)
    parser.add_argument(
        "--no-gpu-round-robin",
        action="store_true",
        help="Use --gpu-start for every job instead of assigning consecutive GPUs.",
    )
    parser.add_argument(
        "--continue-on-preflight-error",
        action="store_true",
        help="Run valid jobs even if another job fails preflight.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all child preflights and generate the manifest, but do not train.",
    )
    return parser


def _run(command: Sequence[str]) -> int:
    return subprocess.run(command, cwd=str(REPO_ROOT), check=False).returncode


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.parallel <= 0:
        parser.error("--parallel must be positive.")
    if args.resume_from and args.output_dir:
        parser.error("--resume-from cannot be combined with --output-dir.")
    if not args.resume_from and not args.config:
        parser.error("--config is required unless --resume-from is provided.")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.resume_from:
        try:
            manifest_path, manifest = load_previous_manifest(
                Path(args.resume_from)
            )
            records = records_from_previous_manifest(manifest_path, manifest)
        except BatchConfigurationError as error:
            parser.error(str(error))
        output_root = manifest_path.parent
        manifest["schema_version"] = max(int(manifest.get("schema_version", 1)), 2)
        manifest["resumed_at"] = timestamp
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        manifest["resume_parallel"] = args.parallel
        manifest["dry_run"] = args.dry_run
        manifest["jobs"] = [item["record"] for item in records]
        print(f"Resuming previous batch from {manifest_path}")
    else:
        config_path = Path(args.config).resolve()
        try:
            jobs = load_batch_definition(config_path)
        except BatchConfigurationError as error:
            parser.error(str(error))

        # A normal --config invocation resumes the compatible batch that has
        # the most verified work. --output-dir remains an explicit request for
        # a separate destination.
        auto_resume_root = (
            None if args.output_dir else find_auto_resume_root(config_path, jobs)
        )
        output_root = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else (auto_resume_root or BATCH_OUTPUTS_ROOT / timestamp)
        )
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "unlearning_cost_batch_manifest.json"
        previous_manifest = _read_json(manifest_path) if auto_resume_root else None

        records = []
        for index, job in enumerate(jobs):
            explicit_gpu = "gpu" in job
            gpu = int(job["gpu"]) if explicit_gpu else (
                args.gpu_start
                if args.no_gpu_round_robin
                else args.gpu_start + (index % args.parallel)
            )
            job_output = output_root / job["name"]
            command = build_measurement_command(job, job_output, gpu)
            complete = job_output_is_complete(job_output, job)
            record = {
                "index": index,
                "name": job["name"],
                "job": job,
                "gpu": gpu,
                "output_dir": str(job_output),
                "command": command,
                "status": "succeeded" if complete else "pending_preflight",
                "preflight_returncode": 0 if complete else None,
                "run_returncode": 0 if complete else None,
                "error": None,
                "resume_count": (
                    int((previous_manifest or {}).get("resume_count", 0)) + 1
                    if auto_resume_root else 0
                ),
                "completion_verified": complete,
            }
            records.append(
                {
                    "job": job,
                    "explicit_gpu": explicit_gpu,
                    "record": record,
                    "skip": complete,
                }
            )

        manifest = {
            "schema_version": 3,
            "created_at": (previous_manifest or {}).get("created_at", timestamp),
            "config": str(config_path),
            "parallel": args.parallel,
            "dry_run": args.dry_run,
            "jobs": [item["record"] for item in records],
        }
        if auto_resume_root:
            manifest.update(
                {
                    "resumed_at": timestamp,
                    "resume_count": int(
                        (previous_manifest or {}).get("resume_count", 0)
                    ) + 1,
                    "resume_parallel": args.parallel,
                    "auto_resume_from": str(auto_resume_root),
                }
            )
            skipped_names = [
                item["record"]["name"] for item in records if item["skip"]
            ]
            print(
                "Automatically resuming "
                f"{auto_resume_root}; verified completed jobs: "
                f"{', '.join(skipped_names)}"
            )

    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest_path, manifest)

    active_records = [
        item for item in records if not item.get("skip", False)
    ]
    for item in active_records:
        record = item["record"]
        preflight_command = list(record["command"]) + ["--dry-run"]
        print(f"[preflight] {record['name']}: {shlex.join(preflight_command)}")
        record["status"] = "preflight_running"
        atomic_write_json(manifest_path, manifest)
        returncode = _run(preflight_command)
        record["preflight_returncode"] = returncode
        record["status"] = "preflight_ok" if returncode == 0 else "preflight_error"
        if returncode != 0:
            record["error"] = f"Preflight exited with code {returncode}."
        atomic_write_json(manifest_path, manifest)

    preflight_failed = any(
        item["record"]["status"] == "preflight_error"
        for item in active_records
    )
    runnable = [
        item
        for item in active_records
        if item["record"]["status"] == "preflight_ok"
    ]
    if preflight_failed and not args.continue_on_preflight_error:
        for item in runnable:
            item["record"]["status"] = "aborted_due_to_batch_preflight_error"
        atomic_write_json(manifest_path, manifest)
        print(f"Batch aborted before training; see {manifest_path}")
        return 1
    if args.dry_run:
        print(f"Batch dry-run manifest: {manifest_path}")
        return 1 if preflight_failed else 0

    for run_index, item in enumerate(runnable):
        if not item["explicit_gpu"] and not args.resume_from:
            gpu = (
                args.gpu_start
                if args.no_gpu_round_robin
                else args.gpu_start + (run_index % args.parallel)
            )
            item["record"]["gpu"] = gpu
            item["record"]["command"] = build_measurement_command(
                item["job"], Path(item["record"]["output_dir"]), gpu
            )
        item["record"]["status"] = "queued"
    atomic_write_json(manifest_path, manifest)

    pending = list(runnable)
    active_gpus = set()
    future_to_item = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        while pending or future_to_item:
            while len(future_to_item) < args.parallel:
                next_index = next(
                    (
                        index
                        for index, item in enumerate(pending)
                        if item["record"]["gpu"] not in active_gpus
                    ),
                    None,
                )
                if next_index is None:
                    break
                item = pending.pop(next_index)
                record = item["record"]
                record["status"] = "running"
                record["attempts"] = int(record.get("attempts", 0)) + 1
                active_gpus.add(record["gpu"])
                future = executor.submit(_run, record["command"])
                future_to_item[future] = item
                atomic_write_json(manifest_path, manifest)

            done, _ = concurrent.futures.wait(
                future_to_item,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                item = future_to_item.pop(future)
                record = item["record"]
                active_gpus.remove(record["gpu"])
                try:
                    returncode = future.result()
                    record["run_returncode"] = returncode
                    record["status"] = "succeeded" if returncode == 0 else "failed"
                    if returncode != 0:
                        record["error"] = f"Measurement exited with code {returncode}."
                except Exception as error:
                    record["status"] = "failed"
                    record["error"] = f"{type(error).__name__}: {error}"
                atomic_write_json(manifest_path, manifest)
                print(f"[{record['status']}] {record['name']}")

    failed = preflight_failed or any(
        item["record"]["status"] == "failed"
        for item in active_records
    )
    print(f"Batch manifest: {manifest_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())