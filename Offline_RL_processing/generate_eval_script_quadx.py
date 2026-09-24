"""Generate shell scripts for batch QuadX d3rlpy/SB3 evaluation."""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import shlex
import sys
import time
from pathlib import Path

from generator_ratio_filter import normalize_ratio_filter, path_matches_ratio, ratio_filename_suffix

from quadx_path_metadata import (
    build_eval_targets,
    infer_quadx_path_metadata,
    normalize_dataset_tokens,
    parse_ratio_values_from_path,
    ratio_tag_from_values,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EVALUATION_SCRIPT = SCRIPT_DIR / "evaluation_quadx.py"
DEFAULT_ENV_ID_LIST = [
    "retain_random_static_spheres",
    "forget_low_altitude_barrier",
]
AUX_ZIP_MARKERS = ("replay_buffer", "vecnormalize")


def normalize_backend(value: str) -> str:
    normalized = value.lower().replace("_", "-")
    if normalized in {"d3rl", "d3rlpy"}:
        return "d3rl"
    if normalized in {"sb3", "stable-baselines3", "stable-baseline3"}:
        return "sb3"
    raise ValueError("--d3rl-or-sb3 must be one of: d3rl, d3rlpy, sb3")


def split_env_tokens(values: list[str]) -> list[str]:
    tokens: list[str] = []
    for value in values:
        tokens.extend(token.strip() for token in value.split(",") if token.strip())
    if not tokens:
        raise ValueError("At least one eval environment must be provided.")
    return tokens


def get_max_step_d3rl_model(files: list[str]) -> str | None:
    pattern = re.compile(r"^model_(\d+)\.pt$")
    valid_models: list[tuple[int, str]] = []

    for filename in files:
        match = pattern.match(filename)
        if match:
            valid_models.append((int(match.group(1)), filename))

    if not valid_models:
        return None

    valid_models.sort(key=lambda item: item[0], reverse=True)
    return valid_models[0][1]


def get_sb3_zip_models(
    files: list[str],
    model_glob: str,
    include_aux_zips: bool,
    all_zips: bool,
) -> list[str]:
    zip_files = []
    for filename in files:
        if not filename.endswith(".zip"):
            continue
        if not fnmatch.fnmatch(filename, model_glob):
            continue
        if not include_aux_zips and any(marker in filename for marker in AUX_ZIP_MARKERS):
            continue
        zip_files.append(filename)

    if all_zips:
        return sorted(zip_files)

    step_pattern = re.compile(r"^(?P<prefix>.+)_(?P<steps>\d+)_steps\.zip$")
    latest_by_prefix: dict[str, tuple[int, str]] = {}
    plain_zips = []
    for filename in zip_files:
        match = step_pattern.match(filename)
        if not match:
            plain_zips.append(filename)
            continue
        prefix = match.group("prefix")
        steps = int(match.group("steps"))
        current = latest_by_prefix.get(prefix)
        if current is None or steps > current[0]:
            latest_by_prefix[prefix] = (steps, filename)

    selected = plain_zips + [item[1] for item in latest_by_prefix.values()]
    return sorted(selected)


def extract_seed_from_path(path_str: str) -> int | None:
    match = re.search(r"seed_(\d+)", path_str)
    if match:
        return int(match.group(1))
    return None


def parse_step_from_model_dir(model_dir: str) -> str:
    """Extract train/unlearning budget from common experiment path layouts."""
    if not model_dir:
        return "N/A"

    parts = [part for part in model_dir.replace("\\", "/").split("/") if part]

    def find_explicit_step(start_index: int = 0) -> str:
        for part in parts[start_index:]:
            match = re.fullmatch(r"step_(\d+)", part)
            if match:
                return match.group(1)
        return "N/A"

    if "random_Rewarding_simple_fit" in parts:
        marker_index = parts.index("random_Rewarding_simple_fit")
        step = find_explicit_step(marker_index + 1)
        if step != "N/A":
            return step

    if "trajDeleter" in parts:
        marker_index = parts.index("trajDeleter")
        phase_steps: dict[str, int] = {}
        for part in parts[marker_index + 1:]:
            for phase, value in re.findall(r"phase([12])_(\d+)", part):
                phase_steps[phase] = int(value)

        if "1" in phase_steps and "2" in phase_steps:
            return str(phase_steps["1"] + phase_steps["2"])
        if "1" in phase_steps:
            return str(phase_steps["1"])
        if "2" in phase_steps:
            return str(phase_steps["2"])

    explicit_step = find_explicit_step()
    if explicit_step != "N/A":
        return explicit_step

    for part in parts:
        match = re.search(r"(?:^|_)steps?_(\d+)(?:_|$)", part)
        if match:
            return match.group(1)

    return "N/A"


def step_matches_model_dir(model_dir: str, target_steps: int | None) -> bool:
    if target_steps is None:
        return True
    return parse_step_from_model_dir(model_dir) == str(target_steps)


def generate_output_filename(root_path: str, backend: str, ratio=None) -> str:
    normalized_path = root_path.rstrip("/")
    base_name = normalized_path.replace("/", "_").replace(".", "")
    base_name = re.sub(r"[^\w\-_.]", "", base_name)
    ratio_suffix = ratio_filename_suffix(ratio)
    time_str = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    return f"run_eval_quadx_{backend}_{base_name}{ratio_suffix}_{time_str}.sh"


def optional_arg(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def append_common_eval_args(
    command: list[str],
    args: argparse.Namespace,
    seed: int,
    datasets: list[str],
    retained_ratios: list[float],
    env_id_list: list[str] | None = None,
) -> None:
    command.extend(
        [
            "--datasets",
            *datasets,
            "--retained-ratios",
            *(str(value) for value in retained_ratios),
            "--seed",
            str(seed),
            "--n-trials",
            str(args.n_trials),
            "--gpu",
            str(args.gpu),
            "--save-dir",
            args.save_dir,
            "--dataset-version",
            str(args.dataset_version),
            "--quadx-env-id",
            args.quadx_env_id,
            "--obstacle-obs-mode",
            args.obstacle_obs_mode,
            f"--raycast-elevation-angles={args.raycast_elevation_angles}",
            "--raycast-start-offset",
            str(args.raycast_start_offset),
            "--context-length",
            str(args.context_length),
        ]
    )
    if env_id_list:
        command.extend(["--env-id-list", *env_id_list])
    optional_arg(command, "--num-rays", args.num_rays)
    optional_arg(command, "--num-obstacle-features", args.num_obstacle_features)
    optional_arg(command, "--eval-max-episode-steps", args.eval_max_episode_steps)
    if not args.raycast_include_vertical:
        command.append("--no-raycast-vertical")
    if not args.deterministic:
        command.append("--stochastic")


def build_d3rl_command(
    *,
    args: argparse.Namespace,
    model_dir: str,
    model_filename: str,
    seed: int,
    datasets: list[str],
    retained_ratios: list[float],
    env_id_list: list[str] | None,
) -> str:
    command = [
        args.python_bin,
        args.evaluation_script,
        "--d3rl-or-sb3",
        "d3rl",
        "--model-dir",
        model_dir,
        "--model-filename",
        model_filename,
    ]
    append_common_eval_args(
        command, args, seed, datasets, retained_ratios, env_id_list
    )
    return shlex.join(command)


def build_sb3_command(
    *,
    args: argparse.Namespace,
    model_path: str,
    seed: int,
    datasets: list[str],
    retained_ratios: list[float],
    env_id_list: list[str] | None,
) -> str:
    command = [
        args.python_bin,
        args.evaluation_script,
        "--d3rl-or-sb3",
        "sb3",
        "--model-path",
        model_path,
        "--sb3-algo",
        args.sb3_algo,
        "--sb3-device",
        args.sb3_device,
    ]
    append_common_eval_args(
        command, args, seed, datasets, retained_ratios, env_id_list
    )
    return shlex.join(command)


def resolve_eval_metadata(
    args: argparse.Namespace, model_path: str
) -> tuple[list[str], list[float], list[str] | None, str]:
    manual_env_id_list = split_env_tokens(args.env_id_list) if args.env_id_list else None
    if manual_env_id_list:
        datasets = normalize_dataset_tokens(manual_env_id_list)
        retained_ratios, _ratio_tag, _ratio_source = parse_ratio_values_from_path(
            model_path, allowed_lengths={len(datasets)}
        )
        if not retained_ratios:
            retained_ratios = [1.0] * len(datasets)
    else:
        metadata = infer_quadx_path_metadata(
            model_path, fallback_datasets=DEFAULT_ENV_ID_LIST
        )
        datasets = metadata.datasets
        retained_ratios = metadata.retained_ratios

    build_eval_targets(datasets, retained_ratios, args.dataset_version)
    return datasets, retained_ratios, manual_env_id_list, ratio_tag_from_values(retained_ratios)


def write_serial(commands: list[tuple[str, str]], output_file) -> None:
    output_file.write("# Running in SERIAL mode.\n\n")
    output_file.write("set -e\n")
    output_file.write("set -o xtrace\n\n")

    for command, label in commands:
        output_file.write(command + "\n")
        output_file.write(f"echo 'Finished evaluating: {label}'\n")
        output_file.write("echo '---------------------------------------------'\n\n")

    output_file.write("echo 'All evaluations completed!'\n")


def write_parallel(commands: list[tuple[str, str]], max_jobs: int, output_file) -> None:
    output_file.write(f"# Running in PARALLEL mode (Batches of {max_jobs}).\n\n")
    output_file.write(
        f"echo 'Starting parallel evaluation ({len(commands)} total jobs, "
        f"{max_jobs} per batch)...'\n\n"
    )
    output_file.write("pids=()\n")
    output_file.write("fail_count=0\n\n")

    for index, (command, label) in enumerate(commands):
        batch_index = index // max_jobs + 1
        output_file.write(f"echo '[Batch {batch_index}] Starting: {label}'\n")
        output_file.write(f"({command}) &\n")
        output_file.write("pids+=($!)\n\n")

        is_batch_full = (index + 1) % max_jobs == 0
        is_last_command = (index + 1) == len(commands)
        if is_batch_full or is_last_command:
            output_file.write("# --- Waiting for batch to finish ---\n")
            output_file.write(f"echo 'Waiting for batch {batch_index} to finish...'\n")
            output_file.write("for pid in \"${pids[@]}\"; do\n")
            output_file.write("    wait \"$pid\"\n")
            output_file.write("    status=$?\n")
            output_file.write("    if [ \"$status\" -ne 0 ]; then\n")
            output_file.write(
                "        echo \"WARNING: Job $pid failed with exit code $status\"\n"
            )
            output_file.write("        fail_count=$((fail_count + 1))\n")
            output_file.write("    fi\n")
            output_file.write("done\n")
            output_file.write("echo 'Batch finished.'\n")
            output_file.write("pids=()\n\n")

    output_file.write("echo 'All evaluations completed!'\n")
    output_file.write("if [ \"$fail_count\" -ne 0 ]; then\n")
    output_file.write("    echo \"WARNING: $fail_count jobs failed!\"\n")
    output_file.write("    exit 1\n")
    output_file.write("else\n")
    output_file.write("    echo 'All jobs succeeded.'\n")
    output_file.write("fi\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate batch scripts for QuadX d3rlpy/SB3 evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--d3rl-or-sb3",
        default="d3rl",
        help="Backend selector: d3rl/d3rlpy searches model_*.pt; sb3 searches .zip files.",
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory to search for model checkpoints.",
    )
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument(
        "--env-id-list",
        nargs="+",
        default=None,
        help=(
            "Optional override/fallback QuadX eval families or dataset ids. "
            "When omitted, datasets are inferred from each model path."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fallback seed when no seed_X component is found in a model path.",
    )
    parser.add_argument(
        "--algo",
        default=None,
        type=str,
        help="Optional substring filter for model directories or filenames.",
    )
    parser.add_argument(
        "--step",
        "--steps",
        dest="steps",
        type=int,
        default=None,
        help="Optional: only use model paths whose parsed train/unlearning budget matches this value.",
    )
    parser.add_argument(
        "--ratio",
        nargs="+",
        default=None,
        help="Optional ratio directory filter, e.g. 0.9_1.0_1.0 or 0.9 1.0 1.0.",
    )
    parser.add_argument(
        "--sb3-model-glob",
        default="*.zip",
        help="Filename glob used only in SB3 mode, e.g. ori_model.zip.",
    )
    parser.add_argument(
        "--include-aux-zips",
        action="store_true",
        help="Include replay-buffer or VecNormalize zip files in SB3 mode.",
    )
    parser.add_argument(
        "--sb3-all-zips",
        action="store_true",
        help="Evaluate every matching SB3 zip instead of only plain zips plus latest *_steps.zip per prefix.",
    )
    parser.add_argument("--sb3-algo", default="SAC")
    parser.add_argument("--sb3-device", default="auto")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save-dir", "--save_dir", dest="save_dir", default="evaluation_results")
    parser.add_argument("--dataset-version", type=int, default=0)
    parser.add_argument(
        "--quadx-env-id",
        type=str,
        default="PyFlyt/QuadX-Obstacle-Waypoints-v0",
    )
    parser.add_argument(
        "--obstacle-obs-mode",
        choices=["nearest_k", "raycast"],
        default="raycast",
    )
    parser.add_argument("--num-rays", type=int, default=None)
    parser.add_argument("--num-obstacle-features", type=int, default=None)
    parser.add_argument("--raycast-elevation-angles", default="-30.0,0.0,30.0")
    parser.add_argument(
        "--no-raycast-vertical",
        dest="raycast_include_vertical",
        action="store_false",
    )
    parser.set_defaults(raycast_include_vertical=True)
    parser.add_argument("--raycast-start-offset", type=float, default=0.2)
    parser.add_argument("--context-length", type=int, default=2)
    parser.add_argument("--eval-max-episode-steps", type=int, default=None)
    parser.add_argument(
        "--evaluation-script",
        default=str(DEFAULT_EVALUATION_SCRIPT),
        help="Path to evaluation_quadx.py used in generated commands.",
    )
    parser.add_argument("--python-bin", default="python")
    policy_group = parser.add_mutually_exclusive_group()
    policy_group.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        default=argparse.SUPPRESS,
    )
    policy_group.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        default=argparse.SUPPRESS,
    )
    parser.set_defaults(deterministic=True)
    return parser


def normalize_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    try:
        args.backend = normalize_backend(args.d3rl_or_sb3)
    except ValueError as exc:
        parser.error(str(exc))
    if args.parallel <= 0:
        parser.error("--parallel must be positive.")
    if args.n_trials <= 0:
        parser.error("--n-trials must be positive.")
    if args.dataset_version < 0:
        parser.error("--dataset-version must be non-negative.")
    if args.context_length <= 0:
        parser.error("--context-length must be positive.")
    if args.num_rays is not None and args.num_rays <= 0:
        parser.error("--num-rays must be positive.")
    if args.num_obstacle_features is not None and args.num_obstacle_features <= 0:
        parser.error("--num-obstacle-features must be positive.")
    if args.raycast_start_offset < 0.0:
        parser.error("--raycast-start-offset must be non-negative.")
    if args.eval_max_episode_steps is not None and args.eval_max_episode_steps <= 0:
        parser.error("--eval-max-episode-steps must be positive when provided.")
    if args.steps is not None and args.steps < 0:
        parser.error("--step/--steps must be non-negative when provided.")


def seed_for_path(args: argparse.Namespace, path: str) -> int | None:
    model_seed = extract_seed_from_path(path)
    if model_seed is not None:
        return model_seed
    if args.seed is not None:
        print(f"[Info] Seed not found in '{path}'. Using fallback seed: {args.seed}")
        return args.seed
    print(f"[Warning] Skipping '{path}': no seed_X in path and no fallback --seed.")
    return None


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    normalize_args(parser, args)
    try:
        ratio_filter = normalize_ratio_filter(args.ratio)
    except ValueError as exc:
        parser.error(str(exc))

    search_root = args.root
    max_jobs = args.parallel
    manual_env_id_list = split_env_tokens(args.env_id_list) if args.env_id_list else None

    if not os.path.isdir(search_root):
        print(f"Error: Directory '{search_root}' does not exist.")
        sys.exit(1)

    output_script_name = generate_output_filename(search_root, args.backend, ratio_filter)
    commands: list[tuple[str, str]] = []
    print(f"Searching for {args.backend} models in: {search_root} ...")
    if manual_env_id_list:
        print(f"Eval environment override: {' '.join(manual_env_id_list)}")
    else:
        print("Eval environments will be inferred from each model path.")
    if args.steps is not None:
        print(f"Step filter enabled: steps={args.steps}")
    if ratio_filter is not None:
        print(f"Ratio filter enabled: ratio={ratio_filter}")

    for dirpath, _dirnames, filenames in os.walk(search_root):
        if ratio_filter is not None and not path_matches_ratio(dirpath, ratio_filter):
            continue
        if args.backend == "d3rl":
            if args.algo is not None and args.algo not in dirpath:
                continue
            best_model_file = get_max_step_d3rl_model(filenames)
            if best_model_file is None:
                continue
            if not step_matches_model_dir(dirpath, args.steps):
                continue
            model_seed = seed_for_path(args, dirpath)
            if model_seed is None:
                continue
            try:
                datasets, retained_ratios, env_override, ratio_tag = resolve_eval_metadata(
                    args, dirpath
                )
            except ValueError as exc:
                print(f"[Skip] Could not parse QuadX metadata for {dirpath}: {exc}")
                continue
            command = build_d3rl_command(
                args=args,
                model_dir=dirpath,
                model_filename=best_model_file,
                seed=model_seed,
                datasets=datasets,
                retained_ratios=retained_ratios,
                env_id_list=env_override,
            )
            label = f"{dirpath}/{best_model_file}"
            commands.append((command, label))
            print(
                f"[Added] Seed: {model_seed} | Ratios: {ratio_tag} | "
                f"Datasets: {' '.join(datasets)} | Model: {label}"
            )
        else:
            zip_models = get_sb3_zip_models(
                filenames,
                args.sb3_model_glob,
                args.include_aux_zips,
                args.sb3_all_zips,
            )
            for zip_model in zip_models:
                model_path = os.path.join(dirpath, zip_model)
                if args.algo is not None and args.algo not in model_path:
                    continue
                if not step_matches_model_dir(model_path, args.steps):
                    continue
                model_seed = seed_for_path(args, model_path)
                if model_seed is None:
                    continue
                try:
                    datasets, retained_ratios, env_override, ratio_tag = resolve_eval_metadata(
                        args, model_path
                    )
                except ValueError as exc:
                    print(f"[Skip] Could not parse QuadX metadata for {model_path}: {exc}")
                    continue
                command = build_sb3_command(
                    args=args,
                    model_path=model_path,
                    seed=model_seed,
                    datasets=datasets,
                    retained_ratios=retained_ratios,
                    env_id_list=env_override,
                )
                commands.append((command, model_path))
                print(
                    f"[Added] Seed: {model_seed} | Ratios: {ratio_tag} | "
                    f"Datasets: {' '.join(datasets)} | Model: {model_path}"
                )

    if not commands:
        print("No valid models found matching criteria.")
        return

    with open(output_script_name, "w", encoding="utf-8") as output_file:
        output_file.write("#!/bin/bash\n")
        output_file.write(
            f"# Auto-generated QuadX {args.backend} evaluation script for root: {search_root}\n"
        )
        output_file.write("# Seeds are automatically extracted from directory paths.\n\n")
        if max_jobs <= 1:
            write_serial(commands, output_file)
        else:
            write_parallel(commands, max_jobs, output_file)

    print(f"\nSuccessfully generated {len(commands)} commands.")
    print(f"Script saved to: {output_script_name}")
    if max_jobs > 1:
        print(f"Mode: Parallel (Batches of {max_jobs})")
    else:
        print("Mode: Serial")
    print(f"Run it with: bash {output_script_name}")


if __name__ == "__main__":
    main()
