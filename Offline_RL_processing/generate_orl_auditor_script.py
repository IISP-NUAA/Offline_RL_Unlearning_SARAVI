#!/usr/bin/env python3
import os
import re
import json
import hashlib
import argparse
import shlex
import time
from typing import Dict, List, Optional, Sequence, Tuple

from generator_ratio_filter import normalize_ratio_filter, path_matches_ratio, ratio_filename_suffix

from orl_auditor_core import canonical_algorithm_name, ratio_tag, shadow_policy_output_dir


DEFAULT_SHADOW_SEARCH_DIRS = ("/root/autodl-fs",)

# =============================================================================
# 1. Helper Functions copied/adapted from generate_critic_divergence_measurement_script.py
# =============================================================================

def generate_output_filename(root_dir, ratio=None):
    """Generate a readable shell name with an optional ratio marker."""
    normalized = os.path.normpath(root_dir).replace("\\", "/")
    path_parts = [part for part in normalized.split("/") if part not in {"", ".", ".."}]
    readable = "_".join(path_parts[-3:]) if path_parts else "suspects"
    readable = re.sub(r"[^A-Za-z0-9_-]+", "", readable)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    if len(readable) > 120:
        readable = readable[:120]
    ratio_suffix = ratio_filename_suffix(ratio)
    time_str = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    return f"run_orl_auditor_{readable}_{digest}{ratio_suffix}_{time_str}.sh"


def quote_arg(value):
    return shlex.quote(str(value))


def quote_join(values: Sequence[object]) -> str:
    return " ".join(quote_arg(value) for value in values)


def dataset_components_tag(datasets: Sequence[str]) -> str:
    """Return the directory token used for an ordered datasets configuration."""
    return "_".join(re.sub(r"[^A-Za-z0-9_.-]+", "-", str(item)) for item in datasets)


def dataset_components_aliases(datasets: Sequence[str]) -> set:
    """Return dataset-directory tokens used by the different training pipelines.

    Auditor output directories keep environment version suffixes such as
    ``-v0``. The policy/shadow training pipeline historically dropped those
    suffixes when constructing its joined dataset directory name. Both names
    describe the same ordered dataset configuration.
    """
    exact_tag = dataset_components_tag(datasets).lower()
    versionless_tag = "_".join(
        re.sub(r"-v\d+$", "", str(item), flags=re.IGNORECASE)
        for item in datasets
    ).lower()
    return {
        exact_tag,
        versionless_tag,
        f"datasets_{exact_tag}",
        f"datasets_{versionless_tag}",
    }


def path_matches_dataset_config(path_str: str, dataset: str, datasets: Sequence[str]) -> bool:
    """Require both the task name and the ordered joined dataset names in a model path."""
    parts = [part.lower() for part in path_str.replace("\\", "/").split("/") if part]
    dataset_name = str(dataset).lower()
    dataset_alias = f"audit_split_eval_{dataset_name}"
    # Existing policy checkpoints use the raw joined dataset directory name,
    # while auditor outputs use the explicit datasets_ prefix. Both
    # encode the same ordered configuration and must remain exact matches.
    # Policy checkpoints may additionally omit environment version suffixes
    # such as ``-v0`` from every component.
    component_aliases = dataset_components_aliases(datasets)

    has_dataset = dataset_name in parts or dataset_alias in parts
    has_components = any(part in component_aliases for part in parts)
    return has_dataset and has_components


def parse_step_from_model_dir(model_dir: str) -> str:
    """
    Extract the training/unlearning step from a model path.
    Special unlearning layouts are handled before broad steps_* matching so
    seed_0_steps_2000000 does not hide the actual unlearning step.
    """
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
        phase_steps = {}
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


def step_matches_model_dir(model_dir, target_steps):
    if target_steps is None:
        return True
    return parse_step_from_model_dir(model_dir) == str(target_steps)


def find_model_dirs(root_dir, target_steps=None):
    """Walk a directory and find all subdirectories containing model.pt or model_xxx.pt."""
    model_dirs = []
    pattern = re.compile(r"model_(\d+)\.pt$")

    if not os.path.isdir(root_dir):
        print(f"Error: Search directory does not exist: {root_dir}")
        return []

    for dirpath, _, filenames in os.walk(root_dir):
        if "model.pt" in filenames or any(pattern.match(filename) for filename in filenames):
            if not step_matches_model_dir(dirpath, target_steps):
                continue
            model_dirs.append(dirpath)

    return model_dirs


def extract_seed_from_path(path_str):
    """Extract seed number from path strings like seed_42, seed42, seed-42."""
    match = re.search(r"seed[-_]?(\d+)", path_str, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


RATIO_PREFIXES = ("retained_ratios", "retain_ratios", "learn_ratios", "ratios")
RATIO_TOKEN_PATTERN = re.compile(r"^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")


def normalize_ratio_token(token):
    normalized = token.strip().lower().replace("p", ".")
    if not RATIO_TOKEN_PATTERN.fullmatch(normalized):
        return None
    try:
        value = float(normalized)
    except ValueError:
        return None
    if value < 0.0 or value > 1.0:
        return None
    return normalized


def parse_ratio_segment(path_str, allowed_lengths=None):
    """
    Extract ratio values from path segments such as ratios_0.5_0.5 or
    learn_ratios_0.25_0.5_0.75. The last valid segment is the effective setting.
    """
    matched = (None, None)
    for path_part in re.split(r"[\\/]", path_str):
        lower_part = path_part.lower()
        for prefix in RATIO_PREFIXES:
            marker = prefix + "_"
            start = lower_part.find(marker)
            while start != -1:
                before_ok = start == 0 or not lower_part[start - 1].isalnum()
                if before_ok:
                    tokens = []
                    tail = path_part[start + len(marker):]
                    for raw_token in tail.split("_"):
                        token = normalize_ratio_token(raw_token)
                        if token is None:
                            break
                        tokens.append(token)
                    if tokens and (allowed_lengths is None or len(tokens) in allowed_lengths):
                        matched = (" ".join(tokens), "_".join(tokens))
                start = lower_part.find(marker, start + 1)
    return matched


def parse_ratios_pointmaze(path_str):
    return parse_ratio_segment(path_str, allowed_lengths={3})


def parse_ratios_mujoco(path_str):
    return parse_ratio_segment(path_str, allowed_lengths={1, 2})


def parse_ratios_quadx(path_str):
    return parse_ratio_segment(path_str)


def parse_ratios(path_str, dataset, target_root):
    dataset = (dataset or "").lower()
    target_root = (target_root or "").lower()

    if "quadx" in dataset or "quadx" in target_root:
        return parse_ratios_quadx(path_str)

    if "pointmaze" in dataset or "pointmaze" in target_root:
        return parse_ratios_pointmaze(path_str)

    if (
        "mujoco" in dataset
        or "halfcheetah" in target_root
        or "walker" in target_root
        or "hopper" in target_root
    ):
        return parse_ratios_mujoco(path_str)

    return parse_ratios_quadx(path_str)


def parse_method_name(path_str, fallback_algo):
    if fallback_algo is not None and fallback_algo.upper() not in path_str.upper():
        return None
    path_upper = path_str.upper()
    if "CQL" in path_upper:
        return "CQL"
    if "IQL" in path_upper:
        return "IQL"
    if "TD3" in path_upper and "BC" in path_upper:
        return "TD3PLUSBC"
    if "PLA" in path_upper:
        return "PLASP"
    if "BCQ" in path_upper:
        return "BCQ"
    if "BEAR" in path_upper:
        return "BEAR"
    # if "CRR" in path_upper:
    #     return "CRR"
    return fallback_algo


def _is_model_dir(path, filenames):
    pattern = re.compile(r"model_(\d+)\.pt$")
    return "model.pt" in filenames or any(pattern.match(filename) for filename in filenames)


def read_json_file(path: str) -> Optional[Dict[str, object]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def model_params_path(model_dir: str) -> Optional[str]:
    params_path = os.path.join(model_dir, "params.json")
    return params_path if os.path.isfile(params_path) else None


def algorithm_from_params(params_path: str, fallback: Optional[str] = None) -> Optional[str]:
    config = read_json_file(params_path)
    if not config:
        return fallback
    configured = config.get("algorithm") or config.get("type")
    if configured:
        return canonical_algorithm_name(str(configured))
    return fallback


def shadow_config_matches(
    config: Optional[Dict[str, object]],
    dataset: str,
    datasets: Sequence[str],
    ratio_values: Sequence[float],
    split_seed: int,
    training_seed: int,
    algorithm: str,
    training_steps: int,
    shuffle: bool,
    max_episodes: Optional[int],
) -> bool:
    if not config:
        return False
    try:
        configured_ratios = [float(value) for value in config.get("retained_ratios", [])]
    except (TypeError, ValueError):
        return False
    return (
        config.get("metric_type") == "ORL_Auditor_Shadow_Policy"
        and config.get("trained_only_on") == "D_f"
        and config.get("dataset_name") == dataset
        and list(config.get("dataset_components", [])) == list(datasets)
        and ratios_equal(configured_ratios, ratio_values)
        and int(config.get("split_seed", -1)) == split_seed
        and int(config.get("training_seed", -1)) == training_seed
        and canonical_algorithm_name(str(config.get("algorithm", "")))
        == canonical_algorithm_name(algorithm)
        and int(config.get("training_steps", -1)) == training_steps
        and bool(config.get("shuffle", True)) == shuffle
        and config.get("max_episodes") == max_episodes
        and bool(config.get("completed", False))
    )


def existing_shadow_policy(
    output_dir: str,
    dataset: str,
    datasets: Sequence[str],
    ratio_values: Sequence[float],
    split_seed: int,
    training_seed: int,
    algorithm: str,
    training_steps: int,
    shuffle: bool,
    max_episodes: Optional[int],
) -> bool:
    checkpoint = os.path.join(output_dir, f"model_{training_steps}.pt")
    params_path = os.path.join(output_dir, "params.json")
    config = read_json_file(os.path.join(output_dir, "shadow_config.json"))
    return (
        os.path.isfile(checkpoint)
        and os.path.isfile(params_path)
        and shadow_config_matches(
            config,
            dataset,
            datasets,
            ratio_values,
            split_seed,
            training_seed,
            algorithm,
            training_steps,
            shuffle,
            max_episodes,
        )
    )


def shadow_policy_candidate_dirs(
    root_dir: str,
    dataset: str,
    datasets: Sequence[str],
    ratio_values: Sequence[float],
    split_seed: int,
    training_seed: int,
    algorithm: str,
    training_steps: int,
    shuffle: bool,
) -> List[str]:
    """Return canonical and compact layouts for one shadow-policy cache entry.

    The normal cache includes ``stats_orl_auditor_<dataset>`` below the supplied
    root. Space-saving external storage may instead use the dataset directory
    itself as the root (for example ``/root/autodl-fs/datasets_...``), so both
    layouts are checked without relaxing the metadata validation.
    """
    split_mode = "shuffle" if shuffle else "no_shuffle"
    relative_path = os.path.join(
        f"datasets_{dataset_components_tag(datasets)}",
        f"ratios_{ratio_tag(ratio_values)}",
        "split_D_f",
        split_mode,
        f"split_seed_{split_seed}",
        f"algorithm_{canonical_algorithm_name(algorithm)}",
        f"training_seed_{training_seed}",
        f"steps_{training_steps}",
        "trained_shadow_policy",
    )

    candidates = [
        shadow_policy_output_dir(
            root_dir,
            dataset,
            datasets,
            ratio_values,
            split_seed,
            algorithm,
            training_seed,
            training_steps,
            shuffle=shuffle,
        ),
        os.path.join(os.path.normpath(root_dir), relative_path),
    ]

    unique_candidates = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def find_existing_shadow_policy(
    search_roots: Sequence[str],
    dataset: str,
    datasets: Sequence[str],
    ratio_values: Sequence[float],
    split_seed: int,
    training_seed: int,
    algorithm: str,
    training_steps: int,
    shuffle: bool,
    max_episodes: Optional[int],
) -> Optional[str]:
    """Find an exact completed shadow policy in any configured cache root."""
    for root_dir in search_roots:
        for candidate_dir in shadow_policy_candidate_dirs(
            root_dir,
            dataset,
            datasets,
            ratio_values,
            split_seed,
            training_seed,
            algorithm,
            training_steps,
            shuffle,
        ):
            if existing_shadow_policy(
                candidate_dir,
                dataset,
                datasets,
                ratio_values,
                split_seed,
                training_seed,
                algorithm,
                training_steps,
                shuffle,
                max_episodes,
            ):
                return candidate_dir
    return None


def latest_ckpt_in_dir(directory):
    if not os.path.isdir(directory):
        return None
    ckpts = []
    for filename in os.listdir(directory):
        match = re.fullmatch(r"ckpt_(\d+)\.pt", filename)
        if match:
            ckpts.append((int(match.group(1)), os.path.join(directory, filename)))
    if not ckpts:
        return None
    return max(ckpts, key=lambda item: item[0])[1]


def ratios_equal(left, right):
    try:
        left_values = [float(value) for value in left]
        right_values = [float(value) for value in right]
    except (TypeError, ValueError):
        return False
    if len(left_values) != len(right_values):
        return False
    return all(abs(a - b) < 1e-12 for a, b in zip(left_values, right_values))


def critic_config_matches(
    config,
    dataset,
    datasets,
    ratio_values,
    seed,
    audit_split,
    shuffle,
    max_episodes,
    max_transitions,
):
    return (
        config.get("dataset_name") == dataset
        and list(config.get("dataset_components", [])) == list(datasets)
        and ratios_equal(config.get("retained_ratios", []), ratio_values)
        and config.get("seed") == seed
        and config.get("audit_split") == audit_split
        and bool(config.get("shuffle", True)) == shuffle
        and config.get("max_episodes") == max_episodes
        and config.get("max_transitions") == max_transitions
    )


def find_existing_auditor_critic(
    search_roots,
    dataset,
    datasets,
    ratio_values,
    ratio_tag,
    seed,
    audit_split,
    shuffle,
    max_episodes,
    max_transitions,
):
    """Find a critic with an exactly matching D_f reconstruction configuration.

    Older auditor runs may have written ``ckpt_*.pt`` into the canonical
    dataset/ratio/split/seed directory without writing ``critic_config.json``.
    Such checkpoints cannot be validated from metadata, so they are accepted
    only for the unbounded, shuffled D_f configuration.  This is the default
    configuration used by the existing QuadX critic batch and is deliberately
    not used when an episode or transition cap is requested.
    """
    candidates = []
    target_components = list(datasets)
    datasets_tag = dataset_components_tag(target_components)
    split_mode = "shuffle" if shuffle else "no_shuffle"

    def add_canonical_candidate(canonical_dir):
        config_path = os.path.join(canonical_dir, "critic_config.json")
        if not os.path.isfile(config_path):
            legacy_checkpoint = latest_ckpt_in_dir(canonical_dir)
            if (
                legacy_checkpoint
                and shuffle
                and max_episodes is None
                and max_transitions is None
            ):
                candidates.append(legacy_checkpoint)
            return
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        if critic_config_matches(
            config,
            dataset,
            target_components,
            ratio_values,
            seed,
            audit_split,
            shuffle,
            max_episodes,
            max_transitions,
        ):
            canonical_ckpt = latest_ckpt_in_dir(canonical_dir)
            if canonical_ckpt:
                candidates.append(canonical_ckpt)

    for search_root in search_roots:
        dataset_root = os.path.normpath(search_root)
        expected_name = f"stats_orl_auditor_{dataset}"
        if os.path.basename(dataset_root) != expected_name:
            dataset_root = os.path.join(dataset_root, expected_name)

        canonical_dirs = [
            os.path.join(
                dataset_root,
                f"datasets_{datasets_tag}",
                f"ratios_{ratio_tag}",
                f"split_{audit_split}",
                split_mode,
                f"seed_{seed}",
                "trained_critic_model",
            ),
            # Compatibility with exact configs written before split mode was in the path.
            os.path.join(
                dataset_root,
                f"datasets_{datasets_tag}",
                f"ratios_{ratio_tag}",
                f"split_{audit_split}",
                f"seed_{seed}",
                "trained_critic_model",
            ),
        ]
        for canonical_dir in canonical_dirs:
            add_canonical_candidate(canonical_dir)

        if not os.path.isdir(search_root):
            continue

        for dirpath, _, filenames in os.walk(search_root):
            if "critic_config.json" not in filenames:
                continue
            config_path = os.path.join(dirpath, "critic_config.json")
            try:
                with open(config_path, "r", encoding="utf-8") as handle:
                    config = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue

            if not critic_config_matches(
                config,
                dataset,
                target_components,
                ratio_values,
                seed,
                audit_split,
                shuffle,
                max_episodes,
                max_transitions,
            ):
                continue

            configured_ckpt = config.get("critic_checkpoint")
            if configured_ckpt:
                configured_candidates = [configured_ckpt]
                if not os.path.isabs(configured_ckpt):
                    configured_candidates.append(
                        os.path.join(os.path.dirname(config_path), configured_ckpt)
                    )
                for configured_candidate in configured_candidates:
                    if os.path.exists(configured_candidate):
                        candidates.append(configured_candidate)
                        break

            fallback_ckpt = latest_ckpt_in_dir(dirpath)
            if fallback_ckpt:
                candidates.append(fallback_ckpt)

    candidates = sorted(set(candidates), key=lambda path: os.path.getmtime(path), reverse=True)
    return candidates[0] if candidates else None


# =============================================================================
# 2. Command Builders
# =============================================================================

def build_train_command(args, ratios_str, seed, gpu_expr):
    parts = [
        "python", "train_orl_auditor_critic.py",
        "--dataset", args.dataset,
        "--datasets", *args.datasets,
        "--retained-ratios", *ratios_str.split(),
        "--audit-split", args.audit_split,
        "--seed", seed,
        "--gpu", gpu_expr,
        "--train-epochs", args.critic_train_epochs,
        "--batch-size", args.critic_train_batch_size,
        "--hidden-size", args.critic_hidden_size,
        "--gamma", args.critic_gamma,
        "--learning-rate", args.critic_learning_rate,
        "--train-ratio", args.critic_train_ratio,
        "--save-interval", args.critic_save_interval,
        "--num-workers", args.num_workers,
        "--save-dir", args.critic_save_dir,
    ]
    if args.no_shuffle:
        parts.append("--no-shuffle")
    if args.save_transitions:
        parts.append("--save-transitions")
    if args.max_critic_episodes is not None:
        parts.extend(["--max-episodes", args.max_critic_episodes])
    if args.max_critic_transitions is not None:
        parts.extend(["--max-transitions", args.max_critic_transitions])
    return quote_join(parts)



def build_shadow_train_command(
    args,
    ratios_str,
    split_seed,
    training_seed,
    algorithm,
    base_params,
):
    parts = [
        "python", args.shadow_training_script,
        "--dataset", args.dataset,
        "--datasets", *args.datasets,
        "--retained-ratios", *ratios_str.split(),
        "--split-seed", split_seed,
        "--training-seed", training_seed,
        "--algorithm", algorithm,
        "--base-params", base_params,
        "--training-steps", args.shadow_training_steps,
        "--n-steps-per-epoch", args.shadow_n_steps_per_epoch,
        "--gpu", args.gpu,
        "--save-dir", args.shadow_dir,
    ]
    if args.no_shuffle:
        parts.append("--no-shuffle")
    if args.force_retrain_shadows:
        parts.append("--force")
    if args.max_shadow_episodes is not None:
        parts.extend(["--max-episodes", args.max_shadow_episodes])
    return quote_join(parts)


def build_eval_command(args, critic_var, shadow_dirs, suspect_dir, suspect_name, ratios_str, seed):
    parts = [
        "python", "evaluate_orl_auditor.py",
        "--critic-checkpoint", f"${{{critic_var}}}",
        "--shadow-dirs", *shadow_dirs,
        "--suspect-dirs", suspect_dir,
        "--suspect-names", suspect_name,
        "--suspect-membership", args.suspect_membership,
        "--suspect-roles", args.suspect_role,
        "--dataset", args.dataset,
        "--datasets", *args.datasets,
        "--retained-ratios", *ratios_str.split(),
        "--audit-split", args.audit_split,
        "--seed", seed,
        "--gpu", "{GPU_PLACEHOLDER}",
        "--batch-size", args.eval_batch_size,
        "--trajectory-size", args.trajectory_size,
        "--significance-level", args.significance_level,
        "--save-dir", args.audit_save_dir,
    ]
    if args.no_shuffle:
        parts.append("--no-shuffle")
    if args.num_audited_episodes is not None:
        parts.extend(["--num-audited-episodes", args.num_audited_episodes])
    if args.shadow_require_model is not None:
        parts.extend(["--shadow-require-model", args.shadow_require_model])
    if args.suspect_require_model is not None:
        parts.extend(["--suspect-require-model", args.suspect_require_model])
    return quote_join(parts).replace("'${" + critic_var + "}'", "\"${" + critic_var + "}\"")


# =============================================================================
# 3. Main Script
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate a complete D_f-based ORL-Auditor critic, shadow-policy, and audit workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--suspect-dir", type=str, required=True,
                        help="Root directory to search for suspect models to audit.")
    parser.add_argument("--shadow-dir", type=str, required=True,
                        help="Cache root for same-algorithm policies trained exclusively on the fixed D_f split.")
    parser.add_argument(
        "--shadow-search-dirs",
        type=str,
        nargs="+",
        default=list(DEFAULT_SHADOW_SEARCH_DIRS),
        help="Additional roots searched for completed shadow policies before training missing entries.",
    )
    parser.add_argument("--algo", type=str, default=None,
                        help="Fallback algorithm if it cannot be parsed from the suspect path.")

    parser.add_argument("--dataset", type=str, default="pointmaze",
                        help="Task name, e.g. pointmaze, antmaze, halfcheetah.")
    parser.add_argument("--datasets", nargs="+", default=["large-dense-v2", "umaze-dense-v2", "medium-dense-v2"],
                        help="Sub-dataset names passed to train/evaluate scripts.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional forced seed; otherwise parsed from suspect paths.")
    parser.add_argument("--steps", type=int, default=None,
                        help="Optional: only use suspect model directories whose parsed steps match this value.")
    parser.add_argument("--ratio", nargs="+", default=None,
                        help="Optional ratio directory filter, e.g. 0.9_1.0_1.0 or 0.9 1.0 1.0.")
    parser.add_argument("--audit-split", choices=["D_f"], default="D_f",
                        help="ORL-Auditor critic, shadow policies, and probes are fixed to D_f.")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Disable shuffling in load_merged_dataset split.")

    parser.add_argument("--critic-search-dirs", nargs="+", default=["stats_results_orl_auditor"],
                        help="Roots searched for existing trained auditor critics.")
    parser.add_argument("--critic-save-dir", type=str, default="stats_results_orl_auditor",
                        help="Root passed to train_orl_auditor_critic.py for missing critics.")
    parser.add_argument("--audit-save-dir", type=str, default="stats_results_orl_auditor",
                        help="Root passed to evaluate_orl_auditor.py for audit outputs.")

    parser.add_argument("--num-shadow-student", type=int, default=15,
                        help="Number of independently initialized D_f-only shadow policies (paper default: 15).")
    parser.add_argument("--min-shadow-student", type=int, default=2,
                        help="Minimum supported shadow count for the Grubbs test.")
    parser.add_argument("--shadow-training-seeds", type=int, nargs="+", default=None,
                        help="Independent policy-training seeds; defaults to 0..num-shadow-student-1.")
    parser.add_argument("--shadow-training-steps", type=int, default=100000,
                        help="Offline updates for every D_f-only shadow policy.")
    parser.add_argument("--shadow-n-steps-per-epoch", type=int, default=10000,
                        help="d3rlpy logging/checkpoint epoch length during shadow training.")
    parser.add_argument("--shadow-training-script", default="train_orl_auditor_shadow_policy.py",
                        help="D_f-only shadow-policy training entry point.")
    parser.add_argument("--force-retrain-shadows", action="store_true",
                        help="Retrain exact shadow cache entries instead of reusing them.")
    parser.add_argument("--max-shadow-episodes", type=int, default=None,
                        help="Optional debug cap for D_f episodes used by each shadow.")
    parser.add_argument("--shadow-any-seed", action="store_true",
                        help="Deprecated and ignored; split seed is fixed while training seeds differ.")
    parser.add_argument("--suspect-membership", choices=["member", "nonmember", "unknown"], default="unknown",
                        help="Optional label used only for auditor-quality TPR/TNR evaluation.")
    parser.add_argument(
        "--suspect-role",
        choices=["original", "unlearned", "retrained", "unknown"],
        default="unknown",
        help="Semantic role reported for generated target audit jobs.",
    )
    parser.add_argument("--ignore-no-shuffle", action="store_true",
                        help="Deprecated; no_shuffle suspects are skipped unless --no-shuffle is selected.")
    parser.add_argument("--include-no-shuffle-suspects", action="store_true",
                        help="Allow suspect paths containing no_shuffle without selecting --no-shuffle.")

    parser.add_argument("--gpu", type=int, default=0,
                        help="Base GPU ID to use.")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel audit jobs to run after critics are prepared.")
    parser.add_argument("--no-gpu-round-robin", action="store_true",
                        help="If set, all parallel jobs use the same --gpu ID.")

    parser.add_argument("--critic-train-epochs", type=int, default=200,
                        help="Epochs for missing auditor critic training.")
    parser.add_argument("--critic-train-batch-size", type=int, default=4096,
                        help="Training batch size for missing auditor critics.")
    parser.add_argument("--critic-hidden-size", type=int, default=1024,
                        help="Hidden size for missing auditor critics.")
    parser.add_argument("--critic-gamma", type=float, default=0.99,
                        help="Discount factor for missing auditor critics.")
    parser.add_argument("--critic-learning-rate", type=float, default=1e-3,
                        help="Learning rate for missing auditor critics.")
    parser.add_argument("--critic-train-ratio", type=float, default=0.7,
                        help="Train split ratio for missing auditor critics.")
    parser.add_argument("--critic-save-interval", type=int, default=100,
                        help="Save interval for missing auditor critic checkpoints.")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader workers for missing auditor critic training.")
    parser.add_argument("--save-transitions", action="store_true",
                        help="Save transition_form.npy when training missing critics.")
    parser.add_argument("--max-critic-episodes", type=int, default=None,
                        help="Optional debug cap for critic-training source episodes.")
    parser.add_argument("--max-critic-transitions", type=int, default=None,
                        help="Optional debug cap for critic-training transitions.")

    parser.add_argument("--eval-batch-size", type=int, default=512,
                        help="Batch size for evaluate_orl_auditor.py value estimation.")
    parser.add_argument("--num-audited-episodes", type=int, default=None,
                        help="Optional cap per suspect; by default audit every D_f trajectory.")
    parser.add_argument("--trajectory-size", type=float, default=1.0,
                        help="Fraction of each trajectory used for audit.")
    parser.add_argument("--significance-level", type=float, default=0.01,
                        help="Grubbs significance level.")
    parser.add_argument("--shadow-require-model", type=int, default=None,
                        help="Force shadow checkpoint model_<N>.pt.")
    parser.add_argument("--suspect-require-model", type=int, default=None,
                        help="Force suspect checkpoint model_<N>.pt.")

    args = parser.parse_args()
    try:
        ratio_filter = normalize_ratio_filter(args.ratio)
    except ValueError as exc:
        parser.error(str(exc))

    if args.num_shadow_student < 2 or args.min_shadow_student < 2:
        raise ValueError("ORL-Auditor Grubbs auditing requires at least two shadow models.")
    if args.num_shadow_student < args.min_shadow_student:
        raise ValueError("--num-shadow-student must be >= --min-shadow-student.")
    if args.shadow_training_steps <= 0 or args.shadow_n_steps_per_epoch <= 0:
        raise ValueError("Shadow training steps and steps per epoch must be positive.")
    if (
        args.shadow_require_model is not None
        and args.shadow_require_model != args.shadow_training_steps
    ):
        raise ValueError(
            "--shadow-require-model must equal --shadow-training-steps for generated shadows."
        )

    requested_shadow_seeds = (
        args.shadow_training_seeds
        if args.shadow_training_seeds is not None
        else list(range(args.num_shadow_student))
    )
    shadow_training_seeds = []
    for training_seed in requested_shadow_seeds:
        if training_seed not in shadow_training_seeds:
            shadow_training_seeds.append(training_seed)
    if len(shadow_training_seeds) < args.num_shadow_student:
        raise ValueError(
            "Provide at least --num-shadow-student distinct --shadow-training-seeds."
        )
    shadow_training_seeds = shadow_training_seeds[:args.num_shadow_student]

    output_script_name = generate_output_filename(args.suspect_dir, ratio_filter)
    newline = os.linesep
    search_roots = []
    for root in list(args.critic_search_dirs) + [args.critic_save_dir]:
        if root not in search_roots:
            search_roots.append(root)

    shadow_search_roots = []
    for root in [args.shadow_dir] + list(args.shadow_search_dirs):
        if root not in shadow_search_roots:
            shadow_search_roots.append(root)

    print(f"Searching for suspect models in: {args.suspect_dir} ...")
    suspect_candidates = find_model_dirs(args.suspect_dir, target_steps=args.steps)
    print(f"Found {len(suspect_candidates)} candidate suspect model directories.")
    if args.steps is not None:
        print(f"Step filter enabled: steps={args.steps}")
    if ratio_filter is not None:
        print(f"Ratio filter enabled: ratio={ratio_filter}")

    commands = []
    critic_specs: Dict[Tuple[object, ...], Dict[str, object]] = {}
    shadow_specs: Dict[Tuple[object, ...], Dict[str, object]] = {}

    for suspect_dir in suspect_candidates:
        if ratio_filter is not None and not path_matches_ratio(suspect_dir, ratio_filter):
            continue
        if (
            "no_shuffle" in suspect_dir
            and not args.no_shuffle
            and not args.include_no_shuffle_suspects
        ):
            continue

        current_seed = args.seed if args.seed is not None else extract_seed_from_path(suspect_dir)
        if current_seed is None:
            print(f"[Skip] Could not extract seed from path: {suspect_dir}")
            continue

        ratios_str, connected_ratios_str = parse_ratios(suspect_dir, args.dataset, args.suspect_dir)
        if not ratios_str:
            print(f"[Skip] Could not parse ratios from path: {suspect_dir}")
            continue
        ratio_values = ratios_str.split()

        # The auditor critic is independent of the student algorithm.  Register
        # this key before shadow matching so a missing critic is still trained
        # even when the corresponding audit job has no usable shadow pool.
        critic_key = (
            args.dataset,
            tuple(args.datasets),
            connected_ratios_str,
            current_seed,
            args.audit_split,
            not args.no_shuffle,
            args.max_critic_episodes,
            args.max_critic_transitions,
        )
        if critic_key not in critic_specs:
            existing_critic = find_existing_auditor_critic(
                search_roots,
                dataset=args.dataset,
                datasets=args.datasets,
                ratio_values=ratio_values,
                ratio_tag=connected_ratios_str,
                seed=current_seed,
                audit_split=args.audit_split,
                shuffle=not args.no_shuffle,
                max_episodes=args.max_critic_episodes,
                max_transitions=args.max_critic_transitions,
            )
            critic_var = f"CRITIC_CKPT_{len(critic_specs)}"
            critic_specs[critic_key] = {
                "var": critic_var,
                "ratios_str": ratios_str,
                "ratio_tag": connected_ratios_str,
                "ratio_values": ratio_values,
                "seed": current_seed,
                "existing": existing_critic,
                "train_cmd_gpu0": build_train_command(args, ratios_str, current_seed, args.gpu),
            }
            if existing_critic:
                print(
                    f"[Critic] Reusing existing critic for dataset={args.dataset}, "
                    f"datasets={args.datasets}, ratio={connected_ratios_str}, seed={current_seed}: "
                    f"{existing_critic}"
                )
            else:
                print(
                    f"[Critic] Will train missing critic for dataset={args.dataset}, "
                    f"datasets={args.datasets}, ratio={connected_ratios_str}, seed={current_seed}"
                )

        base_params = model_params_path(suspect_dir)
        if base_params is None:
            print(f"[Skip] params.json not found in suspect model directory: {suspect_dir}")
            continue
        path_method = parse_method_name(suspect_dir, args.algo)
        method_name = algorithm_from_params(base_params, path_method)
        if method_name is None:
            print(f"[Skip] Could not identify algorithm from {base_params}")
            continue
        method_name = canonical_algorithm_name(method_name)
        if args.algo and method_name != canonical_algorithm_name(args.algo):
            continue

        shadow_dirs = []
        existing_shadow_count = 0
        for training_seed in shadow_training_seeds:
            expected_dir = shadow_policy_output_dir(
                args.shadow_dir,
                args.dataset,
                args.datasets,
                [float(value) for value in ratio_values],
                current_seed,
                method_name,
                training_seed,
                args.shadow_training_steps,
                shuffle=not args.no_shuffle,
            )
            existing_dir = find_existing_shadow_policy(
                shadow_search_roots,
                args.dataset,
                args.datasets,
                [float(value) for value in ratio_values],
                current_seed,
                training_seed,
                method_name,
                args.shadow_training_steps,
                shuffle=not args.no_shuffle,
                max_episodes=args.max_shadow_episodes,
            )
            is_existing = existing_dir is not None and not args.force_retrain_shadows
            selected_dir = existing_dir if is_existing else expected_dir
            shadow_dirs.append(selected_dir)
            if is_existing:
                existing_shadow_count += 1

            shadow_key = (
                args.dataset,
                tuple(args.datasets),
                connected_ratios_str,
                current_seed,
                method_name,
                training_seed,
                args.shadow_training_steps,
                not args.no_shuffle,
                args.max_shadow_episodes,
            )
            if shadow_key not in shadow_specs:
                shadow_specs[shadow_key] = {
                    "dataset": args.dataset,
                    "datasets": list(args.datasets),
                    "ratio": connected_ratios_str,
                    "split_seed": current_seed,
                    "training_seed": training_seed,
                    "algorithm": method_name,
                    "training_steps": args.shadow_training_steps,
                    "expected_dir": expected_dir,
                    "selected_dir": selected_dir,
                    "existing": is_existing,
                    "train_cmd": build_shadow_train_command(
                        args,
                        ratios_str,
                        current_seed,
                        training_seed,
                        method_name,
                        base_params,
                    ),
                }

        print(
            f"[Shadows] {existing_shadow_count}/{len(shadow_dirs)} exact D_f shadows "
            f"already available for split_seed={current_seed}, algorithm={method_name}, "
            f"ratio={connected_ratios_str}; missing entries will be trained."
        )

        critic_var = critic_specs[critic_key]["var"]
        suspect_name = os.path.basename(os.path.normpath(suspect_dir)) or f"suspect_seed_{current_seed}"
        eval_cmd = build_eval_command(
            args,
            critic_var=critic_var,
            shadow_dirs=shadow_dirs,
            suspect_dir=suspect_dir,
            suspect_name=suspect_name,
            ratios_str=ratios_str,
            seed=current_seed,
        )
        commands.append({
            "cmd": eval_cmd,
            "seed": current_seed,
            "algo": method_name,
            "ratio": connected_ratios_str,
            "suspect": suspect_dir,
            "shadow_count": len(shadow_dirs),
            "shadow_seed_policy": "fixed_split_seed_independent_training_seeds",
        })
        print(
            f"[Match] Seed {current_seed}, Algo {method_name}, Ratio {connected_ratios_str}:\n"
            f"   Suspect: ...{suspect_dir[-60:]}\n"
            f"   Shadows: {len(shadow_dirs)} (trained only on D_f)"
        )

    if not commands and not critic_specs:
        print("No valid commands or critic keys generated.")
        return
    if not commands:
        print("No valid audit commands generated; the output script will only prepare registered prerequisites.")

    print(
        f"\nGenerating '{output_script_name}' with {len(commands)} audit commands, "
        f"{len(critic_specs)} critic keys, and {len(shadow_specs)} shadow-policy keys..."
    )

    with open(output_script_name, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated ORL-Auditor execution script\n")
        f.write(f"# Suspect Root: {args.suspect_dir}\n")
        f.write(f"# Shadow Root:  {args.shadow_dir}\n")
        f.write(f"# Shadow Search Roots:  {shadow_search_roots}\n")
        f.write(f"# Audit Split:  {args.audit_split}\n\n")
        f.write("set -e\n\n")

        f.write("CRITIC_SEARCH_DIRS=(")
        f.write(" ".join(quote_arg(root) for root in search_roots))
        f.write(")\n\n")

        f.write(f"DATASETS_TAG={quote_arg(dataset_components_tag(args.datasets))}\n\n")

        f.write("critic_config_matches() {\n")
        f.write("    local config_path=\"$1\"\n")
        f.write("    local dataset=\"$2\"\n")
        f.write("    local ratios_tag=\"$3\"\n")
        f.write("    local split=\"$4\"\n")
        f.write("    local seed=\"$5\"\n")
        f.write("    local split_mode=\"$6\"\n")
        f.write("    local max_episodes=\"$7\"\n")
        f.write("    local max_transitions=\"$8\"\n")
        f.write("    shift 8\n")
        f.write("    python - \"$config_path\" \"$dataset\" \"$ratios_tag\" \"$split\" \"$seed\" \"$split_mode\" \"$max_episodes\" \"$max_transitions\" \"$@\" <<'PY'\n")
        f.write("import json\nimport hashlib\n")
        f.write("import sys\n")
        f.write("config_path, dataset, ratios_tag, split, seed, split_mode, max_episodes, max_transitions = sys.argv[1:9]\n")
        f.write("components = sys.argv[9:]\n")
        f.write("expected_max_episodes = None if max_episodes == 'none' else int(max_episodes)\n")
        f.write("expected_max_transitions = None if max_transitions == 'none' else int(max_transitions)\n")
        f.write("try:\n")
        f.write("    with open(config_path, 'r', encoding='utf-8') as handle:\n")
        f.write("        config = json.load(handle)\n")
        f.write("    expected_ratios = [float(value) for value in ratios_tag.split('_')]\n")
        f.write("    actual_ratios = [float(value) for value in config.get('retained_ratios', [])]\n")
        f.write("except (OSError, ValueError, TypeError, json.JSONDecodeError):\n")
        f.write("    raise SystemExit(1)\n")
        f.write("matches = (\n")
        f.write("    config.get('dataset_name') == dataset\n")
        f.write("    and config.get('dataset_components', []) == components\n")
        f.write("    and len(actual_ratios) == len(expected_ratios)\n")
        f.write("    and all(abs(left - right) < 1e-12 for left, right in zip(actual_ratios, expected_ratios))\n")
        f.write("    and str(config.get('seed')) == seed\n")
        f.write("    and config.get('audit_split') == split\n")
        f.write("    and bool(config.get('shuffle', True)) == (split_mode == 'shuffle')\n")
        f.write("    and config.get('max_episodes') == expected_max_episodes\n")
        f.write("    and config.get('max_transitions') == expected_max_transitions\n")
        f.write(")\n")
        f.write("raise SystemExit(0 if matches else 1)\n")
        f.write("PY\n")
        f.write("}\n\n")

        f.write("find_latest_critic() {\n")
        f.write("    local dataset=\"$1\"\n")
        f.write("    local ratios_tag=\"$2\"\n")
        f.write("    local split=\"$3\"\n")
        f.write("    local seed=\"$4\"\n")
        f.write("    local split_mode=\"$5\"\n")
        f.write("    local max_episodes=\"$6\"\n")
        f.write("    local max_transitions=\"$7\"\n")
        f.write("    shift 7\n")
        f.write("    local components=(\"$@\")\n")
        f.write("    local search_dir dataset_root canonical config_path latest\n")
        f.write("    for search_dir in \"${CRITIC_SEARCH_DIRS[@]}\"; do\n")
        f.write("        dataset_root=\"${search_dir%/}\"\n")
        f.write("        if [ \"$(basename \"$dataset_root\")\" = \"stats_orl_auditor_${dataset}\" ]; then\n")
        f.write("            dataset_root=\"$dataset_root\"\n")
        f.write("        else\n")
        f.write("            dataset_root=\"$dataset_root/stats_orl_auditor_${dataset}\"\n")
        f.write("        fi\n")
        f.write("        for canonical in \"$dataset_root/datasets_${DATASETS_TAG}/ratios_${ratios_tag}/split_${split}/${split_mode}/seed_${seed}/trained_critic_model\" \"$dataset_root/datasets_${DATASETS_TAG}/ratios_${ratios_tag}/split_${split}/seed_${seed}/trained_critic_model\"; do\n")
        f.write("            config_path=\"$canonical/critic_config.json\"\n")
        f.write("            if compgen -G \"$canonical/ckpt_*.pt\" > /dev/null; then\n")
        f.write("                if [ -f \"$config_path\" ] && critic_config_matches \"$config_path\" \"$dataset\" \"$ratios_tag\" \"$split\" \"$seed\" \"$split_mode\" \"$max_episodes\" \"$max_transitions\" \"${components[@]}\"; then\n")
        f.write("                    latest=$(ls -1t \"$canonical\"/ckpt_*.pt | head -n 1)\n")
        f.write("                    echo \"$latest\"\n")
        f.write("                    return 0\n")
        f.write("                elif [ ! -e \"$config_path\" ] && [ \"$max_episodes\" = \"none\" ] && [ \"$max_transitions\" = \"none\" ] && [ \"$split_mode\" = \"shuffle\" ]; then\n")
        f.write("                    latest=$(ls -1t \"$canonical\"/ckpt_*.pt | head -n 1)\n")
        f.write("                    echo \"$latest\"\n")
        f.write("                    return 0\n")
        f.write("                fi\n")
        f.write("            fi\n")
        f.write("        done\n")
        f.write("    done\n")
        f.write("    return 1\n")
        f.write("}\n\n")

        f.write("# -----------------------------------------------------------------------------\n")
        f.write("# Prepare one auditor critic per unique dataset/datasets/ratio/seed/split key.\n")
        f.write("# Critics found during generation are reused directly; missing critics are trained here.\n")
        f.write("# -----------------------------------------------------------------------------\n\n")

        components_args = quote_join(args.datasets)
        for _, spec in critic_specs.items():
            var = spec["var"]
            ratio_tag = spec["ratio_tag"]
            seed = spec["seed"]
            train_cmd = spec["train_cmd_gpu0"]
            split_mode = "shuffle" if not args.no_shuffle else "no_shuffle"
            max_episodes_tag = (
                "none" if args.max_critic_episodes is None else str(args.max_critic_episodes)
            )
            max_transitions_tag = (
                "none" if args.max_critic_transitions is None else str(args.max_critic_transitions)
            )
            find_cmd = (
                f"find_latest_critic {quote_arg(args.dataset)} {quote_arg(ratio_tag)} "
                f"{quote_arg(args.audit_split)} {quote_arg(seed)} "
                f"{quote_arg(split_mode)} {quote_arg(max_episodes_tag)} "
                f"{quote_arg(max_transitions_tag)} {components_args}"
            )
            f.write(
                f"# Critic key: dataset={args.dataset}, datasets={args.datasets}, "
                f"ratios={ratio_tag}, split={args.audit_split}/{split_mode}, seed={seed}\n"
            )
            existing_critic = spec.get("existing")
            if existing_critic:
                f.write(f"{var}={quote_arg(existing_critic)}\n")
                f.write(f"if [ ! -f \"${{{var}}}\" ]; then\n")
                f.write(
                    f"    echo 'ERROR: detected auditor critic is no longer available: "
                    f"{existing_critic}'\n"
                )
                f.write("    exit 1\n")
                f.write("fi\n")
                f.write(f"echo 'Reusing existing auditor critic: '${{{var}}}\n\n")
            else:
                f.write(f"{var}=\"$({find_cmd} || true)\"\n")
                f.write(f"if [ -z \"${{{var}}}\" ]; then\n")
                f.write(
                    f"    echo 'No exactly matching auditor critic found for "
                    f"dataset={args.dataset}, datasets={args.datasets}, ratios={ratio_tag}, "
                    f"seed={seed}; training now.'\n"
                )
                f.write(f"    {train_cmd}\n")
                f.write(f"    {var}=\"$({find_cmd} || true)\"\n")
                f.write("fi\n")
                f.write(f"if [ -z \"${{{var}}}\" ]; then\n")
                f.write(
                    f"    echo 'ERROR: failed to prepare auditor critic for "
                    f"dataset={args.dataset}, datasets={args.datasets}, ratios={ratio_tag}, seed={seed}.'\n"
                )
                f.write("    exit 1\n")
                f.write("fi\n\n")

        f.write("# -----------------------------------------------------------------------------\n")
        f.write("# Prepare one same-algorithm shadow policy per independent training seed.\n")
        f.write("# Every shadow is trained exclusively on the exact D_f reconstructed by split_seed.\n")
        f.write("# The trainer validates and reuses an exact completed cache entry before fitting.\n")
        f.write("# -----------------------------------------------------------------------------\n\n")

        for _, spec in shadow_specs.items():
            expected_dir = spec["expected_dir"]
            selected_dir = spec["selected_dir"]
            checkpoint_dir = selected_dir if spec["existing"] else expected_dir
            checkpoint = os.path.join(
                checkpoint_dir,
                f"model_{spec['training_steps']}.pt",
            )
            f.write(
                f"# Shadow key: dataset={spec['dataset']}, datasets={spec['datasets']}, "
                f"ratios={spec['ratio']}, split_seed={spec['split_seed']}, "
                f"training_seed={spec['training_seed']}, algorithm={spec['algorithm']}, "
                f"steps={spec['training_steps']}\n"
            )
            if spec["existing"]:
                f.write(f"echo 'Reusing existing D_f shadow policy: {selected_dir}'\n")
            else:
                f.write(f"{spec['train_cmd']}\n")
            f.write(f"if [ ! -f {quote_arg(checkpoint)} ]; then\n")
            f.write(
                f"    echo 'ERROR: failed to prepare D_f shadow policy: "
                f"{checkpoint}'\n"
            )
            f.write("    exit 1\n")
            f.write("fi\n\n")

        f.write("# -----------------------------------------------------------------------------\n")
        f.write("# Run audit jobs after critics and all D_f shadow policies are ready.\n")
        f.write("# -----------------------------------------------------------------------------\n\n")

        max_jobs = args.parallel
        if not commands:
            f.write("echo 'ORL-Auditor prerequisites prepared; no valid audit jobs were found.'\n")
        elif max_jobs <= 1:
            f.write("# Running audits in SERIAL mode.\n")
            for i, command in enumerate(commands):
                gpu_to_use = args.gpu
                full_cmd = command["cmd"].replace("{GPU_PLACEHOLDER}", str(gpu_to_use))
                f.write(f"# --- Audit Job {i + 1}/{len(commands)}: seed={command['seed']} ratio={command['ratio']} algo={command['algo']} --- {newline}")
                f.write(f"{full_cmd}\n\n")
            f.write("echo 'All ORL-Auditor jobs completed.'\n")
        else:
            f.write(f"# Running audits in PARALLEL mode (Batches of {max_jobs}).\n\n")
            f.write("pids=()\n")
            f.write("fail_count=0\n\n")
            for i, command in enumerate(commands):
                gpu_to_use = args.gpu if args.no_gpu_round_robin else args.gpu + (i % max_jobs)
                full_cmd = command["cmd"].replace("{GPU_PLACEHOLDER}", str(gpu_to_use))
                f.write(f"echo '[Batch {i // max_jobs + 1}] Starting audit job {i + 1} on GPU {gpu_to_use}'\n")
                f.write(f"({full_cmd}) &\n")
                f.write("pids+=($!)\n\n")

                is_batch_full = (i + 1) % max_jobs == 0
                is_last_command = (i + 1) == len(commands)
                if is_batch_full or is_last_command:
                    f.write("echo 'Waiting for current audit batch to finish...'\n")
                    f.write("for pid in \"${pids[@]}\"; do\n")
                    f.write("    wait $pid\n")
                    f.write("    if [ $? -ne 0 ]; then\n")
                    f.write("        echo \"WARNING: Audit job $pid failed.\"\n")
                    f.write("        fail_count=$((fail_count + 1))\n")
                    f.write("    fi\n")
                    f.write("done\n")
                    f.write("echo 'Audit batch finished.'\n")
                    f.write("pids=()\n\n")

            f.write("echo 'All ORL-Auditor jobs completed.'\n")
            f.write("if [ $fail_count -ne 0 ]; then\n")
            f.write("    echo \"WARNING: $fail_count audit jobs failed!\"\n")
            f.write("    exit 1\n")
            f.write("else\n")
            f.write("    echo 'All audit jobs succeeded.'\n")
            f.write("fi\n")

    try:
        os.chmod(output_script_name, 0o755)
    except OSError as e:
        print(f"Warning: Could not set execute permissions: {e}")

    print(f"Successfully generated '{output_script_name}'.")


if __name__ == "__main__":
    main()
