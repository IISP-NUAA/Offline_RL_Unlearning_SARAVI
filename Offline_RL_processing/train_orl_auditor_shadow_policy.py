#!/usr/bin/env python3
"""Train one ORL-Auditor shadow policy exclusively on a fixed forget split."""

import argparse
import hashlib
import json
import os
from typing import Any, Dict, List

from orl_auditor_core import (
    canonical_algorithm_name,
    load_d3rlpy,
    load_forget_episodes,
    save_json,
    set_global_seeds,
    shadow_policy_output_dir,
)


ALGORITHM_CLASS_ALIASES = {
    "TD3PLUSBC": "TD3PlusBC",
    "PLASP": "PLASWithPerturbation",
}


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_params(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def model_checkpoint_path(output_dir: str, training_steps: int) -> str:
    return os.path.join(output_dir, f"model_{training_steps}.pt")


def config_matches(config: Dict[str, Any], args: argparse.Namespace) -> bool:
    try:
        actual_ratios = [float(value) for value in config.get("retained_ratios", [])]
    except (TypeError, ValueError):
        return False
    expected_ratios = [float(value) for value in args.retained_ratios]
    return (
        config.get("metric_type") == "ORL_Auditor_Shadow_Policy"
        and config.get("trained_only_on") == "D_f"
        and config.get("dataset_name") == args.dataset
        and list(config.get("dataset_components", [])) == list(args.datasets)
        and len(actual_ratios) == len(expected_ratios)
        and all(abs(left - right) < 1e-12 for left, right in zip(actual_ratios, expected_ratios))
        and int(config.get("split_seed", -1)) == args.split_seed
        and int(config.get("training_seed", -1)) == args.training_seed
        and canonical_algorithm_name(config.get("algorithm", "")) == canonical_algorithm_name(args.algorithm)
        and int(config.get("training_steps", -1)) == args.training_steps
        and bool(config.get("shuffle", True)) == (not args.no_shuffle)
        and config.get("max_episodes") == args.max_episodes
        and bool(config.get("completed", False))
    )


def load_algorithm(base_params: str, configured_algorithm: str, gpu: int):
    d3rlpy = load_d3rlpy()
    canonical = canonical_algorithm_name(configured_algorithm)
    class_names: List[str] = [configured_algorithm]
    alias = ALGORITHM_CLASS_ALIASES.get(canonical)
    if alias and alias not in class_names:
        class_names.append(alias)

    for class_name in class_names:
        algo_class = getattr(d3rlpy.algos, class_name, None)
        if algo_class is not None:
            use_gpu = gpu if gpu >= 0 else False
            return algo_class.from_json(base_params, use_gpu=use_gpu)

    raise ValueError(
        f"Algorithm '{configured_algorithm}' is not available in d3rlpy.algos "
        f"(tried {class_names})."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a same-algorithm ORL-Auditor shadow policy on exactly D_f. "
            "The data split seed and policy-training seed are independent."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--retained-ratios", type=float, nargs="+", required=True)
    parser.add_argument("--split-seed", type=int, required=True,
                        help="Seed used only to reconstruct the fixed D_r/D_f split.")
    parser.add_argument("--training-seed", type=int, required=True,
                        help="Seed used only for policy initialization, sampling, and optimization.")
    parser.add_argument("--algorithm", required=True,
                        help="Canonical target algorithm, e.g. CQL, BCQ, IQL, TD3PLUSBC.")
    parser.add_argument("--base-params", required=True,
                        help="Target algorithm params.json used to initialize a fresh shadow policy.")
    parser.add_argument("--training-steps", type=int, default=100000)
    parser.add_argument("--n-steps-per-epoch", type=int, default=10000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--save-dir", default="stats_results_orl_auditor")
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if an exactly matching completed shadow policy exists.")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Optional debug cap; omitted runs on the complete D_f split.")
    args = parser.parse_args()

    if len(args.datasets) != len(args.retained_ratios):
        parser.error("--datasets and --retained-ratios must have the same length.")
    if args.training_steps <= 0:
        parser.error("--training-steps must be positive.")
    if args.n_steps_per_epoch <= 0:
        parser.error("--n-steps-per-epoch must be positive.")
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("--max-episodes must be positive when provided.")
    if not os.path.isfile(args.base_params):
        parser.error(f"--base-params does not exist: {args.base_params}")

    params = load_params(args.base_params)
    configured_algorithm = params.get("algorithm") or params.get("type")
    if not configured_algorithm:
        raise ValueError(f"No algorithm/type field found in {args.base_params}")
    if canonical_algorithm_name(configured_algorithm) != canonical_algorithm_name(args.algorithm):
        raise ValueError(
            f"Algorithm mismatch: --algorithm={args.algorithm}, "
            f"but {args.base_params} configures {configured_algorithm}."
        )

    output_dir = shadow_policy_output_dir(
        args.save_dir,
        args.dataset,
        args.datasets,
        args.retained_ratios,
        args.split_seed,
        args.algorithm,
        args.training_seed,
        args.training_steps,
        shuffle=not args.no_shuffle,
    )
    checkpoint_path = model_checkpoint_path(output_dir, args.training_steps)
    config_path = os.path.join(output_dir, "shadow_config.json")
    output_params_path = os.path.join(output_dir, "params.json")
    params_sha256 = file_sha256(args.base_params)

    if not args.force and os.path.isfile(checkpoint_path) and os.path.isfile(output_params_path):
        try:
            existing_config = load_params(config_path)
        except (OSError, json.JSONDecodeError):
            existing_config = {}
        if config_matches(existing_config, args):
            print(f"Reusing existing D_f shadow policy: {output_dir}")
            return

    forget_episodes = load_forget_episodes(
        args.dataset,
        args.datasets,
        args.retained_ratios,
        args.split_seed,
        shuffle=not args.no_shuffle,
    )
    if args.max_episodes is not None:
        forget_episodes = forget_episodes[:args.max_episodes]
    if not forget_episodes:
        raise ValueError(
            "D_f is empty for this dataset/ratio/split-seed configuration; "
            "a forget-set shadow policy cannot be trained."
        )

    # The split has already been materialized with split_seed. From this point on,
    # only training_seed controls model initialization and optimizer randomness.
    set_global_seeds(args.training_seed)
    algorithm = load_algorithm(args.base_params, configured_algorithm, args.gpu)

    os.makedirs(output_dir, exist_ok=True)
    logdir = os.path.join(output_dir, "training_logs")
    print(
        f"Training {canonical_algorithm_name(args.algorithm)} shadow on {len(forget_episodes)} "
        f"D_f episodes: split_seed={args.split_seed}, training_seed={args.training_seed}, "
        f"steps={args.training_steps}"
    )
    algorithm.fit(
        forget_episodes,
        eval_episodes=None,
        n_steps=args.training_steps,
        n_steps_per_epoch=min(args.n_steps_per_epoch, args.training_steps),
        logdir=logdir,
        experiment_name=f"{canonical_algorithm_name(args.algorithm)}_shadow",
        scorers={},
    )
    algorithm.save_model(checkpoint_path)

    with open(output_params_path, "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=4)

    config = {
        "metric_type": "ORL_Auditor_Shadow_Policy",
        "trained_only_on": "D_f",
        "dataset_name": args.dataset,
        "dataset_components": args.datasets,
        "retained_ratios": args.retained_ratios,
        "split_seed": args.split_seed,
        "training_seed": args.training_seed,
        "algorithm": canonical_algorithm_name(args.algorithm),
        "configured_algorithm": configured_algorithm,
        "training_steps": args.training_steps,
        "n_steps_per_epoch": min(args.n_steps_per_epoch, args.training_steps),
        "shuffle": not args.no_shuffle,
        "num_forget_episodes": len(forget_episodes),
        "max_episodes": args.max_episodes,
        "base_params_path": os.path.abspath(args.base_params),
        "base_params_sha256": params_sha256,
        "model_checkpoint": checkpoint_path,
        "completed": True,
    }
    save_json(config, config_path)
    print(f"Saved D_f shadow policy: {checkpoint_path}")


if __name__ == "__main__":
    main()
