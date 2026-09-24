from __future__ import annotations

import argparse
import glob
import importlib.metadata
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import List

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
PYFLYT_ROOT = WORKSPACE_ROOT / "UAV_RL_Unlearning" / "PyFlyt-master"
LOCAL_MINARI_REF = WORKSPACE_ROOT / "minari_ref"
LOCAL_MINARI_VERSION = "0.5.3"

for path in (PROJECT_ROOT, PYFLYT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import d3rlpy
import gymnasium as gym
import numpy as np
import torch
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from sklearn.model_selection import train_test_split


def _patch_minari_distribution_version() -> None:
    current_version = importlib.metadata.version
    if getattr(current_version, "_uav_minari_patch", False):
        return

    def version(distribution_name: str) -> str:
        if distribution_name == "minari":
            return LOCAL_MINARI_VERSION
        return current_version(distribution_name)

    version._uav_minari_patch = True  # type: ignore[attr-defined]
    importlib.metadata.version = version


def _load_local_minari() -> ModuleType:
    init_file = LOCAL_MINARI_REF / "__init__.py"
    if not init_file.exists():
        raise ImportError(
            "Minari is not installed and local minari_ref/__init__.py was not found."
        )

    _patch_minari_distribution_version()
    spec = importlib.util.spec_from_file_location(
        "minari",
        init_file,
        submodule_search_locations=[str(LOCAL_MINARI_REF)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {init_file}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules["minari"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("minari", None)
        raise
    return module


def import_minari() -> ModuleType:
    try:
        import minari  # type: ignore

        return minari
    except ModuleNotFoundError as exc:
        if exc.name != "minari":
            raise

    return _load_local_minari()


minari = import_minari()

try:
    from minari_dataset_processing.utility import FlattenDictObsWrapper
    from minari_dataset_processing.utility import load_merged_dataset
    from minari_dataset_processing.utility import load_model_with_fix
except ImportError as e:
    print(e)
    print("Error: Could not import from 'minari_dataset_processing'.")
    print(
        "Please ensure 'merged_data_with_episode.py' is in a folder named "
        "'minari_dataset_processing' and accessible."
    )
    sys.exit(1)


DEFAULT_NUM_RAYS = 16
DEFAULT_NUM_OBSTACLE_FEATURES = 3
DEFAULT_RAYCAST_ELEVATION_ANGLES = (-30.0, 0.0, 30.0)
DEFAULT_RAYCAST_INCLUDE_VERTICAL = True
DEFAULT_RAYCAST_START_OFFSET = 0.2
QUADX_ENV_ID = "PyFlyt/QuadX-Obstacle-Waypoints-v0"


def split_dataset_tokens(values: List[str]) -> List[str]:
    tokens: List[str] = []
    for value in values:
        tokens.extend(token.strip() for token in value.split(",") if token.strip())
    if not tokens:
        raise ValueError("At least one dataset must be provided.")
    return tokens


def normalize_quadx_dataset_id(dataset_name: str, version: int) -> str:
    if "/" in dataset_name:
        return dataset_name
    if re.search(r"-v\d+$", dataset_name):
        return f"quadx/{dataset_name}"
    return f"quadx/{dataset_name}-v{version}"


def dataset_name_to_family(dataset_name: str) -> str:
    family = dataset_name.rsplit("/", 1)[-1]
    return re.sub(r"-v\d+$", "", family)


def parse_raycast_elevation_angles(value: str | None) -> tuple[float, ...]:
    if value is None:
        return DEFAULT_RAYCAST_ELEVATION_ANGLES

    angles = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not angles:
        raise ValueError("--raycast-elevation-angles cannot be empty.")
    if any(angle < -90.0 or angle > 90.0 for angle in angles):
        raise ValueError("--raycast-elevation-angles values must be in [-90, 90].")
    return angles


def flat_base_obs_dim(context_length: int) -> int:
    # Current QuadX defaults: quaternion attitude state is 21 and each waypoint delta is 3.
    return 21 + 3 * context_length


def raycast_observation_dim(
    num_rays: int,
    raycast_elevation_angles: tuple[float, ...],
    raycast_include_vertical: bool,
) -> int:
    vertical_rays = 2 if raycast_include_vertical else 0
    return num_rays * len(raycast_elevation_angles) + vertical_rays


def infer_quadx_observation_args(
    args: argparse.Namespace, observation_shape: tuple[int, ...]
) -> None:
    if len(observation_shape) != 1:
        raise ValueError(
            f"QuadX expects a flat vector observation, got shape {observation_shape}."
        )

    obs_dim = int(observation_shape[0])
    base_dim = flat_base_obs_dim(args.context_length)
    obstacle_dim = obs_dim - base_dim
    if obstacle_dim < 0:
        raise ValueError(
            f"Dataset observation dim {obs_dim} is smaller than the expected "
            f"QuadX base dim {base_dim} for context_length={args.context_length}."
        )

    if args.obstacle_obs_mode == "raycast":
        if args.num_rays is None:
            vertical_rays = 2 if args.raycast_include_vertical else 0
            horizontal_dim = obstacle_dim - vertical_rays
            if (
                horizontal_dim > 0
                and horizontal_dim % len(args.raycast_elevation_angles) == 0
            ):
                args.num_rays = horizontal_dim // len(args.raycast_elevation_angles)
            else:
                args.num_rays = DEFAULT_NUM_RAYS
        if args.num_obstacle_features is None:
            args.num_obstacle_features = DEFAULT_NUM_OBSTACLE_FEATURES
        expected_obstacle_dim = raycast_observation_dim(
            int(args.num_rays),
            args.raycast_elevation_angles,
            bool(args.raycast_include_vertical),
        )
    else:
        if args.num_obstacle_features is None:
            if obstacle_dim > 0 and obstacle_dim % 8 == 0:
                args.num_obstacle_features = int(obstacle_dim // 8)
            else:
                args.num_obstacle_features = DEFAULT_NUM_OBSTACLE_FEATURES
        if args.num_rays is None:
            args.num_rays = DEFAULT_NUM_RAYS
        expected_obstacle_dim = int(args.num_obstacle_features) * 8

    expected_dim = base_dim + expected_obstacle_dim
    if expected_dim != obs_dim:
        raise ValueError(
            "Evaluation environment observation shape would not match the dataset. "
            f"dataset_obs_dim={obs_dim}, expected_env_dim={expected_dim}, "
            f"context_length={args.context_length}, obstacle_obs_mode={args.obstacle_obs_mode}, "
            f"num_rays={args.num_rays}, num_obstacle_features={args.num_obstacle_features}, "
            f"raycast_elevation_angles={args.raycast_elevation_angles}, "
            f"raycast_include_vertical={args.raycast_include_vertical}."
        )


def make_quadx_eval_env(args: argparse.Namespace, eval_family: str) -> gym.Env:
    try:
        import PyFlyt.gym_envs  # noqa: F401
        from PyFlyt.gym_envs import FlattenWaypointEnv
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Could not import local PyFlyt from {PYFLYT_ROOT}.") from exc

    print(
        "Creating QuadX evaluation environment: "
        f"env_id={args.quadx_env_id}, eval_family={eval_family}"
    )
    base_env = gym.make(
        args.quadx_env_id,
        dataset_split_mode="fixed_eval",
        obstacle_obs_mode=args.obstacle_obs_mode,
        family_scheduler_mode="fixed_list",
        fixed_family_list=[eval_family],
        num_rays=args.num_rays,
        num_obstacle_features=args.num_obstacle_features,
        raycast_elevation_angles=args.raycast_elevation_angles,
        raycast_include_vertical=args.raycast_include_vertical,
        raycast_start_offset=args.raycast_start_offset,
        render_mode=None,
    )
    base_env.action_space.seed(args.seed)
    base_env.observation_space.seed(args.seed)

    flat_env = FlattenWaypointEnv(env=base_env, context_length=args.context_length)
    eval_env = FlattenDictObsWrapper(flat_env, is_antmaze=False)
    if args.eval_max_episode_steps is not None:
        eval_env.max_episode_steps = args.eval_max_episode_steps
    return eval_env


def load_algorithm(args: argparse.Namespace):
    if not args.base_params:
        raise ValueError("--base_params must be provided for a new run")

    print(f"Loading base params from: {args.base_params}")
    if args.algo == "CQL":
        return d3rlpy.algos.CQL.from_json(args.base_params, use_gpu=args.gpu)
    if args.algo == "BCQ":
        return d3rlpy.algos.BCQ.from_json(args.base_params, use_gpu=args.gpu)
    if args.algo == "BEAR":
        return d3rlpy.algos.BEAR.from_json(args.base_params, use_gpu=args.gpu)
    if args.algo.upper() == "TD3PLUSBC":
        return d3rlpy.algos.TD3PlusBC.from_json(args.base_params, use_gpu=args.gpu)
    if args.algo == "IQL":
        return d3rlpy.algos.IQL.from_json(args.base_params, use_gpu=args.gpu)
    if args.algo == "PLASP":
        return d3rlpy.algos.PLASWithPerturbation.from_json(
            args.base_params, use_gpu=args.gpu
        )
    if args.algo == "AWAC":
        return d3rlpy.algos.AWAC.from_json(args.base_params, use_gpu=args.gpu)
    if args.algo == "CRR":
        return d3rlpy.algos.CRR.from_json(args.base_params, use_gpu=args.gpu)
    raise ValueError(f"No available algorithm specified for {args.algo}.")


def resume_algorithm(args: argparse.Namespace):
    print(f"--- Resuming training from {args.resume_from_logdir} ---")

    params_path = os.path.join(args.resume_from_logdir, "params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"params.json not found in {args.resume_from_logdir}")
    print(f"Loading params from: {params_path}")

    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    algo_name = params.get("algorithm", args.algo)

    try:
        AlgoClass = getattr(d3rlpy.algos, algo_name)
        print(f"Initializing algorithm with --gpu {args.gpu}")
        algorithm = AlgoClass.from_json(params_path, use_gpu=args.gpu)
    except AttributeError as exc:
        raise ValueError(f"Algorithm '{algo_name}' not found in d3rlpy.algos") from exc

    model_files = glob.glob(os.path.join(args.resume_from_logdir, "model_*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No model_*.pt files found in {args.resume_from_logdir}")

    step_nums = [
        int(re.search(r"model_(\d+).pt", model_file).group(1))
        for model_file in model_files
    ]
    latest_step = max(step_nums)
    latest_model_path = os.path.join(args.resume_from_logdir, f"model_{latest_step}.pt")

    load_model_with_fix(algorithm, latest_model_path, args.gpu)
    algorithm.set_grad_step(latest_step)

    n_steps_to_run = args.n_steps - latest_step
    print(f"Resuming from step: {latest_step}")
    print(f"Total steps: {args.n_steps} | Remaining steps: {n_steps_to_run}")
    return algorithm, latest_step, n_steps_to_run


def main(args: argparse.Namespace) -> None:
    task = args.dataset
    datasets = split_dataset_tokens(args.datasets)
    args.datasets = datasets
    print(f"--- Loading QuadX datasets for task '{task}': {datasets} ---")

    ratios = args.ratios
    if ratios is None:
        ratios = [1.0] * len(datasets)
    elif len(ratios) != len(datasets):
        raise ValueError(
            f"Must provide {len(datasets)} ratios for datasets {datasets}, "
            f"but got {len(ratios)}."
        )

    dirs = [normalize_quadx_dataset_id(name, args.dataset_version) for name in datasets]
    print(f"Resolved Minari dataset ids: {dirs}")

    dataset_required, _dataset_remained = load_merged_dataset(
        dirs, ratios, args.seed, shuffle=(args.shuffle == 1)
    )
    if len(dataset_required) == 0:
        print("Error: The 'required' (retained) dataset is empty. Cannot train.")
        print("This happens if all ratios are set to 0.0.")
        return

    infer_quadx_observation_args(args, dataset_required.get_observation_shape())
    print(
        "Env observation config: "
        f"mode={args.obstacle_obs_mode}, context_length={args.context_length}, "
        f"num_rays={args.num_rays}, num_obstacle_features={args.num_obstacle_features}, "
        f"raycast_elevation_angles={args.raycast_elevation_angles}, "
        f"raycast_include_vertical={args.raycast_include_vertical}"
    )

    eval_source = args.eval_family or datasets[-1]
    eval_family = dataset_name_to_family(eval_source)
    eval_env = make_quadx_eval_env(args, eval_family)

    d3rlpy.seed(args.seed)
    start_step = 0
    n_steps_to_run = args.n_steps

    if args.resume_from_logdir:
        algorithm, start_step, n_steps_to_run = resume_algorithm(args)
        if n_steps_to_run <= 0:
            print(f"Training already completed ({start_step} steps). Exiting.")
            return
    else:
        print("--- Starting new training run ---")
        algorithm = load_algorithm(args)

    print("Splitting 'required' (retained) dataset (D_r) into 99% train / 1% validation...")
    train_episodes, test_episodes = train_test_split(
        dataset_required,
        random_state=args.seed,
        test_size=0.01,
        shuffle=True,
    )

    print(f"Final training episodes (from D_r): {len(train_episodes)}")
    print(f"Final validation episodes (from D_r): {len(test_episodes)}")

    datasets_str = "_".join(dataset_name_to_family(name) for name in datasets)
    ratios_str = "learn_ratios"
    for ratio in ratios:
        ratios_str = ratios_str + "_" + str(ratio)

    if args.shuffle == 1:
        logdir = (
            f"{args.type}/{args.dataset}/{datasets_str}/"
            f"seed_{args.seed}_steps_{args.n_steps}/{ratios_str}"
        )
    else:
        logdir = (
            f"{args.type}/{args.dataset}/{datasets_str}/"
            f"no_shuffle_seed_{args.seed}_steps_{args.n_steps}/{ratios_str}"
        )

    algorithm.fit(
        train_episodes,
        eval_episodes=test_episodes,
        n_steps=n_steps_to_run,
        n_steps_per_epoch=args.n_steps_per_epoch,
        logdir=logdir,
        experiment_name=args.algo,
        tensorboard_dir="../results_tensorboard",
        scorers={
            "environment": evaluate_on_environment(eval_env),
            "td_error": td_error_scorer,
            "discounted_advantage": discounted_sum_of_advantage_scorer,
            "value_scale": average_value_estimation_scorer,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fully train an offline RL policy on QuadX Minari datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="quadx")
    parser.add_argument("--type", type=str, default="Fully_trained")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["retain_random_static_spheres", "forget_low_altitude_barrier"],
        help=(
            "QuadX dataset names. Values can be obstacle family names "
            "(retain_random_static_spheres), names with versions "
            "(retain_random_static_spheres-v0), full Minari ids "
            "(quadx/retain_random_static_spheres-v0), or comma-separated lists."
        ),
    )
    parser.add_argument(
        "--dataset-version",
        type=int,
        default=0,
        help="Version used when --datasets entries are obstacle family names.",
    )
    parser.add_argument(
        "--base-params",
        type=str,
        default=None,
        help="Path to QuadX base params.json for new runs.",
    )
    parser.add_argument("--algo", type=str, default="CQL")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=2000000)
    parser.add_argument("--n-steps-per-epoch", type=int, default=100000)
    parser.add_argument("--resume-from-logdir", type=str, default=None)
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Retained ratio for each dataset. Order must match --datasets. "
            "If omitted, uses 1.0 for every dataset."
        ),
    )
    parser.add_argument(
        "--shuffle",
        type=int,
        default=1,
        help="Whether to shuffle when splitting each dataset into D_r and D_f.",
    )
    parser.add_argument(
        "--quadx-env-id",
        type=str,
        default=QUADX_ENV_ID,
        help="Gymnasium id for the QuadX obstacle waypoint environment.",
    )
    parser.add_argument(
        "--eval-family",
        type=str,
        default=None,
        help=(
            "Obstacle family used for environment evaluation. If omitted, the "
            "last entry in --datasets is used."
        ),
    )
    parser.add_argument(
        "--obstacle-obs-mode",
        choices=["nearest_k", "raycast"],
        default="raycast",
    )
    parser.add_argument("--num-rays", type=int, default=None)
    parser.add_argument(
        "--num-obstacle-features",
        type=int,
        default=None,
        help="Nearest-obstacle feature rows. If omitted, infer from the dataset shape when possible.",
    )
    parser.add_argument(
        "--raycast-elevation-angles",
        default=",".join(str(v) for v in DEFAULT_RAYCAST_ELEVATION_ANGLES),
        help="Comma-separated raycast elevation angles in degrees.",
    )
    parser.add_argument(
        "--no-raycast-vertical",
        dest="raycast_include_vertical",
        action="store_false",
        help="Disable the two vertical raycast fractions.",
    )
    parser.set_defaults(raycast_include_vertical=DEFAULT_RAYCAST_INCLUDE_VERTICAL)
    parser.add_argument(
        "--raycast-start-offset",
        type=float,
        default=DEFAULT_RAYCAST_START_OFFSET,
    )
    parser.add_argument("--context-length", type=int, default=2)
    parser.add_argument(
        "--eval-max-episode-steps",
        type=int,
        default=None,
        help="Optional compatibility attribute for d3rlpy environment evaluation.",
    )
    return parser


def normalize_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.dataset_version < 0:
        parser.error("--dataset-version must be non-negative.")
    if args.n_steps <= 0:
        parser.error("--n-steps must be positive.")
    if args.n_steps_per_epoch <= 0:
        parser.error("--n-steps-per-epoch must be positive.")
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

    try:
        args.raycast_elevation_angles = parse_raycast_elevation_angles(
            args.raycast_elevation_angles
        )
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    parser = build_parser()
    parsed_args = parser.parse_args()
    normalize_args(parser, parsed_args)
    print(parsed_args.shuffle == 1)
    torch.manual_seed(parsed_args.seed)
    np.random.seed(parsed_args.seed)
    main(parsed_args)
