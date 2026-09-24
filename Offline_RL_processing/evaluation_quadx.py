"""Evaluate d3rlpy or Stable-Baselines3 checkpoints on QuadX obstacle environments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_quadx_runtime_paths() -> tuple[Path, Path]:
    candidates: list[Path] = []
    for base_dir in (SCRIPT_DIR, *SCRIPT_DIR.parents):
        candidates.append(base_dir / "PyFlyt-master")
        candidates.append(base_dir / "UAV_RL_Unlearning" / "PyFlyt-master")

    for pyflyt_root in candidates:
        if (pyflyt_root / "PyFlyt").is_dir():
            project_root = pyflyt_root.parent
            return pyflyt_root, project_root

    searched = "\n".join(str(path) for path in candidates)
    raise RuntimeError(f"Could not find PyFlyt-master. Searched:\n{searched}")


PYFLYT_ROOT, QUADX_PROJECT_ROOT = resolve_quadx_runtime_paths()
for import_root in (QUADX_PROJECT_ROOT, PYFLYT_ROOT):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

import numpy as np

from quadx_path_metadata import (
    build_eval_targets,
    infer_quadx_path_metadata,
    normalize_dataset_tokens,
    normalize_quadx_family_name,
    ratio_tag_from_values,
)


DEFAULT_NUM_RAYS = 16
DEFAULT_NUM_OBSTACLE_FEATURES = 3
DEFAULT_RAYCAST_ELEVATION_ANGLES = (-30.0, 0.0, 30.0)
DEFAULT_RAYCAST_INCLUDE_VERTICAL = True
DEFAULT_RAYCAST_START_OFFSET = 0.2
QUADX_ENV_ID = "PyFlyt/QuadX-Obstacle-Waypoints-v0"
DEFAULT_ENV_ID_LIST = [
    "retain_random_static_spheres",
    "forget_low_altitude_barrier",
]


class D3RLGymCompatibilityWrapper:
    """Expose Gymnasium envs through the older Gym API expected by d3rlpy scorers."""

    def __init__(self, env: Any):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs):
        obs, _info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, bool(terminated or truncated), info

    def close(self):
        return self.env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


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
        raise ValueError("At least one environment id/family must be provided.")
    return tokens


def normalize_quadx_dataset_id(env_name: str, version: int) -> str:
    if "/" in env_name:
        return env_name
    if re.search(r"-v\d+$", env_name):
        return f"quadx/{env_name}"
    return f"quadx/{env_name}-v{version}"


def env_name_to_family(env_name: str) -> str:
    family = env_name.rsplit("/", 1)[-1]
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
    return 21 + 3 * context_length


def raycast_observation_dim(
    num_rays: int,
    raycast_elevation_angles: tuple[float, ...],
    raycast_include_vertical: bool,
) -> int:
    vertical_rays = 2 if raycast_include_vertical else 0
    return num_rays * len(raycast_elevation_angles) + vertical_rays


def infer_quadx_observation_args(
    args: argparse.Namespace, observation_shape: tuple[int, ...] | None
) -> None:
    if observation_shape is None:
        if args.num_rays is None:
            args.num_rays = DEFAULT_NUM_RAYS
        if args.num_obstacle_features is None:
            args.num_obstacle_features = DEFAULT_NUM_OBSTACLE_FEATURES
        return

    if len(observation_shape) != 1:
        raise ValueError(
            f"QuadX expects a flat vector observation, got shape {observation_shape}."
        )

    obs_dim = int(observation_shape[0])
    base_dim = flat_base_obs_dim(args.context_length)
    obstacle_dim = obs_dim - base_dim
    if obstacle_dim < 0:
        raise ValueError(
            f"Model observation dim {obs_dim} is smaller than QuadX base dim "
            f"{base_dim} for context_length={args.context_length}."
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
            "Evaluation environment observation shape would not match the model. "
            f"model_obs_dim={obs_dim}, expected_env_dim={expected_dim}, "
            f"context_length={args.context_length}, obstacle_obs_mode={args.obstacle_obs_mode}, "
            f"num_rays={args.num_rays}, num_obstacle_features={args.num_obstacle_features}, "
            f"raycast_elevation_angles={args.raycast_elevation_angles}, "
            f"raycast_include_vertical={args.raycast_include_vertical}."
        )


def make_quadx_eval_env(args: argparse.Namespace, env_name: str, backend: str) -> Any:
    try:
        import gymnasium as gym
        from gymnasium.wrappers import TimeLimit
        import PyFlyt.gym_envs  # noqa: F401
        import env as quadx_local_envs  # noqa: F401
        from PyFlyt.gym_envs import FlattenWaypointEnv
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Could not import QuadX runtime from PyFlyt root {PYFLYT_ROOT} "
            f"and project root {QUADX_PROJECT_ROOT}."
        ) from exc

    dataset_id = normalize_quadx_dataset_id(env_name, args.dataset_version)
    family = env_name_to_family(dataset_id)
    print(
        "Creating QuadX evaluation environment: "
        f"env_id={args.quadx_env_id}, family={family}, backend={backend}"
    )

    base_env = gym.make(
        args.quadx_env_id,
        dataset_split_mode="fixed_eval",
        obstacle_obs_mode=args.obstacle_obs_mode,
        family_scheduler_mode="fixed_list",
        fixed_family_list=[family],
        num_rays=args.num_rays,
        num_obstacle_features=args.num_obstacle_features,
        raycast_elevation_angles=args.raycast_elevation_angles,
        raycast_include_vertical=args.raycast_include_vertical,
        raycast_start_offset=args.raycast_start_offset,
        render_mode=None,
    )
    base_env.action_space.seed(args.seed)
    base_env.observation_space.seed(args.seed)

    env_for_flattening = base_env
    if args.eval_max_episode_steps is not None:
        print(f"Enforcing TimeLimit with max_episode_steps={args.eval_max_episode_steps}")
        env_for_flattening = TimeLimit(
            base_env, max_episode_steps=args.eval_max_episode_steps
        )

    flat_env = FlattenWaypointEnv(
        env=env_for_flattening, context_length=args.context_length
    )
    if backend == "d3rl_scorer":
        return D3RLGymCompatibilityWrapper(flat_env)
    return flat_env


def get_model_observation_shape(params: dict[str, Any]) -> tuple[int, ...] | None:
    raw_shape = params.get("observation_shape")
    if raw_shape is None:
        return None
    return tuple(int(dim) for dim in raw_shape)


def get_model_action_size(params: dict[str, Any]) -> int | None:
    raw_size = params.get("action_size")
    if raw_size is None:
        return None
    return int(raw_size)


def load_d3rl_model_with_fix(algorithm: Any, model_path: Path, gpu_id: int) -> None:
    import torch
    from d3rlpy.torch_utility import set_state_dict

    print(f"--- Manually loading d3rlpy model from: {model_path} ---")
    if gpu_id >= 0:
        map_location = lambda storage, _loc: storage.cuda(gpu_id)
    else:
        map_location = lambda storage, _loc: storage.cpu()

    chkpt = torch.load(str(model_path), map_location=map_location)
    set_state_dict(algorithm.impl, chkpt)
    print("Model weights loaded successfully.")


def load_d3rl_algorithm(args: argparse.Namespace, params: dict[str, Any]):
    import d3rlpy

    algo_name = params.get("algorithm") or params.get("type")
    if not algo_name:
        raise ValueError("Could not find 'algorithm' or 'type' in params.json.")

    try:
        AlgoClass = getattr(d3rlpy.algos, algo_name)
    except AttributeError as exc:
        raise ValueError(f"Algorithm '{algo_name}' not found in d3rlpy.algos") from exc

    params_path = Path(args.model_dir) / "params.json"
    model_path = Path(args.model_dir) / args.model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Initializing d3rlpy algorithm '{algo_name}' (GPU: {args.gpu})")
    algorithm = AlgoClass.from_json(str(params_path), use_gpu=args.gpu)
    load_d3rl_model_with_fix(algorithm, model_path, args.gpu)
    return algorithm, algo_name


def load_sb3_model(args: argparse.Namespace):
    try:
        import stable_baselines3 as sb3
    except ModuleNotFoundError as exc:
        raise RuntimeError("Stable-Baselines3 is required for --d3rl-or-sb3 sb3.") from exc

    model_path = Path(args.model_path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"SB3 model zip not found: {model_path}")
    if model_path.suffix != ".zip":
        raise ValueError(f"SB3 model path must be a .zip file: {model_path}")

    try:
        AlgoClass = getattr(sb3, args.sb3_algo)
    except AttributeError as exc:
        raise ValueError(f"SB3 algorithm '{args.sb3_algo}' not found.") from exc

    print(f"Loading SB3 {args.sb3_algo} model from: {model_path}")
    try:
        return AlgoClass.load(str(model_path), device=args.sb3_device), args.sb3_algo
    except Exception as exc:
        if "Weights only load failed" not in str(exc):
            raise

        import torch as th

        original_load = th.load

        def compatibility_load(*load_args, **load_kwargs):
            load_kwargs["weights_only"] = False
            return original_load(*load_args, **load_kwargs)

        th.load = compatibility_load
        try:
            return AlgoClass.load(str(model_path), device=args.sb3_device), args.sb3_algo
        finally:
            th.load = original_load


def sb3_model_observation_shape(model: Any) -> tuple[int, ...] | None:
    shape = getattr(getattr(model, "observation_space", None), "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


def sb3_model_action_size(model: Any) -> int | None:
    shape = getattr(getattr(model, "action_space", None), "shape", None)
    if shape is None:
        return None
    return int(np.prod(shape))


def validate_env_spaces(
    env: Any,
    model_observation_shape: tuple[int, ...] | None,
    model_action_size: int | None,
    env_name: str,
) -> None:
    if model_observation_shape is not None:
        env_shape = tuple(env.observation_space.shape)
        if env_shape != model_observation_shape:
            raise ValueError(
                f"Eval env shape {env_shape} does not match model shape "
                f"{model_observation_shape} for {env_name}."
            )
    if model_action_size is not None and hasattr(env.action_space, "shape"):
        env_action_size = int(np.prod(env.action_space.shape))
        if env_action_size != model_action_size:
            raise ValueError(
                f"Eval env action size {env_action_size} does not match model "
                f"action_size {model_action_size}."
            )


def predict_d3rl_action(algorithm: Any, obs: np.ndarray) -> Any:
    batch_obs = np.expand_dims(np.asarray(obs, dtype=np.float32), axis=0)
    action = algorithm.predict(batch_obs)
    if isinstance(action, tuple):
        action = action[0]
    action = np.asarray(action)
    if action.ndim == 0:
        return action.item()
    return action[0]


def predict_sb3_action(model: Any, obs: np.ndarray, deterministic: bool) -> Any:
    action, _state = model.predict(obs, deterministic=deterministic)
    return action


def rollout_episode(
    model: Any,
    env: Any,
    seed: int,
    action_fn: Any,
) -> dict[str, Any]:
    obs, info = env.reset(seed=seed)
    done = False
    episode_return = 0.0
    episode_length = 0
    nearest_distances = []
    obstacle_collision = False
    env_complete = bool(info.get("env_complete", False))
    family = info.get("effective_obstacle_family", "")

    while not done:
        action = action_fn(model, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        episode_length += 1
        done = bool(terminated or truncated)
        obstacle_collision |= bool(info.get("obstacle_collision", False))
        env_complete |= bool(info.get("env_complete", False))
        nearest_distances.append(float(info.get("nearest_obstacle_distance", np.inf)))

    return {
        "family": family,
        "return": episode_return,
        "success": env_complete,
        "collision": obstacle_collision,
        "episode_length": episode_length,
        "nearest_obstacle_distance": float(np.min(nearest_distances))
        if nearest_distances
        else np.inf,
    }


def aggregate_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        return {
            "average_return": np.nan,
            "success_rate": np.nan,
            "collision_rate": np.nan,
            "episode_length": np.nan,
            "nearest_obstacle_distance": np.nan,
            "episodes": [],
        }

    return {
        "average_return": float(np.mean([episode["return"] for episode in episodes])),
        "success_rate": float(np.mean([episode["success"] for episode in episodes])),
        "collision_rate": float(np.mean([episode["collision"] for episode in episodes])),
        "episode_length": float(np.mean([episode["episode_length"] for episode in episodes])),
        "nearest_obstacle_distance": float(
            np.mean([episode["nearest_obstacle_distance"] for episode in episodes])
        ),
        "episodes": episodes,
    }


def evaluate_on_quadx_environment(
    model: Any,
    env: Any,
    args: argparse.Namespace,
    action_fn: Any,
) -> dict[str, Any]:
    episodes = []
    for trial in range(args.n_trials):
        episodes.append(rollout_episode(model, env, args.seed + trial, action_fn))
    return aggregate_episodes(episodes)


def safe_path_component(value: str, max_length: int = 120) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip("/"))
    component = re.sub(r"_+", "_", component).strip("._-") or "unknown"
    if len(component) <= max_length:
        return component
    digest = hashlib.sha1(component.encode("utf-8")).hexdigest()[:8]
    return f"{component[: max_length - 9].rstrip('._-')}_{digest}"


def short_quadx_dataset_name(value: str) -> str:
    family = normalize_quadx_family_name(value)
    family = re.sub(r"^(?:retain|forget)_", "", family)
    return safe_path_component(family, max_length=80)


def dataset_scope_component(datasets: list[str]) -> str:
    shortened = [short_quadx_dataset_name(dataset) for dataset in datasets]
    return safe_path_component("__".join(shortened), max_length=160)


def clean_path_parts(path_value: str) -> list[str]:
    return [
        part
        for part in path_value.replace("\\", "/").split("/")
        if part and part not in {".", ".."}
    ]


def output_method_component(model_location: str) -> str:
    parts = clean_path_parts(model_location)
    if "unlearning_processing" in parts:
        index = parts.index("unlearning_processing")
        if index + 1 < len(parts):
            return safe_path_component(parts[index + 1])
    for category in ("Retrain", "Unlearned", "Fully_trained"):
        if category in parts:
            return category
    if "UDRU" in model_location or "corr" in model_location:
        for part in parts:
            if "corr" in part or "UDRU" in part:
                return safe_path_component(part)
    return safe_path_component(parts[0] if parts else "unknown")


def output_seed_component(model_location: str, seed: int) -> str:
    for part in clean_path_parts(model_location):
        if part.startswith("unlearn_seed_"):
            continue
        if re.fullmatch(r"(?:no_shuffle_)?seed_\d+(?:_steps_\d+)?", part):
            return safe_path_component(part)
    return f"seed_{seed}"


def output_algo_step_component(model_location: str, algorithm: str) -> str:
    for part in clean_path_parts(model_location):
        match = re.fullmatch(r"Unlearning_([^_]+)_steps_(\d+)", part)
        if match:
            return safe_path_component(f"{match.group(1)}_steps_{match.group(2)}")
    return safe_path_component(algorithm or "unknown_algo")


def is_dataset_scope_part(part: str, datasets: list[str]) -> bool:
    normalized_part = normalize_quadx_family_name(part)
    return any(dataset in normalized_part or dataset + "-v" in part for dataset in datasets)


def is_ratio_part(part: str) -> bool:
    lower = part.lower()
    return lower.startswith(("ratios_", "learn_ratios_", "retained_ratios_", "retain_ratios_"))


def output_extra_components(model_location: str, args: argparse.Namespace, algorithm: str) -> list[str]:
    parts = clean_path_parts(model_location)
    method = output_method_component(model_location)
    seed_component = output_seed_component(model_location, args.seed)
    algo_component = output_algo_step_component(model_location, algorithm)
    ignored = {
        "Uncertainty_Driven_RL_Unlearning",
        "Offline_RL_processing",
        "unlearning_processing",
        "quadx",
        "Retrain",
        "Unlearned",
        "Fully_trained",
        method,
        seed_component,
        algo_component,
    }

    extras: list[str] = []
    for part in parts:
        safe_part = safe_path_component(part, max_length=80)
        if part in ignored or safe_part in ignored:
            continue
        if is_ratio_part(part) or is_dataset_scope_part(part, args.eval_datasets):
            continue
        if re.fullmatch(r"Unlearning_([^_]+)_steps_(\d+)", part):
            continue
        if part.startswith("unlearn_seed_"):
            continue
        extras.append(safe_part)

    # Keep the path informative but bounded; the final run/checkpoint component is usually most useful.
    deduped: list[str] = []
    seen: set[str] = set()
    for item in extras:
        if item in seen:
            continue
        deduped.append(item)
        seen.add(item)
    return deduped[-4:]


def source_model_location(args: argparse.Namespace) -> str:
    if args.backend == "sb3":
        return str(Path(args.model_path).expanduser().parent)
    return str(args.model_dir)


def resolve_eval_targets(args: argparse.Namespace) -> list[dict[str, object]]:
    fallback_datasets = None
    if args.env_id_list:
        fallback_datasets = split_env_tokens(args.env_id_list)

    if args.datasets:
        datasets = normalize_dataset_tokens(args.datasets)
        if not datasets:
            raise ValueError("--datasets did not contain any valid QuadX family names.")
        retained_ratios = args.retained_ratios
        if retained_ratios is None:
            metadata = infer_quadx_path_metadata(
                source_model_location(args),
                fallback_datasets=datasets,
                require_ratio_match=False,
            )
            if len(metadata.retained_ratios) == len(datasets):
                retained_ratios = metadata.retained_ratios
            else:
                retained_ratios = [1.0] * len(datasets)
        dataset_source = "cli"
        ratio_source = "cli" if args.retained_ratios is not None else "path_or_default"
    else:
        metadata = infer_quadx_path_metadata(
            source_model_location(args),
            fallback_datasets=fallback_datasets or DEFAULT_ENV_ID_LIST,
        )
        datasets = metadata.datasets
        retained_ratios = (
            args.retained_ratios
            if args.retained_ratios is not None
            else metadata.retained_ratios
        )
        dataset_source = metadata.dataset_source
        ratio_source = "cli" if args.retained_ratios is not None else metadata.ratio_source

    if len(datasets) != len(retained_ratios):
        raise ValueError(
            f"Expected one retained ratio per dataset, got {len(datasets)} datasets "
            f"and {len(retained_ratios)} ratios."
        )

    targets = build_eval_targets(datasets, retained_ratios, args.dataset_version)
    args.eval_datasets = datasets
    args.eval_retained_ratios = [float(value) for value in retained_ratios]
    args.eval_ratio_group = ratio_tag_from_values(args.eval_retained_ratios)
    args.eval_metadata_dataset_source = dataset_source
    args.eval_metadata_ratio_source = ratio_source
    return targets


def components_for_udrl_path(path_value: str) -> list[str]:
    raw_components = [component for component in Path(path_value).parts if component]
    if "Uncertainty_Driven_RL_Unlearning" not in raw_components:
        return [component for component in path_value.split("/") if component]

    marker_index = raw_components.index("Uncertainty_Driven_RL_Unlearning")
    project_relative = raw_components[marker_index + 1 :]
    if project_relative and project_relative[0] == "Offline_RL_processing":
        project_relative = project_relative[1:]
    return project_relative


def legacy_evaluation_model_dir_name(model_location: str) -> str:
    model_dir_components = components_for_udrl_path(model_location)
    if "Unlearned" in model_location:
        return (
            model_dir_components[-3]
            + "_"
            + model_dir_components[-1]
            + "_"
            + model_dir_components[-5]
            + "_"
            + model_dir_components[-4]
        )
    if "UDRU" in model_location or "corr" in model_location:
        offset = 0 if model_dir_components and model_dir_components[0] == ".." else 1
        return (
            model_dir_components[2 - offset]
            + "_"
            + model_dir_components[5 - offset]
            + "_"
            + model_dir_components[6 - offset]
            + "_"
            + model_dir_components[-2]
            + "_"
            + model_dir_components[-3]
        )
    return model_dir_components[0] + "_" + model_dir_components[-3] + "_" + model_dir_components[-1]


def legacy_evaluation_output_path(
    args: argparse.Namespace, result: dict[str, Any], model_location: str
) -> Path:
    id_components = result["env_id"].split("/")
    task_name_safe = id_components[0] + "_" + id_components[1]
    model_dir_name = legacy_evaluation_model_dir_name(model_location)
    output_dir = Path(args.save_dir) / task_name_safe / model_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"results_{task_name_safe}_seed{args.seed}.json"


def categorized_output_path(
    args: argparse.Namespace, result: dict[str, Any], model_location: str
) -> Path:
    method = output_method_component(model_location)
    algo_step = output_algo_step_component(model_location, str(result.get("algorithm", "")))
    dataset_scope = dataset_scope_component(args.eval_datasets)
    seed_component = output_seed_component(model_location, args.seed)
    ratio_component = safe_path_component(f"ratios_{args.eval_ratio_group}")
    extra_components = output_extra_components(
        model_location, args, str(result.get("algorithm", ""))
    )

    output_dir = (
        Path(args.save_dir)
        / "quadx"
        / args.backend
        / method
        / algo_step
        / "datasets"
        / dataset_scope
        / seed_component
        / ratio_component
    )
    for component in extra_components:
        output_dir = output_dir / component
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"results_quadx_seed{args.seed}.jsonl"


def save_result(args: argparse.Namespace, result: dict[str, Any]) -> Path:
    model_location = source_model_location(args)
    output_path = categorized_output_path(args, result, model_location)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, indent=None, sort_keys=True) + "\n")
    return output_path


def evaluate_d3rl(args: argparse.Namespace, eval_targets: list[dict[str, object]]) -> None:
    import d3rlpy

    d3rlpy.seed(args.seed)
    params_path = Path(args.model_dir) / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"params.json not found in {args.model_dir}")

    print(f"Loading params from: {params_path}")
    with params_path.open("r", encoding="utf-8") as f:
        params = json.load(f)

    model_observation_shape = get_model_observation_shape(params)
    model_action_size = get_model_action_size(params)
    infer_quadx_observation_args(args, model_observation_shape)
    algorithm, algo_name = load_d3rl_algorithm(args, params)

    for target in eval_targets:
        env_name = str(target["family"])
        eval_env = make_quadx_eval_env(args, env_name, backend="d3rl")
        validate_env_spaces(eval_env, model_observation_shape, model_action_size, env_name)
        print(f"Evaluating {env_name} on {args.n_trials} trials... (seed={args.seed})")
        metrics = evaluate_on_quadx_environment(
            algorithm, eval_env, args, predict_d3rl_action
        )
        eval_env.close()
        write_eval_result(args, algo_name, target, metrics)


def evaluate_sb3(args: argparse.Namespace, eval_targets: list[dict[str, object]]) -> None:
    np.random.seed(args.seed)
    model, algo_name = load_sb3_model(args)
    model_observation_shape = sb3_model_observation_shape(model)
    model_action_size = sb3_model_action_size(model)
    infer_quadx_observation_args(args, model_observation_shape)

    for target in eval_targets:
        env_name = str(target["family"])
        eval_env = make_quadx_eval_env(args, env_name, backend="sb3")
        validate_env_spaces(eval_env, model_observation_shape, model_action_size, env_name)
        print(f"Evaluating {env_name} on {args.n_trials} trials... (seed={args.seed})")
        metrics = evaluate_on_quadx_environment(
            model,
            eval_env,
            args,
            lambda sb3_model, obs: predict_sb3_action(
                sb3_model, obs, args.deterministic
            ),
        )
        eval_env.close()
        write_eval_result(args, algo_name, target, metrics)


def write_eval_result(
    args: argparse.Namespace,
    algo_name: str,
    target: dict[str, object],
    metrics: dict[str, Any],
) -> None:
    mean_score = float(metrics["average_return"])
    env_id = str(target["env_id"])
    eval_family = str(target["family"])
    retained_ratio = float(target["retained_ratio"])
    result = {
        "backend": args.backend,
        "model_dir": args.model_dir,
        "model_filename": args.model_filename,
        "model_path": args.model_path,
        "algorithm": algo_name,
        "env_id": env_id,
        "eval_family": eval_family,
        "quadx_env_id": args.quadx_env_id,
        "n_trials": args.n_trials,
        "seed": args.seed,
        "mean_score": float(mean_score),
        "average_return": float(metrics["average_return"]),
        "success_rate": float(metrics["success_rate"]),
        "collision_rate": float(metrics["collision_rate"]),
        "episode_length": float(metrics["episode_length"]),
        "nearest_obstacle_distance": float(metrics["nearest_obstacle_distance"]),
        "episodes": metrics["episodes"],
        "dataset_components": args.eval_datasets,
        "retained_ratios": args.eval_retained_ratios,
        "ratio_group": args.eval_ratio_group,
        "eval_dataset_index": int(target["index"]),
        "eval_retained_ratio": retained_ratio,
        "data_split": str(target["data_split"]),
        "family_group": str(target["family_group"]),
        "metadata_dataset_source": args.eval_metadata_dataset_source,
        "metadata_ratio_source": args.eval_metadata_ratio_source,
        "obstacle_obs_mode": args.obstacle_obs_mode,
        "num_rays": args.num_rays,
        "num_obstacle_features": args.num_obstacle_features,
        "raycast_elevation_angles": args.raycast_elevation_angles,
        "raycast_include_vertical": args.raycast_include_vertical,
        "context_length": args.context_length,
    }
    output_path = save_result(args, result)

    print("\n--- Evaluation Results ---")
    print(f"  Backend: {args.backend}")
    print(f"  Env: {result['env_id']}")
    print(f"  Retained Ratio: {retained_ratio:.6g}")
    print(f"  Data Split: {result['data_split']}")
    print(f"  Mean Score: {mean_score:.4f}")
    print(f"  Collision Rate: {result['collision_rate']:.4f}")
    print(f"  Episode Length: {result['episode_length']:.2f}")
    print(f"  Nearest Obstacle Distance: {result['nearest_obstacle_distance']:.4f}")
    print(f"  Saved: {output_path}")
    print("--------------------------\n")


def main(args: argparse.Namespace) -> None:
    print("--- Starting QuadX Evaluation ---")
    eval_targets = resolve_eval_targets(args)
    print(
        "Eval datasets: "
        + " ".join(
            f"{target['family']}({target['data_split']}={target['retained_ratio']})"
            for target in eval_targets
        )
    )
    if args.backend == "d3rl":
        evaluate_d3rl(args, eval_targets)
    else:
        evaluate_sb3(args, eval_targets)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate d3rlpy or SB3 models on QuadX obstacle waypoint environments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--d3rl-or-sb3",
        default="d3rl",
        help="Backend selector: d3rl/d3rlpy for .pt checkpoints, sb3 for .zip models.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="d3rlpy log directory containing params.json and model_*.pt.",
    )
    parser.add_argument(
        "--model-filename",
        type=str,
        default="model.pt",
        help="d3rlpy checkpoint filename inside --model-dir.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="SB3 .zip model path used when --d3rl-or-sb3 sb3.",
    )
    parser.add_argument("--sb3-algo", default="SAC")
    parser.add_argument("--sb3-device", default="auto")
    parser.add_argument(
        "--env-id-list",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional fallback/override QuadX eval families. Prefer --datasets "
            "for new calls; when both are omitted, paths are inferred."
        ),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help=(
            "QuadX dataset families to evaluate. If omitted, they are inferred "
            "from the model path, falling back to --env-id-list."
        ),
    )
    parser.add_argument(
        "--retained-ratios",
        type=float,
        nargs="+",
        default=None,
        help="Retained ratio for each dataset; 1.0=retain, 0.0=forget, middle=partial.",
    )
    parser.add_argument("--dataset-version", type=int, default=0)
    parser.add_argument("--quadx-env-id", type=str, default=QUADX_ENV_ID)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", "--save_dir", dest="save_dir", default="evaluation_results")
    parser.add_argument(
        "--obstacle-obs-mode",
        choices=["nearest_k", "raycast"],
        default="raycast",
    )
    parser.add_argument("--num-rays", type=int, default=None)
    parser.add_argument("--num-obstacle-features", type=int, default=None)
    parser.add_argument(
        "--raycast-elevation-angles",
        default=",".join(str(v) for v in DEFAULT_RAYCAST_ELEVATION_ANGLES),
    )
    parser.add_argument(
        "--no-raycast-vertical",
        dest="raycast_include_vertical",
        action="store_false",
    )
    parser.set_defaults(raycast_include_vertical=DEFAULT_RAYCAST_INCLUDE_VERTICAL)
    parser.add_argument(
        "--raycast-start-offset",
        type=float,
        default=DEFAULT_RAYCAST_START_OFFSET,
    )
    parser.add_argument("--context-length", type=int, default=2)
    parser.add_argument("--eval-max-episode-steps", type=int, default=None)
    policy_group = parser.add_mutually_exclusive_group()
    policy_group.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Use deterministic SB3 policy actions.",
    )
    policy_group.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Sample stochastic SB3 policy actions.",
    )
    parser.set_defaults(deterministic=True)
    return parser


def normalize_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    try:
        args.backend = normalize_backend(args.d3rl_or_sb3)
    except ValueError as exc:
        parser.error(str(exc))

    if args.backend == "d3rl" and not args.model_dir:
        parser.error("--model-dir is required when --d3rl-or-sb3 is d3rl/d3rlpy.")
    if args.backend == "sb3" and not args.model_path:
        parser.error("--model-path is required when --d3rl-or-sb3 is sb3.")
    if args.dataset_version < 0:
        parser.error("--dataset-version must be non-negative.")
    if args.n_trials <= 0:
        parser.error("--n-trials must be positive.")
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
    if args.retained_ratios is not None:
        invalid_ratios = [value for value in args.retained_ratios if value < 0.0 or value > 1.0]
        if invalid_ratios:
            parser.error(f"--retained-ratios values must be in [0, 1], got {invalid_ratios}.")
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
    main(parsed_args)
