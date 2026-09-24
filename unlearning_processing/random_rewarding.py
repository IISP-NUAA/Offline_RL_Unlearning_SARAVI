import sys
import os
import glob
import re
from typing import List, Dict, Set, Optional, Callable, Tuple
import json
import d3rlpy
import gymnasium as gym
import numpy as np
from d3rlpy.base import LearnableBase
from gymnasium.spaces import Box
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer, discounted_sum_of_advantage_scorer, average_value_estimation_scorer
from sklearn.model_selection import train_test_split

import minari
from minari.dataset.episode_data import EpisodeData as MinariEpisodeData
import argparse
import torch
from d3rlpy.torch_utility import set_state_dict

#  Import the dataset processing scripts
#  This assumes 'minari_dataset_processing' is on the PYTHONPATH
#  or in a parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from minari_dataset_processing.merged_data_with_episode import MergedMinariDataset, EpisodeNew
    from minari_dataset_processing.merged_data_with_episode import _get_concatenated_obs_space
    from minari_dataset_processing.merged_data_with_episode import InMemoryMinariDataset
    from minari_dataset_processing.merged_data_with_episode import _concatenate_observations
    from minari_dataset_processing.utility import FlattenDictObsWrapper, load_merged_dataset, load_model_with_fix, parse_dataset_dir
    from Offline_RL_processing.fully_training_quadx import (
        DEFAULT_RAYCAST_ELEVATION_ANGLES,
        DEFAULT_RAYCAST_INCLUDE_VERTICAL,
        DEFAULT_RAYCAST_START_OFFSET,
        QUADX_ENV_ID,
        dataset_name_to_family,
        infer_quadx_observation_args,
        make_quadx_eval_env,
        normalize_quadx_dataset_id,
        parse_raycast_elevation_angles,
        split_dataset_tokens,
    )
except ImportError:
    
    print("Error: Could not import from 'minari_dataset_processing'.")
    print(
        "Please ensure 'merged_data_with_episode.py' is in a folder named 'minari_dataset_processing' and accessible.")
    sys.exit(1)

from unlearning_processing.unlearning_cost_runtime import (
    CostMeasurementController,
    CostMeasurementFailed,
    CostMeasurementStop,
    wrap_standard_updates,
)


def is_quadx_dataset(task: str, datasets: List[str]) -> bool:
    return task.lower() == "quadx" or any(
        str(dataset).startswith("quadx/") for dataset in datasets
    )


def resolve_dataset_dirs(args: argparse.Namespace) -> Tuple[List[str], List[str]]:
    datasets = args.datasets
    if is_quadx_dataset(args.dataset, datasets):
        datasets = split_dataset_tokens(datasets)
        args.datasets = datasets
        return datasets, [
            normalize_quadx_dataset_id(name, args.dataset_version)
            for name in datasets
        ]

    return datasets, parse_dataset_dir(args.dataset, datasets)


def build_eval_env(
    args: argparse.Namespace,
    task: str,
    datasets: List[str],
    dataset_required: MergedMinariDataset,
    dataset_remained: MergedMinariDataset,
) -> gym.Env:
    if is_quadx_dataset(task, datasets):
        shape_source = dataset_required if len(dataset_required) else dataset_remained
        infer_quadx_observation_args(args, shape_source.get_observation_shape())
        print(
            "Env observation config: "
            f"mode={args.obstacle_obs_mode}, context_length={args.context_length}, "
            f"num_rays={args.num_rays}, num_obstacle_features={args.num_obstacle_features}, "
            f"raycast_elevation_angles={args.raycast_elevation_angles}, "
            f"raycast_include_vertical={args.raycast_include_vertical}"
        )
        eval_source = args.eval_family or datasets[-1]
        eval_family = dataset_name_to_family(eval_source)
        return make_quadx_eval_env(args, eval_family)

    reference_ds = (
        dataset_required._datasets[-1]
        if dataset_required._datasets
        else dataset_remained._datasets[-1]
    )
    eval_env_original = reference_ds.recover_environment(eval_env=True)
    if hasattr(eval_env_original, 'max_episode_steps'):
        eval_env_original.max_episode_steps = 1000

    is_antmaze_task = (task == 'antmaze')
    print(f"Applying Gym-API compatibility wrapper (FlattenDictObsWrapper), is_antmaze={is_antmaze_task}")
    return FlattenDictObsWrapper(eval_env_original, is_antmaze=is_antmaze_task)


def add_quadx_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--dataset-version',
        type=int,
        default=0,
        help="Version used when QuadX --datasets entries are obstacle family names.",
    )
    parser.add_argument(
        '--quadx-env-id',
        type=str,
        default=QUADX_ENV_ID,
        help="Gymnasium id for the QuadX obstacle waypoint environment.",
    )
    parser.add_argument(
        '--eval-family',
        type=str,
        default=None,
        help=(
            "QuadX obstacle family used for environment evaluation. If omitted, "
            "the last entry in --datasets is used."
        ),
    )
    parser.add_argument(
        '--obstacle-obs-mode',
        choices=['nearest_k', 'raycast'],
        default='raycast',
    )
    parser.add_argument('--num-rays', type=int, default=None)
    parser.add_argument(
        '--num-obstacle-features',
        type=int,
        default=None,
        help="Nearest-obstacle feature rows. If omitted, infer from the dataset shape when possible.",
    )
    parser.add_argument(
        '--raycast-elevation-angles',
        default=','.join(str(v) for v in DEFAULT_RAYCAST_ELEVATION_ANGLES),
        help="Comma-separated raycast elevation angles in degrees.",
    )
    parser.add_argument(
        '--no-raycast-vertical',
        dest='raycast_include_vertical',
        action='store_false',
        help="Disable the two vertical raycast fractions.",
    )
    parser.set_defaults(raycast_include_vertical=DEFAULT_RAYCAST_INCLUDE_VERTICAL)
    parser.add_argument(
        '--raycast-start-offset',
        type=float,
        default=DEFAULT_RAYCAST_START_OFFSET,
    )
    parser.add_argument('--context-length', type=int, default=2)
    parser.add_argument(
        '--eval-max-episode-steps',
        type=int,
        default=None,
        help="Optional compatibility attribute for d3rlpy environment evaluation.",
    )


def normalize_quadx_eval_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
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

    try:
        args.raycast_elevation_angles = parse_raycast_elevation_angles(
            args.raycast_elevation_angles
        )
    except ValueError as exc:
        parser.error(str(exc))


def resolve_model_checkpoint(model_dir: str) -> Tuple[str, Optional[int]]:
    model_files = glob.glob(os.path.join(model_dir, 'model_*.pt'))
    if not model_files:
        model_path_fallback = os.path.join(model_dir, 'model.pt')
        if os.path.exists(model_path_fallback):
            print("Found 'model.pt'.")
            return model_path_fallback, None
        raise FileNotFoundError(f"No model_*.pt or model.pt files found in {model_dir}")

    step_candidates = []
    for model_file in model_files:
        match = re.search(r'model_(\d+)\.pt$', os.path.basename(model_file))
        if match:
            step_candidates.append((int(match.group(1)), model_file))

    if not step_candidates:
        latest_model_path = sorted(model_files)[0]
        print(
            "Warning: Could not parse step numbers. "
            f"Using {os.path.basename(latest_model_path)}"
        )
        return latest_model_path, None

    latest_step, latest_model_path = max(step_candidates, key=lambda item: item[0])
    print(f"Found latest checkpoint: {os.path.basename(latest_model_path)}")
    return latest_model_path, latest_step


def main(args):
    # --- 1. Setup and Data Loading ---
    task = args.dataset
    datasets, dirs = resolve_dataset_dirs(args)

    print(f"Splitting full dataset with {args.retained_ratios} retained...")
    dataset_required, dataset_remained = load_merged_dataset(dirs, args.retained_ratios, args.seed,shuffle=(args.shuffle==1))
    cost_controller = CostMeasurementController.from_path(
        args.cost_measurement_config,
        dataset_required,
        dataset_remained,
    )

    #  Set up evaluation environment
    eval_env = None if cost_controller is not None else build_eval_env(args, task, datasets, dataset_required, dataset_remained)
    # d3rlpy.seed(args.seed)
        
    if hasattr(eval_env, 'action_space'):
        eval_env.action_space.seed(args.seed)
    if hasattr(eval_env, 'observation_space'):
        eval_env.observation_space.seed(args.seed)
    d3rlpy.seed(args.seed)

    # --- 2. Load Models ---
    params_path = os.path.join(args.model_to_unlearn_dir, 'params.json')
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"params.json not found in {args.model_to_unlearn_dir}")
    model_path, model_step = resolve_model_checkpoint(args.model_to_unlearn_dir)

    with open(params_path, 'r') as f:
        params = json.load(f)
    algo_name = params.get("algorithm", args.algo)

    try:
        AlgoClass = getattr(d3rlpy.algos, algo_name)
    except AttributeError:
        print(f"Error: Algorithm '{algo_name}' not found in d3rlpy.algos.")
        return

    print(f"--- Loading model for Random Reward Unlearning (algorithm) ---")
    algorithm = AlgoClass.from_json(params_path, use_gpu=args.gpu)
    load_model_with_fix(algorithm, model_path, args.gpu)

    if model_step is not None:
        algorithm.set_grad_step(model_step)
        print(f"Resuming training from step {model_step}")

    # --- 3. Prepare Datasets & Poisoning ---

    #  Split Retained Set (D_r) into train and validation
    train_episodes_r, eval_episodes_r = train_test_split(
        dataset_required,
        random_state=args.seed,
        test_size=0.01,
        shuffle=True
    )
    print(f"Retained set (D_r) split into: {len(train_episodes_r)} train / {len(eval_episodes_r)} eval episodes.")

    #  Get Forget Set (D_f) and poison it
    print(f"--- Manually loading and poisoning {len(dataset_remained)} 'forget' episodes (D_f) ---")

    d3rlpy_obs_shape = dataset_remained.get_observation_shape()
    d3rlpy_act_size = dataset_remained.get_action_size()
    is_dict_obs = dataset_remained._is_dict_obs
    obs_key_order = dataset_remained._obs_key_order

    unlearning_episodes = []
    for ds in dataset_remained._datasets:
        for ep_data in ds:
            # 1. POISON THE DATA (Random Rewarding)
            #  Using np.random.uniform as in the original TrajDeleter impl.
            poisoned_rewards = np.random.uniform(
                low=ep_data.rewards.min(),
                high=ep_data.rewards.max(),
                size=len(ep_data.rewards[:, ])
            )

            # 2. Reconstruct Episode Data
            new_infos = ep_data.infos.copy()
            new_infos["source_dataset_id"] = ds.id

            episode_data_with_info = MinariEpisodeData(
                id=ep_data.id,
                observations=ep_data.observations,
                actions=ep_data.actions,
                rewards=poisoned_rewards,  # <-- POISONED REWARDS APPLIED HERE
                terminations=ep_data.terminations,
                truncations=ep_data.truncations,
                infos=new_infos
            )

            if is_dict_obs:
                d3rlpy_obs_data = _concatenate_observations(
                    episode_data_with_info.observations,
                    obs_key_order
                )
            else:
                d3rlpy_obs_data = episode_data_with_info.observations.astype(np.float32)

            # 3. Construct EpisodeNew
            ep_new = EpisodeNew(
                episode_data=episode_data_with_info,
                observation_shape=d3rlpy_obs_shape,
                action_size=d3rlpy_act_size,
                d3rlpy_observations=d3rlpy_obs_data
            )
            unlearning_episodes.append(ep_new)

    print(f"Poisoning complete. Loaded {len(unlearning_episodes)} poisoned episodes.")

    # --- 4. Merge Datasets and Standard Fit ---
    #  Merge Retained (D_r) and Poisoned Forget (D_f') datasets
    full_training_data = train_episodes_r + unlearning_episodes
    print(f"Merged dataset size for training: {len(full_training_data)} episodes "
          f"({len(train_episodes_r)} genuine + {len(unlearning_episodes)} poisoned)")

    #  Prepare log directory
    unlearn_dir = str.replace(args.model_to_unlearn_dir, "Fully_trained", "Unlearned")
    ratios_str = "retained_ratios"
    for r in args.retained_ratios:
        ratios_str = ratios_str + "_" + str(r)
    #  Modified logdir suffix to reflect 'simple_fit' if desired, or keep as is.
    if args.shuffle==1:
        logdir = unlearn_dir + "/" +"unlearn_seed_"+str(args.seed)+"/"+ ratios_str + "/" + args.logdir_suffix  + "_simple_fit/step_" + str(args.total_steps)
    else:
        logdir = unlearn_dir + "/" +"no_shuffle_unlearn_seed_"+str(args.seed)+"/"+ ratios_str + "/" + args.logdir_suffix  + "_simple_fit/step_" + str(args.total_steps)
    fit_kwargs = {}
    if cost_controller is not None:
        logdir = cost_controller.training_log_dir
        fit_eval_episodes = None
        fit_scorers = None
        fit_kwargs = {"experiment_name": "training", "with_timestamp": False}
        wrap_standard_updates(algorithm, cost_controller)
    else:
        fit_eval_episodes = eval_episodes_r
        fit_scorers = {
            'environment': evaluate_on_environment(eval_env),
            'td_error': td_error_scorer,
            'discounted_advantage': discounted_sum_of_advantage_scorer,
            'value_scale': average_value_estimation_scorer,
        }

    # Cost mode writes checkpoints only through CostMeasurementController.

    print(f"--- Starting Standard Fit (Fine-tuning) on merged dataset ---")
    print(f"Total training steps: {args.total_steps}")

    #  Use standard .fit() instead of unlearningfit_stage1
    try:
        algorithm.fit(
            full_training_data,
            eval_episodes=fit_eval_episodes,
            n_steps=args.total_steps,
            n_steps_per_epoch=args.n_steps_per_epoch,
            logdir=logdir,
            scorers=fit_scorers,
            **fit_kwargs
        )
        if cost_controller is not None:
            cost_controller.finish_if_incomplete()
    except CostMeasurementStop as stop:
        print(f"Cost measurement stopped normally: {stop}")
    except CostMeasurementFailed:
        raise
    except Exception as error:
        if cost_controller is not None:
            cost_controller.fail(error)
        raise
    print(f"--- Finished Random Reward Fine-tuning. Model saved to {logdir} ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #  Common args
    parser.add_argument('--dataset', type=str, default='pointmaze')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--algo', type=str, default='CQL')
    parser.add_argument('--datasets', type=str, nargs='+', default=['large-dense-v2', 'medium-dense-v2'])
    parser.add_argument('--model-to-unlearn-dir', type=str, required=True)
    parser.add_argument('--retained-ratios', type=float, nargs='+', required=True)
    parser.add_argument('--total-steps', type=int, default=100000, help="Total steps for fine-tuning (.fit)")
    parser.add_argument('--n-steps-per-epoch', type=int, default=50000)
    parser.add_argument('--logdir-suffix', type=str, default="random_Rewarding")

    #  Args that might be unused now but kept for compatibility with your existing run scripts
    parser.add_argument('--unlearn-steps-per-epoch', type=int, default=5000, help="(Unused in simple .fit)")
    parser.add_argument('--unlearn-freq', type=int, default=5000, help="(Unused in simple .fit)")
    parser.add_argument('--alpha', type=float, default=1.0, help="(Unused in simple .fit)")
    parser.add_argument('--shuffle', type=int, default=1,
                        help="Whether apply shuffle in splitting dataset: D_f and D_r."
                        "the original TrajDeleter paper set as False: only pick the components in the last of expert-v0 as D_f."
                        "Note: we still use shuffle = True in iterating the dataset remained.")
    parser.add_argument('--cost-measurement-config', type=str, default=None,
                        help="Internal JSON controller configuration for cost measurement.")
    add_quadx_eval_args(parser)
    args = parser.parse_args()
    normalize_quadx_eval_args(parser, args)
    if args.n_steps_per_epoch > args.total_steps:
        args.n_steps_per_epoch = args.total_steps
        print("args.n_steps_per_epoch is set lower due to unlearning budget." )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args)