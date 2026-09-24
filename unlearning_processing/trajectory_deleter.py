# unlearning_script_modular.py
#
#  This script implements the TrajDeleter logic by
# calling the author's modified functions (unlearningfit_stage1,
# unlearningfit_stage2) directly, as defined in their provided
# base(d3rlpy).py and cql.py files.
#
# It merges the data loading from fully_training.py with the
# training logic from mujoco_trajdeleter.py.

import sys
import os
import glob
import inspect
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
    wrap_negative_reward_updates,
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
    #  Use data loading logic from fully_training.py
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
        
    if hasattr(eval_env, 'action_space'):
        eval_env.action_space.seed(args.seed)
    if hasattr(eval_env, 'observation_space'):
        eval_env.observation_space.seed(args.seed)
    d3rlpy.seed(args.seed)

    # --- 2. Load Models (Original and Unlearning-Target) ---
    #  This follows mujoco_trajdeleter.py logic

    params_path = os.path.join(args.model_to_unlearn_dir, 'params.json')
    if not os.path.exists(params_path):
        print(params_path, ":" + str(os.path.exists(params_path)))
        raise FileNotFoundError(f"params.json not found in {args.model_to_unlearn_dir}")
    model_path, model_step = resolve_model_checkpoint(args.model_to_unlearn_dir)

    with open(params_path, 'r') as f:
        params = json.load(f)
    algo_name = params.get("algorithm", args.algo)

    #  Dynamically get the Algorithm class from d3rlpy.algos
    #  **ASSUMPTION**: User has replaced d3rlpy.algos.cql with the modified cql.py, etc.
    try:
        AlgoClass = getattr(d3rlpy.algos, algo_name)
    except AttributeError:
        print(f"Error: Algorithm '{algo_name}' not found in d3rlpy.algos.")
        print(
            "Please ensure your --algo argument (or params.json) matches the author's modified scripts (e.g., CQL, BCQ).")
        return

    print(f"--- Loading ORIGINAL model (ori_algorithm) ---")
    ori_algorithm = AlgoClass.from_json(params_path, use_gpu=args.gpu)
    # --- MODIFICATION ---
    # ori_algorithm.load_model(model_path) # <-- Buggy line removed
    load_model_with_fix(ori_algorithm, model_path, args.gpu) # <-- FIX
    # --- END MODIFICATION ---
    ori_algorithm.set_grad_step(0)  #  Reset grad step, though it will be frozen

    print(f"--- Loading UNLEARNING model (algorithm) ---")
    algorithm = AlgoClass.from_json(params_path, use_gpu=args.gpu)
    # --- MODIFICATION ---
    # algorithm.load_model(model_path) # <-- Buggy line removed
    load_model_with_fix(algorithm, model_path, args.gpu) # <-- FIX
    # --- END MODIFICATION ---

    if model_step is not None:
        algorithm.set_grad_step(model_step)
        print(f"Resuming training from step {model_step}")

    # --- 3. Step 1 (TrajDeleter): Prepare Datasets ---

    #  Split Retained Set (D_r) into train and validation
    #  We use a 10% validation split, similar to fully_training.py
    train_episodes_r, eval_episodes_r = train_test_split(
        dataset_required,
        random_state=args.seed,
        test_size=0.01,
        shuffle=True  #  Keep shuffle=False for consistency
    )
    print(f"Retained set (D_r) split into: {len(train_episodes_r)} train / {len(eval_episodes_r)} eval episodes.")

    #  Get Forget Set (D_f)
    unlearning_episodes = dataset_remained.episodes

#  Get Forget Set (D_f) by iterating, poisoning, and manually constructing EpisodeNew
    #  We cannot use dataset_remained.episodes as it creates read-only objects
    
    print(f"--- Manually loading and poisoning {len(dataset_remained)} 'forget' episodes (D_f) ---")

    #  Get d3rlpy-compatible shapes from the merged dataset
    d3rlpy_obs_shape = dataset_remained.get_observation_shape()
    d3rlpy_act_size = dataset_remained.get_action_size()
    is_dict_obs = dataset_remained._is_dict_obs      #  Access internal flag
    obs_key_order = dataset_remained._obs_key_order  #  Access internal key order
    
    unlearning_episodes = [] #  This will be our List[EpisodeNew]
    
    #  Iterate through the underlying datasets (MinariDataset or InMemoryMinariDataset)
    for ds in dataset_remained._datasets:
        #  Iterate through the raw MinariEpisodeData
        for ep_data in ds: 
            
            # 1. POISON THE DATA (by creating a new array)
            #  We cannot modify ep_data.rewards due to FrozenInstanceError.
            #  Instead, create a NEW array containing the poisoned values.
            poisoned_rewards = -ep_data.rewards
            
            # 2. MANUALLY REPLICATE THE LOGIC...
            
            #  Add source info
            new_infos = ep_data.infos.copy()
            new_infos["source_dataset_id"] = ds.id
            
            #  Create a new MinariEpisodeData instance to hold the info
            #  Pass the NEWLY CREATED poisoned_rewards array.
            episode_data_with_info = MinariEpisodeData(
                id=ep_data.id,
                observations=ep_data.observations,
                actions=ep_data.actions,
                rewards=poisoned_rewards, # 
                terminations=ep_data.terminations,
                truncations=ep_data.truncations,
                infos=new_infos
            )
            
            #  Process observations
            if is_dict_obs:
                d3rlpy_obs_data = _concatenate_observations(
                    episode_data_with_info.observations,
                    obs_key_order
                )
            else:
                d3rlpy_obs_data = episode_data_with_info.observations.astype(np.float32)

            # 3. CONSTRUCT THE EpisodeNew OBJECT
            #  This will now use the POISONED rewards when calling the parent constructor
            ep_new = EpisodeNew(
                episode_data=episode_data_with_info,
                observation_shape=d3rlpy_obs_shape,
                action_size=d3rlpy_act_size,
                d3rlpy_observations=d3rlpy_obs_data
            )
            unlearning_episodes.append(ep_new)

    if not unlearning_episodes:
        print("Warning: 'Forget' dataset (D_f) is empty. Phase 1 will only train on D_r.")
    else:
        print(f"Poisoning complete. Loaded {len(unlearning_episodes)} poisoned episodes.")

    # --- 4. Step 2 (TrajDeleter): Phase 1 "Forgetting" ---
    print("--- Starting Phase 1: Forgetting ---")

    #  Replicate step calculation from mujoco_trajdeleter.py
    remain_step_per_epoch = args.n_steps_per_epoch
    unlearn_step_per_epoch = args.unlearn_steps_per_epoch
    unlearn_freq = args.unlearn_freq

    #  Calculate steps based on total desired Phase 1 steps
    #  This logic is from
    remain_step = int(args.phase1_total_steps / (1 + unlearn_step_per_epoch / unlearn_freq))
    unlearn_step = int(remain_step / unlearn_freq * unlearn_step_per_epoch)

    print(f"Phase 1 Total Steps: {args.phase1_total_steps}")
    print(f"  -> Calculated Retained Steps: {remain_step}")
    print(f"  -> Calculated Unlearned Steps: {unlearn_step}")
    print(f"  (Using unlearn_freq={unlearn_freq}, unlearn_steps_per_epoch={unlearn_step_per_epoch})")
    unlearn_dir = str.replace(args.model_to_unlearn_dir,"Fully_trained", "Unlearned")
    ratios_str = "retained_ratios"
    for r in args.retained_ratios:
        ratios_str=ratios_str+"_"+str(r)
    if args.shuffle==1:
        logdir_phase1 = unlearn_dir+"/" +"unlearn_seed_"+str(args.seed)+"/"+ratios_str+"/"+args.logdir_suffix +"/phase1_"+str(args.phase1_total_steps)
    else:
        logdir_phase1 = unlearn_dir+"/" +"no_shuffle_unlearn_seed_"+str(args.seed)+"/"+ratios_str+"/"+args.logdir_suffix +"/phase1_"+str(args.phase1_total_steps)
    stage1_kwargs = {}
    start_with_unlearn = False
    stage1_remain_dataset = train_episodes_r
    stage1_unlearn_dataset = unlearning_episodes
    if cost_controller is not None:
        # Cost measurement uses the controller cadence for dataset switching.
        # maximum_steps is the total logical-update budget, not one block size.
        switch_interval = cost_controller.interval
        remain_step_per_epoch = switch_interval
        unlearn_step_per_epoch = switch_interval
        unlearn_freq = switch_interval
        remain_step = args.phase1_total_steps
        unlearn_step = args.phase1_total_steps
        start_with_unlearn = True
        logdir_phase1 = cost_controller.training_log_dir
        stage1_eval_episodes = None
        stage1_scorers = None
        stage1_kwargs = {"experiment_name": "training", "with_timestamp": False,
                         "save_interval": args.phase1_total_steps + 1}
        wrap_negative_reward_updates(algorithm, cost_controller, ori_algorithm)

        # Prefer the explicit stage1 switch supported by the repository copy.
        # Older installed d3rlpy copies do not accept this keyword. For those
        # copies, swap the stage1 dataset inputs and dispatch methods so that
        # the original API executes D_f first without changing its signature.
        stage1_signature = inspect.signature(algorithm.unlearningfit_stage1)
        if "start_with_unlearn" in stage1_signature.parameters:
            start_with_unlearn = True
        else:
            start_with_unlearn = False
            stage1_remain_dataset = unlearning_episodes
            stage1_unlearn_dataset = train_episodes_r
            wrapped_remain_update = algorithm.update_stage1_remain
            wrapped_unlearn_update = algorithm.update_stage1_unlearn

            def update_for_first_forget_block(batch):
                return wrapped_unlearn_update(batch, args.alpha)

            def update_for_retain_block(batch, alpha):
                return wrapped_remain_update(batch)

            algorithm.update_stage1_remain = update_for_first_forget_block
            algorithm.update_stage1_unlearn = update_for_retain_block
    else:
        stage1_eval_episodes = eval_episodes_r
        stage1_scorers = {
            'environment': evaluate_on_environment(eval_env),
            'td_error': td_error_scorer,
            'discounted_advantage': discounted_sum_of_advantage_scorer,
            'value_scale': average_value_estimation_scorer,
        }
    #  This is the key call, using the author's modified function
    # 
    #  This matches the call in
    try:
        algorithm.unlearningfit_stage1(
            remain_dataset=stage1_remain_dataset,
            unlearn_dataset=stage1_unlearn_dataset,
            remain_step_per_epoch=remain_step_per_epoch,
            unlearn_step_per_epoch=unlearn_step_per_epoch,
            unlearn_freq=unlearn_freq,
            alpha=args.alpha,
            eval_episodes=stage1_eval_episodes,
            remain_steps=remain_step,
            unlearn_steps=unlearn_step,
            **({"start_with_unlearn": True} if start_with_unlearn else {}),
            logdir=logdir_phase1,
            scorers=stage1_scorers,
            **stage1_kwargs
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
    print(f"--- Finished Phase 1. Model saved to {logdir_phase1} ---")
    if args.phase2_total_steps >0:
    # --- 5. Step 3 (TrajDeleter): Phase 2 "Convergence" ---
        print("--- Starting Phase 2: Convergence ---")
        if args.shuffle==1:
            logdir_phase2 = unlearn_dir+"/" +"unlearn_seed_"+str(args.seed)+"/"+ratios_str + "/"+args.logdir_suffix +"/phase2_"+str(args.phase2_total_steps)+"_phase1_"+str(args.phase1_total_steps)
        else:
            logdir_phase2 = unlearn_dir+"/" +"no_shuffle_unlearn_seed_"+str(args.seed)+"/"+ratios_str + "/"+args.logdir_suffix +"/phase2_"+str(args.phase2_total_steps)+"_phase1_"+str(args.phase1_total_steps)
        #  This is the second key call, using the author's modified function
        # 
        #  This matches the call in
        algorithm.unlearningfit_stage2(
            remain_dataset=train_episodes_r,  #  Only on retained data
            ori_algo=ori_algorithm,  #  Pass the frozen original model
            eval_episodes=eval_episodes_r,
            n_steps=args.phase2_total_steps,
            n_steps_per_epoch=args.n_steps_per_epoch,
            logdir=logdir_phase2,
            scorers={
                'environment': evaluate_on_environment(eval_env),
                'td_error': td_error_scorer,
                'discounted_advantage': discounted_sum_of_advantage_scorer,
                'value_scale': average_value_estimation_scorer
            }
        )

        print(f"--- Finished Phase 2. Final unlearned model saved to {logdir_phase2} ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #  --- Args from fully_training.py (for data loading)
    parser.add_argument('--dataset', type=str, default='pointmaze')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--algo', type=str, default='CQL',
                        help="Algorithm name (e.g., CQL, BCQ). Must match author's modified scripts.")
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',  # This allows it to accept one or more space-separated values
        default=['large-dense-v2', 'medium-dense-v2'],  # This is your original default
        help="List of sub-dataset names to load (e.g., 'large-dense-v2', 'medium-dense-v2')."
    )
    #  --- Args for Unlearning (from mujoco_trajdeleter.py)
    parser.add_argument('--model-to-unlearn-dir', type=str, required=True,
                        help="Path to log dir of the *fully-trained* model (e.g., Fully_trained/pointmaze/CQL_...)")

    parser.add_argument('--retained-ratios', type=float, nargs='+', required=True,
                        help="List of ratios (0.0 to 1.0) to *keep* (D_r). "
                             "The rest becomes the 'forget' set (D_f).")

    parser.add_argument('--phase1-total-steps', type=int, default=60000,
                        help="Total steps for Phase 1 (e.g., stage1_step in author code)")

    parser.add_argument('--phase2-total-steps', type=int, default=40000,
                        help="Total steps for Phase 2 (e.g., stage2_step in author code)")

    parser.add_argument('--n-steps-per-epoch', type=int, default=5000,
                        help="Steps per epoch for retained data (Phase 1) and all data (Phase 2)")

    parser.add_argument('--unlearn-steps-per-epoch', type=int, default=5000,
                        help="Steps per epoch for unlearned data (Phase 1)")

    parser.add_argument('--unlearn-freq', type=int, default=5000,
                        help="Frequency (in retained steps) to run an unlearning epoch (Phase 1)")

    parser.add_argument('--alpha', type=float, default=1.0,
                        help="Weight for unlearning loss (lamda in author code)")

    parser.add_argument('--logdir-suffix', type=str, default="trajDeleter",
                        help="Suffix to append to the original model's log directory for saving")
    parser.add_argument('--shuffle', type=int, default=1,
                        help="Whether apply shuffle in splitting dataset: D_f and D_r."
                        "the original TrajDeleter paper set as False: only pick the components in the last of expert-v0 as D_f."
                        "Note: we still use shuffle = True in iterating the dataset remained.")
    parser.add_argument('--cost-measurement-config', type=str, default=None,
                        help="Internal JSON controller configuration for cost measurement.")
    add_quadx_eval_args(parser)
    args = parser.parse_args()
    normalize_quadx_eval_args(parser, args)
    if args.cost_measurement_config is None and ((args.phase1_total_steps<args.n_steps_per_epoch) or (args.phase2_total_steps>0 and args.phase2_total_steps<args.n_steps_per_epoch)):
        args.n_steps_per_epoch = min(args.phase1_total_steps, args.phase2_total_steps)
        args.unlearn_steps_per_epoch = min(args.phase1_total_steps, args.phase2_total_steps)
        args.unlearn_freq = min(args.phase1_total_steps, args.phase2_total_steps)
    elif args.cost_measurement_config is None and args.phase1_total_steps<args.n_steps_per_epoch:
        args.n_steps_per_epoch = args.phase1_total_steps
        args.unlearn_steps_per_epoch = args.phase1_total_steps
        args.unlearn_steps_per_epoch = args.phase1_total_steps
    if args.cost_measurement_config is None and args.phase1_total_steps<=args.unlearn_freq*10:
        args.unlearn_freq = 1000
        args.unlearn_steps_per_epoch = 1000
        args.n_steps_per_epoch = 1000
    if args.phase1_total_steps<=2000:
        args.unlearn_freq = 200
        args.unlearn_steps_per_epoch = 200
        args.n_steps_per_epoch = 200
        print('adjust the unlearn_freq, unlearn_steps_per_epoch and n_steps_per_epoch to 200 for phase1_total_steps<=2000')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    #  Check for dependencies
    if not hasattr(d3rlpy.algos.AlgoBase, "unlearningfit_stage1"):
        print("=" * 50)
        print("ERROR: d3rlpy.algos.AlgoBase does not have 'unlearningfit_stage1'.")
        print("This script requires you to use the author's modified d3rlpy files.")
        print("Please replace your d3rlpy installation's base.py with the provided 'base(d3rlpy).py'.")
        print("=" * 50)
        sys.exit(1)

    main(args)

    # --- Example Command ---
    # 
    # This command will:
    # 1. Load the model from 'Fully_trained/pointmaze/CQL_20251029220942'
    # 2. Define D_r as 80% of large-dense, 50% of umaze-dense, 100% of medium-dense
    # 3. Define D_f as the other 20% of large-dense, 50% of umaze-dense, 0% of medium-dense
    # 4. Call algorithm.unlearningfit_stage1(D_r_train, D_f_poisoned, ...)
    # 5. Call algorithm.unlearningfit_stage2(D_r_train, ori_algo, ...)
    #
    # python unlearning_script_modular.py \
    #   --model-to-unlearn-dir Fully_trained/pointmaze/CQL_20251029220942 \
    #   --retained-ratios 0.8 0.5 1.0 \
    #   --algo CQL \
    #   --phase1-total-steps 6000 \
    #   --phase2-total-steps 4000 \
    #   --alpha 1.0 \
    #   --seed 42 \
    #   --gpu 0
    # python trajectory_deleter.py --model-to-unlearn-dir ../Offline_RL_processing/Fully_trained/pointmaze/large-dense-v2_umaze-dense-v2_medium-dense-v2/CQL_seed42_40Epoch --retained-ratios 1.0 1.0 0.0 --algo CQL --phase1-total-steps 6000 --phase2-total-steps 4000 --alpha 1.0 --seed 42 --gpu 0