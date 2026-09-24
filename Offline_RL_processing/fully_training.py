import sys
import os
import glob
import re
from typing import List, Dict, Set, Optional, Callable, Tuple, Union
import json
import d3rlpy
import gymnasium as gym
import numpy as np
from d3rlpy.base import LearnableBase
from gymnasium.spaces import Box
from d3rlpy.datasets import get_atari
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split
from d3rlpy.dataset import Episode
import minari
import argparse

# --- NEW IMPORTS for the fix ---
import torch
from d3rlpy.torch_utility import set_state_dict

# --- END NEW IMPORTS ---


sys.path.insert(0, sys.path[0] + "/../")
try:
    from minari_dataset_processing.merged_data_with_episode import MergedMinariDataset, EpisodeNew
    from minari_dataset_processing.merged_data_with_episode import _get_concatenated_obs_space
    from minari_dataset_processing.merged_data_with_episode import InMemoryMinariDataset
    from minari_dataset_processing.merged_data_with_episode import _concatenate_observations
    from minari_dataset_processing.utility import FlattenDictObsWrapper, load_merged_dataset, load_model_with_fix,parse_dataset_dir
except ImportError as e:
    print(e)
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




def build_eval_environment(task, dataset_required):
    """Build the legacy scorer environment; cost mode never calls this."""
    print("Setting up evaluation environment...")
    is_antmaze_task = task == "antmaze"
    if "quadrupedal" in task:
        print("Detected Quadrupedal task. Manually creating environment via Adapter.")
        eval_env_original = MetaGymToGymnasiumAdapter(task_name="stairstair")
    else:
        try:
            eval_env_original = dataset_required._datasets[-1].recover_environment(
                eval_env=True
            )
        except Exception as error:
            print(
                f"Warning: recover_environment failed ({error}). "
                "Falling back to None."
            )
            eval_env_original = None
    if eval_env_original is None:
        return None, is_antmaze_task
    eval_env_original.max_episode_steps = 10000
    print("Applying Gym-API compatibility wrapper (FlattenDictObsWrapper) to eval_env.")
    return (
        FlattenDictObsWrapper(
            eval_env_original,
            is_antmaze=is_antmaze_task,
            key_order=None if is_antmaze_task else dataset_required.observation_paths,
        ),
        is_antmaze_task,
    )


def main(args):
    # (Setup task, datasets, eval_env... all unchanged)
    task = args.dataset

    # --- MODIFICATION: Get datasets from args ---
    # The hardcoded list has been removed and replaced by the command-line argument
    datasets = args.datasets
    print(f"--- Loading datasets for task '{task}': {datasets} ---")
    # --- END MODIFICATION ---

    # --- MODIFIED: Handle Ratios ---
    ratios = args.ratios
    if ratios is None:
        ratios = [1.0] * len(datasets)  # Default to using 100% of all datasets
    elif len(ratios) != len(datasets):
        raise ValueError(
            f"Must provide {len(datasets)} ratios for datasets {datasets}, but got {len(ratios)}."
        )

    dirs = parse_dataset_dir(task, datasets)
    # --- MODIFIED: Load split datasets ---
    # Load the datasets, splitting them based on ratios and seed
    # dataset_required is the "Retained Set" (D_r)
    # dataset_remained is the "Forget Set" (D_f)

    dataset_required, dataset_remained = load_merged_dataset(dirs, ratios, args.seed, shuffle=(args.shuffle==1))
    cost_controller = CostMeasurementController.from_path(
        args.cost_measurement_config,
        dataset_required,
        dataset_remained,
    )

    # dataset_remained (D_f) is INTENTIONALLY NOT USED from this point forward.

    # Use the "retained" dataset list for env recovery
    # This will be either a MinariDataset or our InMemoryMinariDataset wrapper,
    # both of which support .recover_environment
    if not dataset_required._datasets:
        print("Error: The 'required' (retained) dataset is empty. Cannot recover environment or train.")
        print("This happens if all ratios are set to 0.0.")
        if cost_controller is not None:
            cost_controller.fail(ValueError("The retained dataset is empty."))
        return

    is_antmaze_task = task == "antmaze"
    eval_env = None
    if cost_controller is None:
        eval_env, is_antmaze_task = build_eval_environment(task, dataset_required)

    d3rlpy.seed(args.seed)
    # --- MODIFICATION: Handle Resuming ---
    # --- vvv NEW LOGIC vvv ---
    # Check if we need to apply the AntMaze preprocessing
    if is_antmaze_task:
        print("--- Applying AntMaze preprocessing (Observation/Reward Transform) to dataset ---")
        # This function converts Minari episodes to d3rlpy Episodes
        dataset_to_split = preprocess_antmaze_to_d3rlpy(dataset_required)
        print(f"Converted {len(dataset_required)} Minari episodes to {len(dataset_to_split)} d3rlpy episodes.")
    else:
        # Use the original Minari dataset wrapper (as iterable of MinariEpisodeData)
        dataset_to_split = dataset_required
    # --- ^^^ END NEW LOGIC ^^^ ---
    start_step = 0
    n_steps_to_run = args.n_steps

    if args.resume_from_logdir:
        print(f"--- Resuming training from {args.resume_from_logdir} ---")

        # 1. Load algorithm configuration from params.json
        params_path = os.path.join(args.resume_from_logdir, 'params.json')
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"params.json not found in {args.resume_from_logdir}")
        print(f"Loading params from: {params_path}")

        with open(params_path, 'r') as f:
            params = json.load(f)
        algo_name = params.get("algorithm", args.algo)

        try:
            AlgoClass = getattr(d3rlpy.algos, algo_name)
            # Initialize algorithm, passing integer GPU ID
            print(f"Initializing algorithm with --gpu {args.gpu}")
            algorithm = AlgoClass.from_json(params_path, use_gpu=args.gpu)
        except AttributeError:
            raise ValueError(f"Algorithm '{algo_name}' not found in d3rlpy.algos")

        # 2. Find latest model
        model_files = glob.glob(os.path.join(args.resume_from_logdir, 'model_*.pt'))
        if not model_files:
            raise FileNotFoundError(f"No model_*.pt files found in {args.resume_from_logdir}")
        step_nums = [int(re.search(r'model_(\d+).pt', f).group(1)) for f in model_files]
        latest_step = max(step_nums)
        latest_model_path = os.path.join(args.resume_from_logdir, f'model_{latest_step}.pt')

        # --- THIS IS THE FIX ---
        # Manually load the model to bypass the d3rlpy map_location bug

        # (New code)
        load_model_with_fix(algorithm, latest_model_path, args.gpu)

        # We NO LONGER call this buggy function:
        # algorithm.load_model(latest_model_path)
        # --- END FIX ---

        # 3. Set the internal gradient step counter
        start_step = latest_step
        algorithm.set_grad_step(start_step)

        # 4. Calculate remaining steps
        n_steps_to_run = args.n_steps - start_step
        print(f"Resuming from step: {start_step}")
        print(f"Total steps: {args.n_steps} | Remaining steps: {n_steps_to_run}")

        if n_steps_to_run <= 0:
            print(f"Training already completed ({start_step} steps). Exiting.")
            return

    else:
        # --- Original logic for a new run ---
        # (This part is also modified to use args.gpu)
        print("--- Starting new training run ---")
        if args.dataset == "pen-human-v1":
            if args.algo == "CQL":
                algorithm = d3rlpy.algos.CQL(use_gpu=args.gpu)
            # (Add other algos here)
            else:
                raise ValueError(f"Algorithm {args.algo} not configured for new run on {args.dataset}")
        else:
            if not args.base_params:
                raise ValueError("--base_params must be provided for a new run")
            print(f"Loading base params from: {args.base_params}")
            if args.algo == "CQL":
                algorithm = d3rlpy.algos.CQL.from_json(args.base_params, use_gpu=args.gpu)
            elif args.algo == "BCQ":
                algorithm = d3rlpy.algos.BCQ.from_json(args.base_params, use_gpu=args.gpu)
            elif args.algo == "BEAR":
                algorithm = d3rlpy.algos.BEAR.from_json(args.base_params, use_gpu=args.gpu)
            elif args.algo.upper() == "TD3PLUSBC":
                algorithm = d3rlpy.algos.TD3PlusBC.from_json(args.base_params, use_gpu=args.gpu)
            elif args.algo == "IQL":
                algorithm = d3rlpy.algos.IQL.from_json(args.base_params, use_gpu=args.gpu)
            elif args.algo == "PLASP":
                algorithm = d3rlpy.algos.PLASWithPerturbation.from_json(args.base_params, use_gpu=args.gpu)
            elif args.algo == "AWAC":
                algorithm = d3rlpy.algos.AWAC.from_json(args.base_params, use_gpu=args.gpu)
            else:
                print("No available algorithms specified!")
                return

    # --- MODIFIED: Split the 'retained' dataset (D_r) for validation ---
    # We now split dataset_required (D_r) to get our training and validation sets.
    # dataset_remained (D_f) is not used.
    print(f"Splitting 'required' (retained) dataset (D_r) into 99% train / 1% validation...")
    train_episodes, test_episodes = train_test_split(dataset_required,
                                                     random_state=args.seed,
                                                     test_size=0.01,
                                                     shuffle=True)

    print(f"Final training episodes (from D_r): {len(train_episodes)}")
    print(f"Final validation episodes (from D_r): {len(test_episodes)}")
    # --- END MODIFICATION ---
    datasets_str = "_".join(datasets)
    ratios_str = "learn_ratios"
    for r in ratios:
        ratios_str=ratios_str+"_"+str(r)
    if args.shuffle==1:
        logdir=str(args.type)+"/"+str(args.dataset)+'/'+datasets_str+'/'+'seed_'+str(args.seed)+"_steps_"+str(args.n_steps)+'/'+ratios_str
    else:
        logdir=str(args.type)+"/"+str(args.dataset)+'/'+datasets_str+'/'+"no_shuffle_"+'seed_'+str(args.seed)+"_steps_"+str(args.n_steps)+'/'+ratios_str
    fit_kwargs = {}
    if cost_controller is not None:
        logdir = cost_controller.training_log_dir
        fit_eval_episodes = None
        fit_scorers = None
        fit_steps_per_epoch = args.n_steps
        fit_kwargs = {"experiment_name": "training", "with_timestamp": False}
        wrap_standard_updates(algorithm, cost_controller)
    else:
        fit_eval_episodes = test_episodes
        fit_steps_per_epoch = 100000
        fit_scorers = {
            'environment': evaluate_on_environment(eval_env),
            'td_error': td_error_scorer,
            'discounted_advantage': discounted_sum_of_advantage_scorer,
            'value_scale': average_value_estimation_scorer,
        }
        fit_kwargs = {"experiment_name": args.algo, "tensorboard_dir": '../results_tensorboard'}
    try:
        algorithm.fit(
            train_episodes,
            eval_episodes=fit_eval_episodes,
            n_steps=n_steps_to_run,
            n_steps_per_epoch=fit_steps_per_epoch,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pointmaze')  # antmaze, pointmaze
    parser.add_argument('--type', type=str, default='Fully_trained')
    # --- MODIFICATION: Added the --datasets argument ---
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',  # This allows it to accept one or more space-separated values
        default=['large-dense-v2', 'medium-dense-v2'],  # This is your original default
        help="List of sub-dataset names to load (e.g., 'large-dense-v2', 'medium-dense-v2')."
    )
    # --- END MODIFICATION ---

    parser.add_argument('--base-params', type=str, default='./params/cql_pointmaze_params.json',
                        help="Path to base params.json for new runs")
    parser.add_argument('--algo', type=str, default='CQL')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n-steps', type=int, default=2000000, help="Total steps to train for")
    parser.add_argument('--resume-from-logdir', type=str, default=None,
                        help="Path to log dir to resume (e.g., Fully_trained/antmaze/CQL_...)")

    # --- vvv Add new arguments here vvv ---

    # NEW argument for dataset ratios
    parser.add_argument('--ratios', type=float, nargs='+', default=None,
                        help="List of ratios (0.0 to 1.0) for each sub-dataset. "
                             "Order must match the hardcoded 'datasets' list in main(). "
                             "If None, defaults to 1.0 for all.")

    parser.add_argument('--shuffle', type=int, default=1,
                        help="Whether apply shuffle in splitting dataset: D_f and D_r."
                        "the original TrajDeleter paper set as False: only pick the components in the last of expert-v0 as D_f."
                        "Note: we still use shuffle = True in iterating the dataset remained.")
    parser.add_argument('--cost-measurement-config', type=str, default=None,
                        help="Internal JSON controller configuration for cost measurement.")

    # --- ^^^ End of new arguments ^^^ ---

    args = parser.parse_args()
    print(args.shuffle==1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args)
    # python fully_training.py --resume-from-logdir Fully_trained/pointmaze/CQL_20251029220942

    # Example usage of the new feature:
    # Use 80% of large-dense, 50% of umaze-dense, and 100% of medium-dense
    # This forms the "Retained Set" (D_r).
    # This D_r is then split 90/10 for training/validation.
    # The "Forget Set" (D_f) (20% large, 50% umaze, 0% medium) is never used.
    # python fully_training.py --algo CQL --ratios 0.8 0.5 1.0 --seed 42