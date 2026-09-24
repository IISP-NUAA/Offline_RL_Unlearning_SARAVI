# evaluate_model.py
#
#  This script evaluates a pre-trained d3rlpy model
#  in a specified Minari/Gymnasium environment.
#  It reuses the exact wrapper and model loading fix
#  from your provided 'fully_training.py' and
#  'trajectory_deleter.py' scripts to ensure compatibility.

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Union, Callable, Optional
import numpy as np
import torch
import gymnasium as gym
import d3rlpy
import minari

from d3rlpy.base import LearnableBase
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.torch_utility import set_state_dict
from gymnasium.spaces import Box
from gymnasium.wrappers import TimeLimit #  ADDED: Needed to force episode termination

# # ---  Dependency Import ---
# #  This assumes 'minari_dataset_processing' is one directory up
# #  or on the PYTHONPATH, as in your 'fully_training.py'.
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# try:
#     from minari_dataset_processing.merged_data_with_episode import _get_concatenated_obs_space
# except ImportError:
#     print("Error: Could not import '_get_concatenated_obs_space' from 'minari_dataset_processing'.")
#     print("Please ensure this script is placed correctly relative to your utility folder,")
#     print("or that 'minari_dataset_processing' is on your PYTHONPATH.")
#     sys.exit(1)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from minari_dataset_processing.utility import FlattenDictObsWrapper, load_model_with_fix
except ImportError:
    print("Please ensure this script is placed correctly relative to your utility folder,")
    print("or that 'minari_dataset_processing' is on your PYTHONPATH.")
    sys.exit(1)


def clean_path_parts(path_value: str) -> List[str]:
    return [
        part
        for part in str(path_value).replace("\\", "/").split("/")
        if part and part not in {".", ".."}
    ]


def hierarchical_model_path(model_dir: str, prefix: str = "name_") -> Path:
    model_dir = str(model_dir).replace("PLASWithPerturbation", "PLASP")
    model_dir = model_dir.replace(".", "")
    parts = clean_path_parts(model_dir) or ["unknown_model"]

    output_path = Path(prefix)
    for part in parts:
        output_path = output_path / part
    return output_path


def task_name_from_env_id(env_id: str) -> str:
    id_components = str(env_id).split("/")
    if len(id_components) >= 2:
        return id_components[0] + "_" + id_components[1]
    return str(env_id).replace("/", "_").replace("-", "_")


def evaluation_output_path(args, env_id: str) -> Path:
    task_name_safe = task_name_from_env_id(env_id)
    output_dir = Path(args.save_dir) / task_name_safe / hierarchical_model_path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"results_{task_name_safe}_seed{args.seed}.json"


# ---  Main Evaluation Function ---

def main(args):
    print(f"--- Starting Evaluation ---")
    d3rlpy.seed(args.seed)

    for env_id in args.env_id_list:
                # --- 1. Load Environment ---
        print(f"Loading environment from Minari dataset: {env_id}")
        try:
        #  Load dataset to recover the environment
            dataset = minari.load_dataset(env_id)
        except Exception as e:
            print(f"Error loading Minari dataset: {e}")
            print(f"Please ensure the dataset ID is correct (e.g., 'D4RL/pointmaze/large-dense-v2')")
            print(f"And that you have access to it (e.g., `huggingface-cli login`)")
            return
    
        if args.eval_env_flag == 0:
            eval_env_original = dataset.recover_environment(eval_env=True)
        else:
            eval_env_original = dataset.recover_environment(eval_env=False)
        
        #  --- FIX FOR HANGING EPISODES ---
        #  Explicitly wrap with TimeLimit. Just setting .max_episode_steps 
        #  as an attribute often doesn't enforce it in Gymnasium.
        max_steps = 2000
        print(f"Enforcing TimeLimit with max_episode_steps={max_steps}")
        eval_env_with_limit = TimeLimit(eval_env_original, max_episode_steps=max_steps)

        # --- 2. Wrap Environment ---
        #  Use the compatibility wrapper from fully_training.py
        print("Applying Gym-API compatibility wrapper (FlattenDictObsWrapper).")

        #  Check if the task is antmaze to apply special observation transform
        is_antmaze_task = "antmaze" in env_id.lower()
        if is_antmaze_task:
            print("[Main] Detected 'antmaze' task. Wrapper will apply special observation transform.")

        #  Pass the time-limited env to our flattening wrapper
        eval_env = FlattenDictObsWrapper(eval_env_with_limit, is_antmaze=is_antmaze_task)

        # --- 3. Load Model ---
        params_path = os.path.join(args.model_dir, 'params.json')
        model_path = os.path.join(args.model_dir, args.model_filename)

        if not os.path.exists(params_path):
            raise FileNotFoundError(f"params.json not found in {args.model_dir}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"Loading params from: {params_path}")
        with open(params_path, 'r') as f:
            params = json.load(f)

        algo_name_key = "algorithm"
        if algo_name_key not in params:
            algo_name_key = "type"
        
        algo_name = params.get(algo_name_key)
        if not algo_name:
            raise ValueError(f"Could not find 'algorithm' or 'type' key in {params_path}")
        

        try:
            AlgoClass = getattr(d3rlpy.algos, algo_name)
        except AttributeError:
            raise ValueError(f"Algorithm '{algo_name}' not found in d3rlpy.algos")

        print(f"Initializing algorithm '{algo_name}' (GPU: {args.gpu})")
        algorithm = AlgoClass.from_json(params_path, use_gpu=args.gpu)

        load_model_with_fix(algorithm, model_path, args.gpu)

        # --- 4. Run Evaluation ---
        print(f"Evaluating model on {args.n_trials} trials... (Seed: {args.seed})")

        scorer = evaluate_on_environment(eval_env, n_trials=args.n_trials)
        mean_score = scorer(algorithm)

        print("\n--- Evaluation Results ---")
        print(f"  Mean Score: {mean_score:.4f}")
        print("--------------------------\n")

        # --- 5. Export Results ---
        results_data = {
            'model_dir': args.model_dir,
            'model_filename': args.model_filename,
            'env_id': env_id,
            'n_trials': args.n_trials,
            'seed': args.seed,
            'mean_score': float(mean_score),
        }

        output_path = evaluation_output_path(args, env_id)

        try:
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write("env_name: " +  env_id+ " \n")
                json.dump(results_data, f, indent=2)
            print(f"Results successfully saved to: {output_path}")
        except Exception as e:
            print(f"Error saving results to JSON: {e}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained d3rlpy model.")
    parser.add_argument('--model-dir', type=str, required=True,
                        help=" Path to the log directory containing params.json")
    parser.add_argument('--model-filename', type=str, default='model.pt',
                        help=" Name of the model file to load")
    parser.add_argument('--env-id-list', type=str, 
        nargs='+',  # This allows it to accept one or more space-separated values
        default=['D4RL/pointmaze/umaze-dense-v2', 'D4RL/pointmaze/large-dense-v2'],  # This is your original default
        help="List of dataset to load (e.g., 'large-dense-v2', 'medium-dense-v2').")
    parser.add_argument('--n-trials', type=int, default=10,
                        help=" Number of episodes to run for evaluation")
    parser.add_argument('--gpu', type=int, default=0,
                        help=" GPU ID to use (-1 for CPU)")
    parser.add_argument('--seed', type=int, default=0,
                        help=" Random seed for evaluation reproducibility")
    parser.add_argument('--save_dir', type=str, default='evaluation_results')
    parser.add_argument('--eval_env_flag', type=int, default=0,
                        help="0 means set eval env = True, 1 means set eval env = False")
    args = parser.parse_args()
    main(args)