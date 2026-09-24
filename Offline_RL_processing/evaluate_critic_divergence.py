# evaluate_critic_divergence.py

import sys
import os
import json
import argparse
import glob
import re
import math
from typing import List, Dict

import numpy as np
import torch
import d3rlpy
import minari 
from d3rlpy.base import LearnableBase
from d3rlpy.dataset import Episode, Transition
from tqdm import tqdm 

# --- Project-Specific Imports ---
# This assumes your utility scripts are on the PYTHONPATH
# or in a discoverable location (e.g., 'minari_dataset_processing').
script_dir = os.path.dirname(os.path.abspath(__file__))
util_path = os.path.join(script_dir, "..") 
sys.path.insert(0, util_path)

try:
    from minari_dataset_processing.utility import (
        load_merged_dataset,
        load_model_with_fix
    )
except ImportError as e:
    print(f"Error: Could not import utility functions.")
    print(f"Please ensure 'load_merged_dataset', 'load_model_with_fix',")
    print(f"and 'sklearn' are available.")
    print(f"Details: {e}")
    sys.exit(1)

# ========================================================================
# SECTION 1: HELPER FUNCTIONS
# ========================================================================

def add_prefix_to_keys(stats: Dict[str, float], prefix: str) -> Dict[str, float]:
    """
    Helper to add a prefix (e.g., 'D_f_') to all keys in a dictionary.
    """
    return {f"{prefix}{k}": v for k, v in stats.items()}

def save_results_to_json(results: dict, save_dir: str, filename: str):
    """
    Saves a results dictionary to a JSON file.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    filepath = os.path.join(save_dir, filename)
    
    try:
        def convert(o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4, default=convert)
        print(f"Results saved to {filepath}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")

def compute_critic_diff_stats(
    model1: LearnableBase, 
    model2: LearnableBase, 
    states_array_np: np.ndarray, 
    actions_array_np: np.ndarray,
    batch_size: int, 
    desc_label: str = "Critic Diff Batch"
) -> Dict[str, float]:
    """
    Computes absolute difference of Q-values |Q1(s,a) - Q2(s,a)|.
    """
    if len(states_array_np) == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0, "count": 0}

    print(f"Computing Critic Value Difference over {len(states_array_np)} samples ({desc_label})...")
    
    # Ensure models are in eval mode (though predict_value typically handles this)
    # d3rlpy's predict_value usually handles numpy inputs directly.
    
    all_diffs: List[float] = []
    n_samples = len(states_array_np)
    
    for i in tqdm(range(0, n_samples, batch_size), desc=f"{desc_label}"):
        end_i = min(i + batch_size, n_samples)
        states_batch = states_array_np[i:end_i]
        actions_batch = actions_array_np[i:end_i]
        
        # predict_value returns (batch_size,) numpy array
        # We use standard=False to get the raw Q-value (expectation)
        q1 = model1.predict_value(states_batch, actions_batch)
        q2 = model2.predict_value(states_batch, actions_batch)
        
        # Calculate Absolute Difference (L1 distance between scalars)
        # diff shape: (batch_size,)
        diff = np.abs(q1 - q2)
        
        all_diffs.extend(diff.tolist())
    
    if not all_diffs:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0, "count": 0}
        
    final_diff_np = np.array(all_diffs)
    
    stats_dict = {
        "mean": np.mean(final_diff_np),
        "std": np.std(final_diff_np),
        "median": np.median(final_diff_np),
        "p90": np.percentile(final_diff_np, 90),
        "p99": np.percentile(final_diff_np, 99),
        "max": np.max(final_diff_np),
        "count": len(final_diff_np)
    }
    
    return stats_dict

def load_model_from_dir(model_dir: str, gpu: int, require_model = None) -> LearnableBase:
    """
    Loads a d3rlpy model using the robust logic from utility.
    """
    print(f"Loading model from directory: {model_dir}")
    
    params_path = os.path.join(model_dir, 'params.json')
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"params.json not found in {model_dir}")

    with open(params_path, 'r') as f:
        params = json.load(f)

    # Find algorithm name
    algo_name_key = "algorithm" if "algorithm" in params else "type"
    algo_name = params.get(algo_name_key)
    if not algo_name:
         raise ValueError(f"Could not find 'algorithm' or 'type' key in {params_path}")
    
    try:
        AlgoClass = getattr(d3rlpy.algos, algo_name)
        # Ensure use_gpu is passed correctly
        algo = AlgoClass.from_json(params_path, use_gpu=(gpu >= 0))
    except AttributeError:
        raise ValueError(f"Algorithm '{algo_name}' not found in d3rlpy.algos")

    # Find the latest model checkpoint
    model_files = glob.glob(os.path.join(model_dir, 'model_*.pt'))
    if not model_files:
        model_path_fallback = os.path.join(model_dir, 'model.pt')
        if os.path.exists(model_path_fallback):
            model_files = [model_path_fallback]
        else:
            raise FileNotFoundError(f"No model_*.pt or model.pt files found in {model_dir}")

    if len(model_files) == 1 and model_files[0].endswith('model.pt'):
        latest_model_path = model_files[0]
        print("Found 'model.pt'.")
    else:
        step_nums = [int(re.search(r'model_(\d+).pt', f).group(1)) for f in model_files if
                        re.search(r'model_(\d+).pt', f)]
        if not step_nums:
            latest_model_path = model_files[0]
            print(f"Warning: Could not parse step numbers. Using {os.path.basename(latest_model_path)}")
        else:
            latest_step = max(step_nums)
            latest_model_path = os.path.join(model_dir, f'model_{latest_step}.pt')
            print(f"Found latest checkpoint: model_{latest_step}.pt")
    if require_model is not None:
        latest_model_path = os.path.join(model_dir, f'model_{require_model}.pt')
    load_model_with_fix(algo, latest_model_path, gpu)
    print(f"Model weights loaded successfully from {latest_model_path}")
    
    return algo, latest_model_path.split("/")[-1]

# ========================================================================
# SECTION 2: MAIN EXECUTION
# ========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calculate Critic Divergence (Q-value difference) between two d3rlpy models."
    )
    
    # --- Args for Model Loading ---
    parser.add_argument('--model-path-1', type=str, required=True,
                        help="Path to the directory of the first model (model 1)")
    parser.add_argument('--model-path-2', type=str, required=True,
                        help="Path to the directory of the second model (model 2)")
    
    # --- Args for Data Loading ---
    parser.add_argument('--dataset', type=str, required=True,
                        help="Task name (e.g., 'pointmaze', 'antmaze')")
    parser.add_argument('--datasets', type=str, nargs='+', required=True,
                        help="List of all sub-dataset names involved")
    parser.add_argument(
        '--retained-ratios', 
        type=float, 
        nargs='+', 
        required=True,
        help="List of ratios (0.0 to 1.0) to keep (D_r)."
    )
    parser.add_argument('--seed', type=int, default=42,
                        help="Seed for the data split")

    # --- Args for Execution & Saving ---
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU ID (-1 for CPU)")
    parser.add_argument('--batch-size', type=int, default=512,
                        help="Batch size for calculation (to avoid OOM)")
    parser.add_argument('--require-model', type=int, default=None,
                        help="read the required model, read the lateset if it is None")
    parser.add_argument('--save-dir', type=str, default="stats_results_critic",
                        help="Directory to save the results JSON file.")
    

    args = parser.parse_args()
    
    if len(args.retained_ratios) != len(args.datasets):
        raise ValueError("--retained-ratios list must have the same length as --datasets list.")
    
    d3rlpy.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    print(f"Using device: {device}")

    # --- 1. Load Models ---
    model1, model1_name = load_model_from_dir(args.model_path_1, args.gpu)
    model2, model2_name = load_model_from_dir(args.model_path_2, args.gpu, require_model= args.require_model)

    # --- 2. Load Data ---
    print("Loading and splitting datasets (replicating unlearning script logic)...")
    dirs = []
      
    if args.dataset in ['halfcheetah', 'walker2d', 'hopper']:
        prefix_path = 'mujoco'
    elif args.dataset in ['antmaze', 'pointmaze']: 
        prefix_path = 'D4RL'
    else:
        prefix_path = ""  # Use the dataset name directly if not recognized
    for i in args.datasets:
        if prefix_path == "":
            dirs.append(f"{args.dataset}/{i}")
        else:
            dirs.append(f"{prefix_path}/{args.dataset}/{i}")
            
    print(f"Targeting dataset directories: {dirs}")

    # Perform the split to get D_r (required) and D_f (remained/forget)
    dataset_required, dataset_remained = load_merged_dataset(
        dirs, 
        args.retained_ratios, 
        args.seed
    )
    print(f"Loaded {len(dataset_required)} 'required' (D_r) episodes.")
    print(f"Loaded {len(dataset_remained)} 'forget' (D_f) episodes.")

    # --- 3. Extract States AND Actions ---
    # Critic evaluation requires (s, a) pairs.
    
    # 3.1 Extract from Forget Set (D_f)
    print("Extracting transitions (states & actions) from 'forget' set (D_f)...")
    transitions_f: List[Transition] = []
    for episode in tqdm(dataset_remained, desc="Processing D_f episodes"):
        transitions_f.extend(episode.transitions)
    
    if transitions_f:
        states_f_np = np.array([t.observation for t in transitions_f])
        actions_f_np = np.array([t.action for t in transitions_f])
    else:
        states_f_np = np.array([])
        actions_f_np = np.array([])
        
    print(f"Extracted {len(states_f_np)} samples from D_f.")
    
    # 3.2 Extract from Required Set (D_r)
    print("Extracting transitions (states & actions) from 'required' set (D_r)...")
    transitions_r: List[Transition] = []
    for episode in tqdm(dataset_required, desc="Processing D_r episodes"):
        transitions_r.extend(episode.transitions)

    if transitions_r:
        states_r_np = np.array([t.observation for t in transitions_r])
        actions_r_np = np.array([t.action for t in transitions_r])
    else:
        states_r_np = np.array([])
        actions_r_np = np.array([])
        
    print(f"Extracted {len(states_r_np)} samples from D_r.")

    if len(states_f_np) == 0:
        print("Warning: The 'forget set' (D_f) is empty.")
    if len(states_r_np) == 0:
        print("Warning: The 'required set' (D_r) is empty.")

    # --- 4. Compute Statistics ---
    combined_stats = {}
    metric_used = "CriticValueDiff"

    # Compute for D_f
    stats_f = compute_critic_diff_stats(
        model1, model2, states_f_np, actions_f_np, args.batch_size, desc_label="Critic Diff on D_f"
    )
    # Compute for D_r
    stats_r = compute_critic_diff_stats(
        model1, model2, states_r_np, actions_r_np, args.batch_size, desc_label="Critic Diff on D_r"
    )

    # --- 5. Merge Results ---
    combined_stats.update(add_prefix_to_keys(stats_f, "D_f_"))
    combined_stats.update(add_prefix_to_keys(stats_r, "D_r_"))
    
    # Print Summary
    print(f"\n=== {metric_used} Summary (Mean Absolute Diff) ===")
    print(f"D_f (Forget) Mean:  {combined_stats['D_f_mean']:.6f} (N={combined_stats['D_f_count']})")
    print(f"D_r (Retain) Mean:  {combined_stats['D_r_mean']:.6f} (N={combined_stats['D_r_count']})")
    print(f"===============================\n")

    # --- 6. Save Results ---
    results_data = {
        "model_1_path": args.model_path_1,
        "model_1_name": model1_name,
        "model_2_path": args.model_path_2,
        "model_2_name": model2_name,
        "metric_type": metric_used,
        "dataset_name": args.dataset,
        "dataset_components": args.datasets,
        "retained_ratios": args.retained_ratios,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "critic_diff_statistics": combined_stats 
    }
    
    # Construct save path
    ratios_str = "_".join([str(r) for r in args.retained_ratios])
    folder_prefix = "stats_critic_"
    # e.g. stats_critic_hopper/ratios_0.1/name_...
    save_path = folder_prefix + args.dataset + "/ratios_" + ratios_str + "/name_" + args.model_path_2.replace(".", "")
    
    os.makedirs(os.path.join(args.save_dir, save_path), exist_ok=True)
    json_filename = f"seed{args.seed}.json"
    
    save_results_to_json(results_data, os.path.join(args.save_dir, save_path), json_filename)

if __name__ == "__main__":
    main()