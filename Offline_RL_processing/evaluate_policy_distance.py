# evaluate_critic_divergence.py

import sys
import os
import json
import argparse
import glob
import re
from typing import List, Dict, Optional

import numpy as np
import torch
import d3rlpy
from d3rlpy.base import LearnableBase
from d3rlpy.dataset import Episode, Transition
from tqdm import tqdm 
from scipy.stats import wasserstein_distance

DEFAULT_FORGET_SAMPLE_EPISODES = 100

# --- Project-Specific Imports ---
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
    sys.exit(1)

# ========================================================================
# SECTION 1: HELPER FUNCTIONS
# ========================================================================

def add_prefix_to_keys(stats: Dict[str, float], prefix: str) -> Dict[str, float]:
    """
    Adds a prefix to all keys in a dictionary for structured JSON logging.
    """
    return {f"{prefix}{k}": v for k, v in stats.items()}

def save_results_to_json(results: dict, save_dir: str, filename: str):
    """
    Saves the evaluation metrics dictionary to a JSON file.
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

def compute_critic_divergence(
    critic_model: LearnableBase, 
    policy_orig: LearnableBase,
    policy_retrain: LearnableBase,
    policy_unlearn: LearnableBase,
    states_array_np: np.ndarray, 
    batch_size: int, 
    desc_label: str = "Critic Divergence Batch"
) -> Dict[str, float]:
    """
    Evaluates actions from three different policies using a fixed critic model
    and computes the Wasserstein distance between the resulting Q-value distributions.
    """
    if len(states_array_np) == 0:
        return {
            "W_unlearn_retrain": 0.0, 
            "W_unlearn_orig": 0.0, 
            "mean_Q_unlearn": 0.0, 
            "mean_Q_retrain": 0.0, 
            "mean_Q_orig": 0.0, 
            "count": 0
        }

    print(f"Computing Critic Divergence over {len(states_array_np)} samples ({desc_label})...")
    
    all_q_orig = []
    all_q_retrain = []
    all_q_unlearn = []
    
    n_samples = len(states_array_np)
    
    for i in tqdm(range(0, n_samples, batch_size), desc=desc_label):
        end_i = min(i + batch_size, n_samples)
        states_batch = states_array_np[i:end_i]
        
        # 1. Generate actions using the three distinct policies
        a_orig = policy_orig.predict(states_batch)
        a_retrain = policy_retrain.predict(states_batch)
        a_unlearn = policy_unlearn.predict(states_batch)
        
        # 2. Evaluate Q-values using the ORIGINAL critic model
        q_orig = critic_model.predict_value(states_batch, a_orig)
        q_retrain = critic_model.predict_value(states_batch, a_retrain)
        q_unlearn = critic_model.predict_value(states_batch, a_unlearn)
        
        all_q_orig.extend(q_orig.tolist())
        all_q_retrain.extend(q_retrain.tolist())
        all_q_unlearn.extend(q_unlearn.tolist())
        
    # 3. Compute distribution distances
    w_dist_unlearn_retrain = wasserstein_distance(all_q_unlearn, all_q_retrain)
    w_dist_unlearn_orig = wasserstein_distance(all_q_unlearn, all_q_orig)
    
    stats_dict = {
        "W_unlearn_retrain": float(w_dist_unlearn_retrain),
        "W_unlearn_orig": float(w_dist_unlearn_orig),
        "mean_Q_unlearn": float(np.mean(all_q_unlearn)),
        "mean_Q_retrain": float(np.mean(all_q_retrain)),
        "mean_Q_orig": float(np.mean(all_q_orig)),
        "count": n_samples
    }
    
    return stats_dict

def compute_forget_critic_wasserstein(
    critic_model: LearnableBase,
    policy_retrain: LearnableBase,
    policy_unlearn: LearnableBase,
    states_array_np: np.ndarray,
    batch_size: int,
    desc_label: str = "Policy Distance on sampled D_f",
) -> Dict[str, float]:
    # Cost measurement only needs W(D_f, unlearn, retrain). Avoid the
    # original-policy forward pass and all D_r critic evaluation.
    if len(states_array_np) == 0:
        return {"W_unlearn_retrain": 0.0, "count": 0}

    print(
        f"Computing sampled D_f critic distance over "
        f"{len(states_array_np)} samples ({desc_label})..."
    )
    all_q_retrain = []
    all_q_unlearn = []

    for i in tqdm(
        range(0, len(states_array_np), batch_size),
        desc=desc_label,
    ):
        states_batch = states_array_np[i:i + batch_size]
        a_retrain = policy_retrain.predict(states_batch)
        a_unlearn = policy_unlearn.predict(states_batch)
        q_retrain = critic_model.predict_value(states_batch, a_retrain)
        q_unlearn = critic_model.predict_value(states_batch, a_unlearn)
        all_q_retrain.extend(q_retrain.tolist())
        all_q_unlearn.extend(q_unlearn.tolist())

    return {
        "W_unlearn_retrain": float(
            wasserstein_distance(all_q_unlearn, all_q_retrain)
        ),
        "count": len(states_array_np),
    }

def load_model_from_dir(model_dir: str, gpu: int, require_model=None) -> LearnableBase:
    """
    Loads a d3rlpy model using the robust logic from utility.
    """
    print(f"Loading model from directory: {model_dir}")
    params_path = os.path.join(model_dir, 'params.json')
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"params.json not found in {model_dir}")

    with open(params_path, 'r') as f:
        params = json.load(f)

    algo_name_key = "algorithm" if "algorithm" in params else "type"
    algo_name = params.get(algo_name_key)
    if not algo_name:
         raise ValueError(f"Could not find 'algorithm' or 'type' key in {params_path}")
    
    try:
        AlgoClass = getattr(d3rlpy.algos, algo_name)
        algo = AlgoClass.from_json(params_path, use_gpu=(gpu if gpu >= 0 else False))
    except AttributeError:
        raise ValueError(f"Algorithm '{algo_name}' not found in d3rlpy.algos")

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
        step_nums = [int(re.search(r'model_(\d+).pt', f).group(1)) for f in model_files if re.search(r'model_(\d+).pt', f)]
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


def build_dataset_dirs(dataset: str, datasets: List[str]) -> List[str]:
    """Build Minari dataset identifiers with the same convention as training."""
    if dataset in ['halfcheetah', 'walker2d', 'hopper']:
        prefix_path = 'mujoco'
    elif dataset in ['antmaze', 'pointmaze']:
        prefix_path = 'D4RL'
    else:
        prefix_path = ""
    if prefix_path:
        return [f"{prefix_path}/{dataset}/{name}" for name in datasets]
    return [f"{dataset}/{name}" for name in datasets]


def extract_states_from_split(dataset_required, dataset_remained):
    """Convert an already-created D_r/D_f split into evaluator state arrays."""
    print("Extracting states from 'forget' set (D_f)...")
    transitions_f: List[Transition] = []
    for episode in tqdm(dataset_remained, desc="Processing D_f episodes"):
        transitions_f.extend(episode.transitions)

    print("Extracting states from 'required' set (D_r)...")
    transitions_r: List[Transition] = []
    for episode in tqdm(dataset_required, desc="Processing D_r episodes"):
        transitions_r.extend(episode.transitions)

    states_f_np = (
        np.array([transition.observation for transition in transitions_f])
        if transitions_f else np.array([])
    )
    states_r_np = (
        np.array([transition.observation for transition in transitions_r])
        if transitions_r else np.array([])
    )
    return states_f_np, states_r_np


def extract_sampled_forget_states(
    dataset_remained,
    sample_episodes: int = DEFAULT_FORGET_SAMPLE_EPISODES,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    # Sample complete trajectories so the estimate is based on 100 D_f episodes,
    # while retaining every transition within each selected trajectory.
    if sample_episodes <= 0:
        raise ValueError("sample_episodes must be positive")

    episodes = list(dataset_remained)
    if len(episodes) > sample_episodes:
        rng = np.random.default_rng(random_seed)
        selected = np.sort(
            rng.choice(len(episodes), size=sample_episodes, replace=False)
        )
        episodes = [episodes[int(index)] for index in selected]

    transitions = []
    for episode in tqdm(episodes, desc="Processing sampled D_f episodes"):
        transitions.extend(episode.transitions)

    print(
        f"Using {len(episodes)} sampled D_f trajectories and "
        f"{len(transitions)} transitions."
    )
    return (
        np.array([transition.observation for transition in transitions])
        if transitions else np.array([])
    )


def evaluate_policy_distance(
    model_orig: LearnableBase,
    model_retrain: LearnableBase,
    model_unlearn: LearnableBase,
    states_f_np: np.ndarray,
    states_r_np: np.ndarray,
    batch_size: int = 512,
) -> Dict[str, float]:
    """Evaluate already-loaded models; shared by the CLI and cost runs."""
    combined_stats: Dict[str, float] = {}
    stats_f = compute_critic_divergence(
        model_orig,
        model_orig,
        model_retrain,
        model_unlearn,
        states_f_np,
        batch_size,
        desc_label="Critic Divergence on D_f",
    )
    stats_r = compute_critic_divergence(
        model_orig,
        model_orig,
        model_retrain,
        model_unlearn,
        states_r_np,
        batch_size,
        desc_label="Critic Divergence on D_r",
    )
    combined_stats.update(add_prefix_to_keys(stats_f, "D_f_"))
    combined_stats.update(add_prefix_to_keys(stats_r, "D_r_"))
    return combined_stats


def evaluate_policy_distance_forget_only(
    model_orig: LearnableBase,
    model_retrain: LearnableBase,
    model_unlearn: LearnableBase,
    states_f_np: np.ndarray,
    batch_size: int = 512,
) -> Dict[str, float]:
    stats_f = compute_forget_critic_wasserstein(
        model_orig,
        model_retrain,
        model_unlearn,
        states_f_np,
        batch_size,
    )
    return add_prefix_to_keys(stats_f, "D_f_")


# ========================================================================
# SECTION 2: MAIN EXECUTION
# ========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calculate Policy Distance using Wasserstein Distance among Original, Retrain, and Unlearn models."
    )
    
    # --- Args for Model Loading ---
    parser.add_argument('--original-dir', type=str, required=True,
                        help="Path to the original model (provides the frozen Critic).")
    parser.add_argument('--retrain-dir', type=str, required=True,
                        help="Path to the retrained model (provides baseline target actions).")
    parser.add_argument('--unlearn-dir', type=str, required=True,
                        help="Path to the unlearned model (evaluated policy).")
    
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
    parser.add_argument('--shuffle', type=int, choices=(0, 1), default=1,
                        help="Use the same shuffled (1) or ordered (0) split as training")

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
    model_orig, name_orig = load_model_from_dir(args.original_dir, args.gpu)
    model_retrain, name_retrain = load_model_from_dir(args.retrain_dir, args.gpu, require_model=args.require_model)
    model_unlearn, name_unlearn = load_model_from_dir(args.unlearn_dir, args.gpu, require_model=args.require_model)

    # --- 2. Load Data ---
    print("Loading and splitting datasets (replicating unlearning script logic)...")
    dirs = build_dataset_dirs(args.dataset, args.datasets)
            
    print(f"Targeting dataset directories: {dirs}")

    dataset_required, dataset_remained = load_merged_dataset(
        dirs,
        args.retained_ratios,
        args.seed,
        shuffle=(args.shuffle == 1),
    )
    print(f"Loaded {len(dataset_required)} 'required' (D_r) episodes.")
    print(f"Loaded {len(dataset_remained)} 'forget' (D_f) episodes.")

    # --- 3. Extract States ---
    states_f_np, states_r_np = extract_states_from_split(
        dataset_required, dataset_remained
    )

    # --- 4. Compute Statistics ---
    metric_used = "Wasserstein_Distance"
    combined_stats = evaluate_policy_distance(
        model_orig,
        model_retrain,
        model_unlearn,
        states_f_np,
        states_r_np,
        batch_size=args.batch_size,
    )

    print(f"\n=== {metric_used} Summary ===")
    print(f"D_f W-Dist (Unlearn vs Retrain): {combined_stats['D_f_W_unlearn_retrain']:.6f}")
    print(f"D_f W-Dist (Unlearn vs Orig):    {combined_stats['D_f_W_unlearn_orig']:.6f}")
    print(f"D_r W-Dist (Unlearn vs Retrain): {combined_stats['D_r_W_unlearn_retrain']:.6f}")
    print(f"===============================\n")

    # --- 6. Save Results ---
    results_data = {
        "original_model_dir": args.original_dir,
        "retrain_model_dir": args.retrain_dir,
        "unlearn_model_dir": args.unlearn_dir,
        "metric_type": metric_used,
        "dataset_name": args.dataset,
        "dataset_components": args.datasets,
        "retained_ratios": args.retained_ratios,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "critic_divergence_statistics": combined_stats 
    }
    
    ratios_str = "_".join([str(r) for r in args.retained_ratios])
    folder_prefix = "stats_policy_distance_"
    dir_shrinked = args.unlearn_dir.replace("PLASWithPerturbation", "PLASP")
    dir_shortened = dir_shrinked.replace(".", "")
    
    save_path = folder_prefix + args.dataset + "/ratios_" + ratios_str + "/name_" + dir_shortened
    
    os.makedirs(os.path.join(args.save_dir, save_path), exist_ok=True)
    json_filename = f"seed{args.seed}.json"
    
    save_results_to_json(results_data, os.path.join(args.save_dir, save_path), json_filename)

if __name__ == "__main__":
    main()