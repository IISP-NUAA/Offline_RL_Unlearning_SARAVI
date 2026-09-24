import sys
import os
import argparse
import json
import glob
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import List

# --- D3RLPy Imports ---
import d3rlpy
from d3rlpy.base import LearnableBase
from d3rlpy.torch_utility import TorchMiniBatch
from d3rlpy.algos.base import AlgoBase 
from torch import Tensor
from analyze_evaluation_results import parse_model_dir
from d3rlpy.models.torch import compute_max_with_n_actions_and_indices,compute_max_with_n_actions
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))

try:
    from minari_dataset_processing.utility import (
        load_merged_dataset,
        load_model_with_fix,
        parse_dataset_dir
    )
except ImportError as e:
    print(f"Error importing utility: {e}")
    sys.exit(1)

# ========================================================================
# SECTION 1: V-Function Estimation
# ========================================================================

@torch.no_grad()
@torch.no_grad()
def estimate_v_function(
    algo: AlgoBase, 
    obs_t: Tensor,
    n_action_samples: int = 10,
    using_target = True,
) -> Tensor:
    """
     Estimates V(s) using the provided algorithm's Q-network and policy sampling.
    """
    batch_size = obs_t.shape[0]
    algo_name = algo.__class__.__name__
    policy = algo.impl._policy
    if using_target  and hasattr(algo.impl,"_targ_q_func"):
        q_func = algo.impl._targ_q_func
    else:
        q_func = algo.impl._q_func
    if 'CQL' in algo_name:
        v_values =  algo.impl._compute_policy_is_values(obs_t,obs_t)
        v_proxy = v_values.mean(dim=(0, 2))
        v_proxy = v_proxy.unsqueeze(-1)
        return v_proxy
    if  'BEAR' in algo_name:
        policy_actions, log_prob = policy.sample_n_with_log_prob(obs_t, n_action_samples)
    else:
        policy_actions = policy.sample_n(obs_t, n_action_samples)
    repeat_obs = obs_t.unsqueeze(1).repeat(
        1, n_action_samples, *([1] * (obs_t.dim() - 1))
    )
    repeat_obs = repeat_obs.reshape(-1, *obs_t.shape[1:])
    flat_policy_actions = policy_actions.reshape(-1, *policy_actions.shape[2:])
    
    target_q_values = q_func(
                repeat_obs, flat_policy_actions, "mean"
    )
    reshaped_target_q = target_q_values.view(batch_size, n_action_samples)
    v_values = reshaped_target_q.mean(dim=1)

    if "BEAR" in algo_name:
        v_values, indices = compute_max_with_n_actions_and_indices(
                obs_t, policy_actions, algo.impl._targ_q_func, algo.impl._lam
            )
        max_log_prob = log_prob[torch.arange(batch_size), indices]
        v_values = v_values - algo.impl._log_temp().exp() * max_log_prob
    return v_values

@torch.no_grad()
def estimate_v_function_for_deterministic_policy(
    algo: AlgoBase, 
    obs_t: torch.Tensor,
    n_action_samples: int = 10,
) -> torch.Tensor:
    batch_size = obs_t.shape[0]
    impl = algo.impl
    
    # Case 1: IQL
    if hasattr(impl, "_value_func") and impl._value_func is not None:
        return impl._value_func(obs_t).view(-1)

    # Case 2: BCQ / PLAS
    algo_name = algo.__class__.__name__
    is_bcq_or_plas = "BCQ" in algo_name or "PLAS" in algo_name or hasattr(impl, "_imitator")
    
    if is_bcq_or_plas:
        class MockBatch:
            def __init__(self, obs, device):
                self.next_observations = obs
                self.observations = obs
                self.actions = None
                self.rewards = None
                self.terminals = None
                self.device = device  
        mock_batch = MockBatch(obs_t, impl.device)
        v_values = impl.compute_target(mock_batch)
        return v_values.view(-1)

    # Case 3: TD3 / TD3+BC
    else:
        sigma = 0.2
        clip = 0.5
        if hasattr(impl, "_predict_best_action"):
            actions = impl._predict_best_action(obs_t)
        else:
            actions = impl._policy(obs_t)
        action_dim = actions.shape[1]
        
        obs_expanded = obs_t.unsqueeze(1).repeat(1, n_action_samples, 1).view(-1, *obs_t.shape[1:])
        actions_expanded = actions.unsqueeze(1).repeat(1, n_action_samples, 1)
        noise = torch.randn_like(actions_expanded) * sigma
        noise = noise.clamp(-clip, clip)
        noisy_action = (actions_expanded + noise).clamp(-1.0, 1.0)
        
        flat_action = noisy_action.view(-1, action_dim)
        flat_q = impl._q_func(obs_expanded, flat_action, "mean")
        v_values = flat_q.view(batch_size, n_action_samples).mean(dim=1)
        return v_values

# ========================================================================
# SECTION 2: Helper Functions
# ========================================================================

def load_model(model_dir: str, gpu: int) -> LearnableBase:
    """Robust model loading logic."""
    print(f"Loading model from: {model_dir}")
    params_path = os.path.join(model_dir, 'params.json')
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    algo_name = params.get("algorithm", params.get("type"))
    AlgoClass = getattr(d3rlpy.algos, algo_name)
    algo = AlgoClass.from_json(params_path, use_gpu=(gpu >= 0))
    
    # Find latest weight
    model_files = glob.glob(os.path.join(model_dir, 'model_*.pt'))
    if not model_files:
        model_path = os.path.join(model_dir, 'model.pt')
    else:
        step_nums = [int(re.search(r'model_(\d+).pt', f).group(1)) for f in model_files if re.search(r'model_(\d+).pt', f)]
        latest_step = max(step_nums)
        model_path = os.path.join(model_dir, f'model_{latest_step}.pt')
    
    load_model_with_fix(algo, model_path, gpu)
    
    # Freeze model
    algo.impl._policy.eval()
    algo.impl._q_func.eval()
    if hasattr(algo.impl, "_value_func") and algo.impl._value_func: algo.impl._value_func.eval()
    
    return algo

# ========================================================================
# SECTION 3: Main Calculation Logic
# ========================================================================

def calculate_advantages(
    retrain_model: LearnableBase,
    fully_model: LearnableBase,
    transitions: List,
    device: str,
    batch_size: int = 256,
    desc: str = "Calculating Advantages"
):
    """
    Computes:
    1. Self Advantage: Q_retrain(s, a) - V_retrain(s)
    2. Relative Advantage: Q_retrain(s, a) - V_fully(s)
    """
    self_advantages = []
    relative_advantages = []
    
    is_stochastic = hasattr(retrain_model.impl._policy, "dist")
    n_samples = len(transitions)
    indices = np.arange(n_samples)
    
    print(f"{desc} over {n_samples} transitions...")
    
    for i in tqdm(range(0, n_samples, batch_size), desc=desc):
        batch_idx = indices[i : i + batch_size]
        batch_transitions = [transitions[k] for k in batch_idx]
        
        batch_d3 = d3rlpy.dataset.TransitionMiniBatch(batch_transitions)
        batch_torch = TorchMiniBatch(
            batch_d3,
            device=device,
            scaler=retrain_model.impl.scaler,
            action_scaler=retrain_model.impl.action_scaler,
            reward_scaler=retrain_model.impl.reward_scaler
        )
        
        obs = batch_torch.observations
        actions = batch_torch.actions
        
        # 1. Calculate Q_retrain(s, a)
        q_retrain = retrain_model.impl._q_func(obs, actions, "mean").view(-1)
        
        # 2. Calculate V_retrain(s)
        if is_stochastic:
            v_retrain = estimate_v_function(retrain_model, obs).view(-1)
        else:
            v_retrain = estimate_v_function_for_deterministic_policy(retrain_model, obs).view(-1)
            
        # 3. Calculate V_fully(s)
        if is_stochastic:
            v_fully = estimate_v_function(fully_model, obs).view(-1)
        else:
            v_fully = estimate_v_function_for_deterministic_policy(fully_model, obs).view(-1)
        
        # 4. Compute Diffs
        adv_self = q_retrain - v_retrain
        adv_relative = q_retrain - v_fully
        
        self_advantages.extend(adv_self.detach().cpu().numpy().tolist())
        relative_advantages.extend(adv_relative.detach().cpu().numpy().tolist())
        
    return np.array(self_advantages), np.array(relative_advantages)


# ========================================================================
# SECTION 4: Plotting
# ========================================================================

def plot_histogram_logic(rel_adv, save_dir, prefix, suffix, color, set_name):
    """
    Helper function to maintain the exact original histogram logic.
    Titles are removed and font sizes are adjusted.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(rel_adv, kde=True, color=color, bins=50)
    plt.axvline(0, color='k', linestyle='--', label="Baseline ($V_{ori}$)")
    plt.axvline(np.mean(rel_adv), color='r', linestyle=':', label=f"Mean: {np.mean(rel_adv):.2f}")
    
    # Title removed per user request
    # plt.title(...)
    
    plt.xlabel("Value Difference")
    plt.ylabel("Count")
    plt.legend()
    
    save_path = os.path.join(save_dir, f"{prefix}_relative_advantage_hist_{suffix}.png")
    # Added bbox_inches='tight' to prevent label cutoff
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {set_name} hist plot to {save_path}")

def plot_distributions(self_adv_f, rel_adv_f, self_adv_r, rel_adv_r, save_dir, prefix):
    # Update seaborn theme to globally increase font sizes for all plots in this function
    sns.set_theme(style="whitegrid", rc={
        "axes.labelsize": 24,     # Font size for x and y labels
        "xtick.labelsize": 20,    # Font size for x tick labels
        "ytick.labelsize": 20,    # Font size for y tick labels
        "legend.fontsize": 20     # Font size for legend
    })
    
    # --- Plot 1: Original Overlay (Forget Set Focus) ---
    plt.figure(figsize=(10, 6))
    sns.kdeplot(self_adv_f, fill=True, label=r"$Q_{retrain} - V_{ret}$ (Self)", color="blue", alpha=0.3)
    sns.kdeplot(rel_adv_f, fill=True, label=r"$Q_{retrain} - V_{ori}$ (Relative)", color="red", alpha=0.3)
    
    plt.axvline(0, color='k', linestyle='--')
    
    # Title removed
    plt.xlabel("Advantage Value")
    plt.ylabel("Density")
    plt.legend()
    
    save_path = os.path.join(save_dir, f"{prefix}_advantage_overlay.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved overlay plot to {save_path}")

    # --- Plot 2: Relative Advantage Comparison (Forget vs Retain) ---
    plt.figure(figsize=(10, 6))
    sns.kdeplot(rel_adv_f, fill=True, label=r"Forget Set ($D_f$)", color="purple", alpha=0.3)
    sns.kdeplot(rel_adv_r, fill=True, label=r"Retain Set ($D_r$)", color="green", alpha=0.3)
    
    plt.axvline(0, color='k', linestyle='--', label="Baseline ($V_{ori}$)")
    plt.axvline(np.mean(rel_adv_f), color='purple', linestyle=':', label=f"Mean Forget: {np.mean(rel_adv_f):.2f}")
    plt.axvline(np.mean(rel_adv_r), color='green', linestyle=':', label=f"Mean Retain: {np.mean(rel_adv_r):.2f}")
    
    # Title removed
    plt.xlabel("Value Difference")
    plt.ylabel("Density")
    plt.legend()
    
    save_path = os.path.join(save_dir, f"{prefix}_relative_advantage_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {save_path}")
    
    # --- Plot 3: Self Advantage Comparison (Forget vs Retain) [NEW] ---
    plt.figure(figsize=(10, 6))
    sns.kdeplot(self_adv_f, fill=True, label=r"Forget Set ($D_f$)", color="purple", alpha=0.3)
    sns.kdeplot(self_adv_r, fill=True, label=r"Retain Set ($D_r$)", color="green", alpha=0.3)
    
    plt.axvline(0, color='k', linestyle='--', label="Zero Advantage")
    plt.axvline(np.mean(self_adv_f), color='purple', linestyle=':', label=f"Mean Forget: {np.mean(self_adv_f):.2f}")
    plt.axvline(np.mean(self_adv_r), color='green', linestyle=':', label=f"Mean Retain: {np.mean(self_adv_r):.2f}")
    
    # Title removed
    plt.xlabel("Advantage Value")
    plt.ylabel("Density")
    plt.legend()
    
    save_path = os.path.join(save_dir, f"{prefix}_self_advantage_comparison.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved self advantage comparison plot to {save_path}")

    # --- Plot 4: Original Histogram Logic on Forget Set ---
    plot_histogram_logic(
        rel_adv_f, 
        save_dir, 
        prefix, 
        suffix="forget", 
        color="purple", 
        set_name="Forget Set"
    )

    # --- Plot 5: Original Histogram Logic on Retain Set ---
    plot_histogram_logic(
        rel_adv_r, 
        save_dir, 
        prefix, 
        suffix="retain", 
        color="green", 
        set_name="Retain Set"
    )

# ========================================================================
# SECTION 5: Main
# ========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrain-model-dir', type=str, required=True, help="Path to Unlearned/Retrained model")
    parser.add_argument('--fully-model-dir', type=str, required=True, help="Path to Original Fully Trained model")
    parser.add_argument('--dataset', type=str, required=True, help="e.g. pointmaze")
    parser.add_argument('--datasets', type=str, nargs='+', required=True, help="Sub datasets")
    parser.add_argument('--retained-ratios', type=float, nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save-dir', type=str, default="visualizations_advantage")
    args = parser.parse_args()

    # Setup
    if len(args.retained_ratios) != len(args.datasets):
        raise ValueError("Ratios length mismatch.")
    
    d3rlpy.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device_name = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    
    # 1. Load Data
    print("Loading datasets to isolate Forget Set AND Retain Set...")
    dirs = parse_dataset_dir(args.dataset, args.datasets)
    
    # Unpack both datasets
    dataset_required, dataset_remained = load_merged_dataset(dirs, args.retained_ratios, args.seed, shuffle=True)
    
    # Extract Forget Transitions
    transitions_f = []
    for ep in tqdm(dataset_remained, desc="Extracting Forget Transitions"):
        transitions_f.extend(ep.transitions)

    # Extract Retain Transitions (New)
    transitions_r = []
    for ep in tqdm(dataset_required, desc="Extracting Retain Transitions"):
        transitions_r.extend(ep.transitions)
        
    print(f"Forget Set Size: {len(transitions_f)} transitions")
    print(f"Retain Set Size: {len(transitions_r)} transitions")
    
    if len(transitions_f) == 0:
        print("Error: Forget set is empty.")
        return

    # 2. Load Models
    model_retrain = load_model(args.retrain_model_dir, args.gpu)
    model_fully = load_model(args.fully_model_dir, args.gpu)

    # 3. Calculate Stats for Forget Set
    self_adv_f, rel_adv_f = calculate_advantages(
        model_retrain, model_fully, transitions_f, device_name, desc="Calc Forget Adv"
    )

    # 4. Calculate Stats for Retain Set
    self_adv_r, rel_adv_r = calculate_advantages(
        model_retrain, model_fully, transitions_r, device_name, desc="Calc Retain Adv"
    )

    # 5. Plot
    method_name, ratio, algo_name, _ = parse_model_dir(args.retrain_model_dir)
    dir_name =  f"{args.dataset}/{algo_name}"
    if 'POISONED' in args.fully_model_dir:
        dir_name = "POISONED"+ dir_name
    os.makedirs(args.save_dir+"/"+dir_name,exist_ok=True)
    print(os.path.exists(dir_name))
    run_name = f"/{method_name}_seed{args.seed}_ratios{ratio}"
    task_name = dir_name+run_name
    print(task_name)
    plot_distributions(self_adv_f, rel_adv_f, self_adv_r, rel_adv_r, args.save_dir, task_name)
    
    print("Done.")

if __name__ == "__main__":
    main()