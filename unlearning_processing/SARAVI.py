

import sys
import os
import argparse
import traceback
import csv
from typing import List, Tuple, Dict
from collections import defaultdict
import json
import glob 
import re 
import copy

import d3rlpy
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from tqdm import tqdm
import minari 

# ---  D3RLPy Imports ---
from d3rlpy.base import LearnableBase
from d3rlpy.algos.base import AlgoBase 
from d3rlpy.dataset import Transition, TransitionMiniBatch
from d3rlpy.torch_utility import TorchMiniBatch, set_state_dict 
from d3rlpy.gpu import Device 
from d3rlpy.logger import D3RLPyLogger
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.models.torch import compute_max_with_n_actions_and_indices,compute_max_with_n_actions
from sklearn.model_selection import train_test_split


# ---  Project-Specific Imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))

try:
    from minari_dataset_processing.utility import (
        load_merged_dataset,
        FlattenDictObsWrapper, 
        load_model_with_fix,
        parse_dataset_dir
    )
except ImportError as e:
    print(f"Error: Could not import from local scripts (e.g., utility.py).")
    print(f"Details: {e}")
    sys.exit(1)


# ========================================================================
#  SECTION 1: UTILITIES
# ========================================================================
class BatchIterator:
    """
     A helper class to iterate over dataset indices sequentially with shuffling (Epoch-based).
    """
    def __init__(self, dataset_size: int, batch_size: int, seed: int = 42):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.indices = np.arange(dataset_size)
        self.rng = np.random.default_rng(seed)
        self.rng.shuffle(self.indices)
        self.ptr = 0

    def next_batch(self) -> np.ndarray:
        if self.ptr + self.batch_size > self.dataset_size:
            self.rng.shuffle(self.indices)
            self.ptr = 0
        
        batch_idx = self.indices[self.ptr : self.ptr + self.batch_size]
        self.ptr += self.batch_size
        
        return batch_idx



class _TargetValueStateBatch:
    """Minimal batch adapter for evaluating a target formula at a state s."""

    def __init__(self, observations: Tensor, device: torch.device):
        # d3rlpy target formulas normally read next_observations. Providing the
        # same state in both fields also preserves formulas that inspect
        # observations for their batch size (for example BEAR).
        self.observations = observations
        self.next_observations = observations
        self.device = device


@torch.no_grad()
def estimate_target_style_state_value(algo: AlgoBase, observations: Tensor) -> Tensor:
    """Estimate V(s) by running the algorithm's native target formula at s.

    The underlying target implementation normally calculates V(s_{t+1}); this
    adapter intentionally substitutes the supplied current state. It therefore
    retains each algorithm's target actor/critic choice and reduction.
    """
    impl = algo.impl
    if impl is None or not hasattr(impl, "compute_target"):
        raise ValueError(
            f"{algo.__class__.__name__} does not provide a target-value computation."
        )

    values = impl.compute_target(_TargetValueStateBatch(observations, impl.device))

    if values.ndim == 3:
        values = values.mean(dim=0)
    return values.reshape(observations.shape[0], -1).mean(dim=1)

def calculate_advantage_custom(
    algo: AlgoBase,
    ref_algo: AlgoBase,  #  ABLATION: Added ref_algo
    batch: TorchMiniBatch,
    n_action_samples: int = 10,
) -> Tensor:
    """
     Calculates advantage A(s, a) = Q_curr(s, a) - V_ref(s).
     ABLATION: V(s) is calculated using the FROZEN reference model.
    """
    #  1. Get Q(s,a) from the CURRENT model
    q_values = algo.impl._q_func(
        batch.observations, batch.actions, "mean"
    )
    
    #  2. Get V(s) from the REFERENCE model with its native target formula.
    v_values_estimate = estimate_target_style_state_value(
        ref_algo, batch.observations
    )
    
    advantage = q_values.view(-1) - v_values_estimate.view(-1).detach()
    return advantage

def calc_mad(tensor):
    return (tensor - tensor.mean()).abs().mean()

TRAINING_METRIC_FIELDS = (
    "step",
    "critic_loss/original_retained",
    "actor_loss/retained",
    "imitator_loss",
    "custom_loss/ratio_loss",
    "custom_lossloss_self_stable",
    "custom_loss/loss_alignment",
    "custom_stats/advantage_mean_f",
    "custom_stats/std_adv_ratio",
    "critic_loss/v_loss",
)


def append_training_metrics_csv(
    csv_path: str, step: int, metrics: Dict[str, float]
) -> None:
    """Append one --logging-steps aggregate to the training metrics CSV."""
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    row = {"step": step, **metrics}
    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TRAINING_METRIC_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in TRAINING_METRIC_FIELDS})

# ========================================================================
#  SECTION 2: LOSS FUNCTIONS
# ========================================================================


def compute_adaptive_entropy_based_penalty(
    q_values: torch.Tensor,
    v_values_estimate: torch.Tensor,
    q_values_ref:torch.Tensor,
    v_values_estimate_ref: torch.Tensor, 
    q_values_on_retain: torch.Tensor,
    v_values_estimate_on_retain: torch.Tensor,
    v_values_on_retain_ref:torch.Tensor,
    eps: float = 1e-6,
    dataset_ratio: float = 1.0,
    filter=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    
    # 1. Data Preparation
    v_anchor = v_values_estimate.detach()
    self_advantage = q_values - v_anchor
    self_advantage_ref = (q_values_ref - v_values_estimate_ref).detach()
    rel_advantage = (q_values - v_values_estimate_ref)
    
    self_retain_advantage  = (q_values_on_retain-v_values_estimate_on_retain).detach()
    retain_advantage_ref =  (q_values_on_retain-v_values_on_retain_ref).detach()
    
    std_adv_ratio = (self_advantage.std()/(rel_advantage.std())).detach()
    std_adv_ratio = torch.clamp(std_adv_ratio,min=0.01, max=0.99)
       
    forget_low = torch.quantile(rel_advantage, 0.05)
    forget_high = torch.quantile(rel_advantage, 0.95)
    
    retain_low = torch.quantile(retain_advantage_ref, 0.05)
    retain_high = torch.quantile(retain_advantage_ref, 0.95)
    
    retain_advantage_ref_clipped = torch.clamp(retain_advantage_ref, min=retain_low, max=retain_high)
    rel_advantage_clipped = torch.clamp(rel_advantage, min=forget_low, max=forget_high)
    
    
    mean_r = retain_advantage_ref_clipped.mean()
    std_r = retain_advantage_ref_clipped.std()
    
    mean_f = rel_advantage_clipped.mean()
    std_f = rel_advantage_clipped.std()
# ===========================================================
# \tilde{A}_\text{rel}(\bm{s}, \bm{a}) = \frac{A_\text{rel}(\bm{s}, \bm{a})-\text{mean}(A_\text{rel}(\mathcal{B}))}{\text{std}(A_\text{rel}(\mathcal{B}))+\epsilon} \text{,} (\bm{s},\bm{a})\sim \mathcal{B} 
# ===========================================================

    norm_r = (retain_advantage_ref_clipped - mean_r) / (std_r + eps)
    norm_f = (rel_advantage_clipped - mean_f) / (std_f + eps)
    

    quantiles = torch.linspace(0, 1, 16, device=q_values.device)
    q_vals_r = torch.quantile(norm_r, quantiles)
    q_vals_f = torch.quantile(norm_f, quantiles)

# ===========================================================
# L_{\text{var-1}}(\mathcal{B}_f;\theta_{\text{cur}}) = -\text{log}[\text{std}(A_{\text{rel}}(\mathcal{B}_f)]
# ===========================================================
    ratio_loss = -torch.log(rel_advantage_clipped.std())
    loss_alignment = torch.zeros_like(ratio_loss,device=ratio_loss.device)
# ===========================================================
# L_{\text{sar}}(\mathcal{B}_f, \mathcal{B}_r; \theta_{\text{cur}}) = \frac{1}{C}\sum_{c=1}^{C}|p_c(\tilde{A}_{\text{rel}}(\mathcal{B}_f))-p_c(\tilde{A}_{\text{rel}}(\mathcal{B}_r))|
# ===========================================================
    loss_shape = torch.nn.functional.l1_loss(q_vals_f, q_vals_r)
# ===========================================================
# L_{\text{var-2}}(\mathcal{B}_f, \mathcal{B}_r;\theta_{\text{cur}})  = \left[\text{log}\left(\frac{\text{std}(A_{\text{rel}}(\mathcal{B}_f))}{\text{std}(A_{\text{rel}}(\mathcal{B}_r))}\right)\right]^2
# # ===========================================================
    loss_scale_brake = (torch.log(std_f) - torch.log(std_r)).pow(2)
    loss_self_stable =  (loss_shape+loss_scale_brake)

    lambda_ratio = std_adv_ratio*(1-dataset_ratio)

    lambda_shape = std_adv_ratio*(1-dataset_ratio)*0.5

    lambda_scale = std_adv_ratio*(1-dataset_ratio)*0.5

    total_loss = (lambda_ratio*ratio_loss+lambda_shape*loss_shape +lambda_scale*loss_scale_brake)

    return total_loss, {
        "loss_total": total_loss.item(),
        "loss_alignment": loss_alignment.item(),
        "ratio_loss": ratio_loss.item(),
        "loss_self_stable": loss_self_stable.item(),
        "std_adv_ratio": std_adv_ratio.item(),
    }
# ========================================================================
#  SECTION 3: CUSTOM UNLEARNING LOOP (With Reference Model)
# ========================================================================

    
def custom_unlearning_loop(
    algo: LearnableBase,
    ref_algo: LearnableBase, 
    transitions_r: List[Transition], 
    transitions_f: List[Transition], 
    
    args: argparse.Namespace,
    logger: D3RLPyLogger, 
    eval_env: gym.Env,
    log_dir: str,
    u_or_s: bool = True,
    uncertainties_f: np.ndarray=None,  
):
    """
     Training loop  Frozen V-function.
    """
    print("is stochastic policy: ", str(hasattr(algo.impl._policy, "dist")))
    print(f"Starting ABLATION unlearning loop (Frozen V-Function) for {args.unlearning_steps} steps...")
    if algo.impl is None or ref_algo.impl is None:
        raise ValueError("Algorithm implementation (impl) is not initialized.")

    n_transitions_r = len(transitions_r)
    n_transitions_f = len(transitions_f)
    dataset_ratio = n_transitions_r / (n_transitions_f+n_transitions_r)
    if dataset_ratio > 0.99:
        dataset_ratio = 0.99
    elif dataset_ratio<0.01:
        dataset_ratio = 0.01
    print("dataset ratio: ", str(dataset_ratio))
    print("dataset size of D_r: ", str(n_transitions_r))
    print("dataset size of D_f: ", str(n_transitions_f))
    filter = None
    if 'CQL' in args.algo:
        filter = 'CQL'
    elif "PLASP" in args.algo:
        filter = 'PLASP'
    elif "TD3" in args.algo:
        filter = 'TD3PLUSBC'
    elif 'IQL' in args.algo:
        filter = 'IQL'
    elif 'BEAR' in args.algo:
        filter = 'BEAR'
    elif 'BCQ' in args.algo:
        filter = 'BCQ'
    if n_transitions_f == 0:
        print("Warning: The 'forget set' (D_f) is empty.")
        return
    if filter is not None and ("halfcheetah" in args.dataset or "walker2d" in args.dataset or "hopper" in args.dataset):
        filter+=args.dataset
        print("filter: ", filter)
    batch_size_r = args.batch_size // 2
    # batch_size_r = int(args.batch_size*dataset_ratio)
    
    batch_size_f = args.batch_size - batch_size_r
    print("Batch sizes, D_r: ", str(batch_size_r), ' D_f: ', str(batch_size_f))
    if n_transitions_r == 0:
        print("Error: Retained set (D_r) is empty.")
        return
        
    metrics_csv_path = os.path.join(logger._logdir, "unlearning_metrics.csv")
    total_steps = args.unlearning_steps
    if args.dataset in ["halfcheetah", "hopper", "walker2d"]:
        # to match the process of TrajDeleter in MuJoCo tasks, 
        # we consider an optimization step of SARAVI in MuJoCo tasks as one of:
        # (a critic update on D_f + an actor update on D_r)
        # or (a critic update on D_r + an actor update on D_r)
            total_steps = round(total_steps*0.5)
    step_metrics = defaultdict(list)
    iter_r = BatchIterator(n_transitions_r, batch_size_r, seed=args.seed)
    iter_f = BatchIterator(n_transitions_f, batch_size_f, seed=args.seed + 1)
    pbar = tqdm(range(1, total_steps + 1), desc="Unlearning")
    imitator_loss = None
    v_loss = None
    if hasattr(algo.impl,"update_imitator"):
        print("The imitator is also updated.")
    for step in pbar:
        
        batch_torch_r = None
        
        # 1. --- Sample Batch D_r (Retain) ---
        batch_indices_r = iter_r.next_batch()
        batch_transitions_r = [transitions_r[i] for i in batch_indices_r]
        batch_d3rlpy_r = TransitionMiniBatch(batch_transitions_r)
        batch_torch_r = TorchMiniBatch(
            batch_d3rlpy_r,
            device=algo.impl.device,
            scaler=algo.impl.scaler,
            action_scaler=algo.impl.action_scaler,
            reward_scaler=algo.impl.reward_scaler
        )

        # 2. --- Sample Batch D_f (Forget) ---
        batch_indices_f = iter_f.next_batch()
        batch_transitions_f = [transitions_f[i] for i in batch_indices_f]
        
        # the uncertainty branch is deprecated.
        
        # batch_uncertainty_f = uncertainties_f[batch_indices_f]
        batch_d3rlpy_f = TransitionMiniBatch(batch_transitions_f)
        batch_torch_f = TorchMiniBatch(
            batch_d3rlpy_f,
            device=algo.impl.device,
            scaler=algo.impl.scaler,
            action_scaler=algo.impl.action_scaler,
            reward_scaler=algo.impl.reward_scaler
        )

        if hasattr(algo.impl,"update_imitator"):
            # print(batch_torch_r.observations)
            # print(batch_torch_r.actions)
            """
                The update_imitator() function is decorated with @torch_api, does not support TorchMinibatch...
            """
            imitator_loss = algo.impl.update_imitator(batch_d3rlpy_r)


        # 3. --- MANUAL CRITIC UPDATE ---
        
        # A. Original Critic Loss (on D_r ONLY)

        if hasattr(algo.impl, "update_value_func"):
            algo.impl._value_optim.zero_grad()
            v_loss = algo.impl.compute_value_loss(batch_torch_r)
            v_loss.backward()
            algo.impl._value_optim.step()
        q_tpn_r = algo.impl.compute_target(batch_torch_r)
        original_critic_loss = algo.impl.compute_critic_loss(batch_torch_r, q_tpn_r)
        algo.impl._critic_optim.zero_grad()
        original_critic_loss.backward()
        algo.impl._critic_optim.step()

        try:
            # A. Base Actor Loss 
            # The actor is updated twice on D_r in implementation, to maintain a retained-data update budget
            # comparable to methods operating on a full retained batch
            actor_loss = algo.impl.compute_actor_loss(batch_torch_r) 
            total_actor_loss = actor_loss

            algo.impl._actor_optim.zero_grad() 
            total_actor_loss.backward()
            algo.impl._actor_optim.step() 
            
            actor_loss_val = actor_loss.item()

        except NotImplementedError:
            actor_loss_val = 0.0


        # B. Penalty Loss (on D_f ONLY)

        q_values = algo.impl._q_func(
            batch_torch_f.observations, batch_torch_f.actions, "mean"
        ).view(-1)
        q_values_ref = ref_algo.impl._q_func(
            batch_torch_f.observations, batch_torch_f.actions, "mean"
        ).view(-1).detach()
        q_values_on_retain = algo.impl._q_func(
            batch_torch_r.observations, batch_torch_r.actions, "mean"
        ).view(-1).detach()

        v_values_estimate = estimate_target_style_state_value(
            algo, batch_torch_f.observations
        ).detach()
        v_values_estimate_on_retain = estimate_target_style_state_value(
            algo, batch_torch_r.observations
        ).detach()
        v_values_estimate_ref = estimate_target_style_state_value(
            ref_algo, batch_torch_f.observations
        ).detach()
        v_values_estimate_on_retain_ref = estimate_target_style_state_value(
            ref_algo, batch_torch_r.observations
        ).detach()
        adv_values_f = q_values - v_values_estimate_ref

        if 'quantile' in args.fixed_threshold_or_quantile:
            loss_penalty, description = compute_adaptive_entropy_based_penalty(q_values,v_values_estimate,q_values_ref,v_values_estimate_ref,q_values_on_retain=q_values_on_retain,v_values_estimate_on_retain=v_values_estimate_on_retain,v_values_on_retain_ref=v_values_estimate_on_retain_ref,filter=filter, dataset_ratio=dataset_ratio)
        else:
            loss_penalty, description = compute_adaptive_entropy_based_penalty(q_values,v_values_estimate,q_values_ref,v_values_estimate_ref,q_values_on_retain=q_values_on_retain,v_values_estimate_on_retain=v_values_estimate_on_retain,v_values_on_retain_ref=v_values_estimate_on_retain_ref,filter=filter, dataset_ratio=dataset_ratio)
        

        critic_penalty_on_Df = args.lambda_penalty * loss_penalty 
        
        algo.impl._critic_optim.zero_grad()
        critic_penalty_on_Df.backward()
        algo.impl._critic_optim.step()
        
        total_loss = original_critic_loss+critic_penalty_on_Df




        try:
            # A. Base Actor Loss (Maximizing Q on D_r)
            actor_loss = algo.impl.compute_actor_loss(batch_torch_r) 
            total_actor_loss = actor_loss

            algo.impl._actor_optim.zero_grad() 
            total_actor_loss.backward()
            algo.impl._actor_optim.step() 
            
            actor_loss_val = actor_loss.item()

        except NotImplementedError:
            actor_loss_val = 0.0

            
        
        # 5. --- STANDARD TARGET UPDATE ---
        if step % args.target_update_interval == 0:
            if hasattr(algo.impl, "update_critic_target"):
                algo.impl.update_critic_target()
            if hasattr(algo.impl, "update_actor_target"):
                algo.impl.update_actor_target()
            
            
        # 6. --- Collect Metrics ---
        step_metrics["critic_loss/original_retained"].append(original_critic_loss.item())
        step_metrics["actor_loss/retained"].append(actor_loss_val)
        if imitator_loss is not None:
            step_metrics["imitator_loss"].append(imitator_loss)
        step_metrics["custom_loss/ratio_loss"].append(description['ratio_loss'])
        step_metrics["custom_lossloss_self_stable"].append(description['loss_self_stable'])
        step_metrics["custom_loss/loss_alignment"].append(description["loss_alignment"])
        step_metrics["custom_stats/advantage_mean_f"].append(adv_values_f.mean().item())
        step_metrics["custom_stats/std_adv_ratio"].append(description["std_adv_ratio"])
        if v_loss is not None:
             step_metrics["critic_loss/v_loss"].append(v_loss.mean().item())   
        
        # 7. --- Training Metric Logging & Recording ---
        if step % args.logging_steps == 0:
            mean_metrics = {k: np.mean(v) for k, v in step_metrics.items()}
            append_training_metrics_csv(metrics_csv_path, step, mean_metrics)
            logger.commit(mean_metrics, step)
            step_metrics.clear()
            pbar.set_description(
                                f"Step {step} | Crit_On_Dr: {original_critic_loss.item():.3f} | ratio_loss: {description['ratio_loss']:.3f} | loss_alignment: {description['loss_alignment']:.3f} | loss_self_stable: {description['loss_self_stable']:.3f} | std_adv_ratio: {description['std_adv_ratio']:.3f}"
                            )
        
        # 8. --- Environment Evaluation ---
        if step % args.eval_for_record_interval == 0 and eval_env:
            print(f"\nStep {step}: Evaluating model...")
            eval_scores = evaluate_on_environment(eval_env, n_trials=10)(algo)
            logger.add_metric("evaluation_mean_reward", eval_scores)

        # 9. --- Saving ---
        if step % args.eval_interval == 0:
            save_path = os.path.join(logger._logdir, f"model_{step}.pt")
            algo.save_model(save_path)
            print(f"Model saved: {save_path}")

    print("Custom unlearning loop finished.")



# ========================================================================
#  SECTION 4: MAIN EXECUTION SCRIPT
# ========================================================================

def run_unlearning(args: argparse.Namespace):
    """
    Main function to set up and run the unlearning process.
    """
    
    # --- 1.  Setup Logging ---
    datasets_str = "_".join(args.datasets)
    ratios_str = "_".join([str(r) for r in args.retained_ratios])
    
    #  ABLATION: Modified run_id to indicate this is the frozen_V ablation
    suffix = "UDRU_"
    if args.shuffle==1:
        run_id = (
            f"seed_{args.seed}/Unlearning_{args.algo}_steps_{str(args.unlearning_steps)}/lambda{args.lambda_penalty}_{suffix}/ratios_{ratios_str}"
        )
    else:
        run_id = (
            f"no_shuffle_seed_{args.seed}/Unlearning_{args.algo}_steps_{str(args.unlearning_steps)}/lambda{args.lambda_penalty}_{suffix}/ratios_{ratios_str}"
        )
    uncertainty_type_str = args.uncertainty_file_f.replace('.npy','')
    
    full_log_dir = os.path.join(args.log_dir, args.dataset, datasets_str, run_id, uncertainty_type_str)
    if 'q' in args.component_to_analyze:
        logger = D3RLPyLogger(
            experiment_name="q_func" , 
            root_dir=full_log_dir,
            with_timestamp=True,
        )   
    elif 'q' not in args.component_to_analyze:
        logger = D3RLPyLogger(
        experiment_name="policy_network" , 
        root_dir=full_log_dir,
        with_timestamp=True,
        )
    print(f"Logging directory set up: {full_log_dir}")
    
    
    # --- 2.  Load Algorithm ---
    print(f"--- Resuming model from {args.model_to_unlearn_dir} ---")

    params_path = os.path.join(args.model_to_unlearn_dir, 'params.json')
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"params.json not found in {args.model_to_unlearn_dir}")

    with open(params_path, 'r') as f:
        params = json.load(f)
    algo_name = params.get("algorithm", args.algo)

    try:
        AlgoClass = getattr(d3rlpy.algos, algo_name)
        # Initialize Mutable Model
        algo = AlgoClass.from_json(params_path, use_gpu=args.gpu)
        
        #  *** Initialize Reference Model ***
        print(f"Initializing Reference Model (frozen copy)...")
        ref_algo = AlgoClass.from_json(params_path, use_gpu=args.gpu)
    except AttributeError:
        raise ValueError(f"Algorithm '{algo_name}' not found in d3rlpy.algos")

    # Load Weights
    model_files = glob.glob(os.path.join(args.model_to_unlearn_dir, 'model_*.pt'))
    if not model_files:
        model_path_fallback = os.path.join(args.model_to_unlearn_dir, 'model.pt')
        if os.path.exists(model_path_fallback):
            model_files = [model_path_fallback]
        else:
            raise FileNotFoundError(f"No model_*.pt found.")

    if len(model_files) == 1 and model_files[0].endswith('model.pt'):
        latest_model_path = model_files[0]
    else:
        step_nums = [int(re.search(r'model_(\d+).pt', f).group(1)) for f in model_files if re.search(r'model_(\d+).pt', f)]
        latest_step = max(step_nums)
        latest_model_path = os.path.join(args.model_to_unlearn_dir, f'model_{latest_step}.pt')

    # Load into Mutable Model
    load_model_with_fix(algo, latest_model_path, args.gpu)
    
    # Load into Reference Model
    load_model_with_fix(ref_algo, latest_model_path, args.gpu)
    
    #  *** Freeze Reference Model (FIXED) ***
    # D3RLPy Impl objects are not nn.Modules, so we iterate known sub-modules.
    print("Freezing internal networks of Reference Model...")
    
    networks_to_freeze = [
        getattr(ref_algo.impl, '_policy', None),
        getattr(ref_algo.impl, '_q_func', None),
        getattr(ref_algo.impl, '_targ_q_func', None),
        getattr(ref_algo.impl, '_value_func', None),      # IQL
        getattr(ref_algo.impl, '_imitator', None),          # BCQ / PLAS
        getattr(ref_algo.impl, '_targ_policy', None),       # TD3 / CRR / PLAS
        getattr(ref_algo.impl, '_perturbation', None),       # PLAS with perturbation
        getattr(ref_algo.impl, '_targ_perturbation', None), # PLAS with perturbation
    ]
    
    for net in networks_to_freeze:
        if net is not None:
            # 1. Set to Eval Mode (important for BatchNorm/Dropout)
            net.eval()
            # 2. Disable Gradients
            for p in net.parameters():
                p.requires_grad = False
    
    print("Reference model frozen.")
    
    
    # --- 3.  Load Data & Build Dirs ---
    print("Loading and splitting datasets...")
    task = args.dataset
    datasets = args.datasets
    dirs = parse_dataset_dir(task, datasets)
    print(f"Splitting full dataset with {args.retained_ratios} retained...")
    dataset_required, dataset_remained = load_merged_dataset(
        dirs, 
        args.retained_ratios, 
        args.seed,
        shuffle=(args.shuffle==1)
    )

    # --- 4.  Setup Environment ---
    eval_env = None
    try:
        if dataset_required._datasets:
             reference_ds = dataset_required._datasets[-1]
        elif dataset_remained._datasets:
             reference_ds = dataset_remained._datasets[-1]
        else:
            raise ValueError("No datasets loaded.")
        eval_env_original = reference_ds.recover_environment(eval_env=True)
        is_antmaze_task = (task == 'antmaze')
        eval_env = FlattenDictObsWrapper(eval_env_original, is_antmaze=is_antmaze_task)
        d3rlpy.seed(args.seed)
        if hasattr(eval_env, 'action_space'): eval_env.action_space.seed(args.seed)
        if hasattr(eval_env, 'observation_space'): eval_env.observation_space.seed(args.seed)
    except Exception as e:
        print(f"Warning: Could not create evaluation env. {e}")
    
    # --- 5.  Extract Transitions & Load Uncertainty ---
    if len(dataset_required) > 0:
        train_episodes_r, _ = train_test_split(
            dataset_required, random_state=args.seed, test_size=0.01, shuffle=True 
        )
    else:
        train_episodes_r = []
    
    transitions_r = []
    for episode in tqdm(train_episodes_r, desc="Processing D_r_train"):
        transitions_r.extend(episode.transitions)
    
    transitions_f = []
    for episode in tqdm(dataset_remained, desc="Processing D_f"):
        transitions_f.extend(episode.transitions)
    
    if args.shuffle == 1:
        retained_str = "retain_ratios_"+ratios_str
    else:
        retained_str = "no_shuffle_retain_ratios_"+ratios_str
    if 'q' in args.component_to_analyze:
        uncertainty_f_path = os.path.join(args.model_to_unlearn_dir, retained_str, args.uncertainty_file_f)
    else:
        uncertainty_f_path = os.path.join(args.model_to_unlearn_dir, retained_str, args.component_to_analyze, args.uncertainty_file_f)
    if not os.path.exists(uncertainty_f_path):
        print("previous implementation requires gradient simiarlity file, but we do not need it here.")

    u_or_s = True
    if 'similarity' in args.uncertainty_or_similarity:
        u_or_s = False
    
    print('using uncertainty instead of similarity: ', u_or_s)
    algo.save_params(logger)
    
    # --- 6.  Run the Loop ---   
    try:
        custom_unlearning_loop(
            algo=algo,
            ref_algo=ref_algo, # Pass the reference model
            transitions_r=transitions_r,
            transitions_f=transitions_f,
            uncertainties_f=None,
            args=args,
            logger=logger,
            eval_env=eval_env,
            log_dir=full_log_dir, 
            u_or_s=u_or_s
        )
    except KeyboardInterrupt:
        print("Training interrupted.")
    except Exception as e:
        print(f"Error: {e}\n{traceback.format_exc()}")
    finally:
        if eval_env:
            if isinstance(eval_env, FlattenDictObsWrapper): eval_env.env.close()
            else: eval_env.close()
        print("Finished.")


def main():
    parser = argparse.ArgumentParser()
    
    #  --- Args for Model and Logging ---
    parser.add_argument('--model-to-unlearn-dir', type=str, required=True,
                        help="Path to the *directory* of the pre-trained model")
    parser.add_argument('--log-dir', type=str, default="SARAVI",
                        help="Root directory to save logs")
    
    parser.add_argument('--dataset', type=str, default='pointmaze', help="Task name")
    parser.add_argument('--algo', type=str, default='CQL', help="Algorithm name")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID")
    
    parser.add_argument('--datasets', type=str, nargs='+', default=['large-dense-v2', 'umaze-dense-v2', 'medium-dense-v2'])
    parser.add_argument('--retained-ratios', type=float, nargs='+', default=None)
    parser.add_argument('--seed', type=int, default=42, help="Seed")

    parser.add_argument('--uncertainty-file-f', type=str, default="forget_similarities.npy")
    
    #  --- Args for Custom Loop ---
    parser.add_argument('--unlearning-steps', type=int, default=20000)
    parser.add_argument('--lambda-penalty', type=float, default=1.0)
    parser.add_argument('--noise-beta', type=float, default=1.0)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--logging-steps', type=int, default=500)
    parser.add_argument('--eval-for-record-interval', type=int, default=10000)
    parser.add_argument('--eval-interval', type=int, default=10000)
    parser.add_argument('--target-update-interval', type=int, default=1)
    parser.add_argument('--uncertainty-or-similarity', type=str, default="similarity")
    parser.add_argument('--fixed-threshold-or-quantile', type=str, default="quantile")
    parser.add_argument('--value-for-quantile-or-threshold', type=float, default=0.8)
    parser.add_argument('--component_to_analyze', type=str, default='q_network',
                        help="Model component to analyze: 'q_network' or 'policy_network'")
    parser.add_argument('--shuffle', type=int, default=1)

    args = parser.parse_args()
    if "POISONED" in args.model_to_unlearn_dir:
        dir_components = args.model_to_unlearn_dir.split("/")
        poisoned_type = "POISONED"
        for component in dir_components:
            if "POISONED" in component:
                poisoned_type += component.split("POISONED")[-1]
                print(component)
                break
        args.log_dir = poisoned_type+args.log_dir
    torch.manual_seed(args.seed)
    if args.eval_interval>args.unlearning_steps:
        args.eval_interval = args.unlearning_steps
        print("change eval interval into ", str(args.eval_interval), " for small unlearning budgets.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.retained_ratios is None:
        args.retained_ratios = [1.0] * len(args.datasets)
    elif len(args.retained_ratios) != len(args.datasets):
        raise ValueError("--retained-ratios list must have the same length as --datasets list.")
    
    seed_str = 'seed_' + str(args.seed)
    if seed_str not in args.model_to_unlearn_dir:
        print("Potential seed mismatch check.")
        sys.exit(1)
        
    run_unlearning(args)

if __name__ == "__main__":
    main()