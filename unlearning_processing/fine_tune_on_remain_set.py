# UDRU_retrain_only.py
#
#  BASED ON 'UDRU_fix.py'.
#  MODIFICATION: Pure Retraining/Fine-tuning on Retain Set (D_r).
#
#  Key Changes:
#  1. REMOVED all operations related to Forget Set (D_f).
#     - No uncertainty loading.
#     - No D_f sampling.
#     - No Penalty loss.
#  2. Critic and Actor are updated solely based on their original losses on D_r.

import sys
import os
import argparse
import traceback
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import json
import glob 
import re 

import d3rlpy
import gymnasium as gym
import numpy as np
import torch
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

from unlearning_processing.unlearning_cost_runtime import (
    CostMeasurementController,
    CostMeasurementFailed,
    CostMeasurementStop,
    read_cost_config,
)


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


def _commit_metrics(logger: D3RLPyLogger, metrics: Dict[str, float], step: int) -> None:
    """Add step metrics to D3RLPy's buffer and flush them to CSV."""
    for name, value in metrics.items():
        if value is None:
            continue
        # Some d3rlpy implementations return one-element numpy arrays.
        scalar_value = float(np.asarray(value, dtype=np.float64).mean())
        metric_path = os.path.join(logger.logdir, f"{name}.csv")
        os.makedirs(os.path.dirname(metric_path), exist_ok=True)
        logger.add_metric(name, scalar_value)

    # D3RLPyLogger.commit takes (epoch, step), not (metrics, step).
    logger.commit(step, step)


# ========================================================================
#  SECTION 2: CUSTOM UNLEARNING LOOP (Retain Only)
# ========================================================================

def custom_unlearning_loop(
    algo: LearnableBase,
    transitions_r: List[Transition], 
    # transitions_f: List[Transition], <-- REMOVED
    # uncertainties_f: np.ndarray,     <-- REMOVED
    args: argparse.Namespace,
    logger: D3RLPyLogger, 
    eval_env: gym.Env,
    log_dir: str,
    cost_controller: Optional[CostMeasurementController] = None,
    original_model: Optional[LearnableBase] = None,
    # u_or_s: bool = True              <-- REMOVED
):
    """
     Training loop that optimizes ONLY on the Retain Set (D_r).
    """
    print(f"Starting custom loop (Retain Set Only) for {args.unlearning_steps} steps...")
    
    if not isinstance(algo, AlgoBase):
        raise TypeError("This loop is designed for Q-learning algorithms.")
    
    if algo.impl is None:
        raise ValueError("Algorithm implementation (impl) is not initialized.")

    n_transitions_r = len(transitions_r)

    if n_transitions_r == 0:
        print("Error: Retained set (D_r) is empty. Cannot perform update.")
        return
    
    # Use full batch size for D_r since we are not mixing D_f
    batch_size_r = args.batch_size 
        
    total_steps = args.unlearning_steps
    step_metrics = defaultdict(list)
    iter_r = BatchIterator(n_transitions_r, batch_size_r, seed=args.seed)
    imitator_loss = None
    pbar = tqdm(range(1, total_steps + 1), desc="Finetuning on D_r")
    for step in pbar:
        
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

        if cost_controller is not None:
            cost_controller.start_training()
        if hasattr(algo.impl, "update_value_func"):
            algo.impl._value_optim.zero_grad()
            v_loss = algo.impl.compute_value_loss(batch_torch_r)
            v_loss.backward()
            algo.impl._value_optim.step()
        if hasattr(algo.impl,"update_imitator"):
            # print(batch_torch_r.observations)
            # print(batch_torch_r.actions)
            """
                The update_imitator() function is decorated with @torch_api, does not support TorchMinibatch...
            """
            imitator_loss = algo.impl.update_imitator(batch_d3rlpy_r)
        # 2. --- CRITIC UPDATE (D_r ONLY) ---
        q_tpn_r = algo.impl.compute_target(batch_torch_r)
        critic_loss = algo.impl.compute_critic_loss(batch_torch_r, q_tpn_r)
        
        algo.impl._critic_optim.zero_grad()
        critic_loss.backward()
        algo.impl._critic_optim.step()


        # 3. --- ACTOR UPDATE (D_r ONLY) ---
        try:
            actor_loss = algo.impl.compute_actor_loss(batch_torch_r) 
            
            algo.impl._actor_optim.zero_grad() 
            actor_loss.backward()
            algo.impl._actor_optim.step() 
            
            actor_loss_val = actor_loss.item()
        except NotImplementedError:
            actor_loss_val = 0.0
            
        
        # 4. --- STANDARD TARGET UPDATE ---
        if step % args.target_update_interval == 0:
            if hasattr(algo.impl, "update_critic_target"):
                algo.impl.update_critic_target()
            if hasattr(algo.impl, "update_actor_target"):
                algo.impl.update_actor_target()

        if cost_controller is not None:
            cost_controller.after_update(
                algo,
                logical_step=step,
                original_model=original_model,
            )
            
            
        # 5. --- Collect Metrics ---
        step_metrics["critic_loss/retained"].append(critic_loss.item())
        step_metrics["actor_loss/retained"].append(actor_loss_val)
        if  imitator_loss is not None:
            step_metrics["imitator_loss/retained"].append(imitator_loss) 
        
        # 6. --- Logging ---
        if step % args.logging_steps == 0:
            mean_metrics = {k: np.mean(v) for k, v in step_metrics.items()}
            _commit_metrics(logger, mean_metrics, step)
            step_metrics.clear()
            if imitator_loss is None:
                pbar.set_description(
                    f"Step {step} | Critic: {critic_loss.item():.3f} | Actor: {actor_loss_val:.3f}"
                    )
            else:
                pbar.set_description(
                    f"Step {step} | Critic: {critic_loss.item():.3f} | Actor: {actor_loss_val:.3f} | Imitator: {imitator_loss:.3f} "
                    )                
        # 7. ---  Evaluation & Saving ---
        if (
            args.eval_for_record_interval > 0
            and step % args.eval_for_record_interval == 0
            and eval_env is not None
        ):
            print(f"\nStep {step}: Evaluating model...")
            eval_scores = evaluate_on_environment(eval_env, n_trials=10)(algo)
            _commit_metrics(
                logger,
                {"evaluation_mean_reward": eval_scores},
                step,
            )

        if (
            cost_controller is None
            and args.eval_interval > 0
            and step % args.eval_interval == 0
        ):
            save_path = os.path.join(logger._logdir, f"model_{step}.pt")
            algo.save_model(save_path)
            print(f"Model saved to {save_path}")

    # Flush a final partial logging window when total steps is not divisible
    # by logging_steps.
    if step_metrics:
        mean_metrics = {k: np.mean(v) for k, v in step_metrics.items()}
        _commit_metrics(logger, mean_metrics, total_steps)

    print("Custom loop (Retain Only) finished.")


# ========================================================================
#  SECTION 3: MAIN EXECUTION SCRIPT
# ========================================================================

def run_unlearning(args: argparse.Namespace):
    """
    Main function to set up and run the process.
    """
    
    # --- 1.  Setup Logging ---
    datasets_str = "_".join(args.datasets)
    ratios_str = "_".join([str(r) for r in args.retained_ratios])
    
    if args.shuffle==1:
        run_id = (
            f"seed_{args.seed}/FinetuneOnly_{args.algo}_steps_{str(args.unlearning_steps)}/ratios_{ratios_str}"
        )
    else:
        run_id = (
            f"no_shuffle_seed_{args.seed}/FinetuneOnly_{args.algo}_steps_{str(args.unlearning_steps)}/ratios_{ratios_str}"
        )
    
    full_log_dir = os.path.join(args.log_dir, args.dataset, datasets_str, run_id)
    cost_config = read_cost_config(args.cost_measurement_config)
    if cost_config is not None:
        full_log_dir = str(cost_config["training_log_dir"])
    
    logger = D3RLPyLogger(
        experiment_name="training" if cost_config is not None else "FinetuneOnly",
        root_dir=full_log_dir,
        with_timestamp=(cost_config is None),
    )
    print(f"Logging directory set up: {full_log_dir}")
    
    
    # --- 2.  Load Algorithm ---
    print(f"--- Resuming model from {args.model_to_unlearn_dir} ---")

    params_path = os.path.join(args.model_to_unlearn_dir, 'params.json')
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"params.json not found in {args.model_to_unlearn_dir}")
    print(f"Loading params from: {params_path}")

    with open(params_path, 'r') as f:
        params = json.load(f)

    algo_name = params.get("algorithm", args.algo)

    try:
        AlgoClass = getattr(d3rlpy.algos, algo_name)
        print(f"Initializing algorithm: {algo_name}")
        algo = AlgoClass.from_json(params_path, use_gpu=args.gpu) 
    except AttributeError:
        raise ValueError(f"Algorithm '{algo_name}' not found in d3rlpy.algos")

    model_files = glob.glob(os.path.join(args.model_to_unlearn_dir, 'model_*.pt'))
    if not model_files:
        model_path_fallback = os.path.join(args.model_to_unlearn_dir, 'model.pt')
        if os.path.exists(model_path_fallback):
            model_files = [model_path_fallback]
        else:
            raise FileNotFoundError(f"No model_*.pt or model.pt files found in {args.model_to_unlearn_dir}")

    if len(model_files) == 1 and model_files[0].endswith('model.pt'):
        latest_model_path = model_files[0]
    else:
        step_nums = [int(re.search(r'model_(\d+).pt', f).group(1)) for f in model_files if re.search(r'model_(\d+).pt', f)]
        latest_step = max(step_nums)
        latest_model_path = os.path.join(args.model_to_unlearn_dir, f'model_{latest_step}.pt')

    load_model_with_fix(algo, latest_model_path, args.gpu)
    print("Model weights loaded successfully.")
    
    
    # --- 3.  Load Data & Build Dirs ---
    print("Loading and splitting datasets...")
    task = args.dataset
    datasets = args.datasets
    dirs = parse_dataset_dir(task, datasets)

    print(f"Splitting full dataset with {args.retained_ratios} retained...")
    #  We still use load_merged_dataset to correctly identify D_r vs D_f
    dataset_required, dataset_remained = load_merged_dataset(
        dirs, 
        args.retained_ratios, 
        args.seed,
        shuffle=(args.shuffle==1)
    )
    cost_controller = CostMeasurementController.from_path(
        args.cost_measurement_config,
        dataset_required,
        dataset_remained,
    )
    #  dataset_remained (D_f) is IGNORED in this script.
    print(f"Loaded {len(dataset_required)} 'required' (D_r) episodes.")

    # --- 4. Setup Environment ---
    if cost_controller is not None:
        print(
            "Cost mode: skipping environment evaluation; "
            "CostMeasurementController owns checkpoint/distance evaluation."
        )
        eval_env = None
    else:
        print("Setting up evaluation environment...")
        eval_env = None
        try:
            if dataset_required._datasets:
                reference_ds = dataset_required._datasets[-1]
            else:
                raise ValueError("No datasets loaded.")

            eval_env_original = reference_ds.recover_environment(eval_env=True)
            is_antmaze_task = (task == 'antmaze')
            eval_env = FlattenDictObsWrapper(
                eval_env_original, is_antmaze=is_antmaze_task
            )

            d3rlpy.seed(args.seed)
            if hasattr(eval_env, 'action_space'):
                eval_env.action_space.seed(args.seed)
            if hasattr(eval_env, 'observation_space'):
                eval_env.observation_space.seed(args.seed)
        except Exception as e:
            print(f"Warning: Could not create evaluation env. {e}")

    # --- 5.  Extract Transitions ---
    print(f"Splitting 'required' (D_r) dataset into 99% train / 1% validation (seed={args.seed})...")
    if len(dataset_required) > 0:
        train_episodes_r, _ = train_test_split(
            dataset_required,
            random_state=args.seed,
            test_size=0.01,
            shuffle=True 
        )
    else:
        train_episodes_r = []
    
    print("Extracting transitions from Retained Set (D_r_train)...")
    transitions_r = []
    for episode in tqdm(train_episodes_r, desc="Processing D_r_train"):
        transitions_r.extend(episode.transitions)

    print(f"Loaded {len(transitions_r)} retain samples (D_r_train).")
    
    if not transitions_r:
         print("Error: Retained set is empty.")
         return

    algo.save_params(logger)
    
    # --- 6.  Run the Loop ---   
    try:
        custom_unlearning_loop(
            algo=algo,
            transitions_r=transitions_r,
            args=args,
            logger=logger,
            eval_env=eval_env,
            log_dir=full_log_dir, 
            cost_controller=cost_controller,
        )
        if cost_controller is not None:
            cost_controller.finish_if_incomplete()
    except CostMeasurementStop as stop:
        print(f"Cost measurement stopped normally: {stop}")
    except CostMeasurementFailed:
        raise
    except KeyboardInterrupt:
        if cost_controller is not None:
            cost_controller.fail(KeyboardInterrupt("Training interrupted."))
        else:
            print("Training interrupted by user.")
    except Exception as e:
        if cost_controller is not None:
            cost_controller.fail(e)
        else:
            print(f"Error: {e}\n{traceback.format_exc()}")
    finally:
        if eval_env:
            if isinstance(eval_env, FlattenDictObsWrapper):
                eval_env.env.close()
            else:
                eval_env.close()
        print("Process finished.")


def main():
    parser = argparse.ArgumentParser()
    
    #  --- Args for Model and Logging ---
    parser.add_argument('--model-to-unlearn-dir', type=str, required=True,
                        help="Path to the *directory* of the pre-trained model")
    parser.add_argument('--log-dir', type=str, default="FinetuneOnly_Only",
                        help="Root directory to save logs")
    
    parser.add_argument('--dataset', type=str, default='pointmaze',
                        help="Task name (e.g., 'pointmaze', 'antmaze')")
    parser.add_argument('--algo', type=str, default='CQL',
                        help="Algorithm name (e.g., 'CQL', 'IQL')")
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU ID (-1 for CPU)")
    
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['large-dense-v2', 'umaze-dense-v2', 'medium-dense-v2'],
                        help="List of *all* sub-dataset names involved.")
    parser.add_argument(
        '--retained-ratios', 
        type=float, 
        nargs='+', 
        default=None,
        help="List of ratios (0.0 to 1.0) to *keep* (D_r)."
    )
    parser.add_argument('--seed', type=int, default=42,
                        help="Seed for the train/test split")

    #  REMOVED: uncertainty-file-f, lambda-penalty, uncertainty-or-similarity, etc.
    
    #  --- Args for Custom Loop ---
    parser.add_argument('--unlearning-steps', type=int, default=20000,
                        help="Total steps for the custom unlearning loop")
    parser.add_argument('--batch-size', type=int, default=256,
                        help="Batch size for sampling D_r")
    parser.add_argument('--logging-steps', type=int, default=500,
                        help="How often to commit logs (in steps)")
    parser.add_argument('--eval-for-record-interval', type=int, default=10000,
                        help="How often to evaluate and save the model (in steps)")
    parser.add_argument('--eval-interval', type=int, default=50000,
                        help="How often to evaluate and save the model (in steps)")
    parser.add_argument('--target-update-interval', type=int, default=1,
                        help="How often to update target networks (1 for soft update)")
    parser.add_argument('--shuffle', type=int, default=1,
                        help="Whether apply shuffle in splitting dataset (kept for consistency with other scripts)")
    parser.add_argument('--cost-measurement-config', type=str, default=None,
                        help="Internal JSON controller configuration for cost measurement.")

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
    np.random.seed(args.seed)
    if args.retained_ratios is None:
        args.retained_ratios = [1.0] * len(args.datasets)
    elif len(args.retained_ratios) != len(args.datasets):
        raise ValueError("--retained-ratios list must have the same length as --datasets list.")
    if args.cost_measurement_config is None and args.eval_interval > args.unlearning_steps:
        args.eval_interval = args.unlearning_steps
        print("------------------shrinking eval interval----------------------")
    seed_str = 'seed_' + str(args.seed)
    if seed_str not in args.model_to_unlearn_dir:
        print("There is a potential mismatch between the original model's seed and the unlearning seed, please check.")
        sys.exit(1)
        
    run_unlearning(args)

if __name__ == "__main__":
    main()