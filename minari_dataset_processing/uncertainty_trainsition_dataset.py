import sys
import os
import argparse
from typing import List, Tuple, Iterator, Union

import numpy as np
import minari
from tqdm import tqdm

# ---  Imports needed by load_merged_dataset ---
from sklearn.model_selection import train_test_split

# ---  Imports from User's Scripts ---
#  Use a robust path relative to this script file
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))

try:
    #  Import the custom dataset classes
    from minari_dataset_processing.merged_data_with_episode import (
        MergedMinariDataset, 
        InMemoryMinariDataset, 
        EpisodeNew
    )
    #  Import d3rlpy Transition object for type hinting
    from d3rlpy.dataset import Transition
except ImportError as e:
    print(f"Error: Could not import from 'minari_dataset_processing'.")
    print(f"Details: {e}")
    print("Please ensure 'merged_data_with_episode.py' (in minari_dataset_processing) is accessible.")
    sys.exit(1)


# ---  START: Copy 'load_merged_dataset' ---
#  This function is copied from your other scripts (e.g., precompute_uncertainty.py)
#  to make this new script self-contained.
def load_merged_dataset(dirs: List[str], ratios: List[float], seed: int) -> Tuple[
    MergedMinariDataset, MergedMinariDataset]:
    """
     Loads and splits datasets into 'required' (D_r) and 'remained' (D_f).
    (Copied from precompute_uncertainty.py)
    """
    if len(dirs) != len(ratios):
        raise ValueError(f"Mismatch in length: {len(dirs)} directories and {len(ratios)} ratios provided.")
    required_datasets_list = []
    remained_datasets_list = []
    print("--- Loading and Splitting Datasets (Original Loader) ---")
    for dir_name, ratio in zip(dirs, ratios):
        if not (0.0 <= ratio <= 1.0):
            raise ValueError(f"Ratio must be between 0.0 and 1.0, but got {ratio} for {dir_name}")
        print(f"Loading {dir_name} (retained ratio: {ratio * 100:.1f}%)")
        full_dataset = minari.load_dataset(dir_name)
        if ratio == 1.0:
            required_datasets_list.append(full_dataset)
            empty_remained = InMemoryMinariDataset(original_dataset=full_dataset, episodes=[])
            remained_datasets_list.append(empty_remained)
        elif ratio == 0.0:
            empty_required = InMemoryMinariDataset(original_dataset=full_dataset, episodes=[])
            required_datasets_list.append(empty_required)
            remained_datasets_list.append(full_dataset)
        else:
            required_ep_list, remained_ep_list = train_test_split(
                full_dataset, train_size=ratio, random_state=seed, shuffle=True
            )
            required_subset = InMemoryMinariDataset(original_dataset=full_dataset, episodes=required_ep_list)
            remained_subset = InMemoryMinariDataset(original_dataset=full_dataset, episodes=remained_ep_list)
            required_datasets_list.append(required_subset)
            remained_datasets_list.append(remained_subset)
    print("----------------------------------------")
    print("Creating 'required' (retained) merged dataset (D_r)...")
    dataset_required = MergedMinariDataset(required_datasets_list)
    print("Creating 'remained' (forget) merged dataset (D_f)...")
    dataset_remained = MergedMinariDataset(remained_datasets_list)
    return dataset_required, dataset_remained
# ---  END: Copy 'load_merged_dataset' ---


# ========================================================================
#  --- START: New Code Framework ---
# ========================================================================

class UncertaintyTransitionDataset:
    """
     A wrapper class that binds a MergedMinariDataset to a corresponding
     uncertainty score file (.npy).
    
     This class iterates over individual *transitions*, yielding
     a tuple of (transition, uncertainty_score) for each.
    
     If the uncertainty file is not found, it initializes all
     scores to 0.0, as requested.
    """
    
    def __init__(self, 
                 base_dataset: MergedMinariDataset, 
                 uncertainty_filepath: str):
        """
         Initializes the wrapper.
        
        Args:
            base_dataset (MergedMinariDataset): The loaded dataset (e.g., D_r or D_f).
            uncertainty_filepath (str): Path to the .npy file containing scores.
        """
        self.base_dataset = base_dataset
        self.uncertainty_filepath = uncertainty_filepath
        
        #  Load or initialize the uncertainty scores
        self.scores = self._load_or_initialize_scores()

    def _load_or_initialize_scores(self) -> np.ndarray:
        """
         Internal function to load scores from .npy file or create a zero array.
        """
        #  The total number of transitions in the merged dataset.
        #  base_dataset.total_steps is the sum of len(episode.rewards)
        #  which is exactly the number of transitions.
        try:
            #  Get the exact number of transitions this dataset contains
            required_length = self.base_dataset.total_steps
        except Exception as e:
            print(f"Error: Could not get total_steps from base_dataset: {e}")
            print("Base dataset might be empty or invalid.")
            return np.array([], dtype=np.float32)

        if required_length == 0:
            print("Warning: Base dataset has 0 transitions. Initializing empty score array.")
            return np.array([], dtype=np.float32)

        try:
            print(f"Attempting to load uncertainty scores from: {self.uncertainty_filepath}")
            scores = np.load(self.uncertainty_filepath)
            
            #  CRITICAL: Validate the length
            if len(scores) != required_length:
                raise ValueError(
                    f"Length mismatch! Dataset has {required_length} transitions, "
                    f"but file '{self.uncertainty_filepath}' has {len(scores)} scores."
                    "  Check if --ratios/--seed match the precomputation."
                )
            
            print(f"Successfully loaded {len(scores)} scores.")
            return scores.astype(np.float32)

        except FileNotFoundError:
            #  This is the requested behavior: auto-zero if not found.
            #  This is the expected path for D_r (the retain set).
            print(f"Warning: Uncertainty file not found at {self.uncertainty_filepath}.")
            print(f"Initializing {required_length} scores to 0.0.")
            return np.zeros(required_length, dtype=np.float32)
            
        except Exception as e:
            print(f"Error loading or validating uncertainty file: {e}")
            raise e

    def __len__(self) -> int:
        """
         Returns the total number of *transitions* in the dataset.
        """
        return len(self.scores)

    def __iter__(self) -> Iterator[Tuple[Transition, float]]:
        """
         Iterates over all transitions in order, yielding
         (transition, score) tuples.
        """
        transition_index = 0
        
        #  Iterate over episodes (EpisodeNew objects)
        #  The order is deterministic (based on load_merged_dataset)
        for episode in self.base_dataset:
            
            #  Iterate over transitions (d3rlpy.dataset.Transition objects)
            #  This iterates from the first transition to the last in the episode.
            #  Note: 'episode' is a d3rlpy.dataset.Episode, which is iterable.
            for transition in episode:
                
                if transition_index >= len(self.scores):
                    #  This should not happen if length check passed, but as a safeguard.
                    print(f"Error: Transition index {transition_index} out of bounds for scores array (len {len(self.scores)}).")
                    break
                
                #  Yield the transition and its corresponding score
                yield (transition, self.scores[transition_index])
                
                transition_index += 1
        
        #  Final check to ensure consistency
        if transition_index != len(self.scores) and len(self.scores) != 0:
             print(f"Warning: Iterator finished at index {transition_index}, but expected {len(self.scores)} transitions.")


def load_datasets_with_uncertainty(
    dirs: List[str], 
    ratios: List[float], 
    seed: int, 
    uncertainty_file_r: str,
    uncertainty_file_f: str
) -> Tuple[UncertaintyTransitionDataset, UncertaintyTransitionDataset]:
    """
     High-level loader function.
    
     1. Loads D_r and D_f using the original 'load_merged_dataset'.
     2. Wraps each dataset in 'UncertaintyTransitionDataset'.
    
    Args:
        dirs (List[str]): List of Minari dataset directories/IDs.
        ratios (List[float]): List of ratios (0.0 to 1.0) to *keep* (D_r).
        seed (int): The random seed for train_test_split.
        uncertainty_file_r (str): Path to the .npy file for D_r (retained).
                                  (File not existing will result in 0.0 scores)
        uncertainty_file_f (str): Path to the .npy file for D_f (forget).
                                  (File must exist and match length if D_f is not empty)

    Returns:
        Tuple[UncertaintyTransitionDataset, UncertaintyTransitionDataset]:
            - u_dataset_r: The wrapped retained dataset.
            - u_dataset_f: The wrapped forget dataset.
    """
    print("\n--- Loading Datasets with Uncertainty Wrapper ---")
    
    #  1. Call the original loader function
    #  This function is copied at the top of this script
    dataset_r_base, dataset_f_base = load_merged_dataset(dirs, ratios, seed)

    print("\n--- Wrapping Retained Dataset (D_r) ---")
    #  2. Wrap the retained dataset
    u_dataset_r = UncertaintyTransitionDataset(
        base_dataset=dataset_r_base,
        uncertainty_filepath=uncertainty_file_r
    )
    
    print("\n--- Wrapping Forget Dataset (D_f) ---")
    #  3. Wrap the forget dataset
    u_dataset_f = UncertaintyTransitionDataset(
        base_dataset=dataset_f_base,
        uncertainty_filepath=uncertainty_file_f
    )
    
    print("\n--- Loading Complete ---")
    print(f"Retained transitions (D_r): {len(u_dataset_r)}")
    print(f"Forget transitions (D_f):   {len(u_dataset_f)}")
    
    #  4. Return the two new wrapped objects
    return u_dataset_r, u_dataset_f

# ========================================================================
#  --- END: New Code Framework ---
# ========================================================================


if __name__ == '__main__':
    """
     Example usage of the new loader.
    
     This example mimics the setup from 'precompute_uncertainty.py'
     but uses the new loader.
    
     !!! NOTE: This example requires 'minari' and the 
     'pointmaze-large-dense-v2' etc. datasets to be available.
    
     Example command to run this test:
    
    python uncertainty_dataset_loader.py \
        --model-dir "/path/to/your/Fully_trained/pointmaze/CQL_..." \
        --datasets "large-dense-v2" "umaze-dense-v2" \
        --ratios 0.8 0.0 \
        --uncertainty-filename "forget_similarities.npy"
    
     This command would:
     1. Load D_r (80% of large-dense) and D_f (100% of umaze-dense).
     2. Try to load "retain_scores.npy" for D_r (will fail and set to 0).
     3. Try to load "forget_similarities.npy" for D_f (should succeed).
    """
    
    #  --- Setup argparse for the example ---
    parser = argparse.ArgumentParser(description="Example of UncertaintyTransitionDataset loader")
    
    #  Args needed by load_merged_dataset
    parser.add_argument('--dataset', type=str, default='pointmaze',
                        help="Task name (e.g., 'pointmaze', 'antmaze')")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['large-dense-v2', 'umaze-dense-v2'],
                        help="List of *all* sub-dataset names involved.")
    parser.add_argument('--ratios', type=float, nargs='+', default=None,
                        help="List of ratios (0.0 to 1.0) to *keep* (D_r).")
    
    #  Args needed for the new loader
    parser.add_argument('--model-dir', type=str, required=True,
                        help=" Path to the log directory containing the .npy file (e.g., Fully_trained/pointmaze/CQL_...)")
    parser.add_argument('--uncertainty-filename', type=str, default='forget_similarities.npy',
                        help="Name of the uncertainty file for the *forget set* (D_f)")
    
    args = parser.parse_args()

    #  --- 1. Prepare Arguments for the Loader ---
    
    #  Default ratios if not provided
    if args.ratios is None:
        ratios = [1.0] * len(args.datasets)
    elif len(args.ratios) != len(args.datasets):
        raise ValueError(
            f"Must provide {len(args.datasets)} ratios for datasets {args.datasets}, but got {len(args.ratios)}."
        )
    else:
        ratios = args.ratios
        
    #  Get dataset directories
    task = args.dataset
    dirs = []
    if task in ['pointmaze', 'antmaze']:
        for i in args.datasets:
            dir_name = 'D4RL' + '/' + task + '/' + i
            dirs.append(dir_name)
    elif task in ['halfcheetah', 'walker2d']:
        for i in args.datasets:
            dir_name = 'mujoco' + '/' + task + '/' + i
            dirs.append(dir_name)

    #  Define the paths for the uncertainty files
    
    #  Path for the FORGET set (D_f)
    forget_uncertainty_path = os.path.join(args.model_dir, args.uncertainty_filename)
    
    #  Path for the RETAIN set (D_r).
    #  We assume this file doesn't exist, so scores will be 0.
    retain_uncertainty_path = os.path.join(args.model_dir, "retain_uncertainty.npy") 
    
    print("--- Example Run Starting ---")
    print(f"Loading datasets: {dirs}")
    print(f"Retain ratios: {ratios}")
    print(f"Forget set .npy path: {forget_uncertainty_path}")
    print(f"Retain set .npy path: {retain_uncertainty_path} (expected to fail and set to 0)")

    #  --- 2. Call the New Loader Function ---
    try:
        u_dataset_r, u_dataset_f = load_datasets_with_uncertainty(
            dirs=dirs,
            ratios=ratios,
            seed=args.seed,
            uncertainty_file_r=retain_uncertainty_path,
            uncertainty_file_f=forget_uncertainty_path
        )
        
        #  --- 3. Demonstrate Iteration (as requested, no filtering) ---
        
        print(f"\n--- Iterating over FORGET set (D_f) (first 5 samples) ---")
        count_f = 0
        total_uncertainty_f = 0.0
        
        #  Iterate using tqdm for progress
        for (transition, uncertainty) in tqdm(u_dataset_f, desc="Processing D_f"):
            if count_f < 5:
                #  Print details for the first few samples
                print(f"  Sample {count_f}: Obs shape={transition.observation.shape}, Action={transition.action[0]:.2f}, Reward={transition.reward:.2f}, Uncertainty={uncertainty:.6f}")
            
            #  Example of processing all samples
            total_uncertainty_f += uncertainty
            count_f += 1
            
        print(f"Total forget transitions processed: {count_f} (should match {len(u_dataset_f)})")
        if count_f > 0:
            print(f"Average forget uncertainty: {total_uncertainty_f / count_f:.6f}")

        print(f"\n--- Iterating over RETAIN set (D_r) (first 5 samples) ---")
        count_r = 0
        total_uncertainty_r = 0.0
        
        #  Iterate using tqdm for progress
        for (transition, uncertainty) in tqdm(u_dataset_r, desc="Processing D_r"):
            if count_r < 5:
                #  The uncertainty here should be 0.0
                print(f"  Sample {count_r}: Obs shape={transition.observation.shape}, Action={transition.action[0]:.2f}, Reward={transition.reward:.2f}, Uncertainty={uncertainty:.6f}")
            
            #  Example of processing all samples
            total_uncertainty_r += uncertainty
            count_r += 1

        print(f"Total retain transitions processed: {count_r} (should match {len(u_dataset_r)})")
        if count_r > 0:
            print(f"Average retain uncertainty: {total_uncertainty_r / count_r:.6f} (should be 0.0)")
        
        print("\n--- Example Run Successful ---")

    except Exception as e:
        print(f"\n--- Example Run FAILED ---")
        print(f"An error occurred: {e}")
        print("Please check your paths and dataset availability.")
        sys.exit(1)