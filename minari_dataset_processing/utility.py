import os
import sys
from typing import List, Dict, Tuple, Union, Optional, Any
import numpy as np
import torch
import gymnasium as gym
import d3rlpy
import minari

from d3rlpy.base import LearnableBase
from d3rlpy.dataset import Episode
from d3rlpy.torch_utility import set_state_dict
from sklearn.model_selection import train_test_split
import random # Added random import

from minari_dataset_processing.merged_data_with_episode import MergedMinariDataset, EpisodeNew
from minari_dataset_processing.merged_data_with_episode import InMemoryMinariDataset
from minari_dataset_processing.merged_data_with_episode import (
    _get_concatenated_obs_space,
    _get_value_at_path,
)
from minari.dataset.episode_data import EpisodeData as MinariEpisodeData

# ==============================================================================
# 1. Observation Processing Helpers
# ==============================================================================

def _flatten_single_obs_dict(
        obs_dict: Dict[str, Any], key_order: List[Tuple[str, ...]]
) -> np.ndarray:
    """Flatten one possibly nested Dict observation in a prescribed path order."""
    flattened_obs_list = [
        np.asarray(_get_value_at_path(obs_dict, path)).reshape(-1)
        for path in key_order
    ]
    return np.concatenate(flattened_obs_list, axis=0, dtype=np.float32)


class MazeExplorerGymnasiumWrapper(gym.Env):
    """
    Adapter: Converts the MazeExplorer environment to the Gymnasium interface.
    """
    def __init__(self, number_maps=1, size=(10, 10), scaled_resolution=(42, 42), seed=None, **kwargs):
        super().__init__()
        
        # Ensure mazeexplorer is installed
        try:
            from mazeexplorer import MazeExplorer
        except ImportError:
            raise ImportError("Please install mazeexplorer package first.")
        
        # Explicitly pass kwargs to ensure parameters like random_key_positions are handled
        self.env = MazeExplorer(
            number_maps=number_maps, 
            size=size, 
            scaled_resolution=scaled_resolution, 
            seed=seed,
            **kwargs 
        )
        
        # Add spec info, which is required by Minari when creating datasets
        self.spec = gym.envs.registration.EnvSpec(
            id="MazeExplorer-Custom-v0",
            entry_point="mazeexplorer:MazeExplorer",
            max_episode_steps=2000 # Estimated max steps
        )
        
        # Convert Observation Space
        h, w, c = self.env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(h, w, c), dtype=np.uint8)
        
        # Convert Action Space
        if hasattr(self.env.action_space, 'n'):
            self.action_space = gym.spaces.Discrete(self.env.action_space.n)
        else:
            raise NotImplementedError("Only Discrete action spaces are currently supported.")
            
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # print("seed : ", seed) 
        if seed is not None:
            
            np.random.seed(seed)
            random.seed(seed)
            if hasattr(self.env, 'seed'):
                 if callable(self.env.seed):
                    self.env.seed(seed)
                 else:
                    self.env.seed = seed
        
        obs = self.env.reset()
        obs = obs.astype(np.uint8)
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = obs.astype(np.uint8)
        terminated = done
        truncated = False 
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()
    
    # Allow access to underlying env methods/attributes
    def __getattr__(self, name):
        return getattr(self.env, name)

class FlattenDictObsWrapper(gym.ObservationWrapper):
    """
    A wrapper to:
    1. Flatten a gymnasium.spaces.Dict observation space into a gymnasium.spaces.Box.
       - For 'antmaze', it applies the specific transform:
         [achieved_goal, desired_goal - achieved_goal, observation]
    2. Convert the gymnasium API (5-tuple step, 2-tuple reset) back to
       the old gym API (4-tuple step, obs reset) for d3rlpy compatibility.
    """

    def __init__(self, env: gym.Env, is_antmaze: bool = False,
                 key_order: Optional[List[Tuple[str, ...]]] = None):
        super().__init__(env)
        self._is_antmaze = is_antmaze
        self._is_dict_space = isinstance(env.observation_space, gym.spaces.Dict)

        if self._is_dict_space:
            # print("[Wrapper] Detected Dict space.")
            self.original_space = env.observation_space

            if self._is_antmaze:
                print("[Wrapper] Applying AntMaze-specific observation space.")
                try:
                    obs_shape = self.original_space.spaces['observation'].shape[0]
                    goal_shape = self.original_space.spaces['achieved_goal'].shape[0]
                    # New shape is [ach_goal, direction, obs]
                    new_shape = (goal_shape + goal_shape + obs_shape,)
                    self.observation_space = gym.spaces.Box(
                        shape=new_shape, low=-np.inf, high=np.inf, dtype=np.float32
                    )
                except KeyError:
                    print("[Wrapper] ERROR: is_antmaze=True, but space lacks 'observation' or 'achieved_goal'.")
                    print("[Wrapper] Falling back to simple flattening.")
                    self._is_antmaze = False
                    self.observation_space, self._key_order = _get_concatenated_obs_space(
                        self.original_space, key_order=key_order
                    )
            else:
                # print("[Wrapper] Applying simple dictionary flattening.")
                self.observation_space, self._key_order = _get_concatenated_obs_space(
                    self.original_space, key_order=key_order
                )
        else:
            print("[Wrapper] Detected non-Dict space. Applying Gym-API compatibility fix.")
            self.observation_space = env.observation_space
            self._key_order = []

    def observation(self, obs: Union[Dict[str, np.ndarray], np.ndarray]) -> np.ndarray:
        if self._is_dict_space:
            if self._is_antmaze:
                direction = obs["desired_goal"] - obs["achieved_goal"]
                return np.concatenate(
                    [obs["achieved_goal"], direction, obs["observation"]],
                    axis=0, dtype=np.float32
                )
            else:
                return _flatten_single_obs_dict(obs, self._key_order)
        return obs

    def reset(self, **kwargs):
        if 'seed' in kwargs:
            super().reset(seed=kwargs['seed'])
        obs, info = super().reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated
        return obs, reward, done, info


def preprocess_antmaze_to_d3rlpy(
        minari_dataset: Union[MergedMinariDataset, InMemoryMinariDataset]
) -> List[Episode]:
    """
    Converts a Minari dataset into a list of d3rlpy.dataset.Episode objects,
    applying specific AntMaze preprocessing (obs transform, reward -1, terminal handling).
    """
    print(f"  -> Iterating over {len(minari_dataset)} Minari episodes for AntMaze conversion...")
    d3rlpy_episodes = []
    obs_shape = minari_dataset.get_observation_shape()
    act_size = minari_dataset.get_action_size()

    for i, minari_ep in enumerate(minari_dataset):
        original_minari_data = minari_ep._minari_data
        obs_dict = original_minari_data.observations
        try:
            direction = obs_dict["desired_goal"] - obs_dict["achieved_goal"]
            stacked_obs = np.concatenate(
                [obs_dict["achieved_goal"], direction, obs_dict["observation"]],
                axis=1, dtype=np.float32
            )
        except (KeyError, IndexError, TypeError):
            # Fallback if not strictly AntMaze structure
            stacked_obs = minari_ep.observations.astype(np.float32)

        rewards = original_minari_data.rewards.astype(np.float32) - 1.0
        terminals = original_minari_data.terminations.astype(np.float32)

        d3rlpy_ep = Episode(
            obs_shape, act_size, stacked_obs,
            original_minari_data.actions.astype(np.float32),
            rewards, terminal=bool(terminals[-1])
        )
        d3rlpy_episodes.append(d3rlpy_ep)

    print(f"  -> Conversion complete. Created {len(d3rlpy_episodes)} d3rlpy Episodes.")
    return d3rlpy_episodes


# ==============================================================================
# 2. Dataset Loading Helpers
# ==============================================================================

def load_merged_dataset(dirs: List[str], ratios: List[float] = None, seed: int = 0, shuffle: bool = True) -> Tuple[
    MergedMinariDataset, MergedMinariDataset]:
    """
    Loads and merges multiple Minari datasets, splitting each one according to ratios.
    Returns (dataset_required, dataset_remained).
    """
    if ratios is None:
         ratios = [1.0] * len(dirs)
    if len(dirs) != len(ratios):
        raise ValueError(f"Mismatch: {len(dirs)} directories and {len(ratios)} ratios.")

    required_datasets_list = []
    remained_datasets_list = []

    print("--- Loading and Splitting Datasets ---")
    for dir_name, ratio in zip(dirs, ratios):
        if not (0.0 <= ratio <= 1.0):
            raise ValueError(f"Ratio must be [0.0, 1.0], got {ratio} for {dir_name}")

        print(f"Loading {dir_name} (ratio: {ratio * 100:.1f}%)")
        full_dataset = minari.load_dataset(dir_name)

        if ratio == 1.0:
            required_datasets_list.append(full_dataset)
            remained_datasets_list.append(InMemoryMinariDataset(full_dataset, episodes=[]))
        elif ratio == 0.0:
            required_datasets_list.append(InMemoryMinariDataset(full_dataset, episodes=[]))
            remained_datasets_list.append(full_dataset)
        else:
            req_eps, rem_eps = train_test_split(full_dataset, train_size=ratio, random_state=seed, shuffle=shuffle)
            print("Shuffle state: ", str(shuffle), " False indicates take the last part of dataset as forget set.")
            required_datasets_list.append(InMemoryMinariDataset(full_dataset, episodes=req_eps))
            remained_datasets_list.append(InMemoryMinariDataset(full_dataset, episodes=rem_eps))

    print("----------------------------------------")
    dataset_required = MergedMinariDataset(required_datasets_list)
    dataset_remained = MergedMinariDataset(remained_datasets_list)
    print(f"Total 'required' (Retained) episodes: {len(dataset_required)}")
    print(f"Total 'remained' (Forget) episodes: {len(dataset_remained)}")
    return dataset_required, dataset_remained


# ==============================================================================
# 3. Model Loading Helpers
# ==============================================================================

def load_model_with_fix(algorithm: LearnableBase, model_path: str, gpu_id: int):
    """
    Manually loads a model checkpoint to bypass the d3rlpy map_location bug.
    """
    print(f"--- Manually loading model from: {model_path} ---")
    if gpu_id >= 0:
        map_location = lambda storage, loc: storage.cuda(gpu_id)
    else:
        # print("Mapping model checkpoint to CPU")
        map_location = lambda storage, loc: storage.cpu()

    try:
        chkpt = torch.load(model_path, map_location=map_location)
        set_state_dict(algorithm.impl, chkpt)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"ERROR loading model weights: {e}")
        raise e


def parse_dataset_dir(task: str, datasets: List[str]):
    """
    offer dir list.
    """
    dirs = []
    if task in ['pointmaze', 'antmaze', 'kitchen', 'door']:
        for i in datasets:
            dirs.append('D4RL' + '/' + task + '/' + i)
    elif task in ['halfcheetah', 'walker2d','hopper']:
        for i in datasets:
            dirs.append('mujoco' + '/' + task + '/' + i)
    elif task in ['quadx']:
        for i in datasets:
            dirs.append(task + '/' + i)
    return dirs

import numpy as np


class PoisoningManager:
    def __init__(self, seed: int):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def apply_poisoning(self, episodes: list, strategy: str = 'invert_reward', noise_scale: float = 0.5):
        """
        Applies poisoning by creating NEW EpisodeNew objects with modified MinariEpisodeData.
        
        Args:
            episodes: List of EpisodeNew objects.
            strategy: Poisoning strategy.
                      - 'invert_reward': Rewards * -1 (TrajDeleter Defense)
                      - 'zero_reward': Rewards = 0
                      - 'linear_reward': Linspace from min to max (TrajDeleter Attack: Reward Noise)
                      - 'random_action': Uniform random actions
                      - 'action_noise': Add Gaussian noise
                      - 'action_bias': Set all actions to Global Mean * 1.5 (TrajDeleter Attack: Label Poisoning)
            noise_scale: Noise level for action_noise.
            
        Returns:
            List[EpisodeNew]: A list of new, poisoned episodes.
        """
        print(f"--- [PoisoningManager] Applying strategy: {strategy} with seed {self.seed} ---")
        
        poisoned_episodes = []
        
        # --- Pre-calculation for Global Strategies ---
        # TrajDeleter's 'action_bias' requires the global mean of the dataset being poisoned.
        bias_value = 0.0
        if strategy == 'action_bias':
            # Gather all actions from the provided episodes to compute the global mean
            all_actions_list = [ep._minari_data.actions for ep in episodes]
            if all_actions_list:
                # Concatenate to shape (Total_Steps, Action_Dim)
                all_actions_concat = np.concatenate(all_actions_list, axis=0)
                # Calculate scalar global mean across all dimensions and steps
                global_mean = np.mean(all_actions_concat)
                bias_value = global_mean * 1.5
                print(f"   [Action Bias] Calculated global action mean: {global_mean:.4f}, Target bias: {bias_value:.4f}")
            else:
                print("   [Action Bias] Warning: No episodes provided, skipping calculation.")

        for ep in episodes:
            # 1. Retrieve original data from the wrapped Minari object
            original_minari_data = ep._minari_data
            
            # Copy data to modify
            new_actions = original_minari_data.actions.copy()
            new_rewards = original_minari_data.rewards.copy()
            
            # 2. Apply Strategy
            if strategy == 'invert_reward':
                # [Method 1] Invert rewards (Used as 'Defense' in TrajDeleter paper)
                new_rewards = -new_rewards
                
            elif strategy == 'zero_reward':
                # [Method 2] Zero out rewards
                new_rewards = np.zeros_like(new_rewards)
            
            elif strategy == 'linear_reward':
                # [Method 3] Linear Reward Noise (TrajDeleter Attack Implementation)
                # Source: mujoco_random_reward.py
                # Logic: Replace rewards with a linear interpolation from min to max
                if len(new_rewards) > 1:
                    r_min = new_rewards.min()
                    r_max = new_rewards.max()
                    new_rewards = np.linspace(r_min, r_max, len(new_rewards)).astype(np.float32)
                
            elif strategy == 'random_action':
                # [Method 4] Replace actions with uniform noise
                new_actions = self.rng.uniform(-1.0, 1.0, size=new_actions.shape).astype(np.float32)
                
            elif strategy == 'action_noise':
                # [Method 5] Add Gaussian noise
                noise = self.rng.normal(0, noise_scale, size=new_actions.shape).astype(np.float32)
                new_actions = np.clip(new_actions + noise, -1.0, 1.0)
            
            elif strategy == 'action_bias':
                # [Method 6] Action Label Poisoning (TrajDeleter Attack Implementation)
                # Source: poisoning_training.py
                # Logic: Set ALL actions to (Mean * 1.5). 
                # Note: This broadcasts the scalar bias_value to the shape (N, Dim)
                new_actions = np.full_like(new_actions, bias_value)
                
            else:
                raise ValueError(f"Unknown poisoning strategy: {strategy}")

            # 3. Update Infos
            new_infos = original_minari_data.infos.copy()
            new_infos['poisoned'] = True
            new_infos['poison_strategy'] = strategy

            # 4. Create a NEW MinariEpisodeData object
            new_minari_data = MinariEpisodeData(
                id=original_minari_data.id,
                observations=original_minari_data.observations,
                actions=new_actions,
                rewards=new_rewards,
                terminations=original_minari_data.terminations,
                truncations=original_minari_data.truncations,
                infos=new_infos
            )

            # 5. Wrap it in EpisodeNew
            new_ep = EpisodeNew(
                episode_data=new_minari_data,
                observation_shape=ep.get_observation_shape(),
                action_size=ep.get_action_size(),
                d3rlpy_observations=ep.observations 
            )
            
            poisoned_episodes.append(new_ep)

        return poisoned_episodes