import numpy as np
import copy
from d3rlpy.dataset import Episode

# Import EpisodeNew if available, to handle custom dataset structures
try:
    from minari_dataset_processing.merged_data_with_episode import EpisodeNew
except ImportError:
    pass

class PoisoningManager:
    def __init__(self, seed: int):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def apply_poisoning(self, episodes: list, strategy: str = 'invert_reward', noise_scale: float = 0.5):
        """
        Applies poisoning strategies to a list of episodes.
        Note: Performs a deep copy to ensure the original dataset is not modified.
        
        Args:
            episodes: List of episodes to be poisoned (D_poison).
            strategy: Poisoning strategy ('invert_reward', 'zero_reward', 'random_action', 'action_noise').
            noise_scale: Noise magnitude for 'action_noise' strategy.
        
        Returns:
            poisoned_episodes: List of processed/poisoned episodes.
        """
        print(f"--- [PoisoningManager] Applying strategy: {strategy} with seed {self.seed} ---")
        
        poisoned_episodes = []
        
        for ep in episodes:
            # 1. Deep copy to preserve the original dataset integrity (crucial for reproducibility)
            # Ensure numpy arrays inside the Episode object are copied
            new_observations = ep.observations.copy()
            new_actions = ep.actions.copy()
            new_rewards = ep.rewards.copy()
            new_terminal = ep.terminal # bool
            
            # 2. Modify data based on the selected strategy
            if strategy == 'invert_reward':
                # Logic from TrajDeleter: r = -r
                new_rewards = -new_rewards
                
            elif strategy == 'zero_reward':
                # Set all rewards to zero
                new_rewards = np.zeros_like(new_rewards)
                
            elif strategy == 'random_action':
                # Replace actions with random uniform noise (assuming action space is -1 to 1)
                # Using self.rng ensures determinism with the fixed seed
                new_actions = self.rng.uniform(-1.0, 1.0, size=new_actions.shape).astype(np.float32)
                
            elif strategy == 'action_noise':
                # Add Gaussian noise to the original actions
                noise = self.rng.normal(0, noise_scale, size=new_actions.shape).astype(np.float32)
                new_actions = np.clip(new_actions + noise, -1.0, 1.0)
            
            else:
                raise ValueError(f"Unknown poisoning strategy: {strategy}")

            # 3. Re-encapsulate into an Episode object
            # Use EpisodeNew if your environment relies on it (from merged_data_with_episode.py)
            # Otherwise, default to d3rlpy's standard Episode class
            try:
                # Standard d3rlpy interface
                poisoned_ep = Episode(
                    observations=new_observations,
                    actions=new_actions,
                    rewards=new_rewards,
                    terminal=new_terminal
                )
            except TypeError:
                # Handle EpisodeNew or different d3rlpy versions requiring different args
                # Assuming EpisodeNew constructor signature based on context
                poisoned_ep = EpisodeNew(
                    observations=new_observations,
                    actions=new_actions,
                    rewards=new_rewards,
                    terminations=np.full(new_rewards.shape, False), # Assumption
                    truncations=np.full(new_rewards.shape, False)   # Assumption
                )
                pass

            poisoned_episodes.append(poisoned_ep)

        return poisoned_episodes

    @staticmethod
    def merge_datasets(dataset_clean: list, dataset_poisoned: list):
        """Helper to merge the clean retained set with the poisoned set."""
        return dataset_clean + dataset_poisoned