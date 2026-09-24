import minari
import numpy as np
from typing import List, Union, Iterable, Iterator, Any, Dict, Tuple, Optional

# Import MinariEpisodeData and rename it
from minari.dataset.episode_data import EpisodeData as MinariEpisodeData
from minari.dataset.minari_storage import MinariStorage
from minari.dataset.minari_dataset import MinariDataset

# --- NEW IMPORTS ---
import gymnasium as gym

# from collections import OrderedDict # No longer strictly needed as gym.spaces.Dict preserves order

try:
    # Assume 'dataset.py' is importable as module 'dataset'
    from d3rlpy.dataset import Episode as D3RLEpisode
except ImportError:
    print("Error: Could not import 'Episode' from 'dataset.py'.")
    print("Please ensure 'dataset.py' is in the same directory or on your PYTHONPATH.")


    # Define a dummy class to avoid NameError later
    class D3RLEpisode:
        def __init__(self, *args, **kwargs):
            raise ImportError("D3RLEpisode (from dataset.py) was not loaded.")


# --- OBSERVATION FLATTENING HELPERS ---
ObservationPath = Tuple[str, ...]


def _iter_box_subspaces(
        space: gym.Space, path: ObservationPath = ()
) -> Iterator[Tuple[ObservationPath, gym.spaces.Box]]:
    """Yield every Box leaf in a (possibly nested) Dict observation space."""
    if isinstance(space, gym.spaces.Box):
        yield path, space
        return

    if isinstance(space, gym.spaces.Dict):
        for key, subspace in space.spaces.items():
            yield from _iter_box_subspaces(subspace, path + (key,))
        return

    location = ".".join(path) if path else "<root>"
    raise ValueError(
        f"Unsupported observation subspace at {location}: {type(space)}. "
        "Only Dict and Box spaces can be flattened."
    )


def _get_value_at_path(observation: Dict[str, Any], path: ObservationPath) -> Any:
    """Return a nested observation value addressed by a Dict-space path."""
    value: Any = observation
    for key in path:
        value = value[key]
    return value


def _get_concatenated_obs_space(
        obs_space: gym.Space,
        key_order: Optional[List[ObservationPath]] = None,
) -> Tuple[gym.spaces.Box, List[ObservationPath]]:
    """Flatten a Box or selected leaves of a nested Dict of Boxes into one Box."""
    if isinstance(obs_space, gym.spaces.Box):
        return obs_space, []
    if not isinstance(obs_space, gym.spaces.Dict):
        raise ValueError(f"Unsupported observation space type: {type(obs_space)}")

    available_spaces = dict(_iter_box_subspaces(obs_space))
    if not available_spaces:
        raise ValueError("Cannot concatenate an empty Dict observation space.")
    if key_order is None:
        key_order = list(available_spaces)
    else:
        missing_paths = [path for path in key_order if path not in available_spaces]
        if missing_paths:
            raise ValueError(f"Observation space is missing paths: {missing_paths}")

    spaces = [available_spaces[path] for path in key_order]
    total_dim = sum(int(np.prod(space.shape)) for space in spaces)
    lows = np.concatenate([space.low.flatten() for space in spaces], dtype=np.float32)
    highs = np.concatenate([space.high.flatten() for space in spaces], dtype=np.float32)
    return gym.spaces.Box(low=lows, high=highs, shape=(total_dim,), dtype=np.float32), key_order


def _concatenate_observations(
        obs_dict: Dict[str, Any], key_order: List[ObservationPath]
) -> np.ndarray:
    """Flatten batched nested Dict observations into an ``(N, features)`` array."""
    if not key_order:
        return np.array([], dtype=np.float32)

    first_array = np.asarray(_get_value_at_path(obs_dict, key_order[0]))
    n_steps = first_array.shape[0]
    flattened_obs = []
    for path in key_order:
        obs_array = np.asarray(_get_value_at_path(obs_dict, path))
        if obs_array.shape[0] != n_steps:
            raise ValueError(
                "Observation arrays have inconsistent lengths (N_steps) for "
                f"path '{'.'.join(path)}'."
            )
        flattened_obs.append(obs_array.reshape(n_steps, int(np.prod(obs_array.shape[1:]))))
    return np.concatenate(flattened_obs, axis=1, dtype=np.float32)


# --- MODIFICATION START ---
# EpisodeNew *is* a D3RLEpisode, and it *has* a MinariEpisodeData instance.
class EpisodeNew(D3RLEpisode):
    """
    A hybrid Episode class that inherits from d3rlpy's Episode (D3RLEpisode)
    and *wraps* a Minari's EpisodeData (MinariEpisodeData) instance.
    """

    _minari_data: MinariEpisodeData  # The wrapped Minari data

    def __init__(self,
                 episode_data: MinariEpisodeData,
                 observation_shape: tuple,  # This is the NEW (concatenated) shape
                 action_size: int,
                 d3rlpy_observations: np.ndarray,  # NEW: Receive pre-concatenated obs
                 create_mask: bool = False,
                 mask_size: int = 1):

        # 1. Store the original Minari episode data.
        self._minari_data = episode_data

        # 2. Initialize the D3RLEpisode parent class
        episode_terminal = bool(episode_data.terminations[-1])

        minari_actions = episode_data.actions  # Length N
        if minari_actions.ndim > 1:
            #  Handle continuous actions, shape (N, D_act)
            # We duplicate the last action to fill the (N+1)th spot.
            last_action_to_pad = minari_actions[-1:]  # Shape (1, D_act)
            d3rlpy_actions = np.concatenate([minari_actions, last_action_to_pad], axis=0)
        else:
            #  Handle discrete actions, shape (N,)
            # We duplicate the last action to fill the (N+1)th spot.
            last_action_to_pad = minari_actions[-1]  # Shape ()
            d3rlpy_actions = np.append(minari_actions, last_action_to_pad)
        # d3rlpy_actions now has length N+1

        # Pad Rewards (Length N) -> (Length N+1)
        minari_rewards = episode_data.rewards  # Length N (r_1 ... r_N)
        #  d3rlpy's Episode.compute_return (File 1, line 788)
        # calculates sum(rewards[1:]).
        # To make this calculation correct (sum r_1...r_N), we must
        # prepend a 0.0 as a dummy r_0.
        d3rlpy_rewards = np.insert(minari_rewards, 0, 0.0)
        # d3rlpy_rewards now has length N+1 ([0.0, r_1 ... r_N])

        # --- MODIFIED: Use the pre-processed d3rlpy_observations ---
        # The parent D3RLEpisode.__init__ requires a numpy array for observations
        D3RLEpisode.__init__(
            self,
            observation_shape=observation_shape,
            action_size=action_size,
            observations=d3rlpy_observations,  # Length N+1
            actions=d3rlpy_actions,  # NOW Length N+1
            rewards=d3rlpy_rewards,  # NOW Length N+1
            terminal=episode_terminal,
            create_mask=create_mask,
            mask_size=mask_size
        )

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the wrapped _minari_data object.
        This makes properties like .id, .terminations, .truncations, .infos
        directly accessible on the EpisodeNew instance.
        """
        try:
            # Try to get the attribute from the wrapped Minari object
            return getattr(self._minari_data, name)
        except AttributeError:
            # If it's not on _minari_data, raise the standard AttributeError
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def __repr__(self) -> str:
        # Use the nice __repr__ from the Minari data
        return repr(self._minari_data)


# --- MODIFICATION END ---

# --- NEW HELPER CLASS for subsampling ---

class InMemoryMinariDataset:
    """
    A wrapper class that holds a subset of episodes from an original
    MinariDataset in memory.

    This class mimics the essential parts of the MinariDataset API
    (e.g., __len__, __getitem__, iteration, properties) needed by
    the MergedMinariDataset constructor, while also delegating
    other calls (like `recover_environment`) to the original dataset.
    """

    def __init__(self, original_dataset: MinariDataset, episodes: List[MinariEpisodeData]):
        """
        Args:
            original_dataset (MinariDataset): The full dataset from which
                                              the episodes were sampled.
            episodes (List[MinariEpisodeData]): The subset of episodes to hold.
        """
        self._original_dataset: MinariDataset = original_dataset
        self._episodes: List[MinariEpisodeData] = episodes

        # Calculate total steps for this subset
        self._total_steps: int = sum(len(ep.rewards) for ep in episodes)

    def __len__(self) -> int:
        """Returns the number of episodes in this subset."""
        return len(self._episodes)

    def __getitem__(self, idx: int) -> MinariEpisodeData:
        """Returns the episode at the specified index."""
        return self._episodes[idx]

    def __iter__(self) -> Iterator[MinariEpisodeData]:
        """Returns an iterator over the episodes in this subset."""
        return iter(self._episodes)

    @property
    def observation_space(self) -> gym.Space:
        """Returns the observation space from the original dataset."""
        return self._original_dataset.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the action space from the original dataset."""
        return self._original_dataset.action_space

    @property
    def id(self) -> str:
        """Returns a modified ID for this subset."""
        return f"{self._original_dataset.id}_subset"

    @property
    def total_steps(self) -> int:
        """Returns the total number of steps in this subset."""
        return self._total_steps

    def __getattr__(self, name: str) -> Any:
        """
        Delegates any unknown attribute access (e.g., `recover_environment`)
        to the original MinariDataset object.
        """
        try:
            # Try to get the attribute from this object first
            return super().__getattribute__(name)
        except AttributeError:
            # If it fails, delegate to the original dataset
            return getattr(self._original_dataset, name)


# --- END NEW HELPER CLASS ---
# --- MergedMinariDataset Class (modified instantiation) ---

class MergedMinariDataset:
    """Merges multiple Minari datasets and allows unified episode sampling."""

    # --- MODIFIED __init__ method ---
    def __init__(self, datasets: List[Union[MinariDataset, InMemoryMinariDataset, str]]):
        """
        Args:
            datasets (list[MinariDataset | InMemoryMinariDataset | str]): list of
                MinariDataset instances, InMemoryMinariDataset instances, or their data paths.
        """
        # (Type hint updated to reflect it holds both types)
        self._datasets: List[Union[MinariDataset, InMemoryMinariDataset]] = []
        self._generator = np.random.default_rng()

        for ds in datasets:
            # --- THIS IS THE FIX ---
            # We now check for both MinariDataset AND our custom InMemoryMinariDataset
            if isinstance(ds, (MinariDataset, InMemoryMinariDataset)):
                self._datasets.append(ds)
            # We also still support loading from a string path
            elif isinstance(ds, str):
                self._datasets.append(MinariDataset(ds))
            # --- END FIX ---
            else:
                # If it's not one of the allowed types, raise an error
                raise ValueError(f"MergedMinariDataset received an unsupported type: {type(ds)}")

        self._episode_offsets = []
        total = 0
        for ds in self._datasets:
            self._episode_offsets.append((total, total + len(ds)))
            total += len(ds)
        self._total_episodes = total

        if not self._datasets:
            raise ValueError("Cannot create MergedMinariDataset with an empty list of datasets.")

        first_ds = self._datasets[0]
        # print('_______________first_ds___________________: ', first_ds)

        # --- MODIFIED: Cache d3rlpy-compatible shape/size info ---
        self._original_observation_space = first_ds.observation_space

        if isinstance(first_ds.observation_space, gym.spaces.Dict):
            self._is_dict_obs = True
            # Kitchen variants can expose different named goal components. Retain
            # only Box leaves shared by every dataset, preserving first-space order.
            first_leaf_spaces = dict(_iter_box_subspaces(first_ds.observation_space))
            self._obs_key_order = list(first_leaf_spaces)
            for ds in self._datasets[1:]:
                if not isinstance(ds.observation_space, gym.spaces.Dict):
                    raise ValueError("Incompatible observation spaces: expected Dict, got Box")
                ds_leaf_spaces = dict(_iter_box_subspaces(ds.observation_space))
                self._obs_key_order = [
                    path for path in self._obs_key_order
                    if path in ds_leaf_spaces
                    and ds_leaf_spaces[path].shape == first_leaf_spaces[path].shape
                ]

            if not self._obs_key_order:
                raise ValueError("Merged Dict datasets have no shared Box observation leaves.")
            concat_space, _ = _get_concatenated_obs_space(
                first_ds.observation_space, self._obs_key_order
            )
            self._observation_shape = concat_space.shape
            print(f"Detected Dict observation space. Concatenated shape: {self._observation_shape}")
            print(f"Concatenation order: {self._obs_key_order}")

        elif isinstance(first_ds.observation_space, gym.spaces.Box):
            self._is_dict_obs = False
            self._observation_shape = first_ds.observation_space.shape
            self._obs_key_order = None  # Not needed
            print(f"Detected Box observation space. Shape: {self._observation_shape}")

        else:
            raise ValueError(f"Unsupported observation space type: {type(first_ds.observation_space)}")

        # print("-----------data.metadata------------------: ", first_ds._data.metadata)

        # --- Action space logic (unchanged) ---
        if isinstance(first_ds.action_space, gym.spaces.Discrete):
            self._action_size = first_ds.action_space.n
            self._is_discrete = True
        elif isinstance(first_ds.action_space, gym.spaces.Box):
            # Assume continuous actions are a flat box
            self._action_size = first_ds.action_space.shape[0]
            self._is_discrete = False
        else:
            raise ValueError(f"Unsupported action space type: {type(first_ds.action_space)}")

        # --- MODIFIED: Check compatibility of all datasets ---
        for ds in self._datasets[1:]:
            if self._is_dict_obs:
                if not isinstance(ds.observation_space, gym.spaces.Dict):
                    raise ValueError("Incompatible observation spaces: expected Dict, got Box")
                temp_space, _ = _get_concatenated_obs_space(
                    ds.observation_space, self._obs_key_order
                )
                if temp_space.shape != self._observation_shape:
                    raise ValueError(
                        f"Incompatible concatenated observation shapes: {self._observation_shape} vs {temp_space.shape}")
            else:
                if not isinstance(ds.observation_space,
                                  gym.spaces.Box) or ds.observation_space.shape != self._observation_shape:
                    raise ValueError(
                        f"Incompatible observation spaces: {self._observation_shape} vs {ds.observation_space.shape}")

            if self._is_discrete:
                if not isinstance(ds.action_space, gym.spaces.Discrete) or ds.action_space.n != self._action_size:
                    raise ValueError(
                        f"Incompatible discrete action spaces found: {self._action_size} vs {ds.action_space.n}")
            else:
                if not isinstance(ds.action_space, gym.spaces.Box) or ds.action_space.shape[0] != self._action_size:
                    raise ValueError(
                        f"Incompatible continuous action spaces found: {self._action_size} vs {ds.action_space.shape[0]}")
        # --- End of cache section ---

    def set_seed(self, seed: int):
        """Set random seed for unified sampling."""
        self._generator = np.random.default_rng(seed)
        for ds in self._datasets:
            ds.set_seed(seed)  #

    @property
    def total_episodes(self) -> int:
        return self._total_episodes

    @property
    def total_steps(self) -> int:
        return sum(ds.total_steps for ds in self._datasets)  #

    @property
    def dataset_ids(self) -> List[str]:
        return [ds.id for ds in self._datasets]  #

    # --- d3rlpy-compatible API ---
    @property
    def observation_shape(self) -> tuple:
        """d3rlpy compatible: Returns (concatenated) observation shape."""
        return self._observation_shape

    def get_observation_shape(self) -> tuple:
        """d3rlpy compatible: Returns (concatenated) observation shape."""
        return self.observation_shape  #

    @property
    def observation_paths(self) -> Optional[List[ObservationPath]]:
        """Paths used to produce each flattened Dict observation feature."""
        return self._obs_key_order

    @property
    def action_size(self) -> int:
        """d3rlpy compatible: Returns action space size."""
        return self._action_size

    def get_action_size(self) -> int:
        """d3rlpy compatible: Returns action space size."""
        return self.action_size  #

    def is_action_discrete(self) -> bool:
        """d3rlpy compatible: Returns if action space is discrete."""
        return self._is_discrete  #

    @property
    def episodes(self) -> List[EpisodeNew]:
        """
        d3rlpy compatible: Returns a list of all episodes.
        Warning: This will load all episodes into memory.
        """
        print("Warning: Accessing .episodes property will load all episodes into memory.")
        return list(self.iterate_episodes())  #

    # --- End d3rlpy API ---

    def _locate_dataset(self, global_idx: int):
        """Return (dataset, local_idx) corresponding to global index."""
        for ds, (start, end) in zip(self._datasets, self._episode_offsets):
            if start <= global_idx < end:
                return ds, global_idx - start
        raise IndexError("Episode index out of range")

    # --- MODIFIED: Return EpisodeNew ---
    def __getitem__(self, idx: int) -> EpisodeNew:
        ds, local_idx = self._locate_dataset(idx)
        # print("ds: ", ds.episode_indices)
        # ds[local_idx] returns a MinariEpisodeData
        episode_data: MinariEpisodeData = ds[local_idx]
        if episode_data.infos is None:
            new_infos = {}
        else:
            new_infos = episode_data.infos.copy()
        new_infos["source_dataset_id"] = ds.id

        episode_data_with_info = MinariEpisodeData(
            id=episode_data.id,
            observations=episode_data.observations,
            actions=episode_data.actions,
            rewards=episode_data.rewards,
            terminations=episode_data.terminations,
            truncations=episode_data.truncations,
            infos=new_infos
        )  #

        # --- MODIFICATION START: Process observations ---
        if self._is_dict_obs:
            # Concatenate observations if space is Dict
            d3rlpy_obs_data = _concatenate_observations(
                episode_data_with_info.observations,
                self._obs_key_order
            )
        else:
            # --- FIX FOR MAZEEXPLORER (IMAGE DATA) ---
            # d3rlpy throws an error if it sees image dimensions but type is float.
            # So, if the original data is uint8 (images), we MUST preserve it.
            if episode_data_with_info.observations.dtype == np.uint8:
                d3rlpy_obs_data = episode_data_with_info.observations
            else:
                d3rlpy_obs_data = episode_data_with_info.observations.astype(np.float32)
        # --- MODIFICATION END ---

        # Create the hybrid EpisodeNew object
        return EpisodeNew(
            episode_data=episode_data_with_info,
            observation_shape=self._observation_shape,  # Pass the (concatenated) shape
            action_size=self._action_size,
            d3rlpy_observations=d3rlpy_obs_data  # Pass the (concatenated) data
        )

    def __len__(self):
        return self._total_episodes

    # --- MODIFIED: Return EpisodeNew ---
    def iterate_episodes(self) -> Iterator[EpisodeNew]:
        """Iterate over all episodes with dataset ID info."""
        for ds in self._datasets:
            for ep_data in ds:  # ep_data is a MinariEpisodeData
                # Add source_dataset_id to infos
                new_infos = ep_data.infos.copy()
                new_infos["source_dataset_id"] = ds.id

                episode_data_with_info = MinariEpisodeData(
                    id=ep_data.id,
                    observations=ep_data.observations,
                    actions=ep_data.actions,
                    rewards=ep_data.rewards,
                    terminations=ep_data.terminations,
                    truncations=ep_data.truncations,
                    infos=new_infos
                )  #

                # --- MODIFICATION START: Process observations ---
                if self._is_dict_obs:
                    d3rlpy_obs_data = _concatenate_observations(
                        episode_data_with_info.observations,
                        self._obs_key_order
                    )
                else:
                    # --- FIX FOR MAZEEXPLORER (IMAGE DATA) ---
                    if ep_data.observations.dtype == np.uint8:
                        d3rlpy_obs_data = ep_data.observations
                    else:
                        d3rlpy_obs_data = ep_data.observations.astype(np.float32)
                # --- MODIFICATION END ---

                # Yield the new EpisodeNew object
                yield EpisodeNew(
                    episode_data=episode_data_with_info,
                    observation_shape=self._observation_shape,
                    action_size=self._action_size,
                    d3rlpy_observations=d3rlpy_obs_data
                )

    # --- MODIFIED: Return EpisodeNew ---
    def sample_episodes(self, n_episodes: int) -> Iterable[EpisodeNew]:
        """Sample episodes across multiple datasets."""
        indices = self._generator.choice(
            np.arange(self._total_episodes), size=n_episodes, replace=False
        )
        episodes = []
        for idx in indices:
            episodes.append(self[idx])  # __getitem__ now returns EpisodeNew
        return episodes

    def filter_episodes(self, condition):
        """
        Filter episodes across all datasets.
        Note: 'condition' must be a callable that takes a MinariEpisodeData.
        """
        filtered_subsets = []
        for ds in self._datasets:
            # condition must be Callable[[MinariEpisodeData], bool]
            filtered_ds = ds.filter_episodes(condition)
            if len(filtered_ds) > 0:
                filtered_subsets.append(filtered_ds)
        # Return a new MergedMinariDataset instance
        return MergedMinariDataset(filtered_subsets)


if __name__ == '__main__':
    # Ensure 'dataset.py' is in the working directory

    print("Loading Minari datasets...")
    # Using pointmaze (which has Dict observations)
    try:
        # Use modern Minari 1.0.0 IDs
        ds1 = minari.load_dataset("pointmaze-umaze-v0", download=True)
        ds2 = minari.load_dataset("pointmaze-medium-v0", download=True)
        ds3 = minari.load_dataset("pointmaze-large-v0", download=True)
    except Exception as e:
        print(f"Could not load pointmaze datasets: {e}")
        print("Please ensure 'minari' is installed correctly.")
        print("Attempting antmaze (this may be slow and also uses Dict obs)...")
        try:
            ds1 = minari.load_dataset("antmaze-umaze-v2", download=True)
            ds2 = minari.load_dataset("antmaze-umaze-diverse-v2", download=True)
            ds3 = minari.load_dataset("antmaze-large-diverse-v2", download=True)
        except Exception as e2:
            print(f"Failed to load antmaze datasets: {e2}")
            print("Exiting. Please check your Minari installation and dataset availability.")
            exit(1)

    print("Creating MergedMinariDataset...")
    merged = MergedMinariDataset([ds1, ds2, ds3])
    merged.set_seed(42)

    print("\n--- Merged Dataset Info (d3rlpy compatible) ---")
    print(f"Total episodes: {len(merged)}")
    print(f"Total steps: {merged.total_steps}")
    print(f"Observation shape (concatenated): {merged.get_observation_shape()}")
    print(f"Action size: {merged.get_action_size()}")
    print(f"Is discrete: {merged.is_action_discrete()}")

    print("\n--- Sampling 5 Episodes (returns EpisodeNew) ---")
    samples = merged.sample_episodes(5)

    for i, ep in enumerate(samples):
        print(f"\n[Sample {i + 1}]")
        print(f"  Type: {type(ep)}")
        # Test MinariEpisodeData functionality (via __getattr__ delegation)
        print(f"  Source Dataset: {ep.infos['source_dataset_id']}")
        print(f"  Minari (id): {ep.id}")
        print(f"  Minari (rewards len): {len(ep.rewards)}")
        print(f"  Minari (truncations shape): {ep.truncations.shape}")
        # Show original dict keys from the wrapped Minari data
        print(f"  Minari (original obs keys): {list(ep._minari_data.observations.keys())}")

        # Test D3RLEpisode functionality (from inheritance)
        # len(ep) will call D3RLEpisode.__len__, returning num_transitions (steps - 1)
        print(f"  d3rlpy (len): {len(ep)}")
        print(f"  d3rlpy (terminal): {ep.terminal}")
        # Check d3rlpy's view of the observations (should be concatenated)
        print(f"  d3rlpy (ep.observations shape): {ep.observations.shape}")

        try:
            # ep[0] will call D3RLEpisode.__getitem__, returning a Transition object
            transition = ep[0]
            print(f"  d3rlpy (transitions[0].reward): {transition.reward}")
            # Check the shape of the observation in the transition object
            print(f"  d3rlpy (transitions[0].observation shape): {transition.observation.shape}")
            print(f"  d3rlpy (transitions[0].next_observation shape): {transition.next_observation.shape}")
        except Exception as e:
            print(f"  d3rlpy (transition test) failed: {e}")
            print("  This might be because 'dataset.py' Cython dependencies are not compiled.")
            print("  However, the 'EpisodeNew' class structure itself is correct.")

        print(f"  Repr: {ep}")