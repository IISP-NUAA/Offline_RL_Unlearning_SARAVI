#!/usr/bin/env python3
"""Shared helpers for ORL-Auditor style dataset auditing.

This module adapts the ORL-Auditor pipeline to this repository's Minari/d3rlpy
data and model layout.  The public scripts are train_orl_auditor_critic.py and
evaluate_orl_auditor.py.
"""

import csv
import glob
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

try:
    from scipy import stats as scipy_stats
    from scipy.stats import wasserstein_distance as scipy_wasserstein_distance
except ImportError:
    scipy_stats = None
    scipy_wasserstein_distance = None


script_dir = os.path.dirname(os.path.abspath(__file__))
util_path = os.path.join(script_dir, "..")
sys.path.insert(0, util_path)


def load_d3rlpy():
    try:
        import d3rlpy
    except ImportError as e:
        raise ImportError(
            "d3rlpy is required to run ORL-Auditor training/evaluation. "
            "Install the repository runtime dependencies before executing this script."
        ) from e
    return d3rlpy


def load_minari_utilities():
    try:
        from minari_dataset_processing.utility import (
            load_merged_dataset,
            load_model_with_fix,
        )
    except ImportError as e:
        raise ImportError(
            "minari_dataset_processing.utility and its runtime dependencies are required "
            "to load datasets and d3rlpy checkpoints."
        ) from e
    return load_merged_dataset, load_model_with_fix

METRIC_NAMES = [
    "l1_distance",
    "l2_distance",
    "cos_distance",
    "wasserstein_distance",
]


class CriticModelWithoutLastActivation(nn.Module):
    """MLP critic used by ORL-Auditor for scalar state-action values."""

    def __init__(self, observation_action_size: int, n_hidden: int = 1024, n_output: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(observation_action_size, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 2 * n_hidden)
        self.fc3 = nn.Linear(2 * n_hidden, n_hidden)
        self.fc4 = nn.Linear(n_hidden, n_output)

    def forward(self, obs_act: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.fc1(obs_act))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return torch.squeeze(out, dim=1)


@dataclass
class LoadedModel:
    model: Any
    checkpoint_name: str
    checkpoint_path: str


@dataclass
class LoadedCritic:
    model: CriticModelWithoutLastActivation
    checkpoint_path: str
    input_dim: int
    hidden_size: int


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def save_json(data: Dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=json_default)
    print(f"Saved JSON to {output_path}")


def set_global_seeds(seed: int) -> None:
    d3rlpy = load_d3rlpy()
    d3rlpy.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_device(gpu: int) -> torch.device:
    if torch.cuda.is_available() and gpu >= 0:
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def build_dataset_dirs(dataset: str, datasets: Sequence[str]) -> List[str]:
    if dataset in ["halfcheetah", "walker2d", "hopper"]:
        prefix_path = "mujoco"
    elif dataset in ["antmaze", "pointmaze"]:
        prefix_path = "D4RL"
    else:
        prefix_path = ""

    dirs = []
    for sub_dataset in datasets:
        if prefix_path:
            dirs.append(f"{prefix_path}/{dataset}/{sub_dataset}")
        else:
            dirs.append(f"{dataset}/{sub_dataset}")
    return dirs


def ratio_tag(retained_ratios: Sequence[float]) -> str:
    return "_".join(str(r) for r in retained_ratios)


def dataset_components_tag(datasets: Sequence[str]) -> str:
    return "_".join(re.sub(r"[^A-Za-z0-9_.-]+", "-", str(item)) for item in datasets)


def canonical_algorithm_name(name: str) -> str:
    """Return the stable algorithm token used by shadow-policy cache keys."""
    normalized = re.sub(r"[^A-Za-z0-9]+", "", str(name)).upper()
    aliases = {
        "TD3BC": "TD3PLUSBC",
        "TD3PLUSBC": "TD3PLUSBC",
        "PLAS": "PLASP",
        "PLASP": "PLASP",
        "PLASWITHPERTURBATION": "PLASP",
    }
    return aliases.get(normalized, normalized)


def auditor_dataset_root(root_dir: str, dataset: str) -> str:
    """Normalize either an auditor root or a task-specific auditor root."""
    root = os.path.normpath(root_dir)
    expected_name = f"stats_orl_auditor_{dataset}"
    if os.path.basename(root) == expected_name:
        return root
    return os.path.join(root, expected_name)


def shadow_policy_output_dir(
    root_dir: str,
    dataset: str,
    datasets: Sequence[str],
    retained_ratios: Sequence[float],
    split_seed: int,
    algorithm: str,
    training_seed: int,
    training_steps: int,
    shuffle: bool = True,
) -> str:
    """Build the exact cache directory for one D_f-trained shadow policy."""
    split_mode = "shuffle" if shuffle else "no_shuffle"
    return os.path.join(
        auditor_dataset_root(root_dir, dataset),
        f"datasets_{dataset_components_tag(datasets)}",
        f"ratios_{ratio_tag(retained_ratios)}",
        "split_D_f",
        split_mode,
        f"split_seed_{split_seed}",
        f"algorithm_{canonical_algorithm_name(algorithm)}",
        f"training_seed_{training_seed}",
        f"steps_{training_steps}",
        "trained_shadow_policy",
    )


def load_dataset_splits(
    dataset: str,
    datasets: Sequence[str],
    retained_ratios: Sequence[float],
    seed: int,
    shuffle: bool = True,
) -> Dict[str, List[Any]]:
    if len(retained_ratios) != len(datasets):
        raise ValueError("--retained-ratios list must have the same length as --datasets list.")

    load_merged_dataset, _ = load_minari_utilities()
    dirs = build_dataset_dirs(dataset, datasets)
    print(f"Targeting dataset directories: {dirs}")
    dataset_required, dataset_remained = load_merged_dataset(
        dirs,
        list(retained_ratios),
        seed,
        shuffle=shuffle,
    )

    required_eps = [dataset_required[i] for i in range(len(dataset_required))]
    forget_eps = [dataset_remained[i] for i in range(len(dataset_remained))]
    return {
        "D_r": required_eps,
        "D_f": forget_eps,
        "all": required_eps + forget_eps,
    }


def load_forget_episodes(
    dataset: str,
    datasets: Sequence[str],
    retained_ratios: Sequence[float],
    seed: int,
    shuffle: bool = True,
) -> List[Any]:
    """Load only the materialized D_f episode list for the fixed audit split."""
    if len(retained_ratios) != len(datasets):
        raise ValueError("--retained-ratios list must have the same length as --datasets list.")

    load_merged_dataset, _ = load_minari_utilities()
    dirs = build_dataset_dirs(dataset, datasets)
    print(f"Targeting dataset directories: {dirs}")
    _, dataset_remained = load_merged_dataset(
        dirs,
        list(retained_ratios),
        seed,
        shuffle=shuffle,
    )
    forget_episodes = [dataset_remained[index] for index in range(len(dataset_remained))]
    print(f"Selected {len(forget_episodes)} episodes from audit split D_f.")
    return forget_episodes


def choose_audit_episodes(split_map: Dict[str, List[Any]], audit_split: str) -> List[Any]:
    if audit_split not in split_map:
        valid = ", ".join(sorted(split_map))
        raise ValueError(f"Unknown audit split '{audit_split}'. Valid values: {valid}")
    episodes = split_map[audit_split]
    print(f"Selected {len(episodes)} episodes from audit split {audit_split}.")
    return episodes


def _flatten_dict_observations(observations: Dict[str, np.ndarray]) -> np.ndarray:
    keys = sorted(observations.keys())
    pieces = []
    length = None
    for key in keys:
        arr = np.asarray(observations[key])
        if length is None:
            length = arr.shape[0]
        pieces.append(arr.reshape(arr.shape[0], -1))
    if not pieces:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(pieces, axis=1).astype(np.float32)


def normalize_observations(observations: Any) -> np.ndarray:
    if isinstance(observations, dict):
        return _flatten_dict_observations(observations)
    obs = np.asarray(observations)
    if obs.ndim == 0:
        obs = obs.reshape(1, 1)
    return obs.astype(np.float32)


def flatten_batch(arr: Any) -> np.ndarray:
    arr_np = np.asarray(arr)
    if arr_np.ndim == 0:
        arr_np = arr_np.reshape(1, 1)
    elif arr_np.ndim == 1:
        arr_np = arr_np.reshape(-1, 1)
    else:
        arr_np = arr_np.reshape(arr_np.shape[0], -1)
    return arr_np.astype(np.float32)


def action_batch(actions: Any) -> np.ndarray:
    return flatten_batch(actions)


def terminal_array_from_episode(episode: Any, length: int) -> np.ndarray:
    terminal_like = None
    for attr in ["terminations", "terminals", "episode_terminals"]:
        if hasattr(episode, attr):
            terminal_like = np.asarray(getattr(episode, attr)).reshape(-1)
            break

    if terminal_like is None:
        terminal_like = np.zeros(length, dtype=bool)
    else:
        terminal_like = terminal_like.astype(bool)
        if terminal_like.shape[0] < length:
            terminal_like = np.pad(terminal_like, (0, length - terminal_like.shape[0]))
        terminal_like = terminal_like[:length]

    if hasattr(episode, "truncations"):
        truncations = np.asarray(getattr(episode, "truncations")).reshape(-1).astype(bool)
        if truncations.shape[0] < length:
            truncations = np.pad(truncations, (0, length - truncations.shape[0]))
        terminal_like = np.logical_or(terminal_like, truncations[:length])

    return terminal_like


def episode_arrays(episode: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if hasattr(episode, "observations") and hasattr(episode, "actions"):
        observations = normalize_observations(episode.observations)
        actions = action_batch(episode.actions)
        rewards = np.asarray(getattr(episode, "rewards", np.zeros(len(actions))), dtype=np.float32).reshape(-1)
        terminals = terminal_array_from_episode(episode, len(rewards))
        return observations, actions, rewards, terminals

    if hasattr(episode, "transitions"):
        observations = []
        actions = []
        rewards = []
        terminals = []
        for transition in episode.transitions:
            observations.append(np.asarray(transition.observation))
            actions.append(np.asarray(transition.action))
            rewards.append(getattr(transition, "reward", 0.0))
            terminals.append(bool(getattr(transition, "terminal", False)))
        return (
            normalize_observations(np.asarray(observations)),
            action_batch(np.asarray(actions)),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminals, dtype=bool),
        )

    raise TypeError(f"Unsupported episode type: {type(episode)}")


def episodes_to_transition_array(
    episodes: Sequence[Any],
    max_episodes: Optional[int] = None,
    max_transitions: Optional[int] = None,
) -> np.ndarray:
    rows = []
    selected = episodes[:max_episodes] if max_episodes is not None else episodes

    for episode in tqdm(selected, desc="Converting episodes to ORL transitions"):
        observations, actions, rewards, terminals = episode_arrays(episode)
        usable = min(
            observations.shape[0] - 1,
            actions.shape[0] - 1,
            rewards.shape[0] - 1,
        )
        if usable <= 0:
            continue

        for idx in range(usable):
            if terminals[idx]:
                continue
            row = np.concatenate(
                [
                    observations[idx].reshape(-1),
                    actions[idx].reshape(-1),
                    observations[idx + 1].reshape(-1),
                    actions[idx + 1].reshape(-1),
                    np.asarray([rewards[idx]], dtype=np.float32),
                ]
            )
            rows.append(row.astype(np.float32))
            if max_transitions is not None and len(rows) >= max_transitions:
                break
        if max_transitions is not None and len(rows) >= max_transitions:
            break

    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32)


def episodes_to_observation_trajectories(
    episodes: Sequence[Any],
    num_audited_episodes: Optional[int],
    trajectory_size: float,
) -> List[np.ndarray]:
    if trajectory_size <= 0.0 or trajectory_size > 1.0:
        raise ValueError("--trajectory-size must be in (0, 1].")
    if num_audited_episodes is not None and num_audited_episodes <= 0:
        raise ValueError("--num-audited-episodes must be positive when specified.")

    trajectories = []
    for episode in episodes:
        observations, _, _, _ = episode_arrays(episode)
        if observations.shape[0] == 0:
            continue
        take = max(1, int(observations.shape[0] * trajectory_size))
        trajectories.append(observations[:take])
        if (
            num_audited_episodes is not None
            and len(trajectories) >= num_audited_episodes
        ):
            break
    return trajectories


def find_latest_model_path(model_dir: str, require_model: Optional[int] = None) -> str:
    if require_model is not None:
        required_path = os.path.join(model_dir, f"model_{require_model}.pt")
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"Required model checkpoint not found: {required_path}")
        return required_path

    model_files = glob.glob(os.path.join(model_dir, "model_*.pt"))
    fallback = os.path.join(model_dir, "model.pt")
    if not model_files and os.path.exists(fallback):
        return fallback
    if not model_files:
        raise FileNotFoundError(f"No model_*.pt or model.pt files found in {model_dir}")

    step_paths = []
    for path in model_files:
        match = re.search(r"model_(\d+)\.pt$", path)
        if match:
            step_paths.append((int(match.group(1)), path))
    if step_paths:
        return max(step_paths, key=lambda item: item[0])[1]
    return sorted(model_files)[-1]


def load_model_from_dir(model_dir: str, gpu: int, require_model: Optional[int] = None) -> LoadedModel:
    print(f"Loading d3rlpy model from directory: {model_dir}")
    d3rlpy = load_d3rlpy()
    _, load_model_with_fix = load_minari_utilities()
    params_path = os.path.join(model_dir, "params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"params.json not found in {model_dir}")

    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    algo_name_key = "algorithm" if "algorithm" in params else "type"
    algo_name = params.get(algo_name_key)
    if not algo_name:
        raise ValueError(f"Could not find 'algorithm' or 'type' key in {params_path}")

    try:
        algo_cls = getattr(d3rlpy.algos, algo_name)
        use_gpu = gpu if gpu >= 0 else False
        algo = algo_cls.from_json(params_path, use_gpu=use_gpu)
    except AttributeError:
        raise ValueError(f"Algorithm '{algo_name}' not found in d3rlpy.algos")

    model_path = find_latest_model_path(model_dir, require_model=require_model)
    load_model_with_fix(algo, model_path, gpu)
    print(f"Loaded weights from {model_path}")
    return LoadedModel(
        model=algo,
        checkpoint_name=os.path.basename(model_path),
        checkpoint_path=model_path,
    )


def train_orl_critic(
    transition_array: np.ndarray,
    output_dir: str,
    device: torch.device,
    train_epochs: int = 200,
    gamma: float = 0.99,
    hidden_size: int = 1024,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    train_ratio: float = 0.7,
    save_interval: int = 100,
    seed: int = 0,
    num_workers: int = 2,
) -> Tuple[str, List[Dict[str, float]]]:
    if transition_array.size == 0:
        raise ValueError("No transitions available for critic training.")
    if transition_array.shape[1] < 3:
        raise ValueError(f"Transition array has invalid shape: {transition_array.shape}")

    replay_dataset = transition_array.astype(np.float32).copy()
    max_abs_reward = float(np.max(np.abs(replay_dataset[:, -1])))
    if max_abs_reward > 0.0:
        replay_dataset[:, -1] = replay_dataset[:, -1] / max_abs_reward

    rng = np.random.default_rng(seed)
    indices = np.arange(replay_dataset.shape[0])
    rng.shuffle(indices)
    train_len = max(1, int(train_ratio * replay_dataset.shape[0]))
    train_indices = indices[:train_len]
    test_indices = indices[train_len:]
    if test_indices.shape[0] == 0:
        test_indices = train_indices

    tensor_data = torch.as_tensor(replay_dataset, dtype=torch.float32)
    observation_action_size = int((tensor_data.shape[-1] - 1) / 2)
    observation_action = tensor_data[:, :observation_action_size]
    next_observation_action_and_reward = tensor_data[:, observation_action_size:]

    train_dataset = TensorDataset(
        observation_action[train_indices],
        next_observation_action_and_reward[train_indices],
    )
    test_dataset = TensorDataset(
        observation_action[test_indices],
        next_observation_action_and_reward[test_indices],
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    critic_model = CriticModelWithoutLastActivation(
        observation_action_size,
        n_hidden=hidden_size,
        n_output=1,
    ).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(critic_model.parameters(), lr=learning_rate, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    os.makedirs(output_dir, exist_ok=True)
    history: List[Dict[str, float]] = []
    latest_path = ""

    for epoch in range(train_epochs):
        critic_model.train()
        train_loss = 0.0
        for observation_action_batch, next_batch in train_loader:
            observation_action_batch = observation_action_batch.to(device)
            next_batch = next_batch.to(device)

            with torch.no_grad():
                next_q = critic_model(next_batch[:, :-1])
                target_q = next_batch[:, -1] + gamma * next_q

            optimizer.zero_grad()
            q_values = critic_model(observation_action_batch)
            loss = criterion(q_values, target_q)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        critic_model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for observation_action_batch, next_batch in test_loader:
                observation_action_batch = observation_action_batch.to(device)
                next_batch = next_batch.to(device)
                next_q = critic_model(next_batch[:, :-1])
                target_q = next_batch[:, -1] + gamma * next_q
                q_values = critic_model(observation_action_batch)
                loss = criterion(q_values, target_q)
                test_loss += float(loss.item())

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_record)
        print(
            f"[epoch: {epoch + 1}] train loss: {train_loss:.6f} "
            f"test loss: {test_loss:.6f}"
        )

        should_save = save_interval > 0 and ((epoch + 1) % save_interval == 0)
        should_save = should_save or (epoch + 1 == train_epochs)
        if should_save:
            latest_path = os.path.join(output_dir, f"ckpt_{epoch + 1}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": critic_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "observation_action_size": observation_action_size,
                    "hidden_size": hidden_size,
                    "gamma": gamma,
                    "max_abs_reward": max_abs_reward,
                },
                latest_path,
            )
            print(f"Saved critic checkpoint to {latest_path}")

        scheduler.step()

    return latest_path, history


def write_training_history(history: Sequence[Dict[str, float]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["epoch", "train_loss", "test_loss", "lr"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)
    print(f"Saved training history to {output_path}")


def load_orl_critic(critic_checkpoint: str, device: torch.device) -> LoadedCritic:
    checkpoint = torch.load(critic_checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    input_dim = int(checkpoint.get("observation_action_size", state_dict["fc1.weight"].shape[1]))
    hidden_size = int(checkpoint.get("hidden_size", state_dict["fc1.weight"].shape[0]))

    model = CriticModelWithoutLastActivation(input_dim, n_hidden=hidden_size, n_output=1).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return LoadedCritic(
        model=model,
        checkpoint_path=critic_checkpoint,
        input_dim=input_dim,
        hidden_size=hidden_size,
    )


def predict_actions(policy: Any, observations: np.ndarray) -> np.ndarray:
    actions = policy.predict(observations)
    return action_batch(actions)


def concat_state_action(observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
    obs_flat = flatten_batch(observations)
    act_flat = action_batch(actions)
    n = min(obs_flat.shape[0], act_flat.shape[0])
    if n == 0:
        return np.empty((0, obs_flat.shape[1] + act_flat.shape[1]), dtype=np.float32)
    return np.concatenate([obs_flat[:n], act_flat[:n]], axis=1).astype(np.float32)


def estimate_values(
    critic: CriticModelWithoutLastActivation,
    states_actions: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if states_actions.shape[0] == 0:
        return np.asarray([], dtype=np.float32)

    values = []
    with torch.no_grad():
        for start in range(0, states_actions.shape[0], batch_size):
            end = min(start + batch_size, states_actions.shape[0])
            batch = torch.as_tensor(states_actions[start:end], dtype=torch.float32, device=device)
            values.append(critic(batch).detach().cpu().numpy())
    return np.concatenate(values, axis=0).astype(np.float32)


def policy_values_on_trajectory(
    policy: Any,
    critic: CriticModelWithoutLastActivation,
    observations: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    actions = predict_actions(policy, observations)
    states_actions = concat_state_action(observations, actions)
    return estimate_values(critic, states_actions, device, batch_size)


def cosine_distance(values1: np.ndarray, values2: np.ndarray) -> float:
    x = np.asarray(values1, dtype=np.float64).reshape(-1)
    y = np.asarray(values2, dtype=np.float64).reshape(-1)
    n = min(x.shape[0], y.shape[0])
    if n == 0:
        return 0.0
    x = x[:n]
    y = y[:n]
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0.0:
        return 0.0 if np.linalg.norm(x - y) == 0.0 else 1.0
    return float(1.0 - np.dot(x, y) / denom)


def wasserstein_distance(values1: np.ndarray, values2: np.ndarray) -> float:
    x = np.asarray(values1, dtype=np.float64).reshape(-1)
    y = np.asarray(values2, dtype=np.float64).reshape(-1)
    if x.shape[0] == 0 or y.shape[0] == 0:
        return 0.0
    if scipy_wasserstein_distance is not None:
        return float(scipy_wasserstein_distance(x, y))

    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    n = min(x_sorted.shape[0], y_sorted.shape[0])
    return float(np.mean(np.abs(x_sorted[:n] - y_sorted[:n])))


def metric_distances_to_mean(value_stack: np.ndarray, mean_values: np.ndarray) -> Dict[str, List[float]]:
    distances = {metric: [] for metric in METRIC_NAMES}
    for idx in range(value_stack.shape[1]):
        values = value_stack[:, idx]
        distances["l1_distance"].append(float(np.sum(np.abs(values - mean_values))))
        distances["l2_distance"].append(float(np.sum(np.square(values - mean_values))))
        distances["cos_distance"].append(cosine_distance(values, mean_values))
        distances["wasserstein_distance"].append(wasserstein_distance(values, mean_values))
    return distances


def metric_distance_to_mean(values: np.ndarray, mean_values: np.ndarray) -> Dict[str, float]:
    n = min(values.shape[0], mean_values.shape[0])
    values = values[:n]
    mean_values = mean_values[:n]
    return {
        "l1_distance": float(np.sum(np.abs(values - mean_values))),
        "l2_distance": float(np.sum(np.square(values - mean_values))),
        "cos_distance": cosine_distance(values, mean_values),
        "wasserstein_distance": wasserstein_distance(values, mean_values),
    }


def compute_orl_audit_rows(
    shadow_models: Sequence[LoadedModel],
    suspect_models: Sequence[LoadedModel],
    suspect_names: Sequence[str],
    suspect_memberships: Sequence[Optional[bool]],
    audit_split: str,
    trajectories: Sequence[np.ndarray],
    critic: CriticModelWithoutLastActivation,
    device: torch.device,
    batch_size: int,
) -> List[Dict[str, Any]]:
    if len(shadow_models) < 2:
        raise ValueError("ORL-Auditor Grubbs testing needs at least two shadow models.")

    shadow_episode_features = []
    for episode_idx, observations in enumerate(tqdm(trajectories, desc="Shadow policy values")):
        value_columns = []
        for shadow in shadow_models:
            values = policy_values_on_trajectory(
                shadow.model,
                critic,
                observations,
                device,
                batch_size,
            )
            value_columns.append(values.reshape(-1, 1))

        min_len = min(column.shape[0] for column in value_columns)
        value_stack = np.concatenate([column[:min_len] for column in value_columns], axis=1)
        mean_values = np.mean(value_stack, axis=1)
        shadow_distances = metric_distances_to_mean(value_stack, mean_values)
        shadow_episode_features.append(
            {
                "episode": episode_idx,
                "observations": observations[:min_len],
                "mean_values": mean_values,
                "shadow_distances": shadow_distances,
            }
        )

    rows = []
    for suspect_idx, suspect in enumerate(suspect_models):
        suspect_name = suspect_names[suspect_idx]
        expected_membership = suspect_memberships[suspect_idx]
        if expected_membership is True:
            actual_buffer_name = audit_split
        elif expected_membership is False:
            actual_buffer_name = f"nonmember:{suspect_name}"
        else:
            actual_buffer_name = f"unknown:{suspect_name}"

        for feature in tqdm(shadow_episode_features, desc=f"Suspect values: {suspect_name}"):
            values = policy_values_on_trajectory(
                suspect.model,
                critic,
                feature["observations"],
                device,
                batch_size,
            )
            distances = metric_distance_to_mean(values, feature["mean_values"])
            row = {
                "episode": feature["episode"],
                "actual_buffer_name": actual_buffer_name,
                "student_name": suspect_name,
                "audit_buffer_name": audit_split,
                "expected_membership": expected_membership,
                "suspect_model_dir": os.path.dirname(suspect.checkpoint_path),
                "suspect_checkpoint": suspect.checkpoint_name,
            }
            for metric in METRIC_NAMES:
                row[f"teacher_student_{metric}"] = distances[metric]
                row[f"shadow_model_{metric}"] = feature["shadow_distances"][metric]
            rows.append(row)
    return rows


def grubbs_test_details(
    shadow_values: Sequence[float],
    suspect_value: float,
    sigma: float,
) -> Dict[str, Any]:
    values = np.asarray(list(shadow_values) + [float(suspect_value)], dtype=np.float64)
    if values.shape[0] < 3:
        raise ValueError("Grubbs test requires at least two shadow values plus one suspect value.")

    std_dev = float(np.std(values))
    mean = float(np.mean(values))
    if std_dev == 0.0:
        return {
            "accepted": bool(float(suspect_value) <= mean),
            "score": 0.0,
            "threshold": None,
        }

    z_score = (float(suspect_value) - mean) / std_dev
    n = values.shape[0]
    if scipy_stats is None:
        return {
            "accepted": bool(z_score <= 3.0),
            "score": float(z_score),
            "threshold": 3.0,
        }

    threshold = scipy_stats.t.isf(sigma / n, n - 2)
    threshold_squared = threshold * threshold
    grubbs_threshold = ((n - 1) / math.sqrt(n)) * math.sqrt(
        threshold_squared / (n - 2 + threshold_squared)
    )
    return {
        "accepted": bool(z_score <= grubbs_threshold),
        "score": float(z_score),
        "threshold": float(grubbs_threshold),
    }


def grubbs_accept_member(shadow_values: Sequence[float], suspect_value: float, sigma: float) -> bool:
    return bool(grubbs_test_details(shadow_values, suspect_value, sigma)["accepted"])


def apply_grubbs_tests(
    rows: Sequence[Dict[str, Any]],
    significance_level: float,
    metrics: Sequence[str] = METRIC_NAMES,
) -> List[Dict[str, Any]]:
    tested_rows = []
    for row in rows:
        out = dict(row)
        for metric in metrics:
            shadow_values = np.asarray(row[f"shadow_model_{metric}"], dtype=np.float64)
            target_distance = float(row[f"teacher_student_{metric}"])
            shadow_mean = float(np.mean(shadow_values))
            shadow_std = float(np.std(shadow_values))
            standardized_distance = None
            # With two shadows, leave-in distances to their mean are symmetric,
            # so floating-point noise can masquerade as a usable variance.
            if shadow_values.size >= 3 and shadow_std > 0.0:
                standardized_distance = float((target_distance - shadow_mean) / shadow_std)

            grubbs = grubbs_test_details(shadow_values, target_distance, significance_level)
            accepted = bool(grubbs["accepted"])
            out[f"{metric}_audit_result"] = accepted
            out[f"{metric}_audit_positive"] = accepted
            out[f"{metric}_shadow_mean"] = shadow_mean
            out[f"{metric}_shadow_std"] = shadow_std
            out[f"{metric}_standardized_distance"] = standardized_distance
            out[f"{metric}_grubbs_score"] = grubbs["score"]
            out[f"{metric}_grubbs_threshold"] = grubbs["threshold"]
            if row["expected_membership"] is None:
                out[f"{metric}_correct"] = None
            else:
                out[f"{metric}_correct"] = bool(accepted == row["expected_membership"])
        tested_rows.append(out)
    return tested_rows


def summarize_audit_rows(
    rows: Sequence[Dict[str, Any]],
    metrics: Sequence[str] = METRIC_NAMES,
) -> Dict[str, Any]:
    summary = {
        "primary_audit_results": {},
        "optional_auditor_quality": {},
    }

    def describe(values: Sequence[float]) -> Dict[str, Optional[float]]:
        array = np.asarray(list(values), dtype=np.float64)
        if array.size == 0:
            return {"Mean": None, "Std": None, "Median": None, "Min": None, "Max": None}
        return {
            "Mean": float(np.mean(array)),
            "Std": float(np.std(array)),
            "Median": float(np.median(array)),
            "Min": float(np.min(array)),
            "Max": float(np.max(array)),
        }

    suspect_names = sorted({row["student_name"] for row in rows})
    for suspect_name in suspect_names:
        suspect_rows = [row for row in rows if row["student_name"] == suspect_name]
        first_row = suspect_rows[0]
        suspect_result = {
            "Suspect Model Directory": first_row["suspect_model_dir"],
            "Suspect Role": first_row.get("suspect_role", "unknown"),
            "Expected Membership": first_row["expected_membership"],
            "Audited Trajectories": len(suspect_rows),
            "Audit Buffer": first_row["audit_buffer_name"],
            "Metrics": {},
        }

        for metric in metrics:
            positives = [bool(row[f"{metric}_audit_result"]) for row in suspect_rows]
            target_distances = [float(row[f"teacher_student_{metric}"]) for row in suspect_rows]
            shadow_distances = [
                float(value)
                for row in suspect_rows
                for value in row[f"shadow_model_{metric}"]
            ]
            standardized = [
                float(row[f"{metric}_standardized_distance"])
                for row in suspect_rows
                if row[f"{metric}_standardized_distance"] is not None
            ]
            suspect_result["Metrics"][metric] = {
                "Audit Positive Rate": float(sum(positives) / len(positives)),
                "Audit Positive Count": int(sum(positives)),
                "Audited Trajectories": len(positives),
                "Target-to-Shadow-Mean Distance": describe(target_distances),
                "Shadow Distance Distribution": describe(shadow_distances),
                "Standardized Distance": describe(standardized),
                "Standardized Distance Valid Count": len(standardized),
            }

        summary["primary_audit_results"][suspect_name] = suspect_result

    labeled_rows = [row for row in rows if row["expected_membership"] is not None]
    if labeled_rows:
        for metric in metrics:
            member_rows = [row for row in labeled_rows if row["expected_membership"] is True]
            nonmember_rows = [row for row in labeled_rows if row["expected_membership"] is False]
            summary["optional_auditor_quality"][metric] = {
                "True Positive Rate": (
                    float(sum(bool(row[f"{metric}_audit_result"]) for row in member_rows) / len(member_rows))
                    if member_rows else None
                ),
                "True Negative Rate": (
                    float(sum(not bool(row[f"{metric}_audit_result"]) for row in nonmember_rows) / len(nonmember_rows))
                    if nonmember_rows else None
                ),
                "Member Count": len(member_rows),
                "Nonmember Count": len(nonmember_rows),
            }

    return summary


def default_suspect_names(suspect_dirs: Sequence[str]) -> List[str]:
    names = []
    used = set()
    for path in suspect_dirs:
        base = os.path.basename(os.path.normpath(path)) or "suspect"
        if base in used:
            base = f"{base}_{len(used)}"
        used.add(base)
        names.append(base)
    return names


def parse_membership_labels(labels: Optional[Sequence[str]], count: int) -> List[Optional[bool]]:
    if labels is None:
        return [None] * count
    if len(labels) != count:
        raise ValueError("--suspect-membership must have the same length as --suspect-dirs.")

    parsed: List[Optional[bool]] = []
    for label in labels:
        normalized = label.lower()
        if normalized in ["member", "in", "true", "1", "yes"]:
            parsed.append(True)
        elif normalized in ["nonmember", "out", "false", "0", "no"]:
            parsed.append(False)
        elif normalized in ["unknown", "none", "na", "n/a"]:
            parsed.append(None)
        else:
            raise ValueError(f"Unknown membership label: {label}")
    return parsed

