#!/usr/bin/env python3
"""Train an ORL-Auditor critic on this repository's retained/forget split."""

import argparse
import os

import numpy as np

from orl_auditor_core import (
    episodes_to_transition_array,
    get_device,
    load_forget_episodes,
    ratio_tag,
    dataset_components_tag,
    save_json,
    set_global_seeds,
    train_orl_critic,
    write_training_history,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train the ORL-Auditor MLP critic from Minari/d3rlpy split data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dataset", type=str, required=True,
                        help="Task name, e.g. pointmaze, antmaze, halfcheetah.")
    parser.add_argument("--datasets", type=str, nargs="+", required=True,
                        help="Sub-dataset names used by load_merged_dataset.")
    parser.add_argument("--retained-ratios", type=float, nargs="+", required=True,
                        help="Retain ratios matching --datasets.")
    parser.add_argument("--audit-split", choices=["D_f"], default="D_f",
                        help="ORL-Auditor critics are trained exclusively on D_f.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for data split and critic training.")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Disable shuffling in load_merged_dataset split.")

    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU id, or -1 for CPU.")
    parser.add_argument("--train-epochs", type=int, default=200,
                        help="Number of critic training epochs.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Bellman target discount factor.")
    parser.add_argument("--hidden-size", type=int, default=1024,
                        help="Hidden size used by the ORL-Auditor critic MLP.")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Critic training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="AdamW learning rate.")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="Fraction of transitions used for critic train split.")
    parser.add_argument("--save-interval", type=int, default=100,
                        help="Save ckpt_N.pt every N epochs; final epoch is always saved.")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader workers.")

    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Optional debug cap on source episodes.")
    parser.add_argument("--max-transitions", type=int, default=None,
                        help="Optional debug cap on converted transitions.")
    parser.add_argument("--save-transitions", action="store_true",
                        help="Also save the ORL transition array as transition_form.npy.")
    parser.add_argument("--save-dir", type=str, default="stats_results_orl_auditor",
                        help="Root directory for critic outputs.")

    args = parser.parse_args()

    set_global_seeds(args.seed)
    device = get_device(args.gpu)
    print(f"Using device: {device}")

    audit_episodes = load_forget_episodes(
        args.dataset,
        args.datasets,
        args.retained_ratios,
        args.seed,
        shuffle=not args.no_shuffle,
    )

    transitions = episodes_to_transition_array(
        audit_episodes,
        max_episodes=args.max_episodes,
        max_transitions=args.max_transitions,
    )
    print(f"Converted transition array shape: {transitions.shape}")
    if transitions.size == 0:
        raise ValueError("No transitions were produced; cannot train ORL-Auditor critic.")

    ratios_str = ratio_tag(args.retained_ratios)
    output_dir = os.path.join(
        args.save_dir,
        f"stats_orl_auditor_{args.dataset}",
        f"datasets_{dataset_components_tag(args.datasets)}",
        f"ratios_{ratios_str}",
        f"split_{args.audit_split}",
        "shuffle" if not args.no_shuffle else "no_shuffle",
        f"seed_{args.seed}",
        "trained_critic_model",
    )
    os.makedirs(output_dir, exist_ok=True)

    if args.save_transitions:
        transition_path = os.path.join(output_dir, "transition_form.npy")
        np.save(transition_path, transitions)
        print(f"Saved ORL transition array to {transition_path}")
    else:
        transition_path = None

    latest_checkpoint, history = train_orl_critic(
        transitions,
        output_dir=output_dir,
        device=device,
        train_epochs=args.train_epochs,
        gamma=args.gamma,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_ratio=args.train_ratio,
        save_interval=args.save_interval,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    history_path = os.path.join(output_dir, "critic_train_history.csv")
    write_training_history(history, history_path)

    config = {
        "metric_type": "ORL_Auditor_Critic",
        "dataset_name": args.dataset,
        "dataset_components": args.datasets,
        "retained_ratios": args.retained_ratios,
        "audit_split": args.audit_split,
        "seed": args.seed,
        "shuffle": not args.no_shuffle,
        "transition_shape": list(transitions.shape),
        "transition_path": transition_path,
        "max_episodes": args.max_episodes,
        "max_transitions": args.max_transitions,
        "critic_checkpoint": latest_checkpoint,
        "train_epochs": args.train_epochs,
        "gamma": args.gamma,
        "hidden_size": args.hidden_size,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_ratio": args.train_ratio,
        "history_path": history_path,
    }
    save_json(config, os.path.join(output_dir, "critic_config.json"))
    print(f"Latest critic checkpoint: {latest_checkpoint}")


if __name__ == "__main__":
    main()
