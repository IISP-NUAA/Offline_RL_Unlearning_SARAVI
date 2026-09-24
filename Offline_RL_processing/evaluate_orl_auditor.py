#!/usr/bin/env python3
"""Run ORL-Auditor style membership auditing with local d3rlpy models."""

import argparse
import os
import time

from orl_auditor_core import (
    METRIC_NAMES,
    apply_grubbs_tests,
    compute_orl_audit_rows,
    default_suspect_names,
    dataset_components_tag,
    episodes_to_observation_trajectories,
    get_device,
    load_forget_episodes,
    load_model_from_dir,
    load_orl_critic,
    parse_membership_labels,
    ratio_tag,
    save_json,
    set_global_seeds,
    summarize_audit_rows,
)


def make_output_dir(args, suspect_names):
    ratios_str = ratio_tag(args.retained_ratios)
    base_dir = os.path.join(
        args.save_dir,
        f"stats_orl_auditor_{args.dataset}",
        f"datasets_{dataset_components_tag(args.datasets)}",
        f"ratios_{ratios_str}",
        f"split_{args.audit_split}",
        "shuffle" if not args.no_shuffle else "no_shuffle",
    )
    if args.output_name:
        run_name = args.output_name
    elif len(suspect_names) == 1:
        run_name = f"name_{args.suspect_dirs[0].replace('.', '')}"
    else:
        run_name = "multi_suspects_" + time.strftime("%m-%d-%H-%M-%S", time.localtime())
    return os.path.join(base_dir, run_name)


def main():
    parser = argparse.ArgumentParser(
        description="Audit suspect d3rlpy models with the ORL-Auditor shadow-model workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--critic-checkpoint", type=str, required=True,
                        help="Path to ckpt_N.pt produced by train_orl_auditor_critic.py.")
    parser.add_argument("--shadow-dirs", type=str, nargs="+", required=True,
                        help="Same-algorithm D_f-only shadow policy directories. Each contains params.json and model_*.pt.")
    parser.add_argument("--suspect-dirs", type=str, nargs="+", required=True,
                        help="Suspect model directories to audit.")
    parser.add_argument("--suspect-names", type=str, nargs="+", default=None,
                        help="Optional display names for suspect dirs.")
    parser.add_argument("--suspect-membership", type=str, nargs="+", default=None,
                        help="Optional auditor-quality labels: member, nonmember, or unknown.")
    parser.add_argument(
        "--suspect-roles",
        choices=["original", "unlearned", "retrained", "unknown"],
        nargs="+",
        default=None,
        help="Semantic comparison role for each target; this is not a membership label.",
    )
    parser.add_argument("--shadow-require-model", type=int, default=None,
                        help="Force shadow checkpoint model_<N>.pt.")
    parser.add_argument("--suspect-require-model", type=int, default=None,
                        help="Force suspect checkpoint model_<N>.pt.")

    parser.add_argument("--dataset", type=str, required=True,
                        help="Task name, e.g. pointmaze, antmaze, halfcheetah.")
    parser.add_argument("--datasets", type=str, nargs="+", required=True,
                        help="Sub-dataset names used by load_merged_dataset.")
    parser.add_argument("--retained-ratios", type=float, nargs="+", required=True,
                        help="Retain ratios matching --datasets.")
    parser.add_argument("--audit-split", choices=["D_f"], default="D_f",
                        help="ORL-Auditor evaluates trajectories from D_f only.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for data split.")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Disable shuffling in load_merged_dataset split.")

    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU id, or -1 for CPU.")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size for critic value estimation.")
    parser.add_argument("--num-audited-episodes", type=int, default=None,
                        help="Optional cap on audit trajectories; by default use every D_f trajectory.")
    parser.add_argument("--trajectory-size", type=float, default=1.0,
                        help="Fraction of each trajectory used by the audit.")
    parser.add_argument("--significance-level", type=float, default=0.01,
                        help="ORL-Auditor Grubbs significance level.")
    parser.add_argument("--save-dir", type=str, default="stats_results_orl_auditor",
                        help="Root directory for audit outputs.")
    parser.add_argument("--output-name", type=str, default=None,
                        help="Optional output folder name under the audit split directory.")

    args = parser.parse_args()

    if args.suspect_names is not None and len(args.suspect_names) != len(args.suspect_dirs):
        raise ValueError("--suspect-names must have the same length as --suspect-dirs.")
    if args.suspect_roles is not None and len(args.suspect_roles) != len(args.suspect_dirs):
        raise ValueError("--suspect-roles must have the same length as --suspect-dirs.")

    set_global_seeds(args.seed)
    device = get_device(args.gpu)
    print(f"Using device: {device}")

    suspect_names = args.suspect_names or default_suspect_names(args.suspect_dirs)
    suspect_roles = args.suspect_roles or ["unknown"] * len(args.suspect_dirs)
    suspect_memberships = parse_membership_labels(args.suspect_membership, len(args.suspect_dirs))

    audit_episodes = load_forget_episodes(
        args.dataset,
        args.datasets,
        args.retained_ratios,
        args.seed,
        shuffle=not args.no_shuffle,
    )
    trajectories = episodes_to_observation_trajectories(
        audit_episodes,
        num_audited_episodes=args.num_audited_episodes,
        trajectory_size=args.trajectory_size,
    )
    if not trajectories:
        raise ValueError("No audit trajectories were produced.")
    print(f"Prepared {len(trajectories)} audit trajectories.")

    loaded_critic = load_orl_critic(args.critic_checkpoint, device)
    print(
        f"Loaded ORL critic from {loaded_critic.checkpoint_path} "
        f"(input_dim={loaded_critic.input_dim}, hidden={loaded_critic.hidden_size})"
    )

    shadow_models = [
        load_model_from_dir(path, args.gpu, require_model=args.shadow_require_model)
        for path in args.shadow_dirs
    ]
    suspect_models = [
        load_model_from_dir(path, args.gpu, require_model=args.suspect_require_model)
        for path in args.suspect_dirs
    ]

    raw_rows = compute_orl_audit_rows(
        shadow_models=shadow_models,
        suspect_models=suspect_models,
        suspect_names=suspect_names,
        suspect_memberships=suspect_memberships,
        audit_split=args.audit_split,
        trajectories=trajectories,
        critic=loaded_critic.model,
        device=device,
        batch_size=args.batch_size,
    )
    role_by_name = dict(zip(suspect_names, suspect_roles))
    for row in raw_rows:
        row["suspect_role"] = role_by_name[row["student_name"]]
    tested_rows = apply_grubbs_tests(raw_rows, args.significance_level, metrics=METRIC_NAMES)
    summary = summarize_audit_rows(tested_rows, metrics=METRIC_NAMES)

    output_dir = make_output_dir(args, suspect_names)
    os.makedirs(output_dir, exist_ok=True)
    summary_json_path = os.path.join(output_dir, f"audit_summary_seed{args.seed}.json")

    result_data = {
        "metric_type": "ORL_Auditor_Grubbs",
        "dataset_name": args.dataset,
        "dataset_components": args.datasets,
        "retained_ratios": args.retained_ratios,
        "audit_split": args.audit_split,
        "seed": args.seed,
        "shuffle": not args.no_shuffle,
        "critic_checkpoint": args.critic_checkpoint,
        "shadow_model_dirs": args.shadow_dirs,
        "suspect_model_dirs": args.suspect_dirs,
        "suspect_names": suspect_names,
        "suspect_roles": suspect_roles,
        "suspect_membership": args.suspect_membership,
        "num_shadow_student": len(args.shadow_dirs),
        "num_of_audited_episode": len(trajectories),
        "trajectory_size": args.trajectory_size,
        "significance_level": args.significance_level,
        "batch_size": args.batch_size,
        "row_count": len(tested_rows),
        "orl_audit_statistics": summary,
    }
    save_json(result_data, summary_json_path)

    print("\n=== ORL-Auditor Target Results ===")
    for suspect_name, suspect_result in summary["primary_audit_results"].items():
        print(f"Suspect: {suspect_name}")
        for metric, metric_result in suspect_result["Metrics"].items():
            positive_rate = metric_result["Audit Positive Rate"]
            raw_mean = metric_result["Target-to-Shadow-Mean Distance"]["Mean"]
            standardized_mean = metric_result["Standardized Distance"]["Mean"]
            standardized_text = "N/A" if standardized_mean is None else f"{standardized_mean:.4f}"
            print(
                f"  {metric}: positive_rate={positive_rate:.4f}, "
                f"target_distance_mean={raw_mean:.4f}, "
                f"standardized_distance_mean={standardized_text}"
            )
    print(f"Summary JSON:  {summary_json_path}")


if __name__ == "__main__":
    main()
