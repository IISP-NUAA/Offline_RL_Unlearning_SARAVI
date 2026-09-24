#!/usr/bin/env python3
import os
import re
import argparse
import sys
import glob
from pathlib import Path
import time
from generator_ratio_filter import normalize_ratio_filter, path_matches_ratio, ratio_filename_suffix

# =============================================================================
# 1. Helper Functions
# =============================================================================

def generate_output_filename(root_dir, ratio=None):
    """Generate a dynamic output filename with an optional ratio marker."""
    base_name = root_dir.strip("/").replace("/", "_").replace("..", "").replace(".", "")
    ratio_suffix = ratio_filename_suffix(ratio)
    time_str = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    base_name = base_name + ratio_suffix + "_" + time_str
    return f"run_policy_distance_evaluation_{base_name}.sh"

def parse_step_from_model_dir(model_dir: str) -> str:
    """
    Extract the training/unlearning step from a model path.
    Special unlearning layouts are handled before broad steps_* matching so
    seed_0_steps_2000000 does not hide the actual unlearning step.
    """
    if not model_dir:
        return "N/A"

    parts = [part for part in model_dir.replace("\\", "/").split("/") if part]

    def find_explicit_step(start_index: int = 0) -> str:
        for part in parts[start_index:]:
            match = re.fullmatch(r"step_(\d+)", part)
            if match:
                return match.group(1)
        return "N/A"

    if "random_Rewarding_simple_fit" in parts:
        marker_index = parts.index("random_Rewarding_simple_fit")
        step = find_explicit_step(marker_index + 1)
        if step != "N/A":
            return step

    if "trajDeleter" in parts:
        marker_index = parts.index("trajDeleter")
        phase_steps = {}
        for part in parts[marker_index + 1:]:
            for phase, value in re.findall(r"phase([12])_(\d+)", part):
                phase_steps[phase] = int(value)

        if "1" in phase_steps and "2" in phase_steps:
            return str(phase_steps["1"] + phase_steps["2"])
        if "1" in phase_steps:
            return str(phase_steps["1"])
        if "2" in phase_steps:
            return str(phase_steps["2"])

    explicit_step = find_explicit_step()
    if explicit_step != "N/A":
        return explicit_step

    for part in parts:
        match = re.search(r"(?:^|_)steps?_(\d+)(?:_|$)", part)
        if match:
            return match.group(1)

    return "N/A"

def step_matches_model_dir(model_dir, target_steps):
    if target_steps is None:
        return True
    return parse_step_from_model_dir(model_dir) == str(target_steps)

def find_model_dirs(root_dir, target_steps=None):
    """Walks a directory and finds all subdirectories containing a 'model_xxx.pt' file."""
    model_dirs = []
    pattern = re.compile(r"model_(\d+)\.pt$")
    
    if not os.path.isdir(root_dir):
        print(f"Error: Search directory does not exist: {root_dir}")
        return []
        
    for dirpath, _, filenames in os.walk(root_dir):
        if "model.pt" in filenames or any(pattern.match(f) for f in filenames):
            if not step_matches_model_dir(dirpath, target_steps):
                continue
            model_dirs.append(dirpath)
            
    return model_dirs

def extract_seed_from_path(path_str):
    """Extracts the seed number from a path string."""
    match = re.search(r"seed[-_]?(\d+)", path_str, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

RATIO_PREFIXES = ("retained_ratios", "retain_ratios", "learn_ratios", "ratios")
RATIO_TOKEN_PATTERN = re.compile(r"^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")

def normalize_ratio_token(token):
    normalized = token.strip().lower().replace("p", ".")
    if not RATIO_TOKEN_PATTERN.fullmatch(normalized):
        return None
    try:
        value = float(normalized)
    except ValueError:
        return None
    if value < 0.0 or value > 1.0:
        return None
    return normalized

def parse_ratio_segment(path_str, allowed_lengths=None):
    """
    Extract ratio values from path segments such as ratios_0.5_0.5 or
    learn_ratios_0.25_0.5_0.75. When a nested run path has multiple ratio
    segments, the last valid segment is the effective setting for that model.
    """
    matched = (None, None)
    for path_part in re.split(r"[\\/]", path_str):
        lower_part = path_part.lower()
        for prefix in RATIO_PREFIXES:
            marker = prefix + "_"
            start = lower_part.find(marker)
            while start != -1:
                before_ok = start == 0 or not lower_part[start - 1].isalnum()
                if before_ok:
                    tokens = []
                    tail = path_part[start + len(marker):]
                    for raw_token in tail.split("_"):
                        token = normalize_ratio_token(raw_token)
                        if token is None:
                            break
                        tokens.append(token)
                    if tokens and (allowed_lengths is None or len(tokens) in allowed_lengths):
                        matched = (" ".join(tokens), "_".join(tokens))
                start = lower_part.find(marker, start + 1)
    return matched

def parse_ratios_pointmaze(path_str):
    return parse_ratio_segment(path_str, allowed_lengths={3})

def parse_ratios_mujoco(path_str):
    return parse_ratio_segment(path_str, allowed_lengths={1, 2})

def parse_ratios_quadx(path_str):
    return parse_ratio_segment(path_str)

def parse_ratios(path_str, dataset, target_root):
    dataset = (dataset or "").lower()
    target_root = (target_root or "").lower()

    if "quadx" in dataset or "quadx" in target_root:
        return parse_ratios_quadx(path_str)

    if "pointmaze" in dataset or "pointmaze" in target_root:
        return parse_ratios_pointmaze(path_str)

    if (
        "mujoco" in dataset
        or "halfcheetah" in target_root
        or "walker" in target_root
        or "hopper" in target_root
    ):
        return parse_ratios_mujoco(path_str)

    return parse_ratios_quadx(path_str)

def parse_method_name(path_str, fallback_algo):
    if fallback_algo is not None and fallback_algo not in path_str:
        return None
    path_upper = path_str.upper()
    if "CQL" in path_upper: return "CQL"
    if "IQL" in path_upper: return "IQL"
    if 'TD3' in path_upper and 'BC' in path_upper: return 'TD3PLUSBC'
    if 'PLA' in path_upper: return "PLASP"
    if "BCQ" in path_upper: return "BCQ"
    if "BEAR" in path_upper: return "BEAR"
    if "CRR" in path_upper: return "CRR"
    return fallback_algo

def find_matching_reference_model(search_root, target_seed, target_algo, ratio=None, ignore_no_shuffle=True):
    """
    Searches inside 'search_root' for a directory that matches seed, algo, and optionally ratio.
    """
    candidates = []
    pt_pattern = re.compile(r"model_(\d+)\.pt$")

    for dirpath, _, filenames in os.walk(search_root):
        if ignore_no_shuffle and "no_shuffle" in dirpath:
            continue
            
        has_model = "model.pt" in filenames or any(pt_pattern.match(f) for f in filenames)
        if not has_model:
            continue
        
        seed_in_path = extract_seed_from_path(dirpath)
        if seed_in_path != target_seed:
            continue
            
        if target_algo.upper() not in dirpath.upper():
            continue
            
        # Optional ratio match (Original model usually skips this since it uses full data)
        if ratio and not path_matches_ratio(dirpath, ratio):
            continue   
            
        candidates.append(dirpath)
    
    if not candidates:
        return None
    
    candidates.sort(key=len, reverse=True)
    return candidates[0]

# =============================================================================
# 2. Main Script
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Wasserstein measurement script with AUTO matching of Original, Retrain, and Unlearn models."
    )
    
    # --- Core Logic Arguments ---
    parser.add_argument("--unlearn-dir", type=str, required=True,
                        help="Root directory for UNLEARNED models (Target).")
    parser.add_argument("--retrain-dir", type=str, required=True,
                        help="Root directory for RETRAIN models (Baseline).")
    parser.add_argument("--original-dir", type=str, required=True,
                        help="Root directory for ORIGINAL models (Provides Critic).")
    
    parser.add_argument("--algo", type=str, default=None,
                        help="Fallback algorithm if it cannot be parsed from the path.")

    # --- Fixed Parameters ---
    parser.add_argument("--dataset", type=str, default="pointmaze",
                        help="Name of the dataset (e.g., 'pointmaze').")
    parser.add_argument("--datasets", nargs='+', 
                        default=['large-dense-v2', 'umaze-dense-v2', 'medium-dense-v2'],
                        help="List of environment datasets.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional: Force a specific seed.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use.")
    parser.add_argument("--steps", type=int, default=None,
                        help="Optional: only use unlearned model directories whose parsed steps match this value.")
    parser.add_argument("--ratio", nargs="+", default=None,
                        help="Optional ratio directory filter, e.g. 0.9_1.0_1.0 or 0.9 1.0 1.0.")

    # --- Execution Control ---
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel jobs to run.")
    parser.add_argument("--no-gpu-round-robin", action="store_true",
                        help="If set, all parallel jobs will use the same --gpu ID.")
    parser.add_argument('--ignore-no-shuffle', type=bool, default=True,
                        help="Ignore directories with 'no_shuffle'.")
                        
    args = parser.parse_args()
    try:
        ratio_filter = normalize_ratio_filter(args.ratio)
    except ValueError as exc:
        parser.error(str(exc))
    
    max_jobs = args.parallel
    output_script_name = generate_output_filename(args.unlearn_dir, ratio_filter)
    datasets_str = ' '.join(args.datasets)
    newline = os.linesep
    
    print(f"Searching for unlearned models in: {args.unlearn_dir} ...")
    model_unlearn_candidates = find_model_dirs(args.unlearn_dir, target_steps=args.steps)
    print(f"Found {len(model_unlearn_candidates)} candidate unlearned model directories.")
    if args.steps is not None:
        print(f"Step filter enabled: steps={args.steps}")
    if ratio_filter is not None:
        print(f"Ratio filter enabled: ratio={ratio_filter}")
    
    if not model_unlearn_candidates:
        print("No model directories found. Exiting.")
        return

    commands = []
    
    for model_path_unlearn in model_unlearn_candidates:
        if ratio_filter is not None and not path_matches_ratio(model_path_unlearn, ratio_filter):
            continue
        current_seed = args.seed if args.seed is not None else extract_seed_from_path(model_path_unlearn)
        
        if current_seed is None:
            print(f"[Skip] Could not extract seed from path: {model_path_unlearn}")
            continue

        ratios_str, connected_ratios_str = parse_ratios(model_path_unlearn, args.dataset, args.unlearn_dir)
        if not ratios_str:
            print(f"[Skip] Could not parse ratios from path: {model_path_unlearn}")
            continue

        method_name = parse_method_name(model_path_unlearn, args.algo)
        if method_name is None:
            continue
            
        # Match Retrain model (requires ratio match)
        model_path_retrain = find_matching_reference_model(
            args.retrain_dir, current_seed, method_name, ratio=connected_ratios_str, ignore_no_shuffle=args.ignore_no_shuffle
        )
        
        # Match Original model (does NOT require ratio match as it represents full data)
        model_path_orig = find_matching_reference_model(
            args.original_dir, current_seed, method_name, ratio=None, ignore_no_shuffle=args.ignore_no_shuffle
        )

        if not model_path_retrain or not model_path_orig:
            print(f"[Warning] Missing reference models for Seed={current_seed}, Algo={method_name}")
            continue
            
        if args.ignore_no_shuffle and ("no_shuffle" in model_path_unlearn):
             continue

        cmd = (
            f"python evaluate_policy_distance.py \\{newline}"
            f"    --dataset {args.dataset} \\{newline}"
            f"    --datasets {datasets_str} \\{newline}"
            f"    --original-dir {model_path_orig} \\{newline}"
            f"    --retrain-dir {model_path_retrain} \\{newline}"
            f"    --unlearn-dir {model_path_unlearn} \\{newline}"
            f"    --retained-ratios {ratios_str} \\{newline}"
            f"    --seed {current_seed} \\{newline}"
        )
        commands.append(cmd)
        print(f"[Match] Seed {current_seed}: \n   Unlearn: ...{model_path_unlearn[-40:]}\n   Retrain: ...{model_path_retrain[-40:]}\n   Orig:    ...{model_path_orig[-40:]}")

    if not commands:
        print("No valid commands generated.")
        return

    print(f"\nGenerating '{output_script_name}' with {len(commands)} commands...")
    
    with open(output_script_name, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated Policy Distance Measurement Script\n")
        f.write(f"# Original Base: {args.original_dir}\n")
        f.write(f"# Retrain Base:  {args.retrain_dir}\n")
        f.write(f"# Unlearn Root:  {args.unlearn_dir}\n\n")

        if max_jobs <= 1:
            f.write("# Running in SERIAL mode.\n")
            f.write("set -e\n")
            for i, cmd in enumerate(commands):
                f.write(f"# --- Job {i+1}/{len(commands)} --- {newline}")
                f.write(f"{cmd} \\{newline}")
                f.write(f"    --gpu {args.gpu}\n\n")
            f.write("echo 'All jobs completed.'\n")
        else:
            f.write(f"# Running in PARALLEL mode (Batches of {max_jobs}).\n\n")
            f.write("pids=()\nfail_count=0\n\n")
            for i, cmd in enumerate(commands):
                gpu_to_use = args.gpu if args.no_gpu_round_robin else args.gpu + (i % max_jobs)
                full_cmd = f"({cmd} \\{newline}    --gpu {gpu_to_use}) "
                
                f.write(f"echo '[Batch {i // max_jobs + 1}] Starting job {i+1} on GPU {gpu_to_use}'\n")
                f.write(f"{full_cmd} &\n")
                f.write("pids+=($!)\n\n")
                
                if (i + 1) % max_jobs == 0 or (i + 1) == len(commands):
                    f.write("echo 'Waiting for current batch jobs to finish...'\n")
                    f.write("for pid in \"${pids[@]}\"; do\n    wait $pid\n    if [ $? -ne 0 ]; then\n        echo \"WARNING: Job $pid failed.\"\n        fail_count=$((fail_count + 1))\n    fi\ndone\n")
                    f.write("echo 'Batch finished.'\npids=()\n\n")
            
            f.write("echo 'All jobs completed.'\n")
            f.write("if [ $fail_count -ne 0 ]; then\n    echo \"WARNING: $fail_count jobs failed!\"\n    exit 1\nelse\n    echo 'All jobs succeeded.'\nfi\n")

    try:
        os.chmod(output_script_name, 0o755)
    except OSError as e:
        pass

    print(f"Successfully generated '{output_script_name}'.")

if __name__ == "__main__":
    main()