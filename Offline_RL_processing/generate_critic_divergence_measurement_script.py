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
    return f"run_critic_evlauation_{base_name}{ratio_suffix}{time_str}.sh"

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
    """
    Extracts the seed number from a path string.
    Matches: seed_42, seed42, seed-42
    """
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
    if fallback_algo is not None:
        if fallback_algo not in path_str:
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

def find_matching_reference_model(search_root, target_seed, target_algo, ratio, ignore_no_shuffle = True):
    """
    Searches inside 'search_root' (model-dir-1) for a directory that:
    1. Contains a model file (model.pt or model_xxx.pt).
    2. Contains 'seed_{target_seed}' (or similar) in its path.
    3. Contains '{target_algo}' in its path.
    """
    candidates = []
    pt_pattern = re.compile(r"model_(\d+)\.pt$")

    # Walk through model-dir-1
    for dirpath, _, filenames in os.walk(search_root):
        # 1. Check for model file presence
        if ignore_no_shuffle:
            if "no_shuffle" in dirpath:
                continue
        has_model = "model.pt" in filenames or any(pt_pattern.match(f) for f in filenames)
        if not has_model:
            continue
        
        # 2. Check Seed Match
        seed_in_path = extract_seed_from_path(dirpath)
        if seed_in_path != target_seed:
            continue
            
        # 3. Check Algo Match (Case insensitive)
        if target_algo.upper() not in dirpath.upper():
            continue
        if not path_matches_ratio(dirpath, ratio):
            continue   
        candidates.append(dirpath)
    
    if not candidates:
        return None
    
    # Heuristic: If multiple matches, prefer the one with the longest path 
    # (often implies a specific timestamp folder inside the algo folder)
    # or prefer the one that might explicitly mention 'learn_ratios' if comparing against retrained.
    # For now, sorting by length descending is a safe default for "deepest valid folder".
    candidates.sort(key=len, reverse=True)
    return candidates[0]

# =============================================================================
# 2. Main Script
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate KL/Wasserstein measurement script with AUTO matching of Seed and Algo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Core Logic Arguments ---
    parser.add_argument("--model-dir-2", type=str, required=True,
                        help="Root directory to search for UNLEARNED models (Target).")
    parser.add_argument("--model-dir-1", type=str, required=True,
                        help="Root directory to search for REFERENCE models (Source).")
    
    parser.add_argument("--algo", type=str, default=None,
                        help="Fallback algorithm if it cannot be parsed from the path.")

    # --- Fixed Parameters ---
    parser.add_argument("--dataset", type=str, default="pointmaze",
                        help="Name of the dataset (e.g., 'pointmaze').")
    parser.add_argument("--datasets", nargs='+', 
                        default=['large-dense-v2', 'umaze-dense-v2', 'medium-dense-v2'],
                        help="List of environment datasets.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional: Force a specific seed. If not set, seed is extracted from directory names.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use.")
    parser.add_argument("--steps", type=int, default=None,
                        help="Optional: only use target model directories whose parsed steps match this value.")
    parser.add_argument("--ratio", nargs="+", default=None,
                        help="Optional ratio directory filter, e.g. 0.9_1.0_1.0 or 0.9 1.0 1.0.")

    # --- Execution Control ---
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel jobs to run.")
    parser.add_argument("--no-gpu-round-robin", action="store_true",
                        help="If set, all parallel jobs will use the same --gpu ID.")
    parser.add_argument('--Wasserstein-or-KL', type=str, default="KL",
                        help="Decide using Wasserstein Distance or KL divergence")
    parser.add_argument('--ignore-no-shuffle', type=bool, default=True,
                        help="Decide using Wasserstein Distance or KL divergence")
    args = parser.parse_args()
    try:
        ratio_filter = normalize_ratio_filter(args.ratio)
    except ValueError as exc:
        parser.error(str(exc))
    
    # --- Setup ---
    max_jobs = args.parallel
    output_script_name = generate_output_filename(args.model_dir_2, ratio_filter)
    datasets_str = ' '.join(args.datasets)
    newline = os.linesep
    
    # --- Find all candidate models (Model 2) ---
    print(f"Searching for unlearned models in: {args.model_dir_2} ...")
    model_2_candidates = find_model_dirs(args.model_dir_2, target_steps=args.steps)
    print(f"Found {len(model_2_candidates)} candidate unlearned model directories.")
    if args.steps is not None:
        print(f"Step filter enabled: steps={args.steps}")
    if ratio_filter is not None:
        print(f"Ratio filter enabled: ratio={ratio_filter}")
    
    if not model_2_candidates:
        print("No model directories found. Exiting.")
        return

    commands = []
    
    # --- Build all commands ---
    for model_path_2 in model_2_candidates:
        if ratio_filter is not None and not path_matches_ratio(model_path_2, ratio_filter):
            continue
        # 1. Extract Seed (Auto or Manual)
        current_seed = args.seed
        
        if current_seed is None:
            current_seed = extract_seed_from_path(model_path_2)
        
        if current_seed is None:
            print(f"[Skip] Could not extract seed from path: {model_path_2}")
            continue

        # 2. Parse Ratios
        ratios_str, connected_ratios_str = parse_ratios(model_path_2, args.dataset, args.model_dir_2)
        if not ratios_str:
            print(f"[Skip] Could not parse ratios from path: {model_path_2}")
            continue

        # 3. Parse Method Name
        method_name = parse_method_name(model_path_2, args.algo)
        if method_name is None:
            continue
        # 4. Find Matching Reference Model (Model 1)
        model_path_1 = find_matching_reference_model(args.model_dir_1, current_seed, method_name,connected_ratios_str)

        if not model_path_1:
            print(f"[Warning] No matching reference model found in {args.model_dir_1}")
            print(f"          Query: Seed={current_seed}, Algo={method_name}")
            print(f"          Source Path: {model_path_2}")
            continue
        if args.ignore_no_shuffle:
            if "no_shuffle" in model_path_1 or "no_shuffle" in model_path_2:
                print("you are in ignore no shuffle mode, ignore dir with no_shuffle.")
                continue
        # if "no_shuffle" in model_path_2:
        #     if "no_shuffle" not in model_path_1:
        #         print("shuffle setting mismatch.")
        #         print(model_path_1)
        #         print(model_path_2)
        #         continue
        # elif "no_shuffle" not in model_path_2:
        #     if "no_shuffle" in model_path_1:
        #         print("shuffle setting mismatch.")
        #         print(model_path_1)
        #         print(model_path_2)
        # 5. Construct the command
        cmd = (
            f"python evaluate_critic_divergence.py \\{newline}"
            f"    --dataset {args.dataset} \\{newline}"
            f"    --datasets {datasets_str} \\{newline}"
            f"    --model-path-1 {model_path_1} \\{newline}"
            f"    --model-path-2 {model_path_2} \\{newline}"
            f"    --retained-ratios {ratios_str} \\{newline}"
            f"    --seed {current_seed} \\{newline}"

        )
        commands.append(cmd)
        print(f"[Match] Seed {current_seed}: \n   Target: ...{model_path_2[-50:]}\n   Ref:    ...{model_path_1[-50:]}")

    if not commands:
        print("No valid commands generated.")
        return

    # --- Write the script file ---
    print(f"\nGenerating '{output_script_name}' with {len(commands)} commands...")
    
    with open(output_script_name, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated KL/Wasserstein measurement script\n")
        f.write(f"# Reference Base: {args.model_dir_1}\n")
        f.write(f"# Search Root: {args.model_dir_2}\n\n")

        if max_jobs <= 1:
            # --- SERIAL MODE ---
            f.write("# Running in SERIAL mode.\n")
            f.write("set -e\n")
            
            for i, cmd in enumerate(commands):
                f.write(f"# --- Job {i+1}/{len(commands)} --- {newline}")
                f.write(f"{cmd} \\{newline}")
                f.write(f"    --gpu {args.gpu}\n\n")
            
            f.write("echo 'All jobs completed.'\n")

        else:
            # --- PARALLEL BATCH MODE ---
            f.write(f"# Running in PARALLEL mode (Batches of {max_jobs}).\n\n")
            f.write("pids=()\n")
            f.write("fail_count=0\n\n")

            for i, cmd in enumerate(commands):
                if args.no_gpu_round_robin:
                    gpu_to_use = args.gpu
                else:
                    gpu_to_use = args.gpu + (i % max_jobs)
                
                full_cmd = (
                    f"({cmd} \\{newline}"
                    f"    --gpu {gpu_to_use}) "
                )
                
                f.write(f"echo '[Batch {i // max_jobs + 1}] Starting job {i+1} on GPU {gpu_to_use}'\n")
                f.write(f"{full_cmd} &\n")
                f.write("pids+=($!)\n\n")
                
                is_batch_full = (i + 1) % max_jobs == 0
                is_last_command = (i + 1) == len(commands)
                
                if is_batch_full or is_last_command:
                    f.write("# --- Waiting for batch to finish ---\n")
                    f.write(f"echo 'Waiting for current batch jobs to finish...'\n")
                    f.write("for pid in \"${pids[@]}\"; do\n")
                    f.write("    wait $pid\n")
                    f.write("    if [ $? -ne 0 ]; then\n")
                    f.write("        echo \"WARNING: Job $pid failed.\"\n")
                    f.write("        fail_count=$((fail_count + 1))\n")
                    f.write("    fi\n")
                    f.write("done\n")
                    f.write("echo 'Batch finished.'\n")
                    f.write("pids=()\n\n")
            
            f.write("echo 'All jobs completed.'\n")
            f.write("if [ $fail_count -ne 0 ]; then\n")
            f.write(f"    echo \"WARNING: $fail_count jobs failed!\"\n")
            f.write("    exit 1\n")
            f.write("else\n")
            f.write("    echo 'All jobs succeeded.'\n")
            f.write("fi\n")

    try:
        os.chmod(output_script_name, 0o755)
    except OSError as e:
        print(f"Warning: Could not set execute permissions: {e}")

    print(f"Successfully generated '{output_script_name}'.")

if __name__ == "__main__":
    main()