#!/usr/bin/env python3
import os
import argparse
import itertools
from pathlib import Path
import sys
import re

# --- Specify the combination and ratios---

COMBINATIONS = [
# for quadx tasks
    # (
    #     ["retain_random_static_spheres-v0", "retain_wide_corridor_right-v0", "forget_narrow_symmetric_corridor-v0", "forget_crossing_at_goal_path-v0", "forget_low_altitude_barrier-v0"],
    #     ["1.0 1.0 0.0 1.0 1.0"]
    # ),
    # (
    #     ["retain_random_static_spheres-v0", "retain_wide_corridor_right-v0", "forget_narrow_symmetric_corridor-v0", "forget_crossing_at_goal_path-v0", "forget_low_altitude_barrier-v0"],
    #     ["1.0 1.0 1.0 0.0 1.0"]
    # ),

# for mujoco tasks
    # (
    #     ['expert-v0', 'medium-v0'], 
    #     ['1.0', '0.9']
    # ),

    # (
    #     ['expert-v0', 'medium-v0'], 
    #     ['0.9', '1.0']
    # ),

    # (
    #     ['expert-v0', 'medium-v0'], 
    #     ['0.9', '0.9']
    # ),

#  for pointmaze
    # (
    #     ['large-dense-v2', 'umaze-dense-v2', 'medium-dense-v2'], 
    #     ['1.0', '0.0', '1.0']
    # ),

    # (
    #     ['large-dense-v2', 'umaze-dense-v2', 'medium-dense-v2'], 
    #     ['0.0', '1.0', '1.0']
    # ),

    # (
    #     ['large-dense-v2', 'umaze-dense-v2', 'medium-dense-v2'], 
    #     ['0.9', '1.0', '1.0']
    # ),

# for walker2d, single dataset
    (
        ['expert-v0'],
        ['0.85']
    ),
    (
        ['expert-v0'],
        ['0.75']
    ),
    
]

# --- Helper Functions ---
def get_max_step_model(files):
    """
    Filters files matching 'model_xxx.pt' and returns the one with the largest number.
    Returns None if no matching files are found.
    """
    pattern = re.compile(r"^model_(\d+)\.pt$")
    valid_models = []
    
    for f in files:
        match = pattern.match(f)
        if match:
            step_count = int(match.group(1))
            valid_models.append((step_count, f))
    
    # Also check for 'model.pt' as a fallback
    if not valid_models and 'model.pt' in files:
        return 'model.pt'
    
    if not valid_models:
        return None
    
    valid_models.sort(key=lambda x: x[0], reverse=True)
    return valid_models[0][1]

def extract_seed_from_path(path_str):
    """
    Extracts the seed number from a path string like '.../seed_42_steps_.../'.
    Returns int(seed) or None if not found.
    """
    match = re.search(r"seed_(\d+)", path_str)
    if match:
        return int(match.group(1))
    return None

def generate_output_filename(args, root_path):
    """
    Generates a dynamic output filename based on root path.
    """
    root_name = root_path.strip('/').split('/')[-1]
    base_name = (f"task_{args.dataset}_"
        f"run_baselines_{args.algo}_"
        f"{root_name}_"
        f"{args.total_steps}steps"
    )
    return f"{base_name}.sh"


def main():
    parser = argparse.ArgumentParser(
        description="Generate a shell script to run unlearning baselines with auto-discovery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--algo", type=str, required=True,
                        help="Algorithm name (e.g., 'IQL', 'CQL').")
    parser.add_argument("--total-steps", type=int, required=True,
                        help="Total unlearning steps (e.g., 200000).")

    parser.add_argument("--dataset", type=str, default="pointmaze",
                        help="Name of the dataset (e.g., 'pointmaze').")
    
    # Modified: This is now the ROOT directory to search
    parser.add_argument("--model-to-unlearn-dir", type=str, required=True,
                        help="Root directory to search for models (e.g. '../Offline_RL_processing/Fully_trained/...').")
    
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Alpha value for the algorithms.")
    
    # Modified: Seed is optional/filter
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional: If provided, only process models with this seed, or use as fallback.")
    
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use (starting ID for parallel mode).")
    parser.add_argument('--shuffle', type=int, default=1,
                        help="Whether apply shuffle in splitting dataset: D_f and D_r.")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel jobs to run. Default is 1 (serial).")
    parser.add_argument("--no-gpu-round-robin", action="store_true",
                        help="If set, all parallel jobs will use the same --gpu ID (DANGEROUS: risks OOM).")
    
    args = parser.parse_args()
    
    # --- 1. Discover Models and Seeds ---
    print(f"Searching for valid models in: {args.model_to_unlearn_dir} ...")
    found_models = [] # List of dicts: {'path': str, 'seed': int}

    if not os.path.isdir(args.model_to_unlearn_dir):
        print(f"Error: Directory '{args.model_to_unlearn_dir}' does not exist.")
        sys.exit(1)

    for dirpath, dirnames, filenames in os.walk(args.model_to_unlearn_dir):
        # Optional: Check if algo name is in path
        if args.algo not in dirpath:
             continue

        # Check for .pt file
        best_model_file = get_max_step_model(filenames)
        
        if best_model_file:
            # Try to extract seed
            model_seed = extract_seed_from_path(dirpath)
            
            final_seed = None
            
            if model_seed is not None:
                # Seed found in path
                if args.seed is not None and model_seed != args.seed:
                    # Filter: User specified a seed, but this folder matches a different one. Skip.
                    continue
                final_seed = model_seed
            else:
                # Seed NOT found in path
                if args.seed is not None:
                    final_seed = args.seed
                    print(f"[Info] Seed not found in path '{dirpath}'. Using fallback: {final_seed}")
                else:
                    print(f"[Warning] Skipping '{dirpath}': No seed in path and no --seed arg provided.")
                    continue
            
            found_models.append({
                'path': dirpath,
                'seed': final_seed
            })

    if not found_models:
        print("No valid models found matching the criteria.")
        sys.exit(1)
    
    print(f"Found {len(found_models)} valid model directories.")

    max_jobs = args.parallel
    output_script_name = generate_output_filename(args, args.model_to_unlearn_dir)
    newline = os.linesep
    
    total_steps = args.total_steps
    # phase1_steps = int(total_steps * 0.8)
    phase1_steps = int(total_steps * 0.8)  # Adjusted to 100% as negative reward.
    phase2_steps = total_steps - phase1_steps
    print(f"Total steps: {total_steps} -> Phase1: {phase1_steps}, Phase2: {phase2_steps}")

    commands = []

    # --- 2. Generate Commands (Found Models x Combinations) ---
    print(f"Generating commands for {len(found_models)} models x {len(COMBINATIONS)} combinations...")
    
    for model_info in found_models:
        curr_model_dir = model_info['path']
        curr_seed = model_info['seed']
        
        for (datasets_list, ratios_list) in COMBINATIONS:
            datasets_str = ' '.join(datasets_list)
            ratios_str = ' '.join(ratios_list)
            
            # --- Command 1: random_rewarding.py ---
            cmd_rewarding = (
                f"python random_rewarding.py \\{newline}"
                f"    --dataset {args.dataset} \\{newline}"
                f"    --datasets {datasets_str} \\{newline}"
                f"    --model-to-unlearn-dir {curr_model_dir} \\{newline}"
                f"    --retained-ratios {ratios_str} \\{newline}"
                f"    --algo {args.algo} \\{newline}"
                f"    --total-steps {total_steps} \\{newline}" 
                f"    --alpha {args.alpha} \\{newline}"
                f"    --shuffle {args.shuffle} \\{newline}"
                f"    --seed {curr_seed}"
            )
            commands.append(cmd_rewarding)
            
            # --- Command 2: trajectory_deleter.py ---
            cmd_deleter = (
                f"python trajectory_deleter.py \\{newline}"
                f"    --dataset {args.dataset} \\{newline}"
                f"    --datasets {datasets_str} \\{newline}"
                f"    --model-to-unlearn-dir {curr_model_dir} \\{newline}"
                f"    --retained-ratios {ratios_str} \\{newline}"
                f"    --algo {args.algo} \\{newline}"
                f"    --phase1-total-steps {phase1_steps} \\{newline}"
                f"    --phase2-total-steps {phase2_steps} \\{newline}"
                f"    --alpha {args.alpha} \\{newline}"
                f"    --shuffle {args.shuffle} \\{newline}"
                f"    --seed {curr_seed}"
            )
            commands.append(cmd_deleter)

    print(f"Generating '{output_script_name}' with {len(commands)} total commands...")
    
    with open(output_script_name, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated script for unlearning baselines (Auto-discovery)\n")
        f.write(f"# Search Root: {args.model_to_unlearn_dir}\n")
        f.write(f"# Total combinations: {len(found_models)} models * {len(COMBINATIONS)} configs * 2 methods = {len(commands)}\n\n")

        if max_jobs <= 1:
            # --- SERIAL MODE ---
            f.write("# Running in SERIAL mode.\n")
            f.write("set -e\n")
            f.write("set -o xtrace\n\n")
            
            for i, cmd in enumerate(commands):
                f.write(f"# --- Job {i+1}/{len(commands)} --- {newline}")
                f.write(f"{cmd} \\{newline}")
                f.write(f"    --gpu {args.gpu}\n\n")
            
            f.write("echo 'All baseline jobs completed.'\n")

        else:
            # --- PARALLEL BATCH MODE ---
            f.write(f"# Running in PARALLEL mode (Batches of {max_jobs}).\n\n")
            f.write("pids=()\n")
            f.write("fail_count=0\n\n")

            for i, cmd in enumerate(commands):
                # GPU Assignment Logic
                if args.no_gpu_round_robin:
                    gpu_to_use = args.gpu
                else:
                    gpu_to_use = args.gpu + (i % max_jobs)
                
                f.write(f"# --- Batch {i // max_jobs + 1}, Job {i+1} (GPU {gpu_to_use}) --- {newline}")
                
                # Add final GPU arg to command
                full_cmd = (
                    f"({cmd} \\{newline}"
                    f"    --gpu {gpu_to_use}) "
                )
                
                f.write(f"echo '[Batch {i // max_jobs + 1}] Starting job {i+1} on GPU {gpu_to_use}'\n")
                f.write(f"{full_cmd} &\n")
                f.write("pids+=($!)\n\n")
                
                is_batch_full = (i + 1) % max_jobs == 0
                is_last_command = (i + 1) == len(commands)
                
                if (is_batch_full or is_last_command):
                    f.write("# --- Waiting for batch to finish ---\n")
                    f.write(f"echo 'Waiting for {len(commands[i-max_jobs+1:i+1]) if is_batch_full else len(commands) % max_jobs} jobs in batch {i // max_jobs + 1} to finish...'\n")
                    
                    f.write("for pid in \"${pids[@]}\"; do\n")
                    f.write("    wait $pid\n")
                    f.write("    if [ $? -ne 0 ]; then\n")
                    f.write("        echo \"WARNING: Job $pid failed with exit code $?\"\n")
                    f.write("        fail_count=$((fail_count + 1))\n")
                    f.write("    fi\n")
                    f.write("done\n")
                    
                    f.write("echo 'Batch finished.'\n")
                    f.write("pids=()\n\n")
            
            # --- Final check ---
            f.write("echo 'All baseline jobs completed.'\n")
            f.write("if [ $fail_count -ne 0 ]; then\n")
            f.write(f"    echo \"WARNING: $fail_count jobs failed!\"\n")
            f.write("    exit 1\n")
            f.write("else\n")
            f.write("    echo 'All jobs succeeded.'\n")
            f.write("fi\n")

    # Make the script executable
    try:
        os.chmod(output_script_name, 0o755)
    except OSError as e:
        print(f"Warning: Could not set execute permissions on {output_script_name}: {e}")

    print(f"\nSuccessfully generated '{output_script_name}'.")
    print("You can now edit the COMBINATIONS list at the top of the .py script or run the .sh script.")
    print(f"To run: ./{output_script_name}")

if __name__ == "__main__":
    main()