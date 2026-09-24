#!/usr/bin/env python3
import os
import argparse
import itertools
import sys
import re

# --- Combinatorial Parameters ---
UNCERTAINTY_FILE_LIST = [
    # 'feature_knn_uncertainty_normalized.npy',
    'forget_similarities.npy',
    # 'feature_density_uncertainty_normalized.npy'
]

# These strings will be inserted directly without quotes
RATIO_LIST = ["0.0 1.0 1.0", "0.9 1.0 1.0", "1.0 0.0 1.0"]
# RATIO_LIST = ["0.75","0.85"]
# RATIO_LIST = ["1.0 1.0 0.0 1.0 1.0","1.0 1.0 1.0 0.0 1.0"]
# RATIO_LIST = [
#     "1.0 0.9",
#     "0.9 0.9"
# ]

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

# --- Helper function for dynamic naming ---
def generate_output_filename(args, root_path):
    """
    Generates a dynamic output filename based on key arguments.
    """
    # Get shorthand for threshold/quantile
    thresh_quant_str = args.fixed_threshold_or_quantile[:6] # "thresh" or "quanti"
    
    # Format values, replacing '.' with 'p' for safe filenames
    val_str = str(args.value_for_quantile_or_threshold).replace('.', 'p')
    lambda_str = str(args.lambda_penalty).replace('.', 'p')
    
    # Use root path name
    root_name = root_path.strip('/').split('/')[-1]
    
    base_name = (
        f"run_SARAVI_{args.algo}_{root_name}_{args.dataset}"
        f"lambda{lambda_str}"
    )
    return f"{base_name}.sh"

# --- Main Script ---
def main():
    parser = argparse.ArgumentParser(
        description="Generate a shell script for UDRU grid search with auto-discovery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Dynamic Parameters (Required) ---
    parser.add_argument("--algo", type=str, required=True,
                        help="Value for --algo (e.g., 'CQL' or 'IQL')")
    parser.add_argument("--lambda-penalty", type=float, default=1.0,
                        help="Value for --lambda-penalty (e.g., 1.0)")
    # the --value-for-quantile-or-threshold parameter is no longer used.
    parser.add_argument("--value-for-quantile-or-threshold", type=float, default=1.0,
                        help="Value for --value-for-quantile-or-threshold (e.g., 0.1)")    
    # --- Fixed Parameters (with Defaults) ---
    parser.add_argument("--datasets", nargs='+', 
                        default=['expert-v0', 'medium-v0'],
                        help="List of datasets.")
    parser.add_argument("--dataset", type=str, 
                        required=True,
                        help="task to perform.")
    
    # Modified: This is now the ROOT directory to search
    parser.add_argument("--model-to-unlearn-dir", type=str, required=True,
                        help="Root directory to search for models (e.g. '../Offline_RL_processing/Fully_trained/...').")
    
    parser.add_argument("--unlearning-steps", type=int, default=200000,
                        help="Number of unlearning steps.")
    
    # Modified: Seed is optional/filter
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional: If provided, only process models with this seed, or use as fallback.")
    
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use (starting ID for parallel mode).")
    parser.add_argument("--fixed-threshold-or-quantile", type=str, default="threshold",
                        help="Value for --fixed-threshold-or-quantile")
    # --- Execution Control ---
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel jobs to run. Default is 1 (serial).")
    parser.add_argument("--no-gpu-round-robin", action="store_true",
                        help="If set, all parallel jobs will use the same --gpu ID (DANGEROUS: risks OOM).")
    parser.add_argument('--shuffle', type=int, default=1,
                        help="Whether apply shuffle in splitting dataset: D_f and D_r.")
    parser.add_argument('--component_to_analyze', type=str, default='q_network',
                        help="Model component to analyze: 'q_network' or 'policy_network'")
    
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

    # --- 2. Prepare Combinations ---
    max_jobs = args.parallel
    output_script_name = generate_output_filename(args, args.model_to_unlearn_dir)
    newline = os.linesep 
    datasets_str = ' '.join(args.datasets)
    
    param_combinations = list(itertools.product(
        UNCERTAINTY_FILE_LIST,
        RATIO_LIST
    ))

    # --- 3. Generate Commands ---
    if args.dataset.lower() in {'halfcheetah', 'hopper', 'walker2d'}:
        saravi_script = 'SARAVI_mujoco.py'
    else:
        saravi_script = 'SARAVI.py'

    commands = []
    print(f"Generating commands for {len(found_models)} models x {len(param_combinations)} param combos...")

    for model_info in found_models:
        curr_model_dir = model_info['path']
        curr_seed = model_info['seed']
        
        for (unc_file, ratios) in param_combinations:
            unc_sim_val = "similarity" if unc_file == 'forget_similarities.npy' else "uncertainty"
            
            cmd = (
                f"python {saravi_script} --dataset {args.dataset} \\{newline}"
                f"    --datasets {datasets_str} \\{newline}"
                f"    --model-to-unlearn-dir {curr_model_dir} \\{newline}"
                f"    --unlearning-steps {args.unlearning_steps} \\{newline}"
                f"    --seed {curr_seed} \\{newline}"
                # gpu added in write loop
                f"    --fixed-threshold-or-quantile {args.fixed_threshold_or_quantile} \\{newline}"
                f"    --algo {args.algo} \\{newline}"
                f"    --lambda-penalty {args.lambda_penalty} \\{newline}"
                f"    --value-for-quantile-or-threshold {args.value_for_quantile_or_threshold} \\{newline}"
                f"    --uncertainty-or-similarity {unc_sim_val} \\{newline}"
                f"    --uncertainty-file-f {unc_file} \\{newline}"
                f"    --component_to_analyze {args.component_to_analyze} \\{newline}"
                f"    --shuffle {args.shuffle} \\{newline}"
                f"    --retained-ratios {ratios}"
            )
            commands.append(cmd)

    # --- Command writing logic ---
    with open(output_script_name, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated training script for UDRU (Auto-discovery)\n")
        f.write(f"# Target Algo: {args.algo}\n")
        f.write(f"# Search Root: {args.model_to_unlearn_dir}\n")
        f.write(f"# Total commands: {len(commands)}\n\n")
        
        if max_jobs <= 1:
            # --- SERIAL MODE ---
            f.write("# Running in SERIAL mode.\n")
            f.write("set -e\n")
            f.write("set -o xtrace\n\n")
            
            for i, cmd in enumerate(commands):
                f.write(f"# --- Job {i+1}/{len(commands)} --- {newline}")
                f.write(f"{cmd} \\{newline}")
                f.write(f"    --gpu {args.gpu}\n\n")
                
                f.write(f"echo \"Finished job {i+1}\"{newline}")
                f.write("echo \"-------------------------------------------------\"\n\n")
            
            f.write("echo \"All UDRU training commands have been executed.\"\n")

        else:
            # --- PARALLEL BATCH MODE ---
            f.write(f"# Running in PARALLEL mode (Batches of {max_jobs}).\n")
            f.write("pids=()\n")
            f.write("fail_count=0\n\n")

            for i, cmd in enumerate(commands):
                # GPU Assignment Logic
                if args.no_gpu_round_robin:
                    gpu_to_use = args.gpu
                else:
                    gpu_to_use = args.gpu + (i % max_jobs) 

                f.write(f"# --- Batch {i // max_jobs + 1}, Job {i+1} (GPU {gpu_to_use}) --- {newline}")
                
                # Full command with GPU
                full_cmd = (
                    f"({cmd} \\{newline}"
                    f"    --gpu {gpu_to_use}) "
                )
                
                # Run command in a subshell and in the background
                f.write(f"echo '[Batch {i // max_jobs + 1}] Starting job {i+1} on GPU {gpu_to_use}'\n")
                f.write(f"{full_cmd} &\n")
                f.write("pids+=($!)\n\n")
                
                is_batch_full = (i + 1) % max_jobs == 0
                is_last_command = (i + 1) == len(commands)
                
                if is_batch_full or is_last_command:
                    f.write("# --- Waiting for batch to finish ---\n")
                    f.write(f'echo "Waiting for ${{#pids[@]}} jobs in batch {i // max_jobs + 1} to finish..."\n')
                    
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
            f.write("echo 'All UDRU training commands have been executed.'\n")
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

    print(f"Successfully generated '{output_script_name}'.")
    print(f"You can now run it using: ./{output_script_name}")

if __name__ == "__main__":
    main()
