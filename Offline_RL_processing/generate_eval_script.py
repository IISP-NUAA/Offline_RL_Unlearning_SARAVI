import os
import re
import argparse
import sys
import time
from generator_ratio_filter import normalize_ratio_filter, path_matches_ratio, ratio_filename_suffix
# Fixed command parts
CMD_PREFIX = "python evaluation.py"
FIXED_ARGS = (
    "--n-trials 50  --gpu 0 --eval_env_flag 1"
)

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

def get_max_step_model(files):
    """
    Filters files matching 'model_xxx.pt' and returns the one with the largest number.
    Returns None if no matching files are found.
    """
    # Regex to match 'model_' followed by digits, ending in '.pt'
    pattern = re.compile(r"^model_(\d+)\.pt$")
    
    valid_models = []
    
    for f in files:
        match = pattern.match(f)
        if match:
            # group(1) captures the digits
            step_count = int(match.group(1))
            valid_models.append((step_count, f))
    
    if not valid_models:
        return None
    
    # Sort by step count (index 0) in descending order
    valid_models.sort(key=lambda x: x[0], reverse=True)
    # Return the filename (index 1) of the best model (first item)
    return valid_models[0][1]

def extract_seed_from_path(path_str):
    """
    Extracts the seed number from a path string like '.../seed_42_steps_.../'.
    Returns int(seed) or None if not found.
    """
    # Searches for 'seed_' followed by one or more digits
    match = re.search(r"seed_(\d+)", path_str)
    if match:
        return int(match.group(1))
    return None

def generate_output_filename(root_path, ratio=None):
    """Generate an output filename with an optional ratio marker."""
    normalized_path = root_path.rstrip('/')
    base_name = normalized_path.replace("/", "_").replace(".", "")
    base_name = re.sub(r'[^\w\-_\.]', '', base_name)
    ratio_suffix = ratio_filename_suffix(ratio)
    time_str = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    return f"run_eval_{base_name}{ratio_suffix}{time_str}.sh"

def main():
    parser = argparse.ArgumentParser(description="Generate evaluation command scripts with auto-detected seeds.")
    parser.add_argument("--root", type=str, required=True, 
                        help="Root directory to search for models (e.g: Unlearned/pointmaze/...)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel jobs in the generated script. Default is 1 (serial).")
    parser.add_argument("--env-id-list", nargs='+', 
                        default=['D4RL/pointmaze/umaze-dense-v2', 'D4RL/pointmaze/large-dense-v2', 'D4RL/pointmaze/medium-dense-v2'],
                        help="List of eval envs.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional: Fallback seed if 'seed_X' is not found in the directory path.")
    parser.add_argument("--algo", default=None, type=str,
                        help="Optional: select only dirs that carry algo name in path")
    parser.add_argument("--steps", type=int, default=None,
                        help="Optional: only use model directories whose parsed steps match this value.")
    parser.add_argument("--ratio", nargs="+", default=None,
                        help="Optional ratio directory filter, e.g. 0.9_1.0_1.0 or 0.9 1.0 1.0.")
    
    args = parser.parse_args()
    try:
        ratio_filter = normalize_ratio_filter(args.ratio)
    except ValueError as exc:
        parser.error(str(exc))

    search_root = args.root
    max_jobs = args.parallel
    env_list_str = ' '.join(args.env_id_list)
    
    if not os.path.isdir(search_root):
        print(f"Error: Directory '{search_root}' does not exist.")
        sys.exit(1)

    output_script_name = generate_output_filename(search_root, ratio_filter)

    commands = []
    print(f"Searching for models in: {search_root} ...")
    if args.steps is not None:
        print(f"Step filter enabled: steps={args.steps}")
    if ratio_filter is not None:
        print(f"Ratio filter enabled: ratio={ratio_filter}")

    for dirpath, dirnames, filenames in os.walk(search_root):
        if ratio_filter is not None and not path_matches_ratio(dirpath, ratio_filter):
            continue
        # 1. Check for model files
        if args.algo is not None:
            if args.algo not in dirpath:
                print("path contains no algo name----", args.algo)
                continue
        best_model_file = get_max_step_model(filenames)
        if best_model_file and not step_matches_model_dir(dirpath, args.steps):
            continue
        
        if best_model_file:
            # 2. Determine Seed
            # First try to extract from path
            model_seed = extract_seed_from_path(dirpath)
            
            # If not found in path, use the CLI argument fallback
            if model_seed is None:
                if args.seed is not None:
                    model_seed = args.seed
                    print(f"[Info] Seed not found in path for '{dirpath}'. Using fallback seed: {model_seed}")
                else:
                    print(f"[Warning] Skipping '{dirpath}': No seed found in path and no fallback --seed provided.")
                    continue
            
            # 3. Construct Command
            cmd = (
                f"{CMD_PREFIX} {FIXED_ARGS} "
                f"--model-dir {dirpath} "
                f"--model-filename {best_model_file} "
                f"--env-id-list {env_list_str} "
                f"--seed {str(model_seed)}"
            )
            commands.append(cmd)
            print(f"[Added] Seed: {model_seed} | Model: .../{os.path.basename(dirpath)}/{best_model_file}")

    if not commands:
        print("No valid models found matching criteria.")
        return

    # Write commands to the shell script
    with open(output_script_name, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Auto-generated evaluation script for root: {search_root}\n")
        f.write("# Logic: Seeds automatically extracted from directory paths.\n\n")
        
        # --- Logic for Serial vs Parallel ---
        
        if max_jobs <= 1:
            # --- SERIAL MODE ---
            f.write("# Running in SERIAL mode.\n\n")
            f.write("set -e\n\n") 
            f.write("set -o xtrace\n\n") # Print each command before executing it
            
            for cmd in commands:
                f.write(cmd + "\n")
                model_dir = cmd.split('--model-dir')[1].split()[0]
                f.write(f"echo 'Finished evaluating: {model_dir}'\n")
                f.write("echo '---------------------------------------------'\n\n")
            
            f.write("echo 'All evaluations completed!'\n")
        
        else:
            # --- PARALLEL BATCH MODE ---
            f.write(f"# Running in PARALLEL mode (Batches of {max_jobs}).\n\n")
            f.write(f"echo 'Starting parallel evaluation ({len(commands)} total jobs, {max_jobs} per batch)...'\n\n")
            
            f.write("pids=()\n")
            f.write("fail_count=0\n\n")

            for i, cmd in enumerate(commands):
                model_dir = cmd.split('--model-dir')[1].split()[0]
                
                f.write(f"echo '[Batch {i // max_jobs + 1}] Starting: {model_dir}'\n")
                # Run command in a subshell and in the background
                f.write(f"({cmd}) &\n")
                # Store its Process ID (PID)
                f.write("pids+=($!)\n\n")
                
                # Check if the batch is full OR if it's the last command
                is_batch_full = (i + 1) % max_jobs == 0
                is_last_command = (i + 1) == len(commands)
                
                if is_batch_full or is_last_command:
                    f.write("# --- Waiting for batch to finish ---\n")
                    f.write(f"echo 'Waiting for {len(commands[i-max_jobs+1:i+1])} jobs in batch {i // max_jobs + 1} to finish...'\n")
                    
                    # Wait for all PIDs in the current batch
                    f.write("for pid in \"${pids[@]}\"; do\n")
                    f.write("    wait $pid\n")
                    # Check the exit code of the job
                    f.write("    if [ $? -ne 0 ]; then\n")
                    f.write("        echo \"WARNING: Job $pid failed with exit code $?\"\n")
                    f.write("        fail_count=$((fail_count + 1))\n")
                    f.write("    fi\n")
                    f.write("done\n")
                    
                    f.write("echo 'Batch finished.'\n")
                    f.write("pids=()\n\n") # Reset the PID list for the next batch
            
            # --- Final check ---
            f.write("echo 'All evaluations completed!'\n")
            f.write("if [ $fail_count -ne 0 ]; then\n")
            f.write(f"    echo \"WARNING: $fail_count jobs failed!\"\n")
            f.write("    exit 1\n")
            f.write("else\n")
            f.write("    echo 'All jobs succeeded.'\n")
            f.write("fi\n")

    print(f"\nSuccessfully generated {len(commands)} commands.")
    print(f"Script saved to: {output_script_name}")
    if max_jobs > 1:
        print(f"Mode: Parallel (Batches of {max_jobs})")
    else:
        print("Mode: Serial")
    print(f"Please run it using: bash {output_script_name}")

if __name__ == "__main__":
    main()