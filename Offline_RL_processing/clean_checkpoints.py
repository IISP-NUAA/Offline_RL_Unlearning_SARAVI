import os
import re
import argparse
import sys


CHECKPOINT_PATTERNS = [
    ("model", re.compile(r"^model_(\d+)\.pt$")),
    ("ckpt", re.compile(r"^ckpt_(\d+)\.pt$")),
]


def get_checkpoint_info(files):
    """
    Parses the file list to find checkpoint files matching known patterns.
    Returns:
        checkpoint_groups (list): A list of (prefix, best_file, files_to_delete)
                                  tuples, one for each matched checkpoint family.
    """
    checkpoint_groups = []

    for prefix, pattern in CHECKPOINT_PATTERNS:
        valid_checkpoints = []

        for f in files:
            match = pattern.match(f)
            if match:
                step_count = int(match.group(1))
                valid_checkpoints.append((step_count, f))

        if not valid_checkpoints:
            continue

        # Sort by step count (index 0) in descending order.
        valid_checkpoints.sort(key=lambda x: x[0], reverse=True)
        best_checkpoint = valid_checkpoints[0][1]
        files_to_delete = [item[1] for item in valid_checkpoints[1:]]
        checkpoint_groups.append((prefix, best_checkpoint, files_to_delete))

    return checkpoint_groups


def main():
    parser = argparse.ArgumentParser(description="Recursively delete old checkpoint files, keeping only the largest-step file for each checkpoint pattern.")
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory to search (e.g., Unlearned/pointmaze/...)")
    parser.add_argument("--dry-run", action="store_true",
                        help="If set, only print files to be deleted without actually deleting them (Recommended for safety).")

    args = parser.parse_args()
    search_root = args.root

    if not os.path.isdir(search_root):
        print(f"Error: Directory '{search_root}' does not exist.")
        sys.exit(1)

    print(f"Searching for checkpoints in: {search_root} ...")
    if args.dry_run:
        print("--- DRY RUN MODE: No files will be deleted ---\n")
    else:
        print("--- WARNING: Files will be PERMANENTLY deleted ---\n")

    deleted_count = 0
    kept_count = 0

    for dirpath, dirnames, filenames in os.walk(search_root):
        checkpoint_groups = get_checkpoint_info(filenames)

        for prefix, best_checkpoint, to_delete in checkpoint_groups:
            kept_count += 1
            if to_delete:
                print(f"Directory: {dirpath}")
                print(f"  Pattern: {prefix}_*.pt")
                print(f"  [KEEP] {best_checkpoint}")

                for del_file in to_delete:
                    full_path = os.path.join(dirpath, del_file)
                    if args.dry_run:
                        print(f"  [WOULD DELETE] {del_file}")
                    else:
                        try:
                            os.remove(full_path)
                            print(f"  [DELETED] {del_file}")
                            deleted_count += 1
                        except OSError as e:
                            print(f"  [ERROR] Could not delete {del_file}: {e}")
                print("-" * 40)

    print("\nSummary:")
    print(f"Checkpoint groups processed: {kept_count}")
    if args.dry_run:
        print("Dry run completed. Remove --dry-run to actually delete files.")
    else:
        print(f"Total files deleted: {deleted_count}")


if __name__ == "__main__":
    main()
