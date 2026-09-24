#  analyze_KL_results.py (MODIFIED)
#  Parses KL/Wasserstein JSONs, extracts Shuffle Mode, and generates Final Summary with Std and Count.

import os
import json
import pandas as pd
import argparse
from typing import List, Dict, Any

# Ensure this import works in your environment, or copy the function here if needed
try:
    from analyze_evaluation_results import (
        add_grouped_bootstrap_confidence_intervals,
        parse_model_dir, parse_step_from_model_dir, sort_with_step,
    )
except ImportError:
    # Fallback if the other file isn't in path, though usually it should be present as per context
    def parse_model_dir(model_dir):
        return "Unknown", "N/A", "N/A", "N/A"

    def parse_step_from_model_dir(model_dir):
        return "N/A"

    def sort_with_step(df, columns):
        valid_columns = [column for column in columns if column in df.columns]
        if not valid_columns:
            return df.reset_index(drop=True)
        return df.sort_values(
            by=valid_columns,
            key=lambda col: pd.to_numeric(col, errors="coerce") if col.name == "Step" else col,
        ).reset_index(drop=True)

    def add_grouped_bootstrap_confidence_intervals(
        results_df, summary_df, group_cols, metric_prefixes,
        summary_mean_columns=None, output_prefixes=None,
    ):
        return summary_df


def get_shuffle_mode(model_dir: str) -> str:
    """Determine shuffle mode from directory string."""
    if "no_shuffle_" in model_dir:
        return "No Shuffle"
    return "Shuffle"

def parse_kl_record(data: Dict[str, Any], file_path: str) -> Dict[str, Any]:
    """
     Parses a single JSON data object.
    """
    row = {}

    # 1.  Extract Metadata
    model_2_path = data.get("model_2_path", "")
    method_name, ratio, algo_name, _ = parse_model_dir(model_2_path)
    row["Method Name"] = method_name
    row["Algo. Name"] = algo_name
    row["Shuffle Mode"] = get_shuffle_mode(model_2_path) 
    step = parse_step_from_model_dir(model_2_path)
    if step == "N/A":
        step = parse_step_from_model_dir(file_path)
    row["Step"] = step

    try:
        components = data.get("dataset_components", [])
        row["Dataset Components"] = "_".join(components)
    except TypeError:
        row["Dataset Components"] = "N/A"

    try:
        ratios = data.get("retained_ratios", [])
        row["Ratios"] = "_".join(map(str, ratios))
    except TypeError:
        row["Ratios"] = "N/A"

    # 2.  Identify Metric Type and Extract Stats
    stats = {}
    metric_type = "Unknown"
    
    if "kl_statistics" in data:
        stats = data.get("kl_statistics", {})
        metric_type = "KL"
    elif "Wasserstein_statistics" in data:
        stats = data.get("Wasserstein_statistics", {})
        metric_type = "Wasserstein"
    elif "js_statistics" in data:
        stats = data.get("js_statistics", {})
        metric_type = "JS"
    elif 'critic_diff_statistics' in data:
        stats = data.get("critic_diff_statistics", {})
        metric_type = "critic_diff"
    else:
        print(f"Warning: No statistics block found in {file_path}.")

    row["KL_or_Wasserstein"] = metric_type
        
    # 3.  Flatten Statistics
    if not stats:
        print(f"Warning: Statistics block was empty in {file_path}.")
        
    for key, value in stats.items():
        row[key] = value 

    return row

def process_kl_files(root_dir: str) -> pd.DataFrame:
    all_records = []
    
    print(f"Starting search from: {root_dir}\n")
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith("seed") and filename.endswith(".json"):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    record = parse_kl_record(data, file_path)
                    seed_str = filename.replace("seed", "").replace(".json", "")
                    record["seed"] = int(seed_str) if seed_str.isdigit() else seed_str
                    
                    all_records.append(record)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    if not all_records:
        return pd.DataFrame()

    return pd.DataFrame(all_records)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, default="stats_results")
    parser.add_argument("--output", type=str, default="summary_stats.csv")
    args = parser.parse_args()

    output_path = "summary"
    os.makedirs(output_path, exist_ok=True)
    
    if not os.path.isdir(args.root_dir):
        print(f"Error: The directory '{args.root_dir}' does not exist.")
        return

    results_df = process_kl_files(args.root_dir)
    
    if not results_df.empty:
        #  --- Column Re-ordering for D_f and D_r ---
        
        #  Base metadata columns
        known_cols = [
            "Method Name", 
            "Algo. Name", 
            "Dataset Components", 
            "Ratios", 
            "KL_or_Wasserstein", 
            "Shuffle Mode", 
            "Step",
            "seed"
        ]
        
        #  Define the desired order for statistics
        prefixes = ["D_f_", "D_r_"]
        base_stats = ["mean", "median", "p90", "p99", "max", "count"]
        
        ordered_stat_cols = []
        for prefix in prefixes:
            for stat in base_stats:
                ordered_stat_cols.append(f"{prefix}{stat}")
        
        #  Filter existing columns against our desired order
        existing_stat_cols = [col for col in ordered_stat_cols if col in results_df.columns]
        
        #  Catch any leftovers 
        all_cols_set = set(results_df.columns)
        known_and_ordered = set(known_cols + existing_stat_cols)
        other_cols = sorted(list(all_cols_set - known_and_ordered))
        
        #  Final composition
        final_columns = [col for col in known_cols if col in results_df.columns]
        final_columns += existing_stat_cols + other_cols
        
        results_df = results_df.reindex(columns=final_columns)
        
        print("\n--- Statistics Summary (Detailed) ---")
        # print(results_df.head().to_string()) 
        
        try:
            out_file = os.path.join(output_path, args.root_dir.replace("/", "_") + "_" + args.output)
            results_df.to_csv(out_file, index=False, encoding='utf-8')
            print(f"Detailed results successfully saved to {out_file}")
        except Exception as e:
            print(f"Error saving CSV: {e}")

        #  Identify numeric columns (statistics) to aggregate
        meta_col_names = set(known_cols)
        numeric_cols = [c for c in results_df.columns if c not in meta_col_names and pd.api.types.is_numeric_dtype(results_df[c])]

        if numeric_cols:
            base, ext = os.path.splitext(args.output)

            # =================================================================
            # NEW: Generate Ratio Summary (Aggregated over Seed, split by Ratio)
            # =================================================================
            print(f"\n--- Ratio Summary (Aggregated over Seed, Split by Step) ---")
            
            # Group by including 'Ratios'
            ratio_group_cols = ['Algo. Name', 'Method Name', 'Shuffle Mode', 'KL_or_Wasserstein', 'Step', 'Ratios']
            valid_ratio_group_cols = [c for c in ratio_group_cols if c in results_df.columns]
            
            # 1. Aggregate Mean and Std for numeric metrics
            ratio_agg_df = results_df.groupby(valid_ratio_group_cols)[numeric_cols].agg(['mean', 'std']).reset_index()
            
            # 2. Compute Sample Count (Group Size) separately
            ratio_count_df = results_df.groupby(valid_ratio_group_cols).size().reset_index(name='Sample Count')
            
            # 3. Flatten MultiIndex columns
            new_cols = []
            for col in ratio_agg_df.columns:
                if isinstance(col, tuple):
                    if col[1] == '':
                        new_cols.append(col[0]) # Group columns
                    elif col[1] == 'mean':
                        new_cols.append(f"{col[0]}_Avg") # Average
                    elif col[1] == 'std':
                        new_cols.append(f"{col[0]}_Std") # Standard Deviation
                else:
                    new_cols.append(col)
            ratio_agg_df.columns = new_cols
            
            # 4. Merge Stats with Count
            ratio_summary_df = pd.merge(ratio_agg_df, ratio_count_df, on=valid_ratio_group_cols, how='left')
            ratio_summary_df = add_grouped_bootstrap_confidence_intervals(
                results_df, ratio_summary_df, valid_ratio_group_cols, ("D_f_", "D_r_", "Gap_")
            )
            
            # 5. Reorder columns to put 'Sample Count' after metadata
            final_ratio_cols = list(ratio_summary_df.columns)
            if 'Sample Count' in final_ratio_cols:
                final_ratio_cols.remove('Sample Count')
                insert_idx = len(valid_ratio_group_cols)
                final_ratio_cols.insert(insert_idx, 'Sample Count')
            ratio_summary_df = ratio_summary_df[final_ratio_cols]
            
            # Sort
            sort_cols = [c for c in ["Algo. Name", "Shuffle Mode", "Method Name", "Step", "Ratios"] if c in ratio_summary_df.columns]
            ratio_summary_df = sort_with_step(ratio_summary_df, sort_cols)
            
            # print(ratio_summary_df.to_string())
            
            try:
                ratio_output_file = f"{base}_ratio_summary{ext}"
                full_ratio_path = os.path.join(output_path, args.root_dir.replace("/", "_") + "_" + ratio_output_file)
                ratio_summary_df.to_csv(full_ratio_path, index=False, encoding='utf-8')
                print(f"\n--- Ratio summary saved to {full_ratio_path} ---")
            except IOError as e:
                print(f"\n--- Error saving ratio summary CSV file: {e} ---")


            # =================================================================
            # 6. Generate Final Summary (Aggregated over Ratio & Seed) [MODIFIED]
            # =================================================================
            print(f"\n--- Final Summary (Aggregated over Ratio & Seed, Split by Step) ---")
            
            # Group by: Algo, Method, Shuffle, MetricType
            group_cols = ['Algo. Name', 'Method Name', 'Shuffle Mode', 'KL_or_Wasserstein', 'Step']
            # Ensure group columns exist
            valid_group_cols = [c for c in group_cols if c in results_df.columns]
            
            # 1. Aggregate Mean and Std for numeric metrics
            agg_stats_df = results_df.groupby(valid_group_cols)[numeric_cols].agg(['mean', 'std']).reset_index()
            
            # 2. Compute Sample Count (Group Size) separately
            count_df = results_df.groupby(valid_group_cols).size().reset_index(name='Sample Count')
            
            # 3. Flatten MultiIndex columns
            new_cols = []
            for col in agg_stats_df.columns:
                if isinstance(col, tuple):
                    if col[1] == '':
                        new_cols.append(col[0]) # Group columns
                    elif col[1] == 'mean':
                        new_cols.append(f"{col[0]}_Avg") # Average
                    elif col[1] == 'std':
                        new_cols.append(f"{col[0]}_Std") # Standard Deviation
                else:
                    new_cols.append(col)
            agg_stats_df.columns = new_cols
            
            # 4. Merge Stats with Count
            final_summary_df = pd.merge(agg_stats_df, count_df, on=valid_group_cols, how='left')
            final_summary_df = add_grouped_bootstrap_confidence_intervals(
                results_df, final_summary_df, valid_group_cols, ("D_f_", "D_r_", "Gap_")
            )
            
            # 5. Reorder columns to put 'Sample Count' after metadata
            final_cols = list(final_summary_df.columns)
            # Remove Sample Count from end and insert after valid_group_cols
            if 'Sample Count' in final_cols:
                final_cols.remove('Sample Count')
                insert_idx = len(valid_group_cols)
                final_cols.insert(insert_idx, 'Sample Count')
            final_summary_df = final_summary_df[final_cols]

            # Sort for readability
            sort_cols = [c for c in ["Algo. Name", "Shuffle Mode", "Method Name", "Step"] if c in final_summary_df.columns]
            final_summary_df = sort_with_step(final_summary_df, sort_cols)
            
            # print(final_summary_df.to_string())
            
            try:
                final_output_file = f"{base}_final_summary{ext}"
                full_final_path = os.path.join(output_path, args.root_dir.replace("/", "_") + "_" + final_output_file)
                final_summary_df.to_csv(full_final_path, index=False, encoding='utf-8')
                print(f"\n--- Final summary saved to {full_final_path} ---")
            except IOError as e:
                print(f"\n--- Error saving final summary CSV file: {e} ---")
        else:
            print("No numeric columns found to aggregate.")

    else:
        print("No data found.")

if __name__ == "__main__":
    main()