import os
import pandas as pd

def add_best_combinations_with_scores_and_reference(main_table_path, analysis_directories, output_path):
    """
    Add the best combination, best deviation combination, and reference algorithm to the main table.

    Args:
        main_table_path (str): Path to the main table (CSV file).
        analysis_directories (list): List of directories containing analysis files.
        output_path (str): Path to save the updated main table.
    """
    # Load the main table
    main_table = pd.read_csv(main_table_path)

    # Prepare new columns
    best_combinations = []
    best_deviation_combinations = []
    reference_algorithms = []

    # Process each dataset directory
    for dataset_dir in analysis_directories:
        dataset_name = os.path.basename(dataset_dir)
        reference_algorithm = None

        for file_name in os.listdir(dataset_dir):
            if file_name.endswith("_relative.csv"):
                file_path = os.path.join(dataset_dir, file_name)

                # Correctly extract the error rate from the file name
                try:
                    error_rate = float(file_name.split("_")[-2].replace("%", ""))
                except ValueError:
                    print(f"Skipping file due to invalid format: {file_name}")
                    continue

                df = pd.read_csv(file_path)

                if "Score" not in df.columns or "Cleaning Algorithm" not in df.columns or "Clustering Method" not in df.columns:
                    print(f"Skipping {file_name}: Required columns missing.")
                    continue

                # Identify the reference algorithm (GT algorithm with Relative Score = 100)
                gt_row = df[(df["Cleaning Algorithm"] == "GroundTruth") & (df["Relative Score (%)"] == 100)]
                if not gt_row.empty and reference_algorithm is None:
                    reference_algorithm = f"{gt_row.iloc[0]['Cleaning Algorithm']} + {gt_row.iloc[0]['Clustering Method']}"

                # Exclude GT (GroundTruth) from consideration
                non_gt_df = df[df["Cleaning Algorithm"] != "GroundTruth"].copy()  # Explicit copy to avoid warnings

                # Find the globally best combination (non-GT only)
                if not non_gt_df.empty:
                    best_row = non_gt_df.loc[non_gt_df["Score"].idxmax()]
                    best_combination = f"{best_row['Cleaning Algorithm']} + {best_row['Clustering Method']}"
                    best_combination_score = best_row["Relative Score (%)"]
                else:
                    best_combination = "N/A"
                    best_combination_score = None

                # Find the best deviation combination (non-GT only)
                if not non_gt_df.empty:
                    non_gt_df["Deviation"] = abs(non_gt_df["Relative Score (%)"] - 100)
                    best_deviation_row = non_gt_df.loc[non_gt_df["Deviation"].idxmin()]
                    best_deviation_combination = f"{best_deviation_row['Cleaning Algorithm']} + {best_deviation_row['Clustering Method']}"
                    best_deviation_combination_score = best_deviation_row["Relative Score (%)"]
                else:
                    best_deviation_combination = "N/A"
                    best_deviation_combination_score = None

                # Add to lists
                best_combinations.append({
                    "Dataset Name": dataset_name,
                    "Error Rate (%)": error_rate,
                    "Best Combination": best_combination,
                    "Best Combination Score (%)": best_combination_score
                })
                best_deviation_combinations.append({
                    "Dataset Name": dataset_name,
                    "Error Rate (%)": error_rate,
                    "Best Deviation Combination": best_deviation_combination,
                    "Best Deviation Combination Score (%)": best_deviation_combination_score
                })

        # Record reference algorithm for the dataset
        reference_algorithms.append({
            "Dataset Name": dataset_name,
            "Reference Algorithm": reference_algorithm
        })

    # Merge results into the main table
    best_combinations_df = pd.DataFrame(best_combinations)
    best_deviation_combinations_df = pd.DataFrame(best_deviation_combinations)
    reference_algorithms_df = pd.DataFrame(reference_algorithms)

    # Merge with the main table
    main_table = main_table.merge(best_combinations_df, on=["Dataset Name", "Error Rate (%)"], how="left")
    main_table = main_table.merge(best_deviation_combinations_df, on=["Dataset Name", "Error Rate (%)"], how="left")
    main_table = main_table.merge(reference_algorithms_df, on=["Dataset Name"], how="left")

    # Ensure output file is writable
    try:
        if os.path.exists(output_path):
            os.remove(output_path)  # Remove the existing file if necessary
        main_table.to_csv(output_path, index=False)
        print(f"Updated main table saved to {output_path}")
    except PermissionError:
        print(f"PermissionError: The file {output_path} is being used by another process. Please close it and try again.")

# Example usage
main_table_path = r"D:\algorithm paper\ML algorithms codes\data_experiments\results\3_analyzed_data\data characterization\characterization_results.csv"
analysis_directories = [
    r"D:\algorithm paper\ML algorithms codes\data_experiments\results\3_analyzed_data\analysis_results_1\beers",
    r"D:\algorithm paper\ML algorithms codes\data_experiments\results\3_analyzed_data\analysis_results_1\flights",
    r"D:\algorithm paper\ML algorithms codes\data_experiments\results\3_analyzed_data\analysis_results_1\hospital",
    r"D:\algorithm paper\ML algorithms codes\data_experiments\results\3_analyzed_data\analysis_results_1\rayyan"
]
output_path = r"D:\algorithm paper\ML algorithms codes\data_experiments\results\3_analyzed_data\data characterization\updated_characterization_results.csv"

add_best_combinations_with_scores_and_reference(main_table_path, analysis_directories, output_path)
