import os
import pandas as pd

def calculate_relative_scores_for_datasets(input_dirs):
    """
    Calculate relative scores for multiple datasets and ensure all algorithm combinations are present.

    Args:
        input_dirs (list): List of directories containing analysis files for each dataset.
    """
    error_files = []  # List to store files with errors

    # Define all possible algorithm combinations
    all_combinations = [
        f"{cleaning} + {clustering}"
        for cleaning in ["mode", "GroundTruth", "raha-baran"]
        for clustering in ["HC", "AffinityPropagation", "GMM", "KMeans", "DBSCAN", "OPTICS"]
    ]

    for input_dir in input_dirs:
        print(f"Processing dataset directory: {input_dir}")
        for file_name in os.listdir(input_dir):
            if file_name.endswith(".csv") and "analysis" in file_name:
                file_path = os.path.join(input_dir, file_name)

                # Extract dataset name and error rate
                dataset_name = os.path.basename(input_dir)
                error_rate = file_name.split("_")[-1].replace(".csv", "").replace("%", "")
                output_file = os.path.join(input_dir, f"{file_name.replace('.csv', '_relative.csv')}")

                try:
                    # Load the data
                    df = pd.read_csv(file_path)

                    if "Cleaning Algorithm" not in df.columns or "Clustering Method" not in df.columns or "Score" not in df.columns:
                        raise ValueError(f"'Cleaning Algorithm', 'Clustering Method', or 'Score' column missing in {file_name}")

                    # Add Algorithm Combination column
                    df["Algorithm Combination"] = df["Cleaning Algorithm"] + " + " + df["Clustering Method"]

                    # Find the maximum score for GroundTruth cleaning method
                    gt_max_score = df[df["Cleaning Algorithm"] == "GroundTruth"]["Score"].max()
                    if pd.isna(gt_max_score):
                        raise ValueError(f"No GroundTruth scores found in {file_name}")

                    # Calculate relative scores
                    df["Relative Score (%)"] = (df["Score"] / gt_max_score) * 100

                    # Ensure all combinations are present
                    missing_combinations = set(all_combinations) - set(df["Algorithm Combination"])
                    for combination in missing_combinations:
                        cleaning, clustering = combination.split(" + ")
                        df = pd.concat([
                            df,
                            pd.DataFrame({
                                "Cleaning Algorithm": [cleaning],
                                "Clustering Method": [clustering],
                                "Algorithm Combination": [combination],
                                "Score": [0],
                                "Relative Score (%)": [0]
                            })
                        ], ignore_index=True)

                    # Save the updated dataframe to a new file
                    df.to_csv(output_file, index=False)
                    print(f"Relative scores written to {output_file}")

                except Exception as e:
                    # Log the file with errors
                    print(f"Error processing {file_name}: {e}")
                    error_files.append(file_name)

    # Print error summary
    if error_files:
        print("\nThe following files had errors and were not processed:")
        for error_file in error_files:
            print(f"  - {error_file}")

# Example usage
input_directories = [
    r"D:\algorithm paper\ML algorithms codes\data_experiments\results\3_analyzed_data\analysis_results_1\beers",
    r"D:\algorithm paper\ML algorithms codes\data_experiments\results\3_analyzed_data\analysis_results_1\flights",
    r"D:\algorithm paper\ML algorithms codes\data_experiments\results\3_analyzed_data\analysis_results_1\hospital",
    r"D:\algorithm paper\ML algorithms codes\data_experiments\results\3_analyzed_data\analysis_results_1\rayyan"
]

calculate_relative_scores_for_datasets(input_directories)
