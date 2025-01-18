import os
import pandas as pd

def analyze_algorithm_combinations(input_directories, output_file, low_error_threshold=25):
    """
    Analyze algorithm combinations across different datasets and generate a summary table with selected metrics.

    Args:
        input_directories (list): List of directories containing raw score files.
        output_file (str): Path to save the output summary CSV file.
        low_error_threshold (int): Threshold for low error rates to filter datasets.
    """
    # Step 1: Initialize summary lists
    summary_all = []
    summary_low_error = []

    # Step 2: Process each dataset directory
    for dataset_dir in input_directories:
        dataset_name = os.path.basename(dataset_dir)

        for file_name in os.listdir(dataset_dir):
            if file_name.endswith("_relative.csv"):
                file_path = os.path.join(dataset_dir, file_name)

                # Load the relative score file
                data = pd.read_csv(file_path)

                # Add Error Rate (%) column if missing
                if "Error Rate (%)" not in data.columns:
                    try:
                        error_rate = float(file_name.split("_")[-2].replace("%", ""))
                        data["Error Rate (%)"] = error_rate
                    except ValueError:
                        raise ValueError(f"Unable to extract Error Rate (%) from file name: {file_name}")

                # Ensure required columns are present
                required_columns = [
                    "Cleaning Algorithm", "Clustering Method", "Relative Score (%)", "Score", "Error Rate (%)"
                ]
                for col in required_columns:
                    if col not in data.columns:
                        raise ValueError(f"Missing required column: {col} in {file_name}")

                # Add Dataset Name and Algorithm Combination columns
                data["Dataset Name"] = dataset_name
                data["Algorithm Combination"] = data["Cleaning Algorithm"] + " + " + data["Clustering Method"]

                # Append data to summary list
                summary_all.append(data)

                # Filter for low error rates
                low_error_data = data[data["Error Rate (%)"] <= low_error_threshold]
                if not low_error_data.empty:
                    summary_low_error.append(low_error_data)

    # Step 3: Combine all datasets
    combined_data_all = pd.concat(summary_all, ignore_index=True)
    combined_data_low_error = pd.concat(summary_low_error, ignore_index=True) if summary_low_error else pd.DataFrame()

    # Step 4: Define a function to calculate metrics
    def calculate_metrics(data):
        results = []
        for combination, group in data.groupby("Algorithm Combination"):
            avg_score = group["Relative Score (%)"].mean()
            std_dev_score = group["Relative Score (%)"].std()
            avg_absolute_score = group["Score"].mean()
            std_dev_absolute_score = group["Score"].std()
            min_deviation = abs(group["Relative Score (%)"] - 100).mean()

            best_row = group.loc[group["Relative Score (%)"].idxmax()]
            best_scenario = f"Dataset: {best_row['Dataset Name']}, Error Rate: {best_row['Error Rate (%)']}%"

            worst_row = group.loc[group["Relative Score (%)"].idxmin()]
            worst_scenario = f"Dataset: {worst_row['Dataset Name']}, Error Rate: {worst_row['Error Rate (%)']}%"

            highest_score = group["Score"].max()
            highest_relative_score = group["Relative Score (%)"].max()
            lowest_score = group["Score"].min()
            lowest_relative_score = group["Relative Score (%)"].min()

            results.append({
                "Algorithm Combination": combination,
                "Percentage Average Score": avg_score,
                "Percentage Score Standard Deviation": std_dev_score,
                "Average Absolute Score": avg_absolute_score,
                "Absolute Score Standard Deviation": std_dev_absolute_score,
                "Average Deviation from 100 (%)": min_deviation,
                "Best Scenario": best_scenario,
                "Worst Scenario": worst_scenario,
                "Highest Score": highest_score,
                "Highest Percentage Score": highest_relative_score,
                "Lowest Score": lowest_score,
                "Lowest Percentage Score": lowest_relative_score
            })
        return pd.DataFrame(results)

    # Step 5: Calculate metrics for all data and low error data
    results_all = calculate_metrics(combined_data_all)
    results_low_error = calculate_metrics(combined_data_low_error) if not combined_data_low_error.empty else pd.DataFrame()

    # Step 6: Save results to CSV files
    results_all.to_csv(output_file.replace(".csv", "_all.csv"), index=False)
    print(f"Summary for all data saved to {output_file.replace('.csv', '_all.csv')}")

    if not results_low_error.empty:
        results_low_error.to_csv(output_file.replace(".csv", "_low_error.csv"), index=False)
        print(f"Summary for low error data saved to {output_file.replace('.csv', '_low_error.csv')}")

# Example usage
input_directories = [
    r"D:\algorithm paper\ML algorithms codes\data_experiments\results\3_analyzed_data\analysis_results_1\beers",
    r"D:\algorithm paper\ML algorithms codes\data_experiments\results\3_analyzed_data\analysis_results_1\flights",
    r"D:\algorithm paper\ML algorithms codes\data_experiments\results\3_analyzed_data\analysis_results_1\hospital",
    r"D:\algorithm paper\ML algorithms codes\data_experiments\results\3_analyzed_data\analysis_results_1\rayyan"
]
output_file = r"D:\algorithm paper\ML algorithms codes\data_experiments\results\algorithm_combinations_summary.csv"
analyze_algorithm_combinations(input_directories, output_file)
