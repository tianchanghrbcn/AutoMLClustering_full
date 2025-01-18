import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_single_dataset(file_path, output_path="results.csv"):
    # Step 1: Handle file path with quotes
    file_path = file_path.strip('"')  # Remove surrounding quotes if present

    # Step 2: Extract error rate and dataset name from file path
    file_name = os.path.basename(file_path)
    dataset_name = os.path.basename(os.path.dirname(file_path))  # Extract dataset folder name
    error_rate = float(file_name.split('%')[0]) if '%' in file_name else 0.0  # Default to 0 if no error rate

    # Step 3: Read dataset
    df = pd.read_csv(file_path)

    # Step 4: Frequency encoding for string columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        freq_map = df[col].value_counts(normalize=True).to_dict()
        df[col] = df[col].map(freq_map)

    # Step 5: Analyze the dataset
    print(f"Analyzing dataset: {dataset_name}, file: {file_name} with error rate: {error_rate}%")

    # Dataset dimensions
    num_samples = df.shape[0]
    num_features = df.shape[1]

    # Missing value ratio
    missing_ratio = df.isnull().sum().sum() / (num_samples * num_features)

    # Data types distribution
    data_types = df.dtypes.value_counts().to_dict()

    # Noise (outlier) detection using IQR (only for numeric columns)
    numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).sum().sum()
    noise_ratio = outliers / (numeric_df.shape[0] * numeric_df.shape[1]) if numeric_df.shape[1] > 0 else 0

    # Collect summary for this file
    summary = {
        "Dataset Name": dataset_name,
        "Error Rate (%)": error_rate,
        "Number of Samples": num_samples,
        "Number of Features": num_features,
        "Missing Value Ratio": missing_ratio,
        "Data Types Distribution": data_types,
        "Noise (Outlier) Ratio": noise_ratio
    }

    # Step 6: Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Step 7: Write results to file
    results_df = pd.DataFrame([summary])

    if not os.path.exists(output_path):
        results_df.to_csv(output_path, index=False)
    else:
        existing_df = pd.read_csv(output_path)
        updated_df = pd.concat([existing_df, results_df], ignore_index=True)
        updated_df.to_csv(output_path, index=False)

    print(f"Results written to {output_path}")

# Example usage:
file_path = input("Enter the absolute path of the dataset: ").strip()  # User inputs the file path
output_path = r"D:\algorithm paper\ML algorithms codes\data_experiments\results\3_analyzed_data\Data characterization\characterization_results.csv"  # Path to save results
analyze_single_dataset(file_path, output_path)
