import os
import pandas as pd


def process_csv_files(folder_path, output_file):
    # List of DataFrames
    dataframes = []

    # Scan folder for CSV files
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)

            # Rimuove le colonne specificate se esistono
            df = df.drop(columns=["alpha_protein_gut", "tau_protein_gut"], errors='ignore')

            dataframes.append(df)

    # Verify if there are any CSV files
    if not dataframes:
        print("No CSV files found in the folder.")
        return

    # Combine all DataFrames and compute the mean
    combined_df = pd.concat(dataframes).groupby(level=0).mean()

    # Save to CSV
    combined_df.to_csv(output_file, index=False)
    print(f"File saved: {output_file}")

folder = "../output/baseline"
output = "../output/baseline/avg_result.csv"
process_csv_files(folder, output)
