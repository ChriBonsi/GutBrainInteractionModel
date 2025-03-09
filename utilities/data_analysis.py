import os

import matplotlib.pyplot as plt
import pandas as pd


def process_conditions(base_folder, conditions, output_folder):
    """Processes multiple conditions, computes mean, std, and CV for each."""

    all_means = {}

    for condition in conditions:
        folder_path = os.path.join(base_folder, condition)
        dataframes = []

        # Read CSVs for the condition
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(folder_path, file))
                df = df.drop(columns=["alpha_protein_gut", "tau_protein_gut"], errors='ignore')
                dataframes.append(df)

        if not dataframes:
            print(f"No data found for {condition}")
            continue

        combined_df = pd.concat(dataframes, keys=range(len(dataframes)))
        mean_df = combined_df.groupby(level=1).mean()
        std_df = combined_df.groupby(level=1).std()

        mean_df["condition"] = condition  # Add condition label
        all_means[condition] = mean_df

        # Save results
        mean_df.to_csv(os.path.join(output_folder, f"{condition}_mean.csv"), index=False)
        std_df.to_csv(os.path.join(output_folder, f"{condition}_std.csv"), index=False)

    # Combine for comparison
    combined_means = pd.concat(all_means.values(), axis=0)
    return combined_means


def plot_comparison(combined_means, variable):
    """Plots Mean ± SD of a variable for all conditions on the same graph."""

    plt.figure(figsize=(10, 5))

    for condition in combined_means["condition"].unique():
        subset = combined_means[combined_means["condition"] == condition]
        plt.plot(subset["tick"], subset[variable], label=f"{condition.title()} diet - Mean")
        plt.fill_between(
            subset["tick"],
            subset[variable] - subset[variable].std(),
            subset[variable] + subset[variable].std(),
            alpha=0.2
        )

    plt.xlabel("Tick")
    plt.ylabel(variable)
    plt.title(f"Comparison of {variable} across diet conditions")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    base_folder = "../output"
    conditions = ["mixed", "healthy", "unhealthy"]
    output_folder = "../output/comparison"
    variable = "damaged_neuron"

    os.makedirs(output_folder, exist_ok=True)
    combined_means = process_conditions(base_folder, conditions, output_folder)

    plot_comparison(combined_means, variable)
