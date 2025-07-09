import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def plot_results(csv_path: str, output_path: str):
    """
    Reads scaling experiment results from a CSV and generates a line plot
    of average latency per decode step versus the number of decode steps.

    Requires pandas, matplotlib, and seaborn.
    Install with: pip install pandas matplotlib seaborn
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.", file=sys.stderr)
        sys.exit(1)

    # Calculate average latency per step
    df['avg_latency_per_step_s'] = df['latency_s'] / df['decode_steps']

    # Create the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(
        data=df,
        x='decode_steps',
        y='avg_latency_per_step_s',
        hue='mode',
        marker='o',
        ax=ax,
        linewidth=2.5
    )

    # Formatting
    ax.set_title('Latency per Decode Step vs. Generation Length', fontsize=16, fontweight='bold')
    ax.set_xlabel('Number of Decode Steps (Generation Length)', fontsize=12)
    ax.set_ylabel('Average Latency per Step (seconds)', fontsize=12)
    ax.legend(title='Execution Mode', title_fontsize='13', fontsize='11')
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to '{output_path}'")

def main():
    if len(sys.argv) != 3:
        print("Usage: python plot_scaling_results.py <input_csv_file> <output_image_file>", file=sys.stderr)
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_image = sys.argv[2]
    plot_results(input_csv, output_image)

if __name__ == "__main__":
    main() 