import os, sys

import pandas as pd
import matplotlib.pyplot as plt

def main(filepath):
    df = pd.read_csv(filepath)

    extracted_counts = df['extracted_label'].value_counts().sort_index()
    true_counts = df['true_label'].value_counts().sort_index()

    # Ensure both Series have the same index
    labels = sorted(set(extracted_counts.index).union(set(true_counts.index)))
    extracted_counts = extracted_counts.reindex(labels, fill_value=0)
    true_counts = true_counts.reindex(labels, fill_value=0)

    # Set positions for bars
    x = range(len(labels))
    width = 0.35

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar([i - width/2 for i in x], true_counts, width=width, label='True Labels')
    plt.bar([i + width/2 for i in x], extracted_counts, width=width, label='Extracted Labels')

    # Customizing
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title('Comparison of True vs Extracted Label Counts')
    plt.xticks(ticks=x, labels=labels, rotation=45)
    plt.legend()
    plt.tight_layout()

    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(filepath))[0]
    plot_path = os.path.join(plot_dir, f'{base_filename}_barplot.png')
    plt.savefig(plot_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_labels.py <csv_file>")
    else:
        main(sys.argv[1])