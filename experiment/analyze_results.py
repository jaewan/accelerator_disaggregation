#!/usr/bin/env python3
"""
Publication-Quality Analysis for Semantic Translation Gap Experiment
Suitable for SOSP/OSDI submission

Analyzes experimental results and generates academic-quality visualizations
demonstrating the semantic translation gap in GPU disaggregation systems.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import argparse

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_and_process_data(csv_file):
    """Load CSV data and compute statistics."""
    df = pd.read_csv(csv_file)
    
    # Convert bytes to GB for readability
    df['net_gb'] = df['net_bytes'] / (1024**3)
    
    # Compute mean and std for each mode/phase combination
    stats = df.groupby(['mode', 'phase']).agg({
        'latency_s': ['mean', 'std'],
        'net_gb': ['mean', 'std'],
        'avg_sm': ['mean', 'std']
    }).round(3)
    
    return df, stats

def print_semantic_gap_analysis(df):
    """Print quantitative analysis for paper."""
    print("="*80)
    print("SEMANTIC TRANSLATION GAP ANALYSIS")
    print("="*80)
    
    # Group by mode and phase
    grouped = df.groupby(['mode', 'phase'])['net_bytes'].mean()
    
    # Calculate semantic gap ratios
    naive_prefill = grouped[('naive', 'prefill')]
    sys_prefill = grouped[('sys_simulated', 'prefill')]
    naive_decode = grouped[('naive', 'decode')]
    sys_decode = grouped[('sys_simulated', 'decode')]
    
    prefill_ratio = naive_prefill / sys_prefill if sys_prefill > 0 else float('inf')
    decode_ratio = naive_decode / sys_decode if sys_decode > 0 else float('inf')
    
    print(f"\nNETWORK TRANSFER ANALYSIS:")
    print(f"Prefill Phase:")
    print(f"  NAÏVE-REMOTE:     {naive_prefill/1e9:.2f} GB")
    print(f"  SYS-SIMULATED:    {sys_prefill/1e9:.2f} GB")
    print(f"  Semantic Gap:     {prefill_ratio:.2f}x")
    
    print(f"\nDecode Phase:")
    print(f"  NAÏVE-REMOTE:     {naive_decode/1e9:.2f} GB")
    print(f"  SYS-SIMULATED:    {sys_decode/1e9:.2f} GB")
    print(f"  Semantic Gap:     {decode_ratio:.2f}x")
    
    # Performance analysis
    latency_grouped = df.groupby(['mode', 'phase'])['latency_s'].mean()
    
    print(f"\nPERFORMANCE ANALYSIS:")
    print(f"Decode Latency:")
    print(f"  LOCAL:            {latency_grouped[('local', 'decode')]:.2f}s")
    print(f"  NAÏVE-REMOTE:     {latency_grouped[('naive', 'decode')]:.2f}s")
    print(f"  SYS-SIMULATED:    {latency_grouped[('sys_simulated', 'decode')]:.2f}s")
    
    naive_overhead = (latency_grouped[('naive', 'decode')] / latency_grouped[('local', 'decode')] - 1) * 100
    sys_overhead = (latency_grouped[('sys_simulated', 'decode')] / latency_grouped[('local', 'decode')] - 1) * 100
    
    print(f"\nLatency Overhead vs Local:")
    print(f"  NAÏVE-REMOTE:     +{naive_overhead:.1f}%")
    print(f"  SYS-SIMULATED:    +{sys_overhead:.1f}%")
    
    print(f"\nKEY FINDINGS FOR PAPER:")
    print(f"• Semantic-aware disaggregation reduces network transfer by {decode_ratio:.1f}x")
    print(f"• Naive approach transfers {naive_decode/1e9:.1f} GB vs {sys_decode/1e9:.1f} GB for decode")
    print(f"• Performance penalty: {naive_overhead:.1f}% (naive) vs {sys_overhead:.1f}% (semantic-aware)")
    print("="*80)

def create_network_transfer_plot(df, output_dir):
    """Create publication-quality network transfer comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Group data
    grouped = df.groupby(['mode', 'phase'])['net_gb'].agg(['mean', 'std'])
    
    # Prefill comparison
    prefill_data = grouped.loc[(slice(None), 'prefill'), :]
    prefill_modes = ['naive', 'sys_simulated']
    prefill_means = [prefill_data.loc[(mode, 'prefill'), 'mean'] for mode in prefill_modes]
    prefill_stds = [prefill_data.loc[(mode, 'prefill'), 'std'] for mode in prefill_modes]
    
    bars1 = ax1.bar(['Naïve-Remote', 'Sys-Simulated'], prefill_means, 
                    yerr=prefill_stds, capsize=5, alpha=0.8,
                    color=['#d62728', '#2ca02c'])
    ax1.set_title('Prefill Phase', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Network Transfer (GB)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars1, prefill_means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{mean:.2f} GB', ha='center', va='bottom', fontweight='bold')
    
    # Decode comparison
    decode_data = grouped.loc[(slice(None), 'decode'), :]
    decode_modes = ['naive', 'sys_simulated']
    decode_means = [decode_data.loc[(mode, 'decode'), 'mean'] for mode in decode_modes]
    decode_stds = [decode_data.loc[(mode, 'decode'), 'std'] for mode in decode_modes]
    
    bars2 = ax2.bar(['Naïve-Remote', 'Sys-Simulated'], decode_means,
                    yerr=decode_stds, capsize=5, alpha=0.8,
                    color=['#d62728', '#2ca02c'])
    ax2.set_title('Decode Phase', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Network Transfer (GB)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels and semantic gap annotation
    for bar, mean in zip(bars2, decode_means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{mean:.2f} GB', ha='center', va='bottom', fontweight='bold')
    
    # Add semantic gap ratio annotation
    ratio = decode_means[0] / decode_means[1]
    ax2.annotate(f'{ratio:.1f}× Reduction', 
                xy=(0.5, max(decode_means) * 0.8), xycoords='axes fraction',
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'network_transfer_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'network_transfer_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved network transfer plot to {output_dir}")

def create_latency_breakdown_plot(df, output_dir):
    """Create latency breakdown visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by mode and compute means
    latency_data = df.groupby(['mode', 'phase'])['latency_s'].agg(['mean', 'std'])
    
    phases = ['prefill', 'decode']
    modes = ['local', 'naive', 'sys_simulated']
    mode_labels = ['Local\n(Baseline)', 'Naïve-Remote', 'Sys-Simulated']
    
    x = np.arange(len(phases))
    width = 0.25
    
    colors = ['#1f77b4', '#d62728', '#2ca02c']
    
    for i, mode in enumerate(modes):
        means = [latency_data.loc[(mode, phase), 'mean'] for phase in phases]
        stds = [latency_data.loc[(mode, phase), 'std'] for phase in phases]
        
        bars = ax.bar(x + i*width, means, width, yerr=stds, capsize=3,
                     label=mode_labels[i], alpha=0.8, color=colors[i])
        
        # Add value labels
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                   f'{mean:.1f}s', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Phase', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Latency Comparison Across Execution Modes', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Prefill', 'Decode'])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_breakdown.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'latency_breakdown.png', dpi=300, bbox_inches='tight')
    print(f"Saved latency breakdown plot to {output_dir}")

def create_combined_analysis_plot(df, output_dir):
    """Create comprehensive analysis plot for paper."""
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Network transfer comparison (spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    
    grouped = df.groupby(['mode', 'phase'])['net_gb'].agg(['mean', 'std'])
    
    phases = ['prefill', 'decode']
    modes = ['naive', 'sys_simulated']
    mode_labels = ['Naïve-Remote', 'Semantic-Aware']
    
    x = np.arange(len(phases))
    width = 0.35
    
    colors = ['#d62728', '#2ca02c']
    
    for i, mode in enumerate(modes):
        means = [grouped.loc[(mode, phase), 'mean'] for phase in phases]
        stds = [grouped.loc[(mode, phase), 'std'] for phase in phases]
        
        bars = ax1.bar(x + i*width, means, width, yerr=stds, capsize=3,
                      label=mode_labels[i], alpha=0.8, color=colors[i])
        
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{mean:.1f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_xlabel('Phase', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Network Transfer (GB)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Network Transfer Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x + width/2)
    ax1.set_xticklabels(['Prefill', 'Decode'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Semantic gap ratio
    ax2 = fig.add_subplot(gs[0, 2])
    
    naive_decode = grouped.loc[('naive', 'decode'), 'mean']
    sys_decode = grouped.loc[('sys_simulated', 'decode'), 'mean']
    ratio = naive_decode / sys_decode
    
    bar = ax2.bar(['Decode Phase'], [ratio], color='orange', alpha=0.8)
    ax2.text(0, ratio + 0.1, f'{ratio:.1f}×', ha='center', va='bottom', 
             fontsize=16, fontweight='bold')
    ax2.set_ylabel('Semantic Gap Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Data Transfer\nReduction', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Latency comparison
    ax3 = fig.add_subplot(gs[1, :])
    
    latency_data = df.groupby(['mode', 'phase'])['latency_s'].agg(['mean', 'std'])
    modes_full = ['local', 'naive', 'sys_simulated']
    mode_labels_full = ['Local (Baseline)', 'Naïve-Remote', 'Semantic-Aware']
    
    x = np.arange(len(phases))
    width = 0.25
    colors_full = ['#1f77b4', '#d62728', '#2ca02c']
    
    for i, mode in enumerate(modes_full):
        means = [latency_data.loc[(mode, phase), 'mean'] for phase in phases]
        stds = [latency_data.loc[(mode, phase), 'std'] for phase in phases]
        
        bars = ax3.bar(x + i*width, means, width, yerr=stds, capsize=3,
                      label=mode_labels_full[i], alpha=0.8, color=colors_full[i])
        
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{mean:.1f}s', ha='center', va='bottom', fontsize=9)
    
    ax3.set_xlabel('Phase', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
    ax3.set_title('(c) End-to-End Latency Comparison', fontsize=13, fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(['Prefill', 'Decode'])
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('Semantic Translation Gap in GPU Disaggregation\n(DialoGPT-Large Model)', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    plt.savefig(output_dir / 'comprehensive_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved comprehensive analysis plot to {output_dir}")

def generate_latex_table(df, output_dir):
    """Generate LaTeX table for paper."""
    grouped = df.groupby(['mode', 'phase']).agg({
        'latency_s': ['mean', 'std'],
        'net_gb': ['mean', 'std']
    }).round(3)
    
    latex_content = """
\\begin{table}[t]
\\centering
\\caption{Experimental Results: Semantic Translation Gap Analysis}
\\label{tab:results}
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Mode} & \\textbf{Phase} & \\textbf{Latency (s)} & \\textbf{Network (GB)} & \\textbf{Gap Ratio} \\\\
\\midrule
"""
    
    # Add data rows
    for phase in ['prefill', 'decode']:
        for mode in ['local', 'naive', 'sys_simulated']:
            if mode == 'local':
                mode_name = "Local"
                net_transfer = "0.00"
                gap_ratio = "—"
            elif mode == 'naive':
                mode_name = "Naïve-Remote"
                lat_mean = grouped.loc[(mode, phase), ('latency_s', 'mean')]
                lat_std = grouped.loc[(mode, phase), ('latency_s', 'std')]
                net_mean = grouped.loc[(mode, phase), ('net_gb', 'mean')]
                net_transfer = f"{net_mean:.2f}"
                gap_ratio = "—"
            else:  # sys_simulated
                mode_name = "Semantic-Aware"
                lat_mean = grouped.loc[(mode, phase), ('latency_s', 'mean')]
                lat_std = grouped.loc[(mode, phase), ('latency_s', 'std')]
                net_mean = grouped.loc[(mode, phase), ('net_gb', 'mean')]
                net_transfer = f"{net_mean:.2f}"
                
                # Calculate gap ratio for decode phase
                if phase == 'decode':
                    naive_net = grouped.loc[('naive', phase), ('net_gb', 'mean')]
                    gap_ratio = f"{naive_net/net_mean:.1f}×"
                else:
                    gap_ratio = "—"
            
            if mode != 'local':
                latency_str = f"{lat_mean:.2f} ± {lat_std:.2f}"
            else:
                lat_mean = grouped.loc[(mode, phase), ('latency_s', 'mean')]
                lat_std = grouped.loc[(mode, phase), ('latency_s', 'std')]
                latency_str = f"{lat_mean:.2f} ± {lat_std:.2f}"
            
            latex_content += f"{mode_name} & {phase.capitalize()} & {latency_str} & {net_transfer} & {gap_ratio} \\\\\n"
        
        if phase == 'prefill':
            latex_content += "\\midrule\n"
    
    latex_content += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    with open(output_dir / 'results_table.tex', 'w') as f:
        f.write(latex_content)
    
    print(f"Saved LaTeX table to {output_dir}/results_table.tex")

def main():
    parser = argparse.ArgumentParser(description='Analyze semantic gap experiment results')
    parser.add_argument('csv_file', help='Path to results CSV file')
    parser.add_argument('--output-dir', default='figures', help='Output directory for plots')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load and process data
    df, stats = load_and_process_data(args.csv_file)
    
    # Print analysis
    print_semantic_gap_analysis(df)
    
    # Generate plots
    create_network_transfer_plot(df, output_dir)
    create_latency_breakdown_plot(df, output_dir)
    create_combined_analysis_plot(df, output_dir)
    
    # Generate LaTeX table
    generate_latex_table(df, output_dir)
    
    print(f"\nAll publication-quality figures saved to {output_dir}/")
    print("Files generated:")
    print("- comprehensive_analysis.pdf (main figure for paper)")
    print("- network_transfer_comparison.pdf")
    print("- latency_breakdown.pdf") 
    print("- results_table.tex (LaTeX table)")

if __name__ == '__main__':
    main() 