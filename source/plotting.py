import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

from preprocessing import extract_data, calculate_distributions


def setup_plot_style():
    """Configures clean scientific publication styling."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'lines.linewidth': 1.8,
        'lines.markersize': 5,
        'grid.alpha': 0.35,
        'grid.linestyle': '--'
    })


def evaluate_smooth_pchip(s_um: np.ndarray, cum_vals: np.ndarray, num_pts: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolates a smooth monotonic curve using PCHIP on log10(diameter)
    to match the continuous curves in the Excel chart sheets.
    """
    # Sieve sizes are strictly positive (10 um to 6000 um)
    q_rev = cum_vals[::-1]
    d_rev = s_um[::-1]
    
    # Remove any potential duplicates in cumulative passing
    q_unique, u_idx = np.unique(q_rev, return_index=True)
    d_unique = d_rev[u_idx]
    
    # Interpolate in log-space for particle diameter
    log_d = np.log10(d_unique)
    pchip = PchipInterpolator(log_d, q_unique)
    
    smooth_d = np.geomspace(d_unique.min(), d_unique.max(), num_pts)
    smooth_q = np.clip(pchip(np.log10(smooth_d)), 0.0, 100.0)
    
    return smooth_d, smooth_q


def plot_psd_curve(ax, s_um: np.ndarray, cum_vals: np.ndarray, label: str, 
                   color: str = None, linestyle: str = '-', marker: str = 'o'):
    """Plots both discrete experimental sieve markers and smooth PCHIP fitted curve."""
    smooth_d, smooth_q = evaluate_smooth_pchip(s_um, cum_vals)
    line = ax.plot(smooth_d, smooth_q, linestyle=linestyle, color=color, label=label)
    actual_color = line[0].get_color()
    # Overlay actual sieve data points
    ax.plot(s_um, cum_vals, linestyle='None', marker=marker, color=actual_color, alpha=0.85)


def finalize_axis(ax, title: str):
    """Sets standard logarithmic x-axis, bounds, labels, and grid."""
    ax.set_xscale('log')
    ax.set_xlim(10, 6000)
    ax.set_ylim(-2, 103)
    ax.set_xlabel('Particle Size, $d$ ($\mu\text{m}$)')
    ax.set_ylabel('Cumulative Passing, $Q$ (%)')
    ax.set_title(title, pad=12, fontweight='bold')
    ax.grid(True, which='both')
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='gray')


# ==============================================================================
# 1. Macro Comparison Charts: Premilled & Milled
# ==============================================================================

def generate_macro_charts(pre_samples: list[dict], milled_samples: list[dict], save_dir: str):
    """Generates 'Premilled Chart' and 'MILLED CHART'."""
    
    # A. Premilled Chart
    fig, ax = plt.subplots(figsize=(9, 6))
    for s in pre_samples:
        if s['Category'] == 'Pure Samples':
            plot_psd_curve(ax, s['Sieves_um'], s['Cum_Passing'], label=s['Sample'])
    finalize_axis(ax, 'Pre-milled Particle Size Distributions')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'premilled_all_fuels.png'), dpi=300)
    plt.close(fig)

    # B. Milled Chart (Pure fuels only)
    fig, ax = plt.subplots(figsize=(9, 6))
    for s in milled_samples:
        if s['Category'] == 'Pure Samples':
            # Skip separate PKS components on the main chart, show combined
            if s['Sample'] in ['PKS Milled', 'PKS Overflow']:
                continue
            plot_psd_curve(ax, s['Sieves_um'], s['Cum_Passing'], label=s['Sample'])
    finalize_axis(ax, 'Milled Particle Size Distributions')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'milled_all_fuels.png'), dpi=300)
    plt.close(fig)


# ==============================================================================
# 2. Pure Biomass Processing Charts: PKS and Pellet
# ==============================================================================

def generate_biomass_charts(pre_samples: list[dict], milled_samples: list[dict], save_dir: str):
    """Generates 'PKS Chart' and 'PKS Chart (2)' (Pellet)."""
    
    # A. PKS Processing Chart (Pre-milled, Milled, Overflow, Milled + Overflow)
    pks_pre = next(s for s in pre_samples if s['Sample'] == 'PKS')
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    plot_psd_curve(ax, pks_pre['Sieves_um'], pks_pre['Cum_Passing'], label='Pre-milled', color='#1f77b4', marker='s')
    
    pks_m_states = {
        'PKS Milled': ('#ff7f0e', '--', '^'),
        'PKS Overflow': ('#2ca02c', ':', 'v'),
        'PKS Milled + Overflow': ('#d62728', '-', 'o')
    }
    for s in milled_samples:
        if s['Sample'] in pks_m_states:
            c, ls, mk = pks_m_states[s['Sample']]
            plot_psd_curve(ax, s['Sieves_um'], s['Cum_Passing'], label=s['Sample'], color=c, linestyle=ls, marker=mk)
            
    finalize_axis(ax, 'Palm Kernel Shell (PKS) Milling Breakdown')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'pks_processing_breakdown.png'), dpi=300)
    plt.close(fig)

    # B. Pellet Processing Chart (Pre-milled vs Milled)
    pellet_pre = next(s for s in pre_samples if s['Sample'] == 'Pellet')
    pellet_mil = next(s for s in milled_samples if s['Sample'] == 'Pellet milled')
    
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    plot_psd_curve(ax, pellet_pre['Sieves_um'], pellet_pre['Cum_Passing'], label='Pre-milled', color='#1f77b4', marker='s')
    plot_psd_curve(ax, pellet_mil['Sieves_um'], pellet_mil['Cum_Passing'], label='Milled', color='#d62728', linestyle='--', marker='o')
    finalize_axis(ax, 'Wood Pellet Pre-milled vs Milled Distribution')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'pellet_processing_breakdown.png'), dpi=300)
    plt.close(fig)


# ==============================================================================
# 3. Coal Blend Comparison Charts (La Loma, Indo, 52, 55, 61, 75, 95)
# ==============================================================================

def generate_blend_comparison_charts(pre_samples: list[dict], milled_samples: list[dict], save_dir: str):
    """
    Generates the 7 fuel-specific comparison charts matching Excel sheets:
    La Loma, Indo, 52, 55, 61, 75, 95 (each showing all 10 pre-milled and milled states).
    """
    coal_fuels = ['La Loma', 'Indo', '52', '55', '61', '75', '95']
    
    # Mapping table for categories and styling
    # Solid lines for pre-milled, dashed for milled
    blend_series_config = [
        # (State, Category, Suffix/Label, Color, Linestyle, Marker)
        ('pre', 'Pure Samples', 'Pure Pre-milled', '#000000', '-', 'o'),
        ('pre', '10% blends PKS', '+10% PKS Pre-milled', '#1f77b4', '-', 's'),
        ('pre', '40% blends PKS', '+40% PKS Pre-milled', '#aec7e8', '-', '^'),
        ('pre', '10% blends Pellet', '+10% Pellet Pre-milled', '#2ca02c', '-', 'd'),
        ('pre', '40% blends Pellet', '+40% Pellet Pre-milled', '#98df8a', '-', 'v'),
        ('milled', 'Pure Samples', 'Pure Milled', '#d62728', '--', 'o'),
        ('milled', '10% PKS', '+10% PKS Milled', '#ff7f0e', '--', 's'),
        ('milled', '40% PKS', '+40% PKS Milled', '#ffbb78', '--', '^'),
        ('milled', '10% Pellet', '+10% Pellet Milled', '#9467bd', '--', 'd'),
        ('milled', '40% Pellet', '+40% Pellet Milled', '#c5b0d5', '--', 'v'),
    ]

    for coal in coal_fuels:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        
        for state, category, label_txt, color, ls, mk in blend_series_config:
            sample_list = pre_samples if state == 'pre' else milled_samples
            
            # Find the matching entry for this coal and category
            match = None
            for s in sample_list:
                s_name = s['Sample'].replace(' milled', '').strip()
                # Handle 'Indonesian' vs 'Indo' nomenclature in the Excel sheet
                if (s_name == coal or (coal == 'Indo' and s_name == 'Indonesian')) and (s['Category'] == category):
                    match = s
                    break
                    
            if match:
                plot_psd_curve(ax, match['Sieves_um'], match['Cum_Passing'], 
                               label=label_txt, color=color, linestyle=ls, marker=mk)

        finalize_axis(ax, f'Fuel Comparison & Milling Analysis: {coal}')
        # Adjust legend to two columns for clean readability
        ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), ncol=2, fontsize=8)
        fig.tight_layout()
        clean_filename = coal.lower().replace(' ', '_')
        fig.savefig(os.path.join(save_dir, f'fuel_chart_{clean_filename}.png'), dpi=300)
        plt.close(fig)


# ==============================================================================
# Main Plotting Runner
# ==============================================================================

if __name__ == "__main__":
    setup_plot_style()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'data', '220125_Particle_Size_Distributions_EL.xlsx')
    plots_dir = os.path.join(current_dir, '..', 'output', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Ingesting and calculating distributions for plotting...")
    # 1. Pre-milled
    df_pre_raw = extract_data(file_path, sheet_name="Pre-milled")
    _, pre_samples = calculate_distributions(df_pre_raw)
    
    # 2. Milled
    df_mil_raw = extract_data(file_path, sheet_name="Milled")
    _, milled_samples = calculate_distributions(df_mil_raw)
    
    print("\nGenerating 11 charts:")
    print(" -> 1. Generating Macro comparison charts (Pre-milled & Milled)...")
    generate_macro_charts(pre_samples, milled_samples, plots_dir)
    
    print(" -> 2. Generating Biomass processing charts (PKS & Pellet)...")
    generate_biomass_charts(pre_samples, milled_samples, plots_dir)
    
    print(" -> 3. Generating 7 Coal blend comparison charts (La Loma, Indo, 52, 55, 61, 75, 95)...")
    generate_blend_comparison_charts(pre_samples, milled_samples, plots_dir)
    
    print(f"\n[+] All 11 plots successfully generated and saved to: {plots_dir}")