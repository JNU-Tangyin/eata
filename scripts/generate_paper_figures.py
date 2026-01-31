import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os

# Style Configuration (Matching STYLE_MAPPING.md)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'cm' # For math expressions
colors = {
    'EATA': '#4169E1',      # RoyalBlue
    'Baseline': '#808080',  # Gray
    'Profit': '#50C878',    # Emerald
    'Risk': '#DC143C',      # Crimson
    'S&P500': '#000000',    # Black
    'Fill_Bull': '#E6F9EE', # Very light green
    'Fill_Bear': '#FDECEC', # Very light red
}

OUTPUT_DIR = '../paper/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def setup_plot(title=None, xlabel=None, ylabel=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    if title: ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    if xlabel: ax.set_xlabel(xlabel, fontsize=12)
    if ylabel: ax.set_ylabel(ylabel, fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.3)
    return fig, ax

def add_placeholder_watermark(ax, text_lines):
    """Adds a red text watermark explaining the figure mapping."""
    text = "\n".join(text_lines)
    ax.text(0.5, 0.5, text, 
            transform=ax.transAxes, 
            ha='center', va='center', 
            fontsize=12, color='red', alpha=0.7,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', boxstyle='round,pad=1'))

# ---------------------------------------------------------
# Figure 1: Cumulative Returns (Main Result)
# ---------------------------------------------------------
def plot_cumulative_returns():
    print("Generating Figure 1: Cumulative Returns...")
    fig, ax = setup_plot(ylabel='Cumulative Return (%)')
    
    # Synthetic Data
    np.random.seed(42)
    days = 1260 # 5 years approx
    t = np.arange(days)
    dates = np.linspace(2020, 2025, days)
    
    # Random walks with drift
    # EATA: High Sharpe (steady up)
    eata_returns = np.random.normal(0.0008, 0.01, days)
    eata_cum = (1 + eata_returns).cumprod() - 1
    
    # Buy & Hold: Market beta (volatile, upward)
    sp500_returns = np.random.normal(0.0004, 0.012, days)
    sp500_cum = (1 + sp500_returns).cumprod() - 1
    
    # LSTM: High vol, okay return
    lstm_returns = np.random.normal(0.0005, 0.015, days)
    lstm_cum = (1 + lstm_returns).cumprod() - 1
    
    # Plotting
    ax.plot(dates, eata_cum * 100, color=colors['EATA'], linewidth=2.5, label='EATA (Ours)')
    ax.plot(dates, sp500_cum * 100, color=colors['S&P500'], linewidth=1.0, linestyle='-', label='Buy & Hold')
    ax.plot(dates, lstm_cum * 100, color=colors['Baseline'], linewidth=1.0, linestyle='--', label='LSTM')
    
    # Annotate Regimes (Visual Logic)
    # Roughly mapping 2020 crash and 2022 bear
    # Just visual shading for style demonstration
    ax.axvspan(2020.15, 2020.3, color=colors['Fill_Bear'], alpha=0.5, label='Bear Regime')
    ax.axvspan(2020.3, 2021.9, color=colors['Fill_Bull'], alpha=0.5, label='Bull Regime')
    
    ax.legend(frameon=False, loc='upper left')
    
    # Watermark for the author
    add_placeholder_watermark(ax, [
        "FILE: fig1_cumulative_returns.pdf",
        "TYPE: Time Series Line Chart",
        "MAPPING: EATA=Blue/Thick, B&H=Black/Thin, Baselines=Gray/Dashed",
        "MESSAGE: EATA demonstrates consistent outperformance & lower drawdown."
    ])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig1_cumulative_returns.pdf')
    plt.savefig(f'{OUTPUT_DIR}/fig1_cumulative_returns.png')
    plt.close()

# ---------------------------------------------------------
# Figure 2: Pareto Frontier (Interpretability vs Performance)
# ---------------------------------------------------------
def plot_pareto_frontier():
    print("Generating Figure 2: Pareto Frontier...")
    fig, ax = setup_plot(xlabel='Complexity (Expression Nodes)', ylabel='Sharpe Ratio')
    
    # Synthetic Data
    # EATA: Low complexity, High Sharpe
    eata_x = np.random.uniform(5, 25, 15)
    eata_y = np.random.uniform(0.7, 1.2, 15)
    
    # GP: High complexity, Med Sharpe
    gp_x = np.random.uniform(30, 80, 20)
    gp_y = np.random.uniform(0.3, 0.8, 20)
    
    # Linear: Very low complexity, Low Sharpe
    lin_x = np.random.uniform(2, 5, 5)
    lin_y = np.random.uniform(0.1, 0.4, 5)
    
    # Deep Learning (Infinite Complexity - plotted as vertical line or far right)
    dl_x = np.ones(10) * 100 # "High"
    dl_y = np.random.uniform(0.5, 0.9, 10)
    
    # Plotting
    ax.scatter(eata_x, eata_y, color=colors['EATA'], s=100, marker='*', label='EATA')
    ax.scatter(gp_x, gp_y, color=colors['Baseline'], s=50, marker='^', alpha=0.6, label='Genetic Programming')
    ax.scatter(lin_x, lin_y, color=colors['Baseline'], s=50, marker='s', alpha=0.6, label='Linear Models')
    
    # Theoretical Deep Learning Zone
    ax.text(95, 0.7, "Deep Learning\n(Black Box)", ha='center', color=colors['Baseline'])
    ax.axvline(x=90, color='gray', linestyle=':', alpha=0.5)
    
    # Pareto Curve
    # Manually drawing a smooth curve over top-left points
    curve_x = np.linspace(2, 25, 100)
    curve_y = 0.4 + 0.6 * np.log(curve_x)/np.log(25) # Mock curve
    ax.plot(curve_x, curve_y + 0.1, color=colors['EATA'], linestyle='--', alpha=0.3, label='Pareto Frontier')

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.5)
    ax.legend(frameon=True)
    
    add_placeholder_watermark(ax, [
        "FILE: fig2_pareto_frontier.pdf",
        "TYPE: Scatter Plot",
        "MAPPING: X=Complexity, Y=Sharpe. EATA=Blue Stars (Top-Left).",
        "MESSAGE: EATA breaks the trade-off, achieving high Sharpe with low complexity."
    ])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig2_pareto_frontier.pdf')
    plt.savefig(f'{OUTPUT_DIR}/fig2_pareto_frontier.png')
    plt.close()

# ---------------------------------------------------------
# Figure 3: Return Distribution (Wasserstein)
# ---------------------------------------------------------
def plot_return_distribution():
    print("Generating Figure 3: Return Distribution...")
    fig, ax = setup_plot(xlabel='Daily Return (%)', ylabel='Density')
    
    # Synthetic Data
    from scipy.stats import norm, t
    
    x = np.linspace(-10, 10, 1000)
    
    # Real Data: Fat tails (Student-t)
    real_dist = t.pdf(x, df=3, loc=0, scale=1.5)
    
    # MSE Prediction: Gaussian (Thin tails)
    mse_dist = norm.pdf(x, loc=0, scale=1.5)
    
    # EATA Prediction: Fits fat tails better
    eata_dist = t.pdf(x, df=4, loc=0.1, scale=1.4) # Slightly shifted mean (alpha)
    
    # Plotting
    ax.fill_between(x, real_dist, color='gray', alpha=0.2, label='Actual Returns (Fat Tails)')
    ax.plot(x, eata_dist, color=colors['EATA'], linewidth=2.5, label='EATA Prediction (Wasserstein)')
    ax.plot(x, mse_dist, color=colors['Risk'], linestyle='--', linewidth=2, label='MSE Baseline (Gaussian)')
    
    ax.set_xlim(-8, 8)
    ax.legend(frameon=False)
    
    add_placeholder_watermark(ax, [
        "FILE: fig3_return_distribution.pdf",
        "TYPE: Density Plot (KDE)",
        "MAPPING: Gray Fill=Actual, Blue Line=EATA, Red Dashed=MSE",
        "MESSAGE: Wasserstein loss enables EATA to capture fat-tail risks."
    ])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig3_return_distribution.pdf')
    plt.savefig(f'{OUTPUT_DIR}/fig3_return_distribution.png')
    plt.close()

# ---------------------------------------------------------
# Figure 4: Search Efficiency
# ---------------------------------------------------------
def plot_search_efficiency():
    print("Generating Figure 4: Search Efficiency...")
    fig, ax = setup_plot(xlabel='Search Time (Minutes)', ylabel='Max Reward (Sharpe)')
    
    # Synthetic Data
    time = np.linspace(0, 60, 100)
    
    # EATA: Fast convergence
    eata_perf = 1.0 - 0.8 * np.exp(-time / 10) + np.random.normal(0, 0.02, 100)
    
    # GP: Slower
    gp_perf = 0.8 - 0.6 * np.exp(-time / 25) + np.random.normal(0, 0.02, 100)
    
    # Random: Flat
    rand_perf = 0.2 + np.random.normal(0, 0.05, 100)
    
    # Plotting
    ax.plot(time, eata_perf, color=colors['EATA'], linewidth=2.5, label='EATA (Neural-Guided)')
    ax.plot(time, gp_perf, color=colors['Baseline'], linewidth=2, linestyle='-', label='Genetic Programming')
    ax.plot(time, rand_perf, color='gray', linewidth=1, linestyle=':', label='Random Search')
    
    ax.set_ylim(0, 1.2)
    ax.legend(frameon=False)
    
    add_placeholder_watermark(ax, [
        "FILE: fig4_search_efficiency.pdf",
        "TYPE: Convergence Plot",
        "MAPPING: X=Time, Y=Reward. Blue=EATA (Fastest).",
        "MESSAGE: Neural guidance significantly accelerates discovery."
    ])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig4_search_efficiency.pdf')
    plt.savefig(f'{OUTPUT_DIR}/fig4_search_efficiency.png')
    plt.close()

if __name__ == "__main__":
    plot_cumulative_returns()
    plot_pareto_frontier()
    plot_return_distribution()
    plot_search_efficiency()
    print("All placeholder figures generated in ../paper/figures/")
