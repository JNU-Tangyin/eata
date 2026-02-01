
import pandas as pd

try:
    df_eata = pd.read_csv('../tables/Batch3-final_metrics.csv')
    df_norl = pd.read_csv('../tables/Batch3_NoRL-final_metrics.csv')
    
    # Merge on Ticker
    df = pd.merge(df_eata[['Ticker', 'Annual Return (AR)', 'Sharpe Ratio', 'Max Drawdown (MDD)']], 
                  df_norl[['Ticker', 'Annual Return (AR)', 'Sharpe Ratio', 'Max Drawdown (MDD)']], 
                  on='Ticker', suffixes=('_EATA', '_NoRL'))
    
    # Sort by Ticker
    df = df.sort_values('Ticker')
    
    print(r'\begin{table*}[htbp]')
    print(r'\centering')
    print(r'\caption{Detailed Performance Metrics per Stock (EATA vs. NoRL Baseline)}')
    print(r'\label{tab:detailed_metrics}')
    print(r'\resizebox{0.9\textwidth}{!}{%')
    print(r'\begin{tabular}{lcccccc}')
    print(r'\toprule')
    print(r'\textbf{Ticker} & \textbf{EATA AR} & \textbf{NoRL AR} & \textbf{EATA SR} & \textbf{NoRL SR} & \textbf{EATA MDD} & \textbf{NoRL MDD} \\')
    print(r'\midrule')
    
    for _, row in df.iterrows():
        t = row['Ticker']
        e_ar = row['Annual Return (AR)_EATA'] * 100
        n_ar = row['Annual Return (AR)_NoRL'] * 100
        e_sr = row['Sharpe Ratio_EATA']
        n_sr = row['Sharpe Ratio_NoRL']
        e_mdd = row['Max Drawdown (MDD)_EATA'] * 100
        n_mdd = row['Max Drawdown (MDD)_NoRL'] * 100
        
        # Improvement check for bolding? Optional. Let's just print standard.
        print(f"{t} & {e_ar:.1f}\\% & {n_ar:.1f}\\% & {e_sr:.2f} & {n_sr:.2f} & {e_mdd:.1f}\\% & {n_mdd:.1f}\\% \\\\")
        
    print(r'\bottomrule')
    print(r'\end{tabular}%')
    print(r'}')
    print(r'\end{table*}')

except Exception as e:
    print(f"Error: {e}")
