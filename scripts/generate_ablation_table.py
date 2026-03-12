
import pandas as pd
import numpy as np
import os

def process_results(input_csv, output_tex):
    # 1. Load Data
    df = pd.read_csv(input_csv)
    
    # 2. Filter out failed experiments (EATA-NoMem)
    df_clean = df[df['variant'] != 'EATA-NoMem'].copy()
    
    # 3. Aggregation using groupby
    metrics = ['annual_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
    df_grouped = df_clean.groupby('variant')[metrics].mean()
    
    # 4. Format Data
    # Convert to percentages
    df_grouped['annual_return'] = df_grouped['annual_return'] * 100
    df_grouped['max_drawdown'] = df_grouped['max_drawdown'] * 100
    df_grouped['win_rate'] = df_grouped['win_rate'] * 100
    
    # Rename Index (Variant names)
    index_mapping = {
        'EATA-Full': r'\textbf{EATA (Full)}',
        'EATA-NoMCTS': 'NoMCTS',
        'EATA-NoNN': 'NoNN',
        'EATA-Simple': 'Simple'
    }
    df_grouped.index = df_grouped.index.map(lambda x: index_mapping.get(x, x))
    
    # Sort: EATA (Full) first, then others, or by SR
    df_grouped = df_grouped.sort_values('sharpe_ratio', ascending=False)

    # 5. Manual LaTeX Generation
    # We construct the string manually to avoid pandas to_latex issues with resizebox
    
    latex_lines = []
    latex_lines.append(r"\begin{table}[htbp]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Component Ablation Analysis. Comparison of EATA variants.}")
    latex_lines.append(r"\label{tab:component_ablation}")
    latex_lines.append(r"\resizebox{0.95\columnwidth}{!}{%")
    latex_lines.append(r"\begin{tabular}{lcccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"\textbf{Variant} & \textbf{AR (\%)} & \textbf{SR} & \textbf{MDD (\%)} & \textbf{Win Rate (\%)} \\")
    latex_lines.append(r"\midrule")
    
    for variant, row in df_grouped.iterrows():
        ar = row['annual_return']
        sr = row['sharpe_ratio']
        mdd = row['max_drawdown']
        wr = row['win_rate']
        
        line = f"{variant} & {ar:.2f} & {sr:.2f} & {mdd:.2f} & {wr:.2f}\\% \\\\"
        latex_lines.append(line)
        
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}%")
    latex_lines.append(r"}")
    latex_lines.append(r"\end{table}")
    
    final_latex = "\n".join(latex_lines)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_tex), exist_ok=True)
    
    with open(output_tex, 'w') as f:
        f.write(final_latex)
    
    print(f"Successfully generated LaTeX table at: {output_tex}")
    print(df_grouped)

if __name__ == "__main__":
    input_csv = '/Users/yin/Desktop/doing/eata/performance_summary_20260207_232901_fixed.csv'
    output_tex = '/Users/yin/Desktop/doing/eata/paper/tables/ablation_component.tex'
    process_results(input_csv, output_tex)
