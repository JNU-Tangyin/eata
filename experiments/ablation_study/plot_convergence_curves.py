"""
从消融实验结果生成搜索效率收敛曲线图
对比 EATA-Full vs EATA-NoNN
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pickle

def plot_convergence_curves(results_dir, output_dir):
    """
    从消融实验结果生成收敛曲线图
    """
    print("="*80)
    print("生成搜索效率收敛曲线图")
    print("="*80)
    
    # 读取结果文件
    results_file = results_dir / 'ablation_results.csv'
    
    if not results_file.exists():
        print(f"❌ 找不到结果文件: {results_file}")
        return None
    
    df = pd.read_csv(results_file)
    print(f"✅ 加载了 {len(df)} 条结果")
    
    # 收集Full和NoNN的收敛曲线
    full_curves = []
    nonn_curves = []
    
    for _, row in df.iterrows():
        variant = row['variant']
        
        # 检查是否有收敛历史数据
        if 'convergence_history' not in row or pd.isna(row['convergence_history']):
            continue
        
        try:
            # 尝试解析收敛历史
            if isinstance(row['convergence_history'], str):
                if row['convergence_history'].startswith('['):
                    convergence = json.loads(row['convergence_history'])
                else:
                    convergence = eval(row['convergence_history'])
            else:
                convergence = row['convergence_history']
            
            if len(convergence) > 0:
                if variant == 'EATA-Full':
                    full_curves.append(convergence)
                elif variant == 'EATA-NoNN':
                    nonn_curves.append(convergence)
        except Exception as e:
            print(f"⚠️ 解析收敛历史失败: {e}")
            continue
    
    print(f"✅ EATA-Full: {len(full_curves)} 条曲线")
    print(f"✅ EATA-NoNN: {len(nonn_curves)} 条曲线")
    
    if len(full_curves) == 0 and len(nonn_curves) == 0:
        print("❌ 没有找到收敛曲线数据")
        return None
    
    # 绘制图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        'Full': '#4169E1',  # Royal Blue
        'NoNN': '#50C878',  # Emerald Green
    }
    
    # 绘制EATA-Full
    if len(full_curves) > 0:
        # 计算平均曲线
        max_len = max(len(c) for c in full_curves)
        min_len = min(len(c) for c in full_curves)
        target_len = min(100, min_len)
        
        interpolated = []
        for curve in full_curves:
            indices = np.linspace(0, len(curve)-1, target_len, dtype=int)
            interpolated.append([curve[i] for i in indices])
        
        mean_curve = np.mean(interpolated, axis=0)
        std_curve = np.std(interpolated, axis=0)
        x = np.arange(len(mean_curve))
        
        ax.plot(x, mean_curve, 
                color=colors['Full'], 
                linewidth=2.5, 
                label='EATA (Full)',
                alpha=0.9)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                        color=colors['Full'], alpha=0.2)
    
    # 绘制EATA-NoNN
    if len(nonn_curves) > 0:
        max_len = max(len(c) for c in nonn_curves)
        min_len = min(len(c) for c in nonn_curves)
        target_len = min(100, min_len)
        
        interpolated = []
        for curve in nonn_curves:
            indices = np.linspace(0, len(curve)-1, target_len, dtype=int)
            interpolated.append([curve[i] for i in indices])
        
        mean_curve = np.mean(interpolated, axis=0)
        std_curve = np.std(interpolated, axis=0)
        x = np.arange(len(mean_curve))
        
        ax.plot(x, mean_curve, 
                color=colors['NoNN'], 
                linewidth=2.5, 
                label='EATA w/o NN',
                alpha=0.9)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                        color=colors['NoNN'], alpha=0.2)
    
    ax.set_xlabel('Search Progress (Window Index)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best Reward (Sharpe Ratio)', fontsize=12, fontweight='bold')
    ax.set_title('Search Efficiency: Convergence Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # 增加横纵轴tick标签字体大小并加粗
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()
    
    # 保存
    pdf_file = output_dir / 'fig4_search_efficiency.pdf'
    png_file = output_dir / 'fig4_search_efficiency.png'
    
    plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 收敛曲线图已保存:")
    print(f"   PDF: {pdf_file}")
    print(f"   PNG: {png_file}")
    
    return pdf_file

if __name__ == '__main__':
    project_root = Path(__file__).parent.parent.parent
    results_dir = project_root / 'experiments' / 'ablation_study' / 'results'
    output_dir = project_root / 'paper' / 'figures'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_convergence_curves(results_dir, output_dir)
