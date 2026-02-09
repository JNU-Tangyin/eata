"""
结果可视化模块
生成消融实验结果的各种图表
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings

# 隐藏警告信息
warnings.filterwarnings('ignore')
plt.rcParams['figure.max_open_warning'] = 0  # 隐藏matplotlib图形数量警告

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class ResultVisualizer:
    """
    结果可视化器，负责生成消融实验的各种图表
    """
    
    def __init__(self, output_dir: Path, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        """
        初始化可视化器
        
        Args:
            output_dir: 图表输出目录
            figsize: 图表大小
            dpi: 图表分辨率
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        self.dpi = dpi
        
        # 设置颜色方案
        self.colors = sns.color_palette("Set2", 10)
        
    def plot_performance_comparison(self, results: List[Dict[str, Any]], 
                                  experiment_id: str = None) -> Path:
        """
        绘制性能对比图
        """
        # 准备数据
        variants = []
        annual_returns = []
        sharpe_ratios = []
        max_drawdowns = []
        
        for result in results:
            if 'error' not in result and 'metrics' in result:
                metrics = result['metrics']
                variants.append(result['variant'])
                annual_returns.append(metrics.get('annual_return', 0))
                sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
                max_drawdowns.append(abs(metrics.get('max_drawdown', 0)))
        
        if not variants:
            return None
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('EATA消融实验性能对比', fontsize=16, fontweight='bold')
        
        # 年化收益率
        axes[0, 0].bar(variants, annual_returns, color=self.colors[:len(variants)])
        axes[0, 0].set_title('年化收益率')
        axes[0, 0].set_ylabel('收益率')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 夏普比率
        axes[0, 1].bar(variants, sharpe_ratios, color=self.colors[:len(variants)])
        axes[0, 1].set_title('夏普比率')
        axes[0, 1].set_ylabel('夏普比率')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 最大回撤
        axes[1, 0].bar(variants, max_drawdowns, color=self.colors[:len(variants)])
        axes[1, 0].set_title('最大回撤')
        axes[1, 0].set_ylabel('回撤幅度')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 风险收益散点图
        axes[1, 1].scatter(max_drawdowns, annual_returns, 
                          c=range(len(variants)), cmap='Set2', s=100)
        for i, variant in enumerate(variants):
            axes[1, 1].annotate(variant, (max_drawdowns[i], annual_returns[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 1].set_xlabel('最大回撤')
        axes[1, 1].set_ylabel('年化收益率')
        axes[1, 1].set_title('风险-收益散点图')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_file = self.output_dir / f"performance_comparison_{experiment_id or 'unknown'}.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def plot_baseline_comparison(self, baseline_comparison: Dict[str, Any], 
                               experiment_id: str = None) -> Path:
        """
        绘制与基准的对比图
        """
        if 'comparisons' not in baseline_comparison:
            return None
        
        comparisons = baseline_comparison['comparisons']
        variants = [comp['variant'] for comp in comparisons]
        
        # 准备数据
        metrics = ['annual_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        changes = {metric: [] for metric in metrics}
        
        for comp in comparisons:
            for metric in metrics:
                change_pct = comp['performance_changes'][metric]['change_pct']
                changes[metric].append(change_pct)
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 准备热力图数据
        heatmap_data = []
        for variant in variants:
            row = []
            for metric in metrics:
                idx = variants.index(variant)
                row.append(changes[metric][idx])
            heatmap_data.append(row)
        
        # 绘制热力图
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
        
        # 设置标签
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(['年化收益率', '夏普比率', '最大回撤', '胜率'])
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels(variants)
        
        # 添加数值标注
        for i in range(len(variants)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{heatmap_data[i][j]:.1f}%',
                              ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title(f'相对于{baseline_comparison.get("baseline_variant", "基准")}的性能变化 (%)', 
                    fontsize=14, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im)
        cbar.set_label('变化百分比 (%)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        # 保存图表
        output_file = self.output_dir / f"baseline_comparison_{experiment_id or 'unknown'}.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def plot_ranking_chart(self, performance_rankings: List[Dict[str, Any]], 
                          experiment_id: str = None) -> Path:
        """
        绘制性能排名图
        """
        if not performance_rankings:
            return None
        
        # 准备数据
        variants = [r['variant'] for r in performance_rankings]
        composite_scores = [r['composite_score'] for r in performance_rankings]
        ranks = [r['rank'] for r in performance_rankings]
        
        # 创建水平条形图
        fig, ax = plt.subplots(figsize=(12, max(8, len(variants) * 0.6)))
        
        bars = ax.barh(range(len(variants)), composite_scores, color=self.colors[:len(variants)])
        
        # 设置标签
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels([f"{rank}. {variant}" for rank, variant in zip(ranks, variants)])
        ax.set_xlabel('综合得分')
        ax.set_title('EATA变体性能排名', fontsize=14, fontweight='bold')
        
        # 添加数值标注
        for i, (bar, score) in enumerate(zip(bars, composite_scores)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{score:.4f}', ha='left', va='center', fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        # 保存图表
        output_file = self.output_dir / f"performance_rankings_{experiment_id or 'unknown'}.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def plot_time_series(self, results: List[Dict[str, Any]], 
                        experiment_id: str = None) -> Optional[Path]:
        """
        绘制时间序列图
        """
        # 准备时间序列数据
        time_series_data = {}
        
        for result in results:
            if 'error' not in result and 'returns' in result:
                returns = result['returns']
                if isinstance(returns, np.ndarray) and len(returns) > 0:
                    cumulative_returns = np.cumprod(1 + returns) - 1
                    time_series_data[result['variant']] = cumulative_returns
        
        if not time_series_data:
            return None
        
        # 绘制累积收益曲线
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, (variant, cum_returns) in enumerate(time_series_data.items()):
            ax.plot(cum_returns, label=variant, color=self.colors[i % len(self.colors)], linewidth=2)
        
        ax.set_xlabel('交易期数')
        ax.set_ylabel('累积收益率')
        ax.set_title('EATA变体累积收益对比', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_file = self.output_dir / f"time_series_{experiment_id or 'unknown'}.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def plot_statistical_significance(self, statistical_results: Dict[str, Any], 
                                    experiment_id: str = None) -> Optional[Path]:
        """
        绘制统计显著性图
        """
        if 'pairwise_tests' not in statistical_results:
            return None
        
        # 准备数据
        variants = []
        p_values = []
        effect_sizes = []
        
        for variant, test_result in statistical_results['pairwise_tests'].items():
            variants.append(variant)
            
            # 获取p值
            if 'paired_t_test' in test_result and 'p_value' in test_result['paired_t_test']:
                p_values.append(test_result['paired_t_test']['p_value'])
            else:
                p_values.append(1.0)
            
            # 获取效应量
            if 'effect_size' in test_result and 'cohens_d' in test_result['effect_size']:
                effect_sizes.append(abs(test_result['effect_size']['cohens_d']))
            else:
                effect_sizes.append(0.0)
        
        if not variants:
            return None
        
        # 创建散点图
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 根据显著性设置颜色
        colors = ['red' if p < 0.05 else 'blue' for p in p_values]
        sizes = [100 + es * 50 for es in effect_sizes]  # 根据效应量调整点的大小
        
        scatter = ax.scatter(effect_sizes, [-np.log10(p) for p in p_values], 
                           c=colors, s=sizes, alpha=0.7)
        
        # 添加标签
        for i, variant in enumerate(variants):
            ax.annotate(variant, (effect_sizes[i], -np.log10(p_values[i])), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 添加显著性线
        ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05')
        ax.axhline(y=-np.log10(0.01), color='red', linestyle='--', alpha=0.7, label='p=0.01')
        
        ax.set_xlabel('效应量 (|Cohen\'s d|)')
        ax.set_ylabel('-log10(p值)')
        ax.set_title('统计显著性与效应量', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_file = self.output_dir / f"statistical_significance_{experiment_id or 'unknown'}.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def create_comprehensive_dashboard(self, analysis_results: Dict[str, Any],
                                     statistical_results: Dict[str, Any],
                                     results: List[Dict[str, Any]],
                                     experiment_id: str = None) -> List[Path]:
        """
        创建综合仪表板
        """
        generated_plots = []
        
        try:
            # 性能对比图
            perf_plot = self.plot_performance_comparison(results, experiment_id)
            if perf_plot:
                generated_plots.append(perf_plot)
            
            # 基准对比图
            if 'baseline_comparison' in analysis_results:
                baseline_plot = self.plot_baseline_comparison(
                    analysis_results['baseline_comparison'], experiment_id
                )
                if baseline_plot:
                    generated_plots.append(baseline_plot)
            
            # 排名图
            if 'performance_ranking' in analysis_results:
                ranking_plot = self.plot_ranking_chart(
                    analysis_results['performance_ranking'], experiment_id
                )
                if ranking_plot:
                    generated_plots.append(ranking_plot)
            
            # 时间序列图
            time_series_plot = self.plot_time_series(results, experiment_id)
            if time_series_plot:
                generated_plots.append(time_series_plot)
            
            # 统计显著性图
            if statistical_results:
                stats_plot = self.plot_statistical_significance(statistical_results, experiment_id)
                if stats_plot:
                    generated_plots.append(stats_plot)
            
        except Exception as e:
            print(f"⚠️ 生成图表时出现错误: {e}")
        
        return generated_plots
    
    def save_plot_index(self, generated_plots: List[Path], 
                       experiment_id: str = None) -> Path:
        """
        保存图表索引
        """
        index_data = []
        
        for plot_path in generated_plots:
            if plot_path and plot_path.exists():
                index_data.append({
                    'experiment_id': experiment_id or 'unknown',
                    'plot_type': plot_path.stem.split('_')[0],
                    'file_name': plot_path.name,
                    'file_path': str(plot_path),
                    'file_size_kb': plot_path.stat().st_size / 1024
                })
        
        df = pd.DataFrame(index_data)
        index_file = self.output_dir / f"plot_index_{experiment_id or 'unknown'}.csv"
        df.to_csv(index_file, index=False, encoding='utf-8')
        
        return index_file
