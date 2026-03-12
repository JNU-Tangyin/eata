"""
性能分析器
分析消融实验结果的性能表现
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json

class PerformanceAnalyzer:
    """
    性能分析器，负责分析消融实验的性能结果
    """
    
    def __init__(self):
        """
        初始化性能分析器
        """
        self.baseline_variant = 'EATA-Full'
        
    def analyze_experiment_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析实验结果
        
        Args:
            results: 实验结果列表
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        # 过滤成功的结果
        successful_results = [r for r in results if 'error' not in r and 'metrics' in r]
        failed_results = [r for r in results if 'error' in r]
        
        if not successful_results:
            return {
                'status': 'failed',
                'message': 'No successful experiments found',
                'failed_count': len(failed_results)
            }
        
        # 基本统计
        analysis = {
            'experiment_summary': {
                'total_variants': len(results),
                'successful_variants': len(successful_results),
                'failed_variants': len(failed_results),
                'success_rate': len(successful_results) / len(results) * 100
            }
        }
        
        # 性能排名
        analysis['performance_ranking'] = self._rank_variants(successful_results)
        
        # 基准对比
        analysis['baseline_comparison'] = self._compare_to_baseline(successful_results)
        
        # 指标分析
        analysis['metrics_analysis'] = self._analyze_metrics(successful_results)
        
        # 变体特征分析
        analysis['variant_characteristics'] = self._analyze_variant_characteristics(successful_results)
        
        return analysis
    
    def _rank_variants(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对变体进行性能排名
        """
        rankings = []
        
        for result in results:
            metrics = result.get('metrics', {})
            rankings.append({
                'variant': result['variant'],
                'annual_return': metrics.get('annual_return', 0.0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                'max_drawdown': metrics.get('max_drawdown', 0.0),
                'win_rate': metrics.get('win_rate', 0.0),
                'composite_score': self._calculate_composite_score(metrics)
            })
        
        # 按综合得分排序
        rankings.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # 添加排名
        for i, ranking in enumerate(rankings):
            ranking['rank'] = i + 1
        
        return rankings
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """
        计算综合得分
        """
        # 权重设置
        weights = {
            'annual_return': 0.3,
            'sharpe_ratio': 0.4,
            'max_drawdown': -0.2,  # 负权重，回撤越小越好
            'win_rate': 0.1
        }
        
        score = 0.0
        for metric, weight in weights.items():
            value = metrics.get(metric, 0.0)
            if metric == 'max_drawdown':
                # 回撤转换为正向指标
                value = -value
            score += weight * value
        
        return score
    
    def _compare_to_baseline(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        与基准进行对比
        """
        # 找到基准结果
        baseline_result = None
        for result in results:
            if result['variant'] == self.baseline_variant:
                baseline_result = result
                break
        
        if not baseline_result:
            return {'error': f'Baseline variant {self.baseline_variant} not found'}
        
        baseline_metrics = baseline_result.get('metrics', {})
        comparisons = []
        
        for result in results:
            if result['variant'] == self.baseline_variant:
                continue
                
            metrics = result.get('metrics', {})
            comparison = {
                'variant': result['variant'],
                'performance_changes': {}
            }
            
            # 计算相对变化
            for metric_name in ['annual_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']:
                baseline_value = baseline_metrics.get(metric_name, 0.0)
                current_value = metrics.get(metric_name, 0.0)
                
                if baseline_value != 0:
                    change_pct = ((current_value - baseline_value) / abs(baseline_value)) * 100
                else:
                    change_pct = 0.0 if current_value == 0 else float('inf')
                
                comparison['performance_changes'][metric_name] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'change_pct': change_pct,
                    'improvement': change_pct > 0 if metric_name != 'max_drawdown' else change_pct < 0
                }
            
            comparisons.append(comparison)
        
        return {
            'baseline_variant': self.baseline_variant,
            'baseline_metrics': baseline_metrics,
            'comparisons': comparisons
        }
    
    def _analyze_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析指标分布
        """
        metrics_data = {}
        
        # 收集所有指标数据
        for result in results:
            metrics = result.get('metrics', {})
            for metric_name, value in metrics.items():
                if metric_name not in metrics_data:
                    metrics_data[metric_name] = []
                metrics_data[metric_name].append(value)
        
        # 计算统计信息
        metrics_stats = {}
        for metric_name, values in metrics_data.items():
            if values:
                metrics_stats[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75)
                }
        
        return metrics_stats
    
    def _analyze_variant_characteristics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析变体特征
        """
        characteristics = {}
        
        for result in results:
            variant_name = result['variant']
            variant_info = result.get('variant_info', {})
            metrics = result.get('metrics', {})
            
            characteristics[variant_name] = {
                'description': variant_info.get('description', ''),
                'hypothesis': variant_info.get('hypothesis', ''),
                'modifications': variant_info.get('modifications', {}),
                'key_metrics': {
                    'annual_return': metrics.get('annual_return', 0.0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                    'max_drawdown': metrics.get('max_drawdown', 0.0)
                },
                'performance_category': self._categorize_performance(metrics)
            }
        
        return characteristics
    
    def _categorize_performance(self, metrics: Dict[str, float]) -> str:
        """
        对性能进行分类
        """
        sharpe_ratio = metrics.get('sharpe_ratio', 0.0)
        annual_return = metrics.get('annual_return', 0.0)
        max_drawdown = metrics.get('max_drawdown', 0.0)
        
        # 简单的性能分类逻辑
        if sharpe_ratio > 1.5 and annual_return > 0.15:
            return 'Excellent'
        elif sharpe_ratio > 1.0 and annual_return > 0.10:
            return 'Good'
        elif sharpe_ratio > 0.5 and annual_return > 0.05:
            return 'Average'
        elif annual_return > 0:
            return 'Below Average'
        else:
            return 'Poor'
    
    def generate_performance_report(self, analysis: Dict[str, Any]) -> str:
        """
        生成性能报告
        """
        report = []
        report.append("# EATA消融实验性能分析报告\n")
        
        # 实验概览
        summary = analysis.get('experiment_summary', {})
        report.append("## 实验概览")
        report.append(f"- 总变体数: {summary.get('total_variants', 0)}")
        report.append(f"- 成功运行: {summary.get('successful_variants', 0)}")
        report.append(f"- 运行失败: {summary.get('failed_variants', 0)}")
        report.append(f"- 成功率: {summary.get('success_rate', 0):.1f}%\n")
        
        # 性能排名
        rankings = analysis.get('performance_ranking', [])
        if rankings:
            report.append("## 性能排名")
            report.append("| 排名 | 变体 | 年化收益 | 夏普比率 | 最大回撤 | 胜率 | 综合得分 |")
            report.append("|------|------|----------|----------|----------|------|----------|")
            
            for ranking in rankings[:10]:  # 显示前10名
                report.append(f"| {ranking['rank']} | {ranking['variant']} | "
                            f"{ranking['annual_return']:.4f} | {ranking['sharpe_ratio']:.4f} | "
                            f"{ranking['max_drawdown']:.4f} | {ranking['win_rate']:.4f} | "
                            f"{ranking['composite_score']:.4f} |")
            report.append("")
        
        # 基准对比
        baseline_comp = analysis.get('baseline_comparison', {})
        if 'comparisons' in baseline_comp:
            report.append("## 基准对比")
            report.append(f"基准变体: {baseline_comp.get('baseline_variant', 'N/A')}\n")
            
            for comp in baseline_comp['comparisons']:
                report.append(f"### {comp['variant']}")
                changes = comp['performance_changes']
                
                for metric, change_data in changes.items():
                    change_pct = change_data['change_pct']
                    improvement = "✅" if change_data['improvement'] else "❌"
                    report.append(f"- {metric}: {change_pct:+.2f}% {improvement}")
                report.append("")
        
        return "\n".join(report)
    
    def export_analysis_to_csv(self, analysis: Dict[str, Any], output_dir: Path) -> List[Path]:
        """
        将分析结果导出为CSV
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        
        # 导出性能排名
        rankings = analysis.get('performance_ranking', [])
        if rankings:
            df_rankings = pd.DataFrame(rankings)
            ranking_file = output_dir / "performance_rankings.csv"
            df_rankings.to_csv(ranking_file, index=False, encoding='utf-8')
            exported_files.append(ranking_file)
        
        # 导出基准对比
        baseline_comp = analysis.get('baseline_comparison', {})
        if 'comparisons' in baseline_comp:
            comp_data = []
            for comp in baseline_comp['comparisons']:
                row = {'variant': comp['variant']}
                for metric, change_data in comp['performance_changes'].items():
                    row[f'{metric}_baseline'] = change_data['baseline']
                    row[f'{metric}_current'] = change_data['current']
                    row[f'{metric}_change_pct'] = change_data['change_pct']
                    row[f'{metric}_improvement'] = change_data['improvement']
                comp_data.append(row)
            
            if comp_data:
                df_comp = pd.DataFrame(comp_data)
                comp_file = output_dir / "baseline_comparison.csv"
                df_comp.to_csv(comp_file, index=False, encoding='utf-8')
                exported_files.append(comp_file)
        
        return exported_files
