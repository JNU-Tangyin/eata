"""
CSV导出器
专门负责将消融实验结果导出为各种CSV格式
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

class CSVExporter:
    """
    CSV导出器，负责将分析结果导出为CSV格式
    """
    
    def __init__(self, output_dir: Path):
        """
        初始化CSV导出器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_performance_summary(self, results: List[Dict[str, Any]], 
                                 experiment_id: str) -> Path:
        """
        导出性能汇总表
        """
        summary_data = []
        
        for result in results:
            if 'error' not in result and 'metrics' in result:
                metrics = result['metrics']
                variant_info = result.get('variant_info', {})
                
                summary_data.append({
                    'experiment_id': experiment_id,
                    'variant': result['variant'],
                    'description': variant_info.get('description', ''),
                    'annual_return': metrics.get('annual_return', 0.0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                    'max_drawdown': metrics.get('max_drawdown', 0.0),
                    'win_rate': metrics.get('win_rate', 0.0),
                    'volatility': metrics.get('volatility', 0.0),
                    'total_return': metrics.get('total_return', 0.0),
                    'rl_reward': result.get('rl_reward', 0.0),
                    'timestamp': result.get('timestamp', datetime.now().isoformat())
                })
        
        df = pd.DataFrame(summary_data)
        output_file = self.output_dir / f"performance_summary_{experiment_id}.csv"
        df.to_csv(output_file, index=False, encoding='utf-8', float_format='%.6f')
        
        return output_file
    
    def export_detailed_comparison(self, results: List[Dict[str, Any]], 
                                 baseline_variant: str = 'EATA-Full',
                                 experiment_id: str = None) -> Path:
        """
        导出详细对比表
        """
        # 找到基准结果
        baseline_result = None
        for result in results:
            if result['variant'] == baseline_variant and 'error' not in result:
                baseline_result = result
                break
        
        if not baseline_result:
            raise ValueError(f"Baseline variant {baseline_variant} not found")
        
        baseline_metrics = baseline_result.get('metrics', {})
        comparison_data = []
        
        for result in results:
            if 'error' not in result and 'metrics' in result:
                metrics = result['metrics']
                variant_info = result.get('variant_info', {})
                
                # 计算相对于基准的变化
                row = {
                    'experiment_id': experiment_id or 'unknown',
                    'variant': result['variant'],
                    'description': variant_info.get('description', ''),
                    'hypothesis': variant_info.get('hypothesis', ''),
                    'modifications': json.dumps(variant_info.get('modifications', {}))
                }
                
                # 添加绝对值和相对变化
                for metric_name in ['annual_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'volatility']:
                    baseline_value = baseline_metrics.get(metric_name, 0.0)
                    current_value = metrics.get(metric_name, 0.0)
                    
                    # 绝对值
                    row[metric_name] = current_value
                    row[f'{metric_name}_baseline'] = baseline_value
                    
                    # 相对变化
                    if baseline_value != 0:
                        change_pct = ((current_value - baseline_value) / abs(baseline_value)) * 100
                    else:
                        change_pct = 0.0 if current_value == 0 else float('inf')
                    
                    row[f'{metric_name}_change_pct'] = change_pct
                    
                    # 是否改善
                    if metric_name == 'max_drawdown':
                        row[f'{metric_name}_improved'] = change_pct < 0  # 回撤越小越好
                    else:
                        row[f'{metric_name}_improved'] = change_pct > 0
                
                comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        output_file = self.output_dir / f"detailed_comparison_{experiment_id or 'unknown'}.csv"
        df.to_csv(output_file, index=False, encoding='utf-8', float_format='%.6f')
        
        return output_file
    
    def export_statistical_results(self, statistical_analysis: Dict[str, Any], 
                                 experiment_id: str = None) -> Path:
        """
        导出统计检验结果
        """
        if 'pairwise_tests' not in statistical_analysis:
            raise ValueError("No pairwise test results found in statistical analysis")
        
        stats_data = []
        
        for variant_name, test_result in statistical_analysis['pairwise_tests'].items():
            row = {
                'experiment_id': experiment_id or 'unknown',
                'variant': variant_name,
                'baseline': statistical_analysis.get('baseline', 'Unknown')
            }
            
            # 描述性统计
            if 'descriptive_stats' in test_result:
                desc_stats = test_result['descriptive_stats']
                row.update({
                    'baseline_mean': desc_stats.get('baseline_mean', 0),
                    'baseline_std': desc_stats.get('baseline_std', 0),
                    'variant_mean': desc_stats.get('variant_mean', 0),
                    'variant_std': desc_stats.get('variant_std', 0),
                    'mean_difference': desc_stats.get('mean_difference', 0)
                })
            
            # t检验结果
            if 'paired_t_test' in test_result and 'error' not in test_result['paired_t_test']:
                t_test = test_result['paired_t_test']
                row.update({
                    'paired_t_statistic': t_test.get('statistic', 0),
                    'paired_t_p_value': t_test.get('p_value', 1),
                    'paired_t_significant': t_test.get('significant', False),
                    'paired_t_interpretation': t_test.get('interpretation', '')
                })
            
            # 独立t检验结果
            if 'independent_t_test' in test_result and 'error' not in test_result['independent_t_test']:
                ind_t_test = test_result['independent_t_test']
                row.update({
                    'independent_t_statistic': ind_t_test.get('statistic', 0),
                    'independent_t_p_value': ind_t_test.get('p_value', 1),
                    'independent_t_significant': ind_t_test.get('significant', False)
                })
            
            # Wilcoxon检验结果
            if 'wilcoxon_test' in test_result and 'error' not in test_result['wilcoxon_test']:
                wilcoxon = test_result['wilcoxon_test']
                row.update({
                    'wilcoxon_statistic': wilcoxon.get('statistic', 0),
                    'wilcoxon_p_value': wilcoxon.get('p_value', 1),
                    'wilcoxon_significant': wilcoxon.get('significant', False)
                })
            
            # 效应量
            if 'effect_size' in test_result and 'error' not in test_result['effect_size']:
                effect = test_result['effect_size']
                row.update({
                    'cohens_d': effect.get('cohens_d', 0),
                    'glass_delta': effect.get('glass_delta', 0),
                    'effect_interpretation': effect.get('cohens_d_interpretation', '')
                })
            
            stats_data.append(row)
        
        df = pd.DataFrame(stats_data)
        output_file = self.output_dir / f"statistical_tests_{experiment_id or 'unknown'}.csv"
        df.to_csv(output_file, index=False, encoding='utf-8', float_format='%.6f')
        
        return output_file
    
    def export_ranking_table(self, performance_rankings: List[Dict[str, Any]], 
                           experiment_id: str = None) -> Path:
        """
        导出性能排名表
        """
        df = pd.DataFrame(performance_rankings)
        output_file = self.output_dir / f"performance_rankings_{experiment_id or 'unknown'}.csv"
        df.to_csv(output_file, index=False, encoding='utf-8', float_format='%.6f')
        
        return output_file
    
    def export_time_series(self, results: List[Dict[str, Any]], 
                          experiment_id: str = None) -> Optional[Path]:
        """
        导出时间序列数据
        """
        time_series_data = []
        
        for result in results:
            if 'error' not in result and 'returns' in result:
                returns = result['returns']
                if isinstance(returns, np.ndarray) and len(returns) > 0:
                    cumulative_returns = np.cumprod(1 + returns)
                    
                    for i, (ret, cum_ret) in enumerate(zip(returns, cumulative_returns)):
                        time_series_data.append({
                            'experiment_id': experiment_id or 'unknown',
                            'variant': result['variant'],
                            'period': i + 1,
                            'return': ret,
                            'cumulative_return': cum_ret - 1,  # 转换为累积收益率
                            'timestamp': result.get('timestamp', datetime.now().isoformat())
                        })
        
        if not time_series_data:
            return None
        
        df = pd.DataFrame(time_series_data)
        output_file = self.output_dir / f"time_series_{experiment_id or 'unknown'}.csv"
        df.to_csv(output_file, index=False, encoding='utf-8', float_format='%.6f')
        
        return output_file
    
    def export_variant_characteristics(self, variant_characteristics: Dict[str, Any], 
                                     experiment_id: str = None) -> Path:
        """
        导出变体特征表
        """
        char_data = []
        
        for variant_name, characteristics in variant_characteristics.items():
            row = {
                'experiment_id': experiment_id or 'unknown',
                'variant': variant_name,
                'description': characteristics.get('description', ''),
                'hypothesis': characteristics.get('hypothesis', ''),
                'modifications': json.dumps(characteristics.get('modifications', {})),
                'performance_category': characteristics.get('performance_category', 'Unknown')
            }
            
            # 添加关键指标
            key_metrics = characteristics.get('key_metrics', {})
            for metric_name, value in key_metrics.items():
                row[f'key_{metric_name}'] = value
            
            char_data.append(row)
        
        df = pd.DataFrame(char_data)
        output_file = self.output_dir / f"variant_characteristics_{experiment_id or 'unknown'}.csv"
        df.to_csv(output_file, index=False, encoding='utf-8', float_format='%.6f')
        
        return output_file
    
    def export_comprehensive_report(self, analysis_results: Dict[str, Any], 
                                  statistical_results: Dict[str, Any],
                                  experiment_id: str = None) -> Dict[str, Path]:
        """
        导出综合报告的所有CSV文件
        
        Returns:
            Dict[str, Path]: 导出文件的路径字典
        """
        exported_files = {}
        
        try:
            # 性能排名
            if 'performance_ranking' in analysis_results:
                ranking_file = self.export_ranking_table(
                    analysis_results['performance_ranking'], experiment_id
                )
                exported_files['rankings'] = ranking_file
            
            # 变体特征
            if 'variant_characteristics' in analysis_results:
                char_file = self.export_variant_characteristics(
                    analysis_results['variant_characteristics'], experiment_id
                )
                exported_files['characteristics'] = char_file
            
            # 统计检验结果
            if statistical_results:
                stats_file = self.export_statistical_results(
                    statistical_results, experiment_id
                )
                exported_files['statistical_tests'] = stats_file
            
        except Exception as e:
            print(f"⚠️ 导出过程中出现错误: {e}")
        
        return exported_files
    
    def create_summary_index(self, exported_files: Dict[str, Path], 
                           experiment_id: str = None) -> Path:
        """
        创建导出文件的索引表
        """
        index_data = []
        
        for file_type, file_path in exported_files.items():
            if file_path and file_path.exists():
                index_data.append({
                    'experiment_id': experiment_id or 'unknown',
                    'file_type': file_type,
                    'file_name': file_path.name,
                    'file_path': str(file_path),
                    'file_size_kb': file_path.stat().st_size / 1024,
                    'created_time': datetime.now().isoformat()
                })
        
        df = pd.DataFrame(index_data)
        index_file = self.output_dir / f"export_index_{experiment_id or 'unknown'}.csv"
        df.to_csv(index_file, index=False, encoding='utf-8')
        
        return index_file
