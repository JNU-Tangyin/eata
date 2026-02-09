"""
统计检验模块
对消融实验结果进行统计显著性检验
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
import warnings

# 隐藏警告信息
warnings.filterwarnings('ignore')
np.seterr(all='ignore')
from itertools import combinations

class StatisticalTester:
    """
    统计检验器，负责对消融实验结果进行统计分析
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        初始化统计检验器
        
        Args:
            significance_level: 显著性水平，默认0.05
        """
        self.significance_level = significance_level
        self.confidence_level = 1 - significance_level
        
    def test_variant_significance(self, baseline_returns: np.ndarray, 
                                 variant_returns: np.ndarray,
                                 variant_name: str) -> Dict[str, Any]:
        """
        测试变体与基准的显著性差异
        
        Args:
            baseline_returns: 基准收益率序列
            variant_returns: 变体收益率序列
            variant_name: 变体名称
            
        Returns:
            Dict[str, Any]: 统计检验结果
        """
        results = {
            'variant_name': variant_name,
            'sample_sizes': {
                'baseline': len(baseline_returns),
                'variant': len(variant_returns)
            }
        }
        
        # 基础统计
        results['descriptive_stats'] = {
            'baseline_mean': np.mean(baseline_returns),
            'baseline_std': np.std(baseline_returns, ddof=1),
            'variant_mean': np.mean(variant_returns),
            'variant_std': np.std(variant_returns, ddof=1),
            'mean_difference': np.mean(variant_returns) - np.mean(baseline_returns)
        }
        
        # 配对t检验（如果样本大小相同）
        if len(baseline_returns) == len(variant_returns):
            results['paired_t_test'] = self._paired_t_test(baseline_returns, variant_returns)
        
        # 独立样本t检验
        results['independent_t_test'] = self._independent_t_test(baseline_returns, variant_returns)
        
        # Wilcoxon秩和检验（非参数）
        results['wilcoxon_test'] = self._wilcoxon_test(baseline_returns, variant_returns)
        
        # Mann-Whitney U检验
        results['mann_whitney_test'] = self._mann_whitney_test(baseline_returns, variant_returns)
        
        # Kolmogorov-Smirnov检验
        results['ks_test'] = self._ks_test(baseline_returns, variant_returns)
        
        # 效应量计算
        results['effect_size'] = self._calculate_effect_size(baseline_returns, variant_returns)
        
        return results
    
    def _paired_t_test(self, baseline: np.ndarray, variant: np.ndarray) -> Dict[str, Any]:
        """
        配对t检验
        """
        try:
            t_stat, p_value = stats.ttest_rel(variant, baseline)
            
            return {
                'statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < self.significance_level,
                'confidence_interval': self._calculate_paired_ci(baseline, variant),
                'interpretation': self._interpret_p_value(p_value)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _independent_t_test(self, baseline: np.ndarray, variant: np.ndarray) -> Dict[str, Any]:
        """
        独立样本t检验
        """
        try:
            # 先进行方差齐性检验
            levene_stat, levene_p = stats.levene(baseline, variant)
            equal_var = levene_p > self.significance_level
            
            t_stat, p_value = stats.ttest_ind(variant, baseline, equal_var=equal_var)
            
            return {
                'statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < self.significance_level,
                'equal_variance_assumed': equal_var,
                'levene_test': {
                    'statistic': float(levene_stat),
                    'p_value': float(levene_p)
                },
                'interpretation': self._interpret_p_value(p_value)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _wilcoxon_test(self, baseline: np.ndarray, variant: np.ndarray) -> Dict[str, Any]:
        """
        Wilcoxon符号秩检验（配对样本）或秩和检验
        """
        try:
            if len(baseline) == len(variant):
                # 配对样本：Wilcoxon符号秩检验
                w_stat, p_value = stats.wilcoxon(variant, baseline)
                test_type = "signed_rank"
            else:
                # 独立样本：Wilcoxon秩和检验（实际上是Mann-Whitney U）
                w_stat, p_value = stats.ranksums(variant, baseline)
                test_type = "rank_sum"
            
            return {
                'test_type': test_type,
                'statistic': float(w_stat),
                'p_value': float(p_value),
                'significant': p_value < self.significance_level,
                'interpretation': self._interpret_p_value(p_value)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _mann_whitney_test(self, baseline: np.ndarray, variant: np.ndarray) -> Dict[str, Any]:
        """
        Mann-Whitney U检验
        """
        try:
            u_stat, p_value = stats.mannwhitneyu(variant, baseline, alternative='two-sided')
            
            return {
                'statistic': float(u_stat),
                'p_value': float(p_value),
                'significant': p_value < self.significance_level,
                'interpretation': self._interpret_p_value(p_value)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _ks_test(self, baseline: np.ndarray, variant: np.ndarray) -> Dict[str, Any]:
        """
        Kolmogorov-Smirnov检验
        """
        try:
            ks_stat, p_value = stats.ks_2samp(baseline, variant)
            
            return {
                'statistic': float(ks_stat),
                'p_value': float(p_value),
                'significant': p_value < self.significance_level,
                'interpretation': self._interpret_p_value(p_value)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_effect_size(self, baseline: np.ndarray, variant: np.ndarray) -> Dict[str, float]:
        """
        计算效应量
        """
        try:
            # Cohen's d
            pooled_std = np.sqrt(((len(baseline) - 1) * np.var(baseline, ddof=1) + 
                                 (len(variant) - 1) * np.var(variant, ddof=1)) / 
                                (len(baseline) + len(variant) - 2))
            
            cohens_d = (np.mean(variant) - np.mean(baseline)) / pooled_std if pooled_std > 0 else 0.0
            
            # Glass's delta
            glass_delta = (np.mean(variant) - np.mean(baseline)) / np.std(baseline, ddof=1) if np.std(baseline, ddof=1) > 0 else 0.0
            
            return {
                'cohens_d': float(cohens_d),
                'glass_delta': float(glass_delta),
                'cohens_d_interpretation': self._interpret_cohens_d(cohens_d)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_paired_ci(self, baseline: np.ndarray, variant: np.ndarray) -> Dict[str, float]:
        """
        计算配对差值的置信区间
        """
        try:
            differences = variant - baseline
            mean_diff = np.mean(differences)
            se_diff = stats.sem(differences)
            
            # t分布的临界值
            df = len(differences) - 1
            t_critical = stats.t.ppf(1 - self.significance_level/2, df)
            
            ci_lower = mean_diff - t_critical * se_diff
            ci_upper = mean_diff + t_critical * se_diff
            
            return {
                'lower': float(ci_lower),
                'upper': float(ci_upper),
                'mean_difference': float(mean_diff)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _interpret_p_value(self, p_value: float) -> str:
        """
        解释p值
        """
        if p_value < 0.001:
            return "极显著 (p < 0.001)"
        elif p_value < 0.01:
            return "高度显著 (p < 0.01)"
        elif p_value < 0.05:
            return "显著 (p < 0.05)"
        elif p_value < 0.1:
            return "边缘显著 (p < 0.1)"
        else:
            return "不显著 (p ≥ 0.1)"
    
    def _interpret_cohens_d(self, d: float) -> str:
        """
        解释Cohen's d效应量
        """
        abs_d = abs(d)
        if abs_d < 0.2:
            return "微小效应"
        elif abs_d < 0.5:
            return "小效应"
        elif abs_d < 0.8:
            return "中等效应"
        else:
            return "大效应"
    
    def multiple_comparison_correction(self, p_values: List[float], 
                                     method: str = 'fdr_bh') -> Dict[str, Any]:
        """
        多重比较校正
        
        Args:
            p_values: p值列表
            method: 校正方法，'bonferroni', 'holm', 'fdr_bh'
            
        Returns:
            Dict[str, Any]: 校正结果
        """
        p_values = np.array(p_values)
        
        if method == 'bonferroni':
            corrected_p = p_values * len(p_values)
            corrected_p = np.minimum(corrected_p, 1.0)
        elif method == 'holm':
            # Holm-Bonferroni方法
            sorted_indices = np.argsort(p_values)
            corrected_p = np.zeros_like(p_values)
            
            for i, idx in enumerate(sorted_indices):
                correction_factor = len(p_values) - i
                corrected_p[idx] = min(1.0, p_values[idx] * correction_factor)
        elif method == 'fdr_bh':
            # Benjamini-Hochberg FDR控制
            sorted_indices = np.argsort(p_values)
            corrected_p = np.zeros_like(p_values)
            
            for i, idx in enumerate(sorted_indices):
                correction_factor = len(p_values) / (i + 1)
                corrected_p[idx] = min(1.0, p_values[idx] * correction_factor)
        else:
            raise ValueError(f"Unknown correction method: {method}")
        
        return {
            'method': method,
            'original_p_values': p_values.tolist(),
            'corrected_p_values': corrected_p.tolist(),
            'significant_after_correction': (corrected_p < self.significance_level).tolist()
        }
    
    def comprehensive_analysis(self, results_dict: Dict[str, np.ndarray], 
                             baseline_key: str = 'EATA-Full') -> Dict[str, Any]:
        """
        对所有变体进行综合统计分析
        
        Args:
            results_dict: {variant_name: returns_array} 的字典
            baseline_key: 基准变体名称
            
        Returns:
            Dict[str, Any]: 综合分析结果
        """
        if baseline_key not in results_dict:
            return {'error': f'Baseline {baseline_key} not found in results'}
        
        baseline_returns = results_dict[baseline_key]
        analysis = {
            'baseline': baseline_key,
            'total_variants': len(results_dict),
            'pairwise_tests': {},
            'summary_statistics': {}
        }
        
        # 对每个变体与基准进行比较
        p_values = []
        variant_names = []
        
        for variant_name, variant_returns in results_dict.items():
            if variant_name == baseline_key:
                continue
                
            test_result = self.test_variant_significance(
                baseline_returns, variant_returns, variant_name
            )
            analysis['pairwise_tests'][variant_name] = test_result
            
            # 收集p值用于多重比较校正
            if 'paired_t_test' in test_result and 'p_value' in test_result['paired_t_test']:
                p_values.append(test_result['paired_t_test']['p_value'])
                variant_names.append(variant_name)
        
        # 多重比较校正
        if p_values:
            analysis['multiple_comparison'] = {
                'bonferroni': self.multiple_comparison_correction(p_values, 'bonferroni'),
                'holm': self.multiple_comparison_correction(p_values, 'holm'),
                'fdr_bh': self.multiple_comparison_correction(p_values, 'fdr_bh')
            }
            analysis['multiple_comparison']['variant_names'] = variant_names
        
        # 汇总统计
        significant_variants = []
        for variant_name, test_result in analysis['pairwise_tests'].items():
            if 'paired_t_test' in test_result and test_result['paired_t_test'].get('significant', False):
                significant_variants.append(variant_name)
        
        analysis['summary_statistics'] = {
            'total_comparisons': len(analysis['pairwise_tests']),
            'significant_variants': significant_variants,
            'significant_count': len(significant_variants),
            'significance_rate': len(significant_variants) / len(analysis['pairwise_tests']) * 100 if analysis['pairwise_tests'] else 0
        }
        
        return analysis
    
    def export_statistical_results(self, analysis: Dict[str, Any], output_file: str) -> None:
        """
        导出统计分析结果到CSV
        """
        if 'pairwise_tests' not in analysis:
            return
        
        # 准备数据
        rows = []
        for variant_name, test_result in analysis['pairwise_tests'].items():
            row = {'variant': variant_name}
            
            # 描述性统计
            if 'descriptive_stats' in test_result:
                stats_data = test_result['descriptive_stats']
                row.update({
                    'baseline_mean': stats_data.get('baseline_mean', 0),
                    'variant_mean': stats_data.get('variant_mean', 0),
                    'mean_difference': stats_data.get('mean_difference', 0)
                })
            
            # t检验结果
            if 'paired_t_test' in test_result:
                t_test = test_result['paired_t_test']
                row.update({
                    't_statistic': t_test.get('statistic', 0),
                    't_p_value': t_test.get('p_value', 1),
                    't_significant': t_test.get('significant', False)
                })
            
            # 效应量
            if 'effect_size' in test_result:
                effect = test_result['effect_size']
                row.update({
                    'cohens_d': effect.get('cohens_d', 0),
                    'effect_interpretation': effect.get('cohens_d_interpretation', '')
                })
            
            rows.append(row)
        
        # 保存到CSV
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"统计分析结果已保存到: {output_file}")
