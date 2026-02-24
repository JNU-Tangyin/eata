"""

EATA消融实验主入口脚本

运行所有6个变体的消融实验

"""



import sys

import os

import pandas as pd

import numpy as np

from datetime import datetime

import json

from pathlib import Path

import warnings



# 隐藏所有警告信息

warnings.filterwarnings('ignore')

pd.options.mode.chained_assignment = None  # 隐藏pandas链式赋值警告



# 添加项目根目录到路径

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))



# 导入配置

from configs.ablation_config import ABLATION_CONFIGS, BASE_CONFIG, EXPERIMENT_SETTINGS, CSV_CONFIG

from configs.experiment_settings import DATA_PATHS, RESULT_PATHS, EATA_DEFAULT_PARAMS



# 导入所有变体

from variants import (

    EATAFull, EATANoNN, EATANoMem, 

    EATASimple, EATANoExplore, EATANoMCTS

)



class AblationStudyRunner:

    """

    EATA消融实验运行器

    """

    

    def __init__(self):

        # 设置消融实验环境变量

        import os

        os.environ['ABLATION_EXPERIMENT_MODE'] = 'true'

        print("🔧 [消融实验] 环境变量ABLATION_EXPERIMENT_MODE已设置为true")

        

        # 完整消融实验：运行6个变体
        self.variants = {
            'EATA-Full': EATAFull,
            'EATA-NoNN': EATANoNN,
            'EATA-NoMCTS': EATANoMCTS,
            'EATA-NoMem': EATANoMem,
            'EATA-NoExplore': EATANoExplore,
            'EATA-Simple': EATASimple,
        }

        

        self.results = []

        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        

        # 确保结果目录存在

        self._ensure_directories()

        

    def _ensure_directories(self):

        """确保所有必要的目录存在"""

        for path in RESULT_PATHS.values():

            path.mkdir(parents=True, exist_ok=True)

            

    def _load_real_stock_data(self, ticker: str):

        """加载真实股票数据"""

        print(f"加载真实股票数据: {ticker}")

        

        # 真实股票数据路径

        data_path = f"d:/下载/分散的20支股票/分散的20支股票/{ticker}.csv"

        

        try:

            # 读取CSV文件

            df = pd.read_csv(data_path)

            

            # 标准化列名

            df.columns = df.columns.str.lower().str.replace(' ', '_')

            

            # 重命名列以匹配期望格式

            column_mapping = {

                'date': 'date',

                'open': 'open', 

                'high': 'high',

                'low': 'low',

                'close': 'close',

                'volume': 'volume'

            }

            

            df = df.rename(columns=column_mapping)

            

            # 转换日期格式

            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')

            

            # 计算amount列（如果不存在）

            if 'amount' not in df.columns:

                df['amount'] = df['close'] * df['volume']

            

            # 按日期排序

            df = df.sort_values('date').reset_index(drop=True)

            

            # 选择最近几年的数据（避免数据过多）

            df = df[df['date'] >= '2020-01-01'].copy()

            

            print(f"成功加载 {ticker} 数据: {len(df)} 条记录")

            print(f"   时间范围: {df['date'].min()} 到 {df['date'].max()}")

            

            return df

            

        except Exception as e:

            print(f"加载 {ticker} 数据失败: {e}")

            raise Exception(f"无法加载真实股票数据 {ticker}，请检查数据文件是否存在")



    def load_all_stock_data(self):

        """

        加载所有20支股票的真实数据

        """

        print("加载20支股票的真实数据...")

        

        # 完整消融实验：10支分散行业股票
        stock_tickers = [
            'BA',     # 航空航天/工业
            'JNJ',    # 医疗保健/制药
            'KO',     # 消费品/饮料
            'NVDA',   # 科技/半导体
            'XOM',    # 能源/石油
            'JPM',    # 金融/银行
            'CAT',    # 工业/机械
            'COST',   # 零售/批发
            'NFLX',   # 媒体/流媒体
            'GE',     # 综合工业
        ]

        

        stocks_data = {}

        

        for ticker in stock_tickers:

            try:

                stock_data = self._load_real_stock_data(ticker)

                

                # 检查数据量是否足够

                if len(stock_data) < 500:

                    print(f"   ⚠️ {ticker}: 数据不足 ({len(stock_data)} 天)，跳过")

                    continue

                

                stocks_data[ticker] = stock_data

                print(f"   ✅ {ticker}: 成功加载 {len(stock_data)} 天数据")

                

            except Exception as e:

                print(f"   ❌ {ticker}: 加载失败 - {str(e)}")

                continue

        

        print(f"成功加载 {len(stocks_data)} 支股票数据")

        return stocks_data

    

    def run_single_variant(self, variant_name, variant_class, train_df, test_df, ticker="TEST"):

        """

        运行单个变体的实验

        """

        print(f"\n🧪 运行消融实验: {variant_name}")

        print(f"   描述: {ABLATION_CONFIGS.get(variant_name, {}).get('description', '基准模型')}")

        

        try:

            # 创建变体实例

            

            # 创建变体实例

            print(f"🔄 运行变体 {variant_name} 在股票 {ticker}...")

            

            try:

                # 运行变体实验

                full_data = pd.concat([train_df, test_df], ignore_index=True)

                variant_instance = variant_class(full_data)

                result = variant_instance.run_backtest(train_df, test_df, ticker)

                

                # 添加实验元数据

                result.update({

                    'variant': variant_name,

                    'ticker': ticker,

                    'experiment_id': self.experiment_id,

                    'timestamp': datetime.now().isoformat(),

                    'error': ''

                })

                

                return result

                

            except Exception as e:

                error_msg = str(e)

                print(f"❌ {variant_name} - {ticker} 失败: {error_msg}")

                

                # 记录失败结果

                error_result = {

                    'variant': variant_name,

                    'ticker': ticker,

                    'experiment_id': self.experiment_id,

                    'timestamp': datetime.now().isoformat(),

                    'error': error_msg,

                    'annual_return': 0.0,

                    'sharpe_ratio': 0.0,

                    'max_drawdown': 0.0,

                    'win_rate': 0.0,

                    'volatility': 0.0,

                    'rl_reward': 0.0

                }

                return error_result

            

        except Exception as e:

            print(f"{variant_name} 实验失败: {str(e)}")

            return {

                'variant': variant_name,

                'ticker': ticker,

                'error': str(e),

                'experiment_id': self.experiment_id,

                'timestamp': datetime.now().isoformat()

            }

    

    def run_all_experiments(self):

        """

        运行所有消融实验 - 使用20支真实股票数据

        """

        print("开始EATA消融实验")

        print(f"   实验ID: {self.experiment_id}")

        print(f"   变体数量: {len(self.variants)}")

        

        # 加载所有股票数据

        stocks_data = self.load_all_stock_data()

        

        if len(stocks_data) == 0:

            print("❌ 没有可用的股票数据，退出实验")

            return

        

        print(f"   股票数量: {len(stocks_data)}")

        print(f"   总实验数: {len(self.variants)} × {len(stocks_data)} = {len(self.variants) * len(stocks_data)}")

        

        # 运行所有变体在所有股票上

        total_experiments = len(self.variants) * len(stocks_data)

        completed_experiments = 0

        

        for variant_name, variant_class in self.variants.items():

            print(f"\n{'='*60}")

            print(f"🔬 开始变体: {variant_name}")

            print(f"{'='*60}")

            

            for ticker, stock_data in stocks_data.items():

                # 分割训练和测试数据 (80% 训练, 20% 测试)

                split_point = int(len(stock_data) * 0.8)

                train_df = stock_data[:split_point].copy()

                test_df = stock_data[split_point:].copy()

                

                print(f"\n📊 {variant_name} - {ticker}")

                print(f"   数据分割: 训练 {len(train_df)} 天, 测试 {len(test_df)} 天")

                

                # 运行单个实验

                result = self.run_single_variant(variant_name, variant_class, train_df, test_df, ticker)

                self.results.append(result)

                

                completed_experiments += 1

                progress = (completed_experiments / total_experiments) * 100

                print(f"   📈 总进度: {completed_experiments}/{total_experiments} ({progress:.1f}%)")

                

        # 保存结果

        self.save_results()

        

        # 生成报告

        self.generate_report()

        

        print(f"\n消融实验完成！")

        print(f"   结果保存在: {RESULT_PATHS['csv_results_dir']}")

    def save_results(self):

        """

        保存实验结果

        """

        print("\n保存实验结果...")

        

        # 保存原始JSON结果

        json_file = RESULT_PATHS['raw_results_dir'] / f"ablation_results_{self.experiment_id}.json"

        with open(json_file, 'w', encoding='utf-8') as f:

            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)

        

        # 生成CSV汇总表

        self.generate_csv_summary()

        

        print(f"   原始结果: {json_file}")

    

    def generate_csv_summary(self):

        """

        生成CSV格式的结果汇总

        """

        summary_data = []

        

        for result in self.results:

            # 修复：检查error字段是否为空字符串或不存在，而不是检查键是否存在

            has_error = result.get('error', '') != ''

            

            if not has_error:

                # 正常结果：提取所有指标

                summary_data.append({

                    'variant': result.get('variant', 'Unknown'),

                    'ticker': result.get('ticker', 'Unknown'),

                    'annual_return': result.get('annual_return', ''),

                    'sharpe_ratio': result.get('sharpe_ratio', ''),

                    'max_drawdown': result.get('max_drawdown', ''),

                    'win_rate': result.get('win_rate', ''),

                    'volatility': result.get('volatility', ''),

                    'rl_reward': result.get('rl_reward', ''),

                    'experiment_id': self.experiment_id,

                    'timestamp': result.get('timestamp', ''),

                    'error': ''

                })

            else:

                # 错误结果：指标为空

                summary_data.append({

                    'variant': result.get('variant', 'Unknown'),

                    'ticker': result.get('ticker', 'Unknown'),

                    'annual_return': '',

                    'sharpe_ratio': '',

                    'max_drawdown': '',

                    'win_rate': '',

                    'volatility': '',

                    'rl_reward': '',

                    'experiment_id': self.experiment_id,

                    'timestamp': result.get('timestamp', ''),

                    'error': result.get('error', 'Unknown error')

                })

        

        # 保存CSV

        summary_df = pd.DataFrame(summary_data)

        csv_file = RESULT_PATHS['csv_results_dir'] / f"performance_summary_{self.experiment_id}.csv"

        summary_df.to_csv(csv_file, index=False, encoding='utf-8')

        

        print(f"   CSV汇总: {csv_file}")

    

    def generate_report(self):

        """

        生成实验报告

        """

        print("\n生成实验报告...")

        

        successful_results = [r for r in self.results if not r.get('error', '')]

        failed_results = [r for r in self.results if r.get('error', '')]

        

        # 按变体统计结果

        variant_stats = {}

        for result in successful_results:

            variant = result.get('variant', 'Unknown')

            if variant not in variant_stats:

                variant_stats[variant] = {

                    'count': 0,

                    'annual_returns': [],

                    'sharpe_ratios': [],

                    'max_drawdowns': [],

                    'win_rates': [],

                    'volatilities': []

                }

            

            variant_stats[variant]['count'] += 1

            variant_stats[variant]['annual_returns'].append(result.get('annual_return', 0))

            variant_stats[variant]['sharpe_ratios'].append(result.get('sharpe_ratio', 0))

            variant_stats[variant]['max_drawdowns'].append(result.get('max_drawdown', 0))

            variant_stats[variant]['win_rates'].append(result.get('win_rate', 0))

            variant_stats[variant]['volatilities'].append(result.get('volatility', 0))

        

        report = f"""

# EATA消融实验报告



## 实验概览

- 实验ID: {self.experiment_id}

- 实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

- 总变体数: {len(self.variants)}

- 成功运行: {len(successful_results)}

- 运行失败: {len(failed_results)}



## 变体配置

"""

        

        for variant_name in self.variants.keys():

            config = ABLATION_CONFIGS.get(variant_name, BASE_CONFIG)

            report += f"\n### {variant_name}\n"

            report += f"- 描述: {config.get('description', 'N/A')}\n"

            report += f"- 假设: {config.get('hypothesis', 'N/A')}\n"

        

        if successful_results:

            report += "\n## 性能结果\n"

            report += "| 变体 | 年化收益 | 夏普比率 | 最大回撤 | 胜率 |\n"

            report += "|------|----------|----------|----------|------|\n"

            

            for result in successful_results:

                variant = result.get('variant', 'Unknown')

                annual_return = result.get('annual_return', 0)

                sharpe_ratio = result.get('sharpe_ratio', 0)

                max_drawdown = result.get('max_drawdown', 0)

                win_rate = result.get('win_rate', 0)

                report += f"| {variant} | {annual_return:.4f} | {sharpe_ratio:.4f} | {max_drawdown:.4f} | {win_rate:.4f} |\n"

        

        if failed_results:

            report += "\n## 失败的实验\n"

            for result in failed_results:

                variant = result.get('variant', 'Unknown')

                error = result.get('error', 'Unknown error')

                report += f"- {variant}: {error}\n"

        

        # 保存报告

        report_file = RESULT_PATHS['processed_results_dir'] / f"experiment_report_{self.experiment_id}.md"

        with open(report_file, 'w', encoding='utf-8') as f:

            f.write(report)

        

        print(f"   实验报告: {report_file}")



def main():

    """

    主函数

    """

    print("EATA消融实验系统")

    print("=" * 50)

    

    # 创建实验运行器

    runner = AblationStudyRunner()

    

    # 运行所有实验

    runner.run_all_experiments()

    

    print("\n消融实验系统运行完成！")



if __name__ == "__main__":

    main()

