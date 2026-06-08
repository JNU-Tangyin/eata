"""
从detailed_outputs提取所有策略的真实数据
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

print('='*80)
print('提取所有策略的波动率和收益率数据')
print('='*80)

detailed_dir = Path('results/comparison_study/baseline_100stocks/detailed_outputs')
files = list(detailed_dir.glob('*.csv'))

print(f'\n找到 {len(files)} 个详细数据文件')

# 存储每个策略的数据
strategy_data = defaultdict(list)

for i, file in enumerate(files):
    if i % 50 == 0:
        print(f'处理进度: {i}/{len(files)}')
    
    try:
        # 解析文件名
        parts = file.stem.split('-')
        ticker = parts[0]
        strategy = parts[1]
        
        # 读取数据
        df = pd.read_csv(file)
        
        if 'portfolio_value' not in df.columns or len(df) < 2:
            continue
        
        pv = df['portfolio_value'].values
        
        # 计算收益率
        returns = np.diff(pv) / pv[:-1]
        
        # 年化收益率
        total_return = (pv[-1] - pv[0]) / pv[0]
        years = len(pv) / 252
        annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # 年化波动率
        volatility = np.std(returns) * np.sqrt(252)
        
        strategy_data[strategy].append({
            'ticker': ticker,
            'annual_return': annual_return * 100,
            'volatility': volatility * 100
        })
        
    except Exception as e:
        continue

print(f'\n处理完成！')

# 保存每个策略的数据
output_dir = Path('results/comparison_study')
for strategy, data in strategy_data.items():
    df = pd.DataFrame(data)
    output_file = output_dir / f'{strategy}_62stocks_volatility.csv'
    df.to_csv(output_file, index=False)
    mean_return = df['annual_return'].mean()
    mean_vol = df['volatility'].mean()
    print(f'✅ {strategy}: {len(df)} 支股票, 平均收益={mean_return:.2f}%, 平均波动率={mean_vol:.2f}%')

print(f'\n数据已保存到: {output_dir}')
