#!/usr/bin/env python3
"""
一键生成论文图表和表格
One-Click Paper Figures and Tables Generator

使用方法:
python generate_paper_outputs.py

输出:
- figures/ 目录下的所有图表 (PDF + PNG)
- tables/ 目录下的所有LaTeX表格
"""

import subprocess
import sys
from pathlib import Path


def main():
    """一键生成所有论文输出"""
    base_dir = Path("/Users/zjt/Desktop/EATA-RL-main")
    
    print("🚀 开始生成论文图表和表格...")
    print("=" * 60)
    
    # 1. 运行实验 (如果需要)
    print("📊 步骤 1: 检查实验数据...")
    experiment_results = base_dir / "experiment_results"
    if not experiment_results.exists() or len(list(experiment_results.glob("*.csv"))) == 0:
        print("⚠️ 未找到实验数据，运行快速实验...")
        try:
            subprocess.run([
                sys.executable, "run_experiments.py", 
                "--mode", "single",
                "--tickers", "AAPL", "MSFT", "GOOGL",
                "--strategies", "eata", "buy_and_hold", "macd", "transformer", "ppo",
                "--runs", "1"
            ], cwd=base_dir, check=True)
            print("✅ 实验数据生成完成")
        except subprocess.CalledProcessError as e:
            print(f"❌ 实验运行失败: {e}")
            return False
    else:
        print("✅ 找到现有实验数据")
    
    # 2. 生成图表和表格
    print("\n🎨 步骤 2: 生成图表和表格...")
    try:
        subprocess.run([
            sys.executable, "experiment_pipeline.py",
            "--mode", "all"
        ], cwd=base_dir, check=True)
        print("✅ 图表和表格生成完成")
    except subprocess.CalledProcessError as e:
        print(f"❌ 图表生成失败: {e}")
        return False
    
    # 3. 检查输出
    print("\n📁 步骤 3: 检查输出文件...")
    
    figures_dir = base_dir / "figures"
    tables_dir = base_dir / "tables"
    
    if figures_dir.exists():
        figure_files = list(figures_dir.glob("*.pdf")) + list(figures_dir.glob("*.png"))
        print(f"📊 生成图表: {len(figure_files)} 个文件")
        for f in sorted(figure_files):
            print(f"  - {f.name}")
    
    if tables_dir.exists():
        table_files = list(tables_dir.glob("*.tex"))
        print(f"📝 生成表格: {len(table_files)} 个文件")
        for f in sorted(table_files):
            print(f"  - {f.name}")
    
    print("\n" + "=" * 60)
    print("🎉 论文图表和表格生成完成！")
    print(f"📁 图表目录: {figures_dir}")
    print(f"📁 表格目录: {tables_dir}")
    print("\n💡 使用提示:")
    print("  - PDF图表适合插入LaTeX论文")
    print("  - PNG图表适合预览和演示")
    print("  - TEX表格可直接插入LaTeX文档")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
