import pandas as pd
import numpy as np

# 核心改动：直接导入我们改造后的Agent
from agent import Agent

class Predictor:
    def __init__(self):
        """
        新版预测器，核心职责是初始化和调用Agent。
        """
        # 注意：Agent的初始化可能需要一个股票列表df，这里我们暂时传入一个空的
        # 在实际应用中，可以传入self.ds.get_watchlist()等
        self.agent = Agent(df=pd.DataFrame())
        print("🤖 新版 Predictor 初始化完成，内含新版 EATA Agent。")

    def predict(self, df: pd.DataFrame) -> int:
        """
        使用Agent对单个数据窗口进行预测。

        Args:
            df (pd.DataFrame): 包含[open, high, low, close, volume, amount]的单个股票数据窗口。

        Returns:
            int: 交易动作 (1: 买入, 0: 持有, -1: 卖出)
        """
        print("\n[Predictor] -> 调用 Agent.criteria 进行决策...")
        action = self.agent.criteria(df)
        action_name = {-1: '卖出', 0: '持有', 1: '买入'}[action]
        print(f"[Predictor] <- Agent决策结果: {action} ({action_name})")
        return action

    def run_for_all_stocks(self, stock_data_dict: dict) -> dict:
        """
        为一个字典中的所有股票数据运行预测。
        这是未来整合到main.py的示例。

        Args:
            stock_data_dict (dict): key为股票代码, value为该股票的DataFrame。

        Returns:
            dict: key为股票代码, value为预测的交易动作。
        """
        results = {}
        print("\n--- 开始为多支股票批量预测 ---")
        for ticker, df in stock_data_dict.items():
            print(f"\n--- 正在预测: {ticker} ---")
            try:
                # 为每支股票独立调用predict，Agent内部会处理语法树继承
                action = self.predict(df)
                results[ticker] = action
            except Exception as e:
                print(f"❌ 预测 {ticker} 失败: {e}")
                results[ticker] = 0 # 出错则持有
        print("\n--- 批量预测完成 ---")
        return results


if __name__ == "__main__":
    print("🚀 启动 EATA 项目核心功能演示")
    print("======================================================")
    print("本脚本现在是项目的主要功能入口和测试平台。")
    print("它将演示如何使用新的Agent对数据进行预测。")
    print("======================================================")

    # 1. 初始化Predictor (它会自动创建新的Agent)
    predictor = Predictor()

    # 2. 创建模拟数据 (与sliding_window_nemots.py中的测试数据类似)
    #    这代表了您为单支股票准备的、用于输入模型的数据。
    print("\n[Main] 准备模拟输入数据...")
    test_data = pd.DataFrame({
        'open': [100 + i*0.1 + np.random.randn()*0.1 for i in range(40)],
        'high': [102 + i*0.1 + np.random.randn()*0.1 for i in range(40)],
        'low': [98 + i*0.1 + np.random.randn()*0.1 for i in range(40)],
        'close': [101 + i*0.1 + np.random.randn()*0.1 for i in range(40)],
        'volume': [1000 + i*10 for i in range(40)]
    })
    test_data['amount'] = test_data['volume'] * test_data['close']
    print(f"[Main] 模拟数据创建完成 ({len(test_data)}条记录)。")

    # 3. 执行预测
    #    在第一次调用时，Agent会使用“重量级”参数进行“冷启动”训练。
    print("\n[Main] === 第一次预测 (冷启动) ===")
    predictor.predict(df=test_data)

    # 4. 模拟数据更新，再次执行预测
    #    在第二次调用时，Agent会检测到已有的语法树，并使用“轻量级”参数进行“热启动”迭代。
    print("\n[Main] === 第二次预测 (热启动/继承) ===")
    # 模拟时间推移，数据发生变化
    updated_data = test_data.iloc[5:].copy() 
    updated_data = pd.concat([updated_data, test_data.tail(5)], ignore_index=True) # 简单模拟数据滚动
    predictor.predict(df=updated_data)

    print("\n🎉 EATA 项目核心功能演示完成！")