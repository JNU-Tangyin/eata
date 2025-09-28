'''
run preditor will return all the predicted action for tickers in watchlist 
支持NEMoTS符号回归预测器集成
'''
import datetime
import pandas as pd
from data import MODEL_PATH, DataStorage
from preprocess import Preprocessor 
from retrying import retry
import pysnooper
import globals

# NEMoTS集成
try:
    from nemots_adapter import NEMoTSPredictor
    NEMOTS_AVAILABLE = True
except ImportError:
    print("NEMoTS适配器不可用，使用传统预测器")
    NEMOTS_AVAILABLE = False

MODEL_PATH = ""

try:
    from .bandwagon import Bandwagon
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        from bandwagon import Bandwagon
    except ImportError:
        print("Bandwagon模块不可用")
        Bandwagon = None

class Predictor:

    def __init__(self, file_name: str = "", use_nemots: bool = True):
        """
        初始化预测器
        
        Args:
            file_name: 模型文件名（传统预测器使用）
            use_nemots: 是否使用NEMoTS预测器
        """
        self.ds = DataStorage()
        self.use_nemots = use_nemots and NEMOTS_AVAILABLE
        
        if self.use_nemots:
            print("🧠 初始化NEMoTS预测器...")
            # 默认使用简化版本，更稳定
            self.nemots_predictor = NEMoTSPredictor(lookback=20, use_full_nemots=False)
            self.is_trained = False
        else:
            print("📊 使用传统Bandwagon预测器...")
            if Bandwagon is not None:
                try:
                    df = pd.read_excel("000016closeweight(5).xls", dtype={'code':'str'}, header = 0)
                    self.bw = Bandwagon(df)
                except Exception as e:
                    print(f"传统预测器初始化失败: {e}")
                    self.bw = None
            else:
                self.bw = None

    def fit(self, df: pd.DataFrame):
        """训练预测器"""
        if self.use_nemots:
            try:
                self.nemots_predictor.fit(df)
                self.is_trained = True
                print("✅ NEMoTS预测器训练完成")
            except Exception as e:
                print(f"❌ NEMoTS训练失败: {e}")
                self.is_trained = False
        else:
            print("传统预测器无需额外训练")

    def predict(self, state=None, df=None):
        """
        预测交易动作
        
        Args:
            state: 传统预测器使用的状态
            df: NEMoTS预测器使用的数据
            
        Returns:
            int: 交易动作 (1: 买入, 0: 持有, -1: 卖出)
        """
        if self.use_nemots and self.is_trained and df is not None:
            try:
                action = self.nemots_predictor.predict_action(df)
                print(f"🧠 NEMoTS预测动作: {action}")
                return action
            except Exception as e:
                print(f"❌ NEMoTS预测失败: {e}")
                # 回退到传统方法
        
        # 传统预测方法
        if self.bw is not None:
            action = 1 if self.bw.vote() > 40 else -1
            print(f"📊 传统预测动作: {action}")
        else:
            action = 0  # 默认持有
            print("⚠️  无可用预测器，默认持有")
        
        self.ds.save_action()
        return action 

    def latest_actions(self)->list[tuple]:
        ''' pretty much the same as 'watch(·)'
            w.r.t. each ticker in watchlist, get the trend(t). latest action is the last row of the dataframe
            this func can also be replaced by:
                result = [(self.end_time, t, t.iloc[-1].action) for t in self.trends(WatchList)]
                df = pd.DataFrame(result,columns=['date','ticker','action'],dtype=int)
        '''
        latest_action = lambda t: self.trend(t).iloc[-1].action
        result = [(self.end_time, t,latest_action(t)) for t in watchlist]
        df = pd.DataFrame(result,columns=['date','ticker','action'],dtype=int)
        self.ds.save_predicted(df[df.action.isin([-1,1])], if_exists = 'append') # save only action in [-1,1]
        return result # or, df as 'st.table(df)' in visualize.py
    
    def save_action(self, a, price):
        '''将本次决策保存在predicted
        a - 决策
        price - 当前close价
        '''
        pass

'''
buy or sell sz50etf by predicting its constituent
'''

if __name__ == "__main__":
    print("🚀 启动NEMoTS预测系统")
    print("=" * 50)
    
    # 创建NEMoTS预测器
    predictor = Predictor(use_nemots=True)
    
    print(f"✅ 预测器初始化完成")
    print(f"   使用NEMoTS: {predictor.use_nemots}")
    
    # 创建测试数据进行预测演示
    import numpy as np
    test_data = pd.DataFrame({
        'open': [100 + i + np.random.randn()*0.1 for i in range(30)],
        'high': [102 + i + np.random.randn()*0.1 for i in range(30)],
        'low': [98 + i + np.random.randn()*0.1 for i in range(30)],
        'close': [101 + i + np.random.randn()*0.1 for i in range(30)],
        'volume': [1000 + i*10 for i in range(30)]
    })
    # 添加amount字段（成交额 = 成交量 * 收盘价）
    test_data['amount'] = test_data['volume'] * test_data['close']
    
    print("\n📊 开始NEMoTS预测演示...")
    try:
        # 训练NEMoTS
        predictor.fit(test_data)
        
        # 进行预测
        action = predictor.predict(df=test_data.tail(10))
        action_name = {-1: '卖出', 0: '持有', 1: '买入'}[action]
        
        print(f"✅ NEMoTS预测结果: {action} ({action_name})")
        
    except Exception as e:
        print(f"⚠️ 预测过程出错: {e}")
    
    print("\n🎉 NEMoTS预测系统运行完成！")
