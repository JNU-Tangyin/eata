import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from globals import summary, test_result
from pathlib import Path

def load_css(file_name:str = "streamlit.css")->None:
    """
    Function to load and render a local stylesheet
    """
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


class WebServer:
    def __init__(self):
        dirs = os.listdir(f"{test_result}")  # Test目录下的子目录
        self.agents = [d for d in dirs if not os.path.isfile(d)]
        self.obj = self.agents[2]    # 这行开始改成循环，即可遍历目录下所有的策略的测试结果

        data_folder = Path(f"{test_result}/{self.obj}")
        files = os.listdir(data_folder)  # 目录下所有文件
        files = [f for f in files if os.path.splitext(f)[1] == '.csv']  # 只选择 .csv 文件,
        files.remove(summary) if summary in files else files
        print(f"Testing strategy {self.obj} with {files}")
        self.perf = pd.read_csv(data_folder/f'{summary}', index_col= 0)
        self.dfs = [pd.read_csv(data_folder/f'{f}', index_col=0) for f in files]
        # 下一步改为根据agents，遍历指定目录

    def process(self, df, days = 250):
        d = df.copy().tail(days)   #选最后250交易日的数据(大致1年），避免最后做出的图过于拥挤。
        d['date'] = pd.to_datetime(d.date)
        self.data_all = d.shape[0]
        # self.df2 = self.df.set_index('date')   #使用streamlit接口画图需要以date作为索引
        self.ticker = d.ticker.iloc[0]
        self.record = str(d.shape[0])  # 一共几天交易日的数据

        # 乘上close，使两天资产线和股票的收盘价线同一起点
        d['change_wo_short'].iloc[0] = 1  # 第一行赋值为1，以这个为起点，后面是相对于上一天比率
        d['change_w_short'].iloc[0] = 1    
        self.asset_wo_short = d.close.iloc[0] * d.change_wo_short.cumprod()
        self.asset_w_short = d.close.iloc[0] * d.change_w_short.cumprod()


        # 计算最后一天的资产
        self.asset_wo = d.close.iloc[0]* self.asset_wo_short.iloc[-1]      #不做空 最新一日资产
        # self.chg_wo = round(df.change_wo_short.iloc[-1],2)
        self.asset_w = d.close.iloc[0]* self.asset_w_short.iloc[-1]     #做空 最新一日资产
        # self.chg_w = round(df.change_w_short.iloc[-1],2)

        # 分别挑出action为买或卖的收盘价，以便于画图标注
        self.action_days = d[d.real_action.isin([1,-1])] # `date` is the index

        # test = d.real_action * d.close
        # self.buy_actions = test.apply(lambda x: x if x>0 else None)     # sell的位置为None，维持序列长度不变
        # self.sell_actions = test.apply(lambda x: -x if x<0 else None)   # buy的位置为None，维持序列长度不变
        # self.buy_ = d[d.real_action== 1]
        # self.sell_ = d[d.real_action== -1]

        self.tick_spacing = 10 #设置横坐标日期的间隙，避免重叠
        self.df = d

    def run(self):
        from itertools import count
        st.title(f"Testing {self.obj}")
        st.header("Summary")
        # histograms of metrics in a 2*3 grid
        f,axes = plt.subplots(nrows=2,ncols=3,figsize=(15,8))
        my_list = ['accuracy','precision','recall', 'f1_score', 'fpr','annual_return']
        for i,m in zip(count(start = 0, step = 1), my_list):
            f.axes[i].hist(self.perf[m], bins = 50, alpha = 0.5) 
            f.axes[i].set_title(m)
        st.pyplot(f)

        st.dataframe(self.perf)
        print(self.perf)
        

        [st.sidebar.text(a) for a in self.agents]

        for df in self.dfs:
            self.process(df)
            # st.subheader("CLOSE & ASSET GRAPH")
            st.metric(label="Ticker", value = self.ticker, delta = 'latest '+ self.record +' days')
            
            # 年化利率计算
            col1, col2 = st.columns(2)
            col1.metric(label="Annual return - Short",
                        value = round(self.asset_w / self.data_all * 250, 2)) #一年的交易日250天
            col2.metric(label="Annual return - No Short",
                        value = round(self.asset_wo / self.data_all * 250, 2))
            
            # 确定绘图的地板和天花板
            floor = min(self.df.close.min(), self.asset_w_short.min(), self.asset_wo_short.min())
            ceiling = max(self.df.close.max(), self.asset_w_short.max(), self.asset_wo_short.max())

            # 画图 close+asset+action
            fig = plt.figure(figsize=(15,8))
            ax1 = fig.add_subplot(111)
            ax1.plot(self.df.date, self.asset_wo_short, 'm-.', label="without short")
            ax1.plot(self.df.date, self.asset_w_short, 'g-', label="with short")
            ax1.legend(loc=1)
            ax1.set_ylabel('Assets change/Close')
            ax2 = ax1 #.twinx()
            ax2.plot(self.df.date, self.df.close, 'b', label = "Price", alpha=0.1)
            ax2.fill_between(self.df.date, floor, self.df.close, color = 'b', alpha = 0.1)
            
            # 1 采用灰底来表示做空时间，白底做多
            d1, d2 = self.action_days.iloc[:-1], self.action_days.iloc[1:]  # 获取索引并配对拼接
            for (a,b) in zip(d1.iterrows(),d2.iterrows()):
                if a[1].real_action == -1:
                    ax2.fill_between(self.df.date, floor, ceiling, (a[1].date <self.df.date) & (self.df.date<b[1].date), color = "k", alpha = 0.1)

            # 2 用^v在close上标记买入或卖出
            # ax2.scatter(self.df.date, self.buy_actions, label='buy', color='red', marker="^")
            # ax2.scatter(self.df.date, self.sell_actions, label='sell', color='green', marker ="v")

            ax2.legend(loc=2)
            # ax2.set_ylabel('Close')
            ax2.set_xlabel('Date')
            st.pyplot(fig)

if __name__=='__main__':
    WebServer().run()