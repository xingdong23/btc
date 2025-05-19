"""
市场数据查看页面

提供交易对的实时行情和历史数据查看功能。
"""
import os
import sys
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exchange.gateio import GateIOExchange
from utils.logger import log


def render_market_data_page():
    """渲染市场数据页面"""
    st.title("市场数据")
    
    # 初始化交易所接口
    exchange = GateIOExchange()
    
    # 交易对选择
    col1, col2 = st.columns([3, 1])
    
    with col1:
        symbol = st.text_input("交易对", "BTC/USDT")
    
    with col2:
        timeframe = st.selectbox(
            "K线周期",
            ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            index=4  # 默认1h
        )
    
    # 日期范围选择
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "开始日期",
            datetime.now() - timedelta(days=7)
        )
    
    with col2:
        end_date = st.date_input(
            "结束日期",
            datetime.now()
        )
    
    # 加载数据按钮
    if st.button("加载数据", type="primary"):
        with st.spinner("正在加载数据..."):
            try:
                # 格式化日期
                start_date_str = start_date.strftime("%Y%m%d")
                end_date_str = end_date.strftime("%Y%m%d")
                
                # 获取历史数据
                data = exchange.get_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000),
                    limit=1000
                )
                
                if data is None or data.empty:
                    st.error("无法获取数据，请检查交易对和日期范围")
                    return
                
                # 过滤日期范围
                data = data[data.index >= pd.Timestamp(start_date)]
                data = data[data.index <= pd.Timestamp(end_date)]
                
                # 显示数据统计
                st.success(f"成功加载 {len(data)} 根K线数据")
                
                # 显示K线图
                st.subheader("K线图")
                
                fig = go.Figure(data=[go.Candlestick(
                    x=data.index,
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name='K线'
                )])
                
                fig.update_layout(
                    title=f'{symbol} {timeframe} K线图',
                    xaxis_title='日期',
                    yaxis_title='价格',
                    height=600,
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 显示成交量
                st.subheader("成交量")
                
                volume_fig = go.Figure(data=[go.Bar(
                    x=data.index,
                    y=data['volume'],
                    name='成交量'
                )])
                
                volume_fig.update_layout(
                    title=f'{symbol} {timeframe} 成交量',
                    xaxis_title='日期',
                    yaxis_title='成交量',
                    height=300
                )
                
                st.plotly_chart(volume_fig, use_container_width=True)
                
                # 显示数据表格
                st.subheader("数据表格")
                
                # 格式化数据
                display_data = data.copy()
                display_data.index = display_data.index.strftime('%Y-%m-%d %H:%M:%S')
                display_data = display_data.reset_index()
                display_data.columns = ['时间', '开盘价', '最高价', '最低价', '收盘价', '成交量']
                
                st.dataframe(display_data, use_container_width=True)
                
                # 显示市场统计
                st.subheader("市场统计")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("最新价格", f"{data['close'].iloc[-1]:.2f}")
                
                with col2:
                    price_change = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0] * 100
                    st.metric("价格变化", f"{price_change:.2f}%")
                
                with col3:
                    st.metric("最高价", f"{data['high'].max():.2f}")
                
                with col4:
                    st.metric("最低价", f"{data['low'].min():.2f}")
                
                # 下载数据按钮
                csv = display_data.to_csv(index=False)
                st.download_button(
                    label="下载CSV数据",
                    data=csv,
                    file_name=f"{symbol.replace('/', '_')}_{timeframe}_{start_date_str}_{end_date_str}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"加载数据失败: {str(e)}")
    
    # 获取实时行情
    st.subheader("实时行情")
    
    try:
        ticker = exchange.get_ticker(symbol)
        
        if ticker:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("最新价格", f"{ticker['last']:.2f}")
            
            with col2:
                change_pct = ticker.get('percentage', 0) * 100
                st.metric("24小时涨跌", f"{change_pct:.2f}%")
            
            with col3:
                st.metric("24小时最高", f"{ticker['high']:.2f}")
            
            with col4:
                st.metric("24小时最低", f"{ticker['low']:.2f}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("24小时成交量", f"{ticker['baseVolume']:.2f}")
            
            with col2:
                st.metric("24小时成交额", f"{ticker['quoteVolume']:.2f}")
            
            with col3:
                st.metric("买一价", f"{ticker['bid']:.2f}")
            
            with col4:
                st.metric("卖一价", f"{ticker['ask']:.2f}")
        else:
            st.warning(f"无法获取 {symbol} 的实时行情")
    
    except Exception as e:
        st.error(f"获取实时行情失败: {str(e)}")
    
    # 获取深度数据
    st.subheader("市场深度")
    
    try:
        depth = exchange.get_order_book(symbol, limit=20)
        
        if depth:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("买单")
                bids_df = pd.DataFrame(depth['bids'], columns=['价格', '数量'])
                st.dataframe(bids_df, use_container_width=True)
            
            with col2:
                st.write("卖单")
                asks_df = pd.DataFrame(depth['asks'], columns=['价格', '数量'])
                st.dataframe(asks_df, use_container_width=True)
            
            # 绘制深度图
            fig = go.Figure()
            
            # 添加买单
            bids_prices = [bid[0] for bid in depth['bids']]
            bids_amounts = [bid[1] for bid in depth['bids']]
            bids_cum = [sum(bids_amounts[:i+1]) for i in range(len(bids_amounts))]
            
            fig.add_trace(go.Scatter(
                x=bids_prices,
                y=bids_cum,
                mode='lines',
                name='买单',
                line=dict(color='green', width=2),
                fill='tozeroy'
            ))
            
            # 添加卖单
            asks_prices = [ask[0] for ask in depth['asks']]
            asks_amounts = [ask[1] for ask in depth['asks']]
            asks_cum = [sum(asks_amounts[:i+1]) for i in range(len(asks_amounts))]
            
            fig.add_trace(go.Scatter(
                x=asks_prices,
                y=asks_cum,
                mode='lines',
                name='卖单',
                line=dict(color='red', width=2),
                fill='tozeroy'
            ))
            
            fig.update_layout(
                title=f'{symbol} 市场深度',
                xaxis_title='价格',
                yaxis_title='累计数量',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"无法获取 {symbol} 的深度数据")
    
    except Exception as e:
        st.error(f"获取深度数据失败: {str(e)}")


if __name__ == "__main__":
    render_market_data_page()
