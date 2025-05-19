"""
账户和资产页面

提供用户账户信息、资产状况和交易历史查看功能。
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
from config.config import config


def render_account_page():
    """渲染账户和资产页面"""
    st.title("账户和资产")
    
    # 检查API密钥是否已配置
    api_key = config.gateio_api_key
    api_secret = config.gateio_secret
    
    if not api_key or not api_secret:
        st.error("API密钥未配置，无法获取账户信息。请检查config.ini文件。")
        return
    
    # 初始化交易所接口
    exchange = GateIOExchange()
    
    # 刷新按钮
    if st.button("刷新数据", type="primary"):
        st.experimental_rerun()
    
    # 账户余额
    st.subheader("账户余额")
    
    try:
        balance = exchange.get_balance()
        
        if balance:
            # 过滤有余额的币种
            non_zero_balance = {
                currency: data for currency, data in balance.items()
                if float(data['total']) > 0
            }
            
            if non_zero_balance:
                # 创建DataFrame
                balance_data = []
                
                for currency, data in non_zero_balance.items():
                    balance_data.append({
                        '币种': currency,
                        '可用': float(data['free']),
                        '冻结': float(data['used']),
                        '总额': float(data['total'])
                    })
                
                balance_df = pd.DataFrame(balance_data)
                
                # 显示余额表格
                st.dataframe(balance_df, use_container_width=True)
                
                # 获取USDT价值
                try:
                    usdt_values = []
                    
                    for currency, data in non_zero_balance.items():
                        if currency == 'USDT':
                            usdt_values.append({
                                '币种': currency,
                                'USDT价值': float(data['total'])
                            })
                        else:
                            # 获取币种对USDT的价格
                            try:
                                ticker = exchange.get_ticker(f"{currency}/USDT")
                                if ticker:
                                    usdt_value = float(data['total']) * ticker['last']
                                    usdt_values.append({
                                        '币种': currency,
                                        'USDT价值': usdt_value
                                    })
                            except:
                                # 如果无法获取价格，跳过
                                pass
                    
                    if usdt_values:
                        # 计算总USDT价值
                        total_usdt = sum(item['USDT价值'] for item in usdt_values)
                        
                        # 创建饼图
                        labels = [item['币种'] for item in usdt_values]
                        values = [item['USDT价值'] for item in usdt_values]
                        
                        fig = go.Figure(data=[go.Pie(
                            labels=labels,
                            values=values,
                            hole=.3
                        )])
                        
                        fig.update_layout(
                            title=f'资产分布 (总计: {total_usdt:.2f} USDT)',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.warning(f"无法计算USDT价值: {str(e)}")
            else:
                st.info("账户中没有余额")
        else:
            st.warning("无法获取账户余额")
    
    except Exception as e:
        st.error(f"获取账户余额失败: {str(e)}")
    
    # 未完成订单
    st.subheader("未完成订单")
    
    try:
        open_orders = exchange.get_open_orders()
        
        if open_orders:
            # 创建DataFrame
            orders_data = []
            
            for order in open_orders:
                orders_data.append({
                    '订单ID': order['id'],
                    '交易对': order['symbol'],
                    '类型': order['side'],
                    '价格': order['price'],
                    '数量': order['amount'],
                    '已成交': order.get('filled', 0),
                    '状态': order['status'],
                    '时间': datetime.fromtimestamp(order['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
                })
            
            orders_df = pd.DataFrame(orders_data)
            
            # 显示订单表格
            st.dataframe(orders_df, use_container_width=True)
        else:
            st.info("没有未完成的订单")
    
    except Exception as e:
        st.error(f"获取未完成订单失败: {str(e)}")
    
    # 交易历史
    st.subheader("交易历史")
    
    # 交易对选择
    symbol = st.text_input("交易对 (留空查询所有)", "")
    
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
    
    # 查询按钮
    if st.button("查询交易历史"):
        with st.spinner("正在查询交易历史..."):
            try:
                # 计算时间戳
                since = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
                
                # 获取交易历史
                trades = exchange.get_my_trades(
                    symbol=symbol if symbol else None,
                    since=since,
                    limit=100
                )
                
                if trades:
                    # 创建DataFrame
                    trades_data = []
                    
                    for trade in trades:
                        trades_data.append({
                            '交易ID': trade['id'],
                            '订单ID': trade.get('order', ''),
                            '交易对': trade['symbol'],
                            '类型': trade['side'],
                            '价格': trade['price'],
                            '数量': trade['amount'],
                            '成交额': trade['cost'],
                            '手续费': trade.get('fee', {}).get('cost', 0),
                            '手续费币种': trade.get('fee', {}).get('currency', ''),
                            '时间': datetime.fromtimestamp(trade['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
                        })
                    
                    trades_df = pd.DataFrame(trades_data)
                    
                    # 过滤日期范围
                    trades_df['时间'] = pd.to_datetime(trades_df['时间'])
                    trades_df = trades_df[trades_df['时间'] >= pd.Timestamp(start_date)]
                    trades_df = trades_df[trades_df['时间'] <= pd.Timestamp(end_date)]
                    trades_df['时间'] = trades_df['时间'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    if not trades_df.empty:
                        # 显示交易历史表格
                        st.dataframe(trades_df, use_container_width=True)
                        
                        # 下载数据按钮
                        csv = trades_df.to_csv(index=False)
                        st.download_button(
                            label="下载CSV数据",
                            data=csv,
                            file_name=f"trades_{start_date}_{end_date}.csv",
                            mime="text/csv"
                        )
                        
                        # 交易统计
                        st.subheader("交易统计")
                        
                        # 按交易对统计
                        symbol_stats = trades_df.groupby('交易对').agg({
                            '交易ID': 'count',
                            '成交额': 'sum'
                        }).reset_index()
                        
                        symbol_stats.columns = ['交易对', '交易次数', '总成交额']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("按交易对统计")
                            st.dataframe(symbol_stats, use_container_width=True)
                        
                        with col2:
                            # 创建交易对成交额饼图
                            fig = go.Figure(data=[go.Pie(
                                labels=symbol_stats['交易对'],
                                values=symbol_stats['总成交额'],
                                hole=.3
                            )])
                            
                            fig.update_layout(
                                title='交易对成交额分布',
                                height=300
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # 按类型统计
                        type_stats = trades_df.groupby('类型').agg({
                            '交易ID': 'count',
                            '成交额': 'sum'
                        }).reset_index()
                        
                        type_stats.columns = ['类型', '交易次数', '总成交额']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("按类型统计")
                            st.dataframe(type_stats, use_container_width=True)
                        
                        with col2:
                            # 创建类型成交额饼图
                            fig = go.Figure(data=[go.Pie(
                                labels=type_stats['类型'],
                                values=type_stats['总成交额'],
                                hole=.3
                            )])
                            
                            fig.update_layout(
                                title='买卖类型成交额分布',
                                height=300
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("所选日期范围内没有交易记录")
                else:
                    st.info("没有交易记录")
            
            except Exception as e:
                st.error(f"获取交易历史失败: {str(e)}")


if __name__ == "__main__":
    render_account_page()
