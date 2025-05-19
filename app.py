#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gate.io量化交易系统 - Web界面

使用Streamlit构建的交易系统Web界面，提供策略配置、市场数据查看和交易监控功能。
"""
import os
import sys
import json
import time
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import config
from utils.logger import setup_logger, log
from exchange.gateio import GateIOExchange
from strategies.base import StrategyBase
from strategies.ma_crossover import MovingAverageCrossover
from strategies.rsi import RSIStrategy
from strategies.bollinger_bands import BollingerBandsStrategy

# 策略映射
STRATEGY_MAP = {
    'ma_crossover': MovingAverageCrossover,
    'rsi': RSIStrategy,
    'bollinger_bands': BollingerBandsStrategy
}

# 设置页面配置
st.set_page_config(
    page_title="Gate.io量化交易系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置应用标题
st.title("Gate.io量化交易系统")

# 侧边栏 - 系统配置
st.sidebar.header("系统配置")

# 选择交易模式
mode = st.sidebar.selectbox(
    "选择交易模式",
    ["回测", "模拟交易", "实盘交易"],
    index=0
)

# 选择交易对
symbol = st.sidebar.text_input("交易对", "BTC/USDT")

# 选择时间周期
timeframe = st.sidebar.selectbox(
    "K线周期",
    ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
    index=4  # 默认1h
)

# 选择策略
strategy_name = st.sidebar.selectbox(
    "交易策略",
    list(STRATEGY_MAP.keys()),
    index=0
)

# 根据选择的策略显示相应的参数设置
strategy_class = STRATEGY_MAP[strategy_name]
default_params = strategy_class.default_params()

st.sidebar.subheader("策略参数")
strategy_params = {}

# 动态生成策略参数输入框
for param_name, param_value in default_params.items():
    if isinstance(param_value, bool):
        strategy_params[param_name] = st.sidebar.checkbox(
            param_name, 
            value=param_value
        )
    elif isinstance(param_value, int):
        strategy_params[param_name] = st.sidebar.number_input(
            param_name, 
            value=param_value,
            step=1
        )
    elif isinstance(param_value, float):
        strategy_params[param_name] = st.sidebar.number_input(
            param_name, 
            value=param_value,
            format="%.4f"
        )
    else:
        strategy_params[param_name] = st.sidebar.text_input(
            param_name, 
            value=str(param_value)
        )

# 初始资金设置
initial_balance = st.sidebar.number_input(
    "初始资金 (USDT)",
    min_value=100.0,
    value=10000.0,
    step=100.0
)

# 回测特定参数
if mode == "回测":
    st.sidebar.subheader("回测设置")
    
    # 回测日期范围
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "开始日期",
            datetime.now() - timedelta(days=30)
        )
    with col2:
        end_date = st.date_input(
            "结束日期",
            datetime.now()
        )
    
    # 手续费率
    fee_rate = st.sidebar.number_input(
        "手续费率",
        min_value=0.0,
        max_value=0.01,
        value=0.001,
        format="%.5f"
    )

# 实盘交易特定参数
if mode == "实盘交易":
    st.sidebar.subheader("实盘设置")
    
    # 测试模式
    test_mode = st.sidebar.checkbox("测试模式 (不执行实际交易)", value=True)
    
    # API密钥状态
    api_key = config.gateio_api_key
    api_secret = config.gateio_secret
    
    if api_key and api_secret:
        st.sidebar.success("API密钥已配置")
    else:
        st.sidebar.error("API密钥未配置，请检查config.ini文件")

# 主界面内容
if mode == "回测":
    st.header("回测模式")
    
    # 回测说明
    st.info("""
    回测模式可以使用历史数据测试交易策略的表现。
    
    1. 在侧边栏选择交易对、时间周期和策略
    2. 设置策略参数和回测日期范围
    3. 点击"开始回测"按钮运行回测
    """)
    
    # 回测按钮
    if st.button("开始回测", type="primary"):
        st.session_state.running_backtest = True
        
        # 显示进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 初始化交易所接口
            exchange = GateIOExchange()
            
            # 加载历史数据
            status_text.text("正在加载历史数据...")
            
            # 格式化日期
            start_date_str = start_date.strftime("%Y%m%d")
            end_date_str = end_date.strftime("%Y%m%d")
            
            # 这里调用main.py中的load_data函数
            from main import load_data
            
            data = load_data(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date_str,
                end_date=end_date_str
            )
            
            progress_bar.progress(25)
            
            if data is None or data.empty:
                st.error("无法加载历史数据，请检查日期范围和交易对")
                st.session_state.running_backtest = False
                return
            
            status_text.text(f"成功加载 {len(data)} 根K线数据")
            progress_bar.progress(50)
            
            # 创建策略实例
            strategy = strategy_class(strategy_params)
            
            # 初始化回测引擎
            from backtest.engine import BacktestEngine
            
            engine = BacktestEngine(
                initial_balance=initial_balance,
                fee_rate=fee_rate
            )
            
            # 设置策略
            engine.set_strategy(strategy)
            
            # 加载数据
            engine.load_data(data, symbol=symbol, timeframe=timeframe)
            
            status_text.text("正在运行回测...")
            progress_bar.progress(75)
            
            # 运行回测
            results = engine.run_backtest()
            
            progress_bar.progress(100)
            status_text.text("回测完成！")
            
            # 显示回测结果
            st.subheader("回测结果")
            
            # 基本指标
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总收益率", f"{results['total_return']:.2f}%")
            with col2:
                st.metric("年化收益率", f"{results['annual_return']:.2f}%")
            with col3:
                st.metric("最大回撤", f"{results['max_drawdown']:.2f}%")
            with col4:
                st.metric("夏普比率", f"{results['sharpe_ratio']:.2f}")
            
            # 交易统计
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总交易次数", f"{results['total_trades']}")
            with col2:
                st.metric("胜率", f"{results['win_rate']:.2f}%")
            with col3:
                st.metric("平均盈利", f"{results['avg_win']:.2f}%")
            with col4:
                st.metric("平均亏损", f"{results['avg_loss']:.2f}%")
            
            # 资金曲线
            st.subheader("资金曲线")
            
            # 转换权益曲线为DataFrame
            equity_df = pd.DataFrame(results['equity_curve'])
            equity_df['datetime'] = pd.to_datetime(equity_df['datetime'])
            equity_df.set_index('datetime', inplace=True)
            
            # 绘制资金曲线
            import plotly.graph_objects as go
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_df.index,
                y=equity_df['equity'],
                mode='lines',
                name='权益曲线',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title='回测权益曲线',
                xaxis_title='日期',
                yaxis_title='权益 (USDT)',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 交易记录
            st.subheader("交易记录")
            
            if results['trades']:
                trades_df = pd.DataFrame(results['trades'])
                trades_df['datetime'] = pd.to_datetime(trades_df['datetime'])
                trades_df = trades_df[['datetime', 'type', 'price', 'size', 'fee']]
                trades_df.columns = ['时间', '类型', '价格', '数量', '手续费']
                
                st.dataframe(trades_df, use_container_width=True)
            else:
                st.info("回测期间没有产生交易")
            
        except Exception as e:
            st.error(f"回测过程中发生错误: {str(e)}")
        finally:
            st.session_state.running_backtest = False

elif mode == "模拟交易":
    st.header("模拟交易模式")
    
    st.info("""
    模拟交易模式使用实时市场数据，但不执行实际交易。
    
    1. 在侧边栏选择交易对、时间周期和策略
    2. 设置策略参数
    3. 点击"开始模拟交易"按钮
    
    注意：模拟交易将在后台运行，您可以随时停止。
    """)
    
    # 检查是否已经在运行模拟交易
    if 'simulate_process_running' not in st.session_state:
        st.session_state.simulate_process_running = False
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not st.session_state.simulate_process_running:
            if st.button("开始模拟交易", type="primary"):
                st.session_state.simulate_process_running = True
                
                # 将策略参数转换为JSON字符串
                params_json = json.dumps(strategy_params)
                
                # 构建命令
                cmd = [
                    "python", "main.py",
                    "--mode", "simulate",
                    "--strategy", strategy_name,
                    "--symbol", symbol,
                    "--timeframe", timeframe,
                    "--initial_balance", str(initial_balance),
                    "--params", params_json
                ]
                
                # 在后台启动模拟交易进程
                import subprocess
                
                try:
                    # 使用Popen启动非阻塞进程
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    st.session_state.simulate_process = process
                    st.success("模拟交易已在后台启动！")
                    
                    # 添加进程ID显示
                    st.info(f"进程ID: {process.pid}")
                    
                except Exception as e:
                    st.error(f"启动模拟交易失败: {str(e)}")
                    st.session_state.simulate_process_running = False
    
    with col2:
        if st.session_state.simulate_process_running:
            if st.button("停止模拟交易", type="secondary"):
                if 'simulate_process' in st.session_state:
                    try:
                        # 终止进程
                        st.session_state.simulate_process.terminate()
                        st.session_state.simulate_process.wait(timeout=5)
                        st.success("模拟交易已停止")
                    except Exception as e:
                        st.error(f"停止模拟交易失败: {str(e)}")
                    finally:
                        st.session_state.simulate_process_running = False
    
    # 显示日志输出
    if st.session_state.simulate_process_running and 'simulate_process' in st.session_state:
        st.subheader("模拟交易日志")
        
        # 创建一个空的占位符用于更新日志
        log_output = st.empty()
        
        # 读取日志文件
        log_file = "logs/trading.log"
        
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    # 读取最后100行
                    lines = f.readlines()[-100:]
                    log_output.code("".join(lines), language="text")
            except Exception as e:
                log_output.error(f"读取日志文件失败: {str(e)}")
        else:
            log_output.info("日志文件不存在")

elif mode == "实盘交易":
    st.header("实盘交易模式")
    
    st.warning("""
    ⚠️ 实盘交易将使用真实资金执行交易，可能导致资金损失，请谨慎操作！
    
    在开始实盘交易前，请确保：
    1. 您已经在回测和模拟交易中充分测试了策略
    2. 您已经在config.ini中正确配置了API密钥
    3. 您了解所有风险并愿意承担责任
    """)
    
    # 检查API密钥是否已配置
    api_key = config.gateio_api_key
    api_secret = config.gateio_secret
    
    if not api_key or not api_secret:
        st.error("API密钥未配置，无法进行实盘交易。请检查config.ini文件。")
    else:
        # 检查是否已经在运行实盘交易
        if 'live_process_running' not in st.session_state:
            st.session_state.live_process_running = False
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not st.session_state.live_process_running:
                # 添加确认对话框
                confirm = st.checkbox("我已了解风险并确认要进行实盘交易")
                
                if confirm:
                    if st.button("开始实盘交易", type="primary"):
                        st.session_state.live_process_running = True
                        
                        # 将策略参数转换为JSON字符串
                        params_json = json.dumps(strategy_params)
                        
                        # 构建命令
                        cmd = [
                            "python", "main.py",
                            "--mode", "live",
                            "--strategy", strategy_name,
                            "--symbol", symbol,
                            "--timeframe", timeframe,
                            "--params", params_json
                        ]
                        
                        # 如果是测试模式，添加测试模式参数
                        if test_mode:
                            cmd.append("--test_mode")
                        
                        # 在后台启动实盘交易进程
                        import subprocess
                        
                        try:
                            # 使用Popen启动非阻塞进程
                            process = subprocess.Popen(
                                cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True
                            )
                            
                            st.session_state.live_process = process
                            
                            if test_mode:
                                st.success("实盘交易（测试模式）已在后台启动！")
                            else:
                                st.success("实盘交易已在后台启动！")
                            
                            # 添加进程ID显示
                            st.info(f"进程ID: {process.pid}")
                            
                        except Exception as e:
                            st.error(f"启动实盘交易失败: {str(e)}")
                            st.session_state.live_process_running = False
        
        with col2:
            if st.session_state.live_process_running:
                if st.button("停止实盘交易", type="secondary"):
                    if 'live_process' in st.session_state:
                        try:
                            # 终止进程
                            st.session_state.live_process.terminate()
                            st.session_state.live_process.wait(timeout=5)
                            st.success("实盘交易已停止")
                        except Exception as e:
                            st.error(f"停止实盘交易失败: {str(e)}")
                        finally:
                            st.session_state.live_process_running = False
        
        # 显示日志输出
        if st.session_state.live_process_running and 'live_process' in st.session_state:
            st.subheader("实盘交易日志")
            
            # 创建一个空的占位符用于更新日志
            log_output = st.empty()
            
            # 读取日志文件
            log_file = "logs/trading.log"
            
            if os.path.exists(log_file):
                try:
                    with open(log_file, "r") as f:
                        # 读取最后100行
                        lines = f.readlines()[-100:]
                        log_output.code("".join(lines), language="text")
                except Exception as e:
                    log_output.error(f"读取日志文件失败: {str(e)}")
            else:
                log_output.info("日志文件不存在")

# 添加页脚
st.markdown("---")
st.markdown("Gate.io量化交易系统 | 版本 1.0.0")
