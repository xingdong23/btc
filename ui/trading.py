"""
交易中心页面

提供策略配置、回测、模拟交易和实盘交易功能。
"""
import os
import sys
import json
import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import config
from utils.logger import log
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


def render_trading_page():
    """渲染交易中心页面"""
    st.title("交易中心")
    
    # 选择交易模式
    mode = st.radio(
        "选择交易模式",
        ["回测", "模拟交易", "实盘交易"],
        horizontal=True
    )
    
    # 交易配置
    st.subheader("交易配置")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbol = st.text_input("交易对", "BTC/USDT")
    
    with col2:
        timeframe = st.selectbox(
            "K线周期",
            ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            index=4  # 默认1h
        )
    
    with col3:
        strategy_name = st.selectbox(
            "交易策略",
            list(STRATEGY_MAP.keys())
        )
    
    # 根据选择的策略显示相应的参数设置
    strategy_class = STRATEGY_MAP[strategy_name]
    default_params = strategy_class.default_params()
    
    st.subheader("策略参数")
    
    # 使用列布局显示参数
    param_cols = st.columns(3)
    strategy_params = {}
    
    # 动态生成策略参数输入框
    for i, (param_name, param_value) in enumerate(default_params.items()):
        with param_cols[i % 3]:
            if isinstance(param_value, bool):
                strategy_params[param_name] = st.checkbox(
                    param_name, 
                    value=param_value
                )
            elif isinstance(param_value, int):
                strategy_params[param_name] = st.number_input(
                    param_name, 
                    value=param_value,
                    step=1
                )
            elif isinstance(param_value, float):
                strategy_params[param_name] = st.number_input(
                    param_name, 
                    value=param_value,
                    format="%.4f"
                )
            else:
                strategy_params[param_name] = st.text_input(
                    param_name, 
                    value=str(param_value)
                )
    
    # 初始资金设置
    st.subheader("资金设置")
    
    initial_balance = st.number_input(
        "初始资金 (USDT)",
        min_value=100.0,
        value=10000.0,
        step=100.0
    )
    
    # 根据交易模式显示不同的内容
    if mode == "回测":
        render_backtest_section(strategy_class, strategy_params, symbol, timeframe, initial_balance)
    elif mode == "模拟交易":
        render_simulation_section(strategy_name, strategy_params, symbol, timeframe, initial_balance)
    elif mode == "实盘交易":
        render_live_trading_section(strategy_name, strategy_params, symbol, timeframe)


def render_backtest_section(strategy_class, strategy_params, symbol, timeframe, initial_balance):
    """渲染回测部分"""
    st.header("回测设置")
    
    # 回测日期范围
    col1, col2 = st.columns(2)
    
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
    fee_rate = st.number_input(
        "手续费率",
        min_value=0.0,
        max_value=0.01,
        value=0.001,
        format="%.5f"
    )
    
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
                
                # 下载交易记录
                csv = trades_df.to_csv(index=False)
                st.download_button(
                    label="下载交易记录",
                    data=csv,
                    file_name=f"backtest_trades_{start_date_str}_{end_date_str}.csv",
                    mime="text/csv"
                )
            else:
                st.info("回测期间没有产生交易")
            
        except Exception as e:
            st.error(f"回测过程中发生错误: {str(e)}")
        finally:
            st.session_state.running_backtest = False


def render_simulation_section(strategy_name, strategy_params, symbol, timeframe, initial_balance):
    """渲染模拟交易部分"""
    st.header("模拟交易设置")
    
    st.info("""
    模拟交易模式使用实时市场数据，但不执行实际交易。
    
    1. 设置好交易对、时间周期、策略和参数
    2. 点击"开始模拟交易"按钮
    3. 系统将在后台运行模拟交易
    
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


def render_live_trading_section(strategy_name, strategy_params, symbol, timeframe):
    """渲染实盘交易部分"""
    st.header("实盘交易设置")
    
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
        # 测试模式
        test_mode = st.checkbox("测试模式 (不执行实际交易)", value=True)
        
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


if __name__ == "__main__":
    render_trading_page()
