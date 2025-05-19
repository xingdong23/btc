"""
交易系统功能测试脚本
"""
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import config
from exchange.gateio import GateIOExchange
from strategies.ma_crossover import MovingAverageCrossover
from backtest.engine import BacktestEngine

def test_backtest():
    """测试回测功能"""
    print("=" * 50)
    print("测试回测功能")
    print("=" * 50)
    
    # 设置回测参数
    symbol = "BTC/USDT"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print("回测参数:")
    print(f"- 交易对: {symbol}")
    print(f"- 开始日期: {start_date.strftime('%Y-%m-%d')}")
    print(f"- 结束日期: {end_date.strftime('%Y-%m-%d')}")
    
    # 将日期转换为字符串格式用于显示
    start_date_str = start_date.strftime("%Y%m%d")
    end_date_str = end_date.strftime("%Y%m%d")
    
    # 初始化策略
    strategy = MovingAverageCrossover({
        'short_ma': 5,
        'long_ma': 20,
        'use_sma': True
    })
    
    # 初始化回测引擎
    engine = BacktestEngine(
        initial_balance=10000,
        fee_rate=0.001,
        slippage=0.0005
    )
    
    # 设置策略
    engine.set_strategy(strategy)
    
    # 加载数据
    print("\n正在加载数据...")
    # 生成模拟的K线数据（每小时1根）
    np.random.seed(42)
    
    # 生成时间序列（每小时1根K线）
    date_range = pd.date_range(start=start_date, end=end_date, freq='1h')
    n = len(date_range)
    
    if n == 0:
        raise ValueError("No data points in the specified date range")
    
    # 生成随机价格序列（几何布朗运动）
    returns = np.random.normal(0, 0.002, n)  # 每日约0.2%的波动
    prices = 100000 * np.exp(np.cumsum(returns))  # 从10万USDT开始
    
    # 创建DataFrame
    data = pd.DataFrame({
        'timestamp': (date_range.astype(np.int64) // 10**6).astype(np.int64),  # 转换为毫秒时间戳
        'open': prices * 0.998,  # 开盘价略低于收盘价
        'high': prices * 1.001,  # 最高价略高于收盘价
        'low': prices * 0.997,   # 最低价略低于开盘价
        'close': prices,         # 收盘价
        'volume': np.random.lognormal(5, 1, n)  # 成交量
    })
    
    # 添加datetime列，用于后续索引
    data['datetime'] = date_range
    
    # 加载数据到回测引擎
    engine.load_data(
        data=data,
        symbol=symbol,
        timeframe='1h',
        start_date=start_date,
        end_date=end_date
    )
    
    # 运行回测
    print("\n正在运行回测...")
    results = engine.run_backtest()
    
    # 输出回测结果
    print("\n回测结果:")
    print(f"初始资金: {results.get('initial_balance', 'N/A')} USDT")
    print(f"最终资金: {results.get('final_balance', 'N/A'):.2f} USDT")
    print(f"总收益率: {results.get('total_return', 'N/A'):.2%}")
    
    # 检查并输出年化收益率（如果存在）
    if 'annualized_return' in results:
        print(f"年化收益率: {results['annualized_return']:.2%}")
    
    # 检查并输出其他指标
    if 'max_drawdown' in results:
        print(f"最大回撤: {results['max_drawdown']:.2%}")
        
    if 'win_rate' in results:
        print(f"胜率: {results['win_rate']:.2%}")
        
    if 'total_trades' in results:
        print(f"总交易次数: {results['total_trades']}")
        
    if 'winning_trades' in results:
        print(f"盈利交易次数: {results['winning_trades']}")
        
    if 'losing_trades' in results:
        print(f"亏损交易次数: {results['losing_trades']}")
    
    return results

def test_live_trading():
    """测试实盘交易功能"""
    print("\n" + "=" * 50)
    print("测试实盘交易功能")
    print("=" * 50)
    
    # 初始化交易所接口
    exchange = GateIOExchange()
    
    # 获取账户余额
    try:
        print("\n获取账户余额...")
        balance = exchange.get_balance()
        if balance:
            print("当前账户余额:")
            for currency, amount in balance.items():
                if float(amount['free']) > 0 or float(amount['used']) > 0:
                    print(f"{currency}: 可用={amount['free']}, 冻结={amount['used']}")
        else:
            print("无法获取账户余额，请检查API密钥和网络连接")
    except Exception as e:
        print(f"获取账户余额时出错: {e}")
    
    # 获取行情数据
    try:
        print("\n获取BTC/USDT行情...")
        ticker = exchange.get_ticker('BTC/USDT')
        if ticker:
            print(f"当前价格: {ticker['last']} USDT")
            print(f"24小时最高价: {ticker['high']} USDT")
            print(f"24小时最低价: {ticker['low']} USDT")
            print(f"24小时成交量: {ticker['baseVolume']} BTC")
    except Exception as e:
        print(f"获取行情数据时出错: {e}")

if __name__ == "__main__":
    # 测试回测功能
    test_backtest()
    
    # 测试实盘交易功能（需要有效的API密钥）
    if not config.gateio_api_key.startswith('mock_'):
        test_live_trading()
    else:
        print("\n检测到模拟API密钥，跳过实盘交易测试")
        print("要测试实盘交易功能，请在config.ini中设置有效的Gate.io API密钥")
