#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gate.io量化交易系统主程序

支持回测、模拟交易和实盘交易模式。
"""
import os
import sys
import argparse
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Type

import pandas as pd

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import config
from utils.logger import setup_logger, log
from exchange.gateio import GateIOExchange
from strategies.base import StrategyBase
from strategies.grid_trading import GateioGridTrading
from strategies.rsi_support_resistance import RSISupportResistanceStrategy
from strategies.vwap_trader import VWAPTraderStrategy
from backtest.engine import BacktestEngine
from risk_management.manager import RiskManager

# 策略映射
STRATEGY_MAP = {
    'grid_trading': GateioGridTrading,
    'rsi_sr': RSISupportResistanceStrategy,
    'vwap': VWAPTraderStrategy
}

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Gate.io量化交易系统')

    # 基本参数
    parser.add_argument('--mode', type=str, required=True,
                       choices=['backtest', 'simulate', 'live'],
                       help='运行模式: backtest(回测), simulate(模拟交易), live(实盘交易)')
    parser.add_argument('--strategy', type=str, required=True,
                       help='策略名称，如 grid_trading')
    parser.add_argument('--symbol', type=str, required=True,
                       help='交易对，如 BTC/USDT, ETH/USDT')

    # 回测参数
    parser.add_argument('--start_date', type=str,
                       help='回测开始日期，格式: YYYYMMDD 或 YYYY-MM-DD')
    parser.add_argument('--end_date', type=str,
                       help='回测结束日期，格式: YYYYMMDD 或 YYYY-MM-DD')
    parser.add_argument('--timeframe', type=str, default='1h',
                       help='K线周期，如 1m, 5m, 15m, 1h, 4h, 1d')
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                       help='初始资金（USDT）')
    parser.add_argument('--fee_rate', type=float, default=0.001,
                       help='手续费率，默认0.1%')

    # 策略参数
    parser.add_argument('--params', type=str, default='{}',
                       help='策略参数，JSON格式字符串')

    # 实盘交易参数
    parser.add_argument('--test_mode', action='store_true',
                       help='测试模式，不执行实际交易，仅记录信号')

    # 日志级别
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='日志级别')

    return parser.parse_args()

def validate_args(args) -> bool:
    """验证参数有效性"""
    # 检查策略是否存在
    if args.strategy not in STRATEGY_MAP:
        log.error(f"不支持的策略: {args.strategy}。可用策略: {', '.join(STRATEGY_MAP.keys())}")
        return False

    # 检查交易对格式
    if '/' not in args.symbol:
        log.error("交易对格式错误，应为 BASE/QUOTE，如 BTC/USDT")
        return False

    # 检查回测参数
    if args.mode == 'backtest' and not args.start_date:
        log.error("回测模式需要指定开始日期 (--start_date)")
        return False

    # 解析策略参数
    try:
        params = json.loads(args.params)
        if not isinstance(params, dict):
            log.error("策略参数必须是一个JSON对象")
            return False
    except json.JSONDecodeError:
        log.error("策略参数必须是有效的JSON格式")
        return False

    return True

def load_data(exchange: GateIOExchange, symbol: str, timeframe: str,
             start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    加载历史数据

    Args:
        exchange: 交易所接口
        symbol: 交易对
        timeframe: 时间周期
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        pd.DataFrame: 包含历史数据的DataFrame
    """
    log.info(f"正在加载 {symbol} {timeframe} 历史数据...")

    # 转换日期格式
    start_dt = pd.to_datetime(start_date) if start_date else None
    end_dt = pd.to_datetime(end_date) if end_date else None

    # 如果结束日期未指定，则使用当前时间
    if end_dt is None:
        end_dt = datetime.now()

    # 如果开始日期未指定，则默认加载最近1000根K线
    if start_dt is None:
        start_dt = end_dt - timedelta(days=30)  # 默认加载30天数据

    # 获取K线数据
    try:
        # 注意：这里简化处理，实际应该分页获取所有历史数据
        ohlcv = exchange.get_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=int(start_dt.timestamp() * 1000),
            limit=1000  # 最大1000根K线
        )

        if ohlcv is None or ohlcv.empty:
            log.error("未能获取到历史数据")
            return None

        # 过滤日期范围
        if start_dt:
            ohlcv = ohlcv[ohlcv.index >= start_dt]
        if end_dt:
            ohlcv = ohlcv[ohlcv.index <= end_dt]

        log.info(f"成功加载 {len(ohlcv)} 根K线数据，时间范围: {ohlcv.index[0]} 到 {ohlcv.index[-1]}")
        return ohlcv

    except Exception as e:
        log.error(f"加载历史数据失败: {str(e)}")
        return None

def run_backtest(strategy_class: Type[StrategyBase], strategy_params: Dict,
                data: pd.DataFrame, initial_balance: float = 10000.0,
                fee_rate: float = 0.001) -> Dict[str, Any]:
    """
    运行回测

    Args:
        strategy_class: 策略类
        strategy_params: 策略参数
        data: 历史数据
        initial_balance: 初始资金
        fee_rate: 手续费率

    Returns:
        Dict: 回测结果
    """
    log.info("开始回测...")

    try:
        # 初始化回测引擎
        engine = BacktestEngine(
            initial_balance=initial_balance,
            fee_rate=fee_rate
        )

        # 创建策略实例
        strategy = strategy_class(strategy_params)

        # 设置策略
        engine.set_strategy(strategy)

        # 加载数据
        engine.load_data(data)

        # 运行回测
        results = engine.run_backtest()

        # 保存回测结果
        output_dir = 'backtest_results'
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(output_dir, f'results_{timestamp}.json')

        # 添加策略参数到结果中
        results['params'] = {
            'strategy': strategy_class.__name__,
            'symbol': data.index.name if hasattr(data.index, 'name') else 'BTC/USDT',
            'start_date': data.index[0].strftime('%Y%m%d') if len(data) > 0 else '',
            'end_date': data.index[-1].strftime('%Y%m%d') if len(data) > 0 else '',
            'strategy_params': strategy_params
        }

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        log.info(f"回测完成，结果已保存到 {results_file}")

        # 打印摘要
        print("\n=== 回测结果 ===")
        print(f"初始资金: {initial_balance:.2f} USDT")
        print(f"结束资金: {results['final_balance']:.2f} USDT")
        print(f"总收益率: {results['total_return']:.2f}%")
        print(f"年化收益率: {results.get('annual_return', 0):.2f}%")
        print(f"最大回撤: {results['max_drawdown']:.2f}%")
        print(f"夏普比率: {results['sharpe_ratio']:.2f}")
        print(f"总交易次数: {results['total_trades']}")
        print(f"胜率: {results.get('win_rate', 0):.2f}%")
        print(f"平均盈利: {results.get('avg_win', 0):.2f}%")
        print(f"平均亏损: {results.get('avg_loss', 0):.2f}%")
        print(f"盈亏比: {results.get('profit_factor', 0):.2f}")

        return results

    except Exception as e:
        log.error(f"回测过程中发生错误: {str(e)}", exc_info=True)
        return {}

def run_simulation(strategy_class: Type[StrategyBase], strategy_params: Dict,
                  symbol: str, timeframe: str, initial_balance: float = 10000.0):
    """
    运行模拟交易（纸交易）

    Args:
        strategy_class: 策略类
        strategy_params: 策略参数
        symbol: 交易对
        timeframe: 时间周期
        initial_balance: 初始资金
    """
    from exchange.gateio import GateIOExchange
    from backtest.engine import BacktestEngine
    import time
    from datetime import datetime, timedelta

    log.info(f"开始模拟交易: {symbol} {timeframe}")
    log.info(f"初始资金: {initial_balance} USDT")
    log.info(f"策略参数: {strategy_params}")

    # 初始化交易所接口
    exchange = GateIOExchange()

    # 初始化回测引擎（用于模拟交易）
    engine = BacktestEngine(initial_balance=initial_balance, fee_rate=0.001)

    # 创建策略实例
    strategy = strategy_class(strategy_params)
    engine.set_strategy(strategy)

    # 获取历史数据用于初始化策略
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)  # 获取30天历史数据

    log.info(f"加载历史数据: {start_time} 至 {end_time}")
    data = load_data(exchange, symbol, timeframe,
                    start_date=start_time.strftime('%Y%m%d'),
                    end_date=end_time.strftime('%Y%m%d'))

    if data is None or len(data) == 0:
        log.error("无法加载历史数据，模拟交易终止")
        return

    # 初始化策略状态
    log.info("初始化策略状态...")
    for _, row in data.iterrows():
        bar = {
            'symbol': symbol,
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume'],
            'datetime': row.name.to_pydatetime()
        }
        strategy.on_bar(bar)

    log.info("策略初始化完成，开始模拟交易...")

    # 模拟交易主循环
    try:
        while True:
            current_time = datetime.now()
            log.info(f"\n=== 模拟交易时间: {current_time} ===")

            # 获取最新K线数据
            ohlcv = exchange.get_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=int((current_time - timedelta(hours=1)).timestamp() * 1000),
                limit=1
            )

            if ohlcv is not None and not ohlcv.empty:
                latest_bar = ohlcv.iloc[-1]
                bar = {
                    'symbol': symbol,
                    'open': latest_bar['open'],
                    'high': latest_bar['high'],
                    'low': latest_bar['low'],
                    'close': latest_bar['close'],
                    'volume': latest_bar['volume'],
                    'datetime': latest_bar.name.to_pydatetime()
                }

                # 更新策略状态
                signal = strategy.on_bar(bar)

                # 模拟执行交易
                if signal and signal.get('signal') in ['buy', 'sell']:
                    log.info(f"生成交易信号: {signal}")
                    # 在实际模拟中，这里会调用交易所API执行交易
                    # 这里我们只记录日志
                    if signal['signal'] == 'buy':
                        log.info(f"[模拟] 买入 {symbol} 数量: {signal.get('size', 0):.6f} 价格: {signal['price']:.2f}")
                    else:
                        log.info(f"[模拟] 卖出 {symbol} 数量: {signal.get('size', 0):.6f} 价格: {signal['price']:.2f}")

                # 显示账户状态
                balance = engine.get_balance()
                position = engine.get_position(symbol)
                log.info(f"账户状态 - 余额: {balance:.2f} USDT | 持仓: {position:.6f} {symbol.split('/')[0]}")

            # 等待下一个周期
            sleep_time = _get_sleep_time(timeframe)
            log.info(f"等待 {sleep_time} 秒后更新...")
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        log.info("\n模拟交易已手动停止")
    except Exception as e:
        log.error(f"模拟交易发生错误: {str(e)}", exc_info=True)
    finally:
        # 显示最终账户状态
        balance = engine.get_balance()
        position = engine.get_position(symbol)
        log.info("\n=== 模拟交易结束 ===")
        log.info(f"最终账户状态:")
        log.info(f"- 余额: {balance:.2f} USDT")
        log.info(f"- 持仓: {position:.6f} {symbol.split('/')[0]}")
        log.info(f"- 总价值: {balance + position * bar['close']:.2f} USDT")

def _get_sleep_time(timeframe: str) -> int:
    """根据时间框架计算睡眠时间"""
    timeframe_seconds = {
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1h': 3600,
        '4h': 14400,
        '1d': 86400
    }
    return timeframe_seconds.get(timeframe.lower(), 60)  # 默认1分钟
    # TODO: 实现模拟交易逻辑

def run_live_trading(strategy_class: Type[StrategyBase], strategy_params: Dict,
                     symbol: str, timeframe: str, test_mode: bool = False):
    """
    运行实盘交易

    Args:
        strategy_class: 策略类
        strategy_params: 策略参数
        symbol: 交易对
        timeframe: 时间周期
        test_mode: 测试模式，不执行实际交易，仅记录信号
    """
    from exchange.gateio import GateIOExchange
    from trading.engine import LiveTradingEngine
    from risk_management.manager import RiskManager

    log.info(f"开始实盘交易: {symbol} {timeframe}")
    log.info(f"策略参数: {strategy_params}")

    if test_mode:
        log.warning("当前为测试模式，不会执行实际交易")
    else:
        log.warning("实盘交易将执行真实交易操作，可能导致资金损失，请确认！")
        confirmation = input("输入 'yes' 确认继续，或任意键取消: ")
        if confirmation.lower() != 'yes':
            log.info("实盘交易已取消")
            return

    try:
        # 初始化交易所接口
        exchange = GateIOExchange()

        # 创建策略实例
        strategy = strategy_class(strategy_params)

        # 初始化风险管理器
        risk_manager = RiskManager()

        # 初始化实盘交易引擎
        engine = LiveTradingEngine(
            exchange=exchange,
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            risk_manager=risk_manager,
            test_mode=test_mode
        )

        # 启动交易引擎
        engine.start()

    except KeyboardInterrupt:
        log.info("实盘交易已手动停止")
    except Exception as e:
        log.error(f"实盘交易发生错误: {str(e)}", exc_info=True)
    finally:
        log.info("实盘交易已结束")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 设置日志级别
    log_level = getattr(logging, args.log_level.upper())
    setup_logger('trading_system', log_level=args.log_level.upper())

    # 验证参数
    if not validate_args(args):
        sys.exit(1)

    # 解析策略参数
    try:
        strategy_params = json.loads(args.params)
    except json.JSONDecodeError:
        log.error("策略参数必须是有效的JSON格式")
        sys.exit(1)

    # 获取策略类
    strategy_class = STRATEGY_MAP[args.strategy]

    # 初始化交易所接口
    exchange = GateIOExchange()

    # 根据模式执行相应操作
    if args.mode == 'backtest':
        # 回测模式
        data = load_data(
            exchange=exchange,
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date
        )

        if data is not None:
            run_backtest(
                strategy_class=strategy_class,
                strategy_params=strategy_params,
                data=data,
                initial_balance=args.initial_balance,
                fee_rate=args.fee_rate
            )

    elif args.mode == 'simulate':
        # 模拟交易模式
        run_simulation(
            strategy_class=strategy_class,
            strategy_params=strategy_params,
            symbol=args.symbol,
            timeframe=args.timeframe,
            initial_balance=args.initial_balance
        )

    elif args.mode == 'live':
        # 实盘交易模式
        run_live_trading(
            strategy_class=strategy_class,
            strategy_params=strategy_params,
            symbol=args.symbol,
            timeframe=args.timeframe,
            test_mode=args.test_mode
        )

    log.info("程序执行完毕")

if __name__ == "__main__":
    main()
