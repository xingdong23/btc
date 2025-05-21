#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试RSI和支撑阻力日内交易策略

使用历史数据进行回测，评估策略性能。
"""
import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import config
from utils.logger import setup_logger, log
from backtest.engine import BacktestEngine
from strategies.rsi_support_resistance import RSISupportResistanceStrategy


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试RSI和支撑阻力日内交易策略')

    # 基本参数
    parser.add_argument('--symbol', type=str, default='BTC_USDT',
                       help='交易对，如 BTC_USDT, ETH_USDT')
    parser.add_argument('--start_date', type=str, default=(datetime.now() - timedelta(days=30)).strftime('%Y%m%d'),
                       help='回测开始日期，格式: YYYYMMDD 或 YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, default=datetime.now().strftime('%Y%m%d'),
                       help='回测结束日期，格式: YYYYMMDD 或 YYYY-MM-DD')
    parser.add_argument('--timeframe', type=str, default='15m',
                       help='K线周期，如 1m, 5m, 15m, 1h, 4h, 1d')
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                       help='初始资金，默认10000 USDT')

    # 策略参数
    parser.add_argument('--rsi_period', type=int, default=14,
                       help='RSI计算周期')
    parser.add_argument('--rsi_oversold', type=float, default=30.0,
                       help='RSI超卖阈值')
    parser.add_argument('--rsi_overbought', type=float, default=70.0,
                       help='RSI超买阈值')
    parser.add_argument('--risk_per_trade', type=float, default=0.02,
                       help='每笔交易风险比例 (0.02表示2%)')
    parser.add_argument('--reward_risk_ratio', type=float, default=2.0,
                       help='盈亏比')

    return parser.parse_args()


def main():
    """主函数"""
    # 导入必要的库
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # 解析命令行参数
    args = parse_args()

    # 设置日志
    # 使用全局日志记录器
    from utils.logger import log

    # 创建策略实例
    strategy_params = {
        'symbol': args.symbol,
        'rsi_period': args.rsi_period,
        'rsi_oversold': args.rsi_oversold,
        'rsi_overbought': args.rsi_overbought,
        'risk_per_trade': args.risk_per_trade,
        'reward_risk_ratio': args.reward_risk_ratio
    }

    strategy = RSISupportResistanceStrategy(strategy_params)

    # 创建回测引擎
    engine = BacktestEngine(initial_balance=args.initial_balance)

    # 设置策略
    engine.set_strategy(strategy)

    # 加载数据
    try:
        # 转换日期格式
        start_dt = pd.to_datetime(args.start_date)
        end_dt = pd.to_datetime(args.end_date)

        log.info(f"使用模拟数据进行回测: {args.symbol}, 周期: {args.timeframe}, 时间范围: {start_dt} - {end_dt}")

        # 生成模拟数据
        # 计算需要生成的K线数量
        if args.timeframe == '15m':
            # 15分钟一根K线，一天有24小时 * 4 = 96根
            bars_per_day = 96
        elif args.timeframe == '1h':
            # 1小时一根K线，一天有24根
            bars_per_day = 24
        else:
            # 默认一天100根K线
            bars_per_day = 100

        # 计算天数
        days = (end_dt - start_dt).days + 1
        # 确保有足够的数据用于支撑阻力检测和指标计算
        min_bars = max(strategy.params['sr_lookback_period'] * 2, 200)
        total_bars = max(days * bars_per_day, min_bars)

        log.info(f"生成 {total_bars} 根模拟 K线数据")

        # 生成时间序列
        timestamps = pd.date_range(start=start_dt, end=end_dt, periods=total_bars)

        # 生成模拟价格数据
        # 使用随机游走模拟价格变化
        np.random.seed(42)  # 设置随机种子，确保可重复性

        # 设置初始价格和波动参数
        initial_price = 60000.0  # 初始价格（比特币价格范围）
        volatility = 0.01  # 每根K线的波动率
        trend = 0.0001  # 微小的上升趋势

        # 生成价格序列
        price_changes = np.random.normal(trend, volatility, total_bars)
        prices = initial_price * (1 + np.cumsum(price_changes))

        # 生成OHLCV数据
        data = []
        for i in range(total_bars):
            # 生成当前K线的高低开收
            close = prices[i]
            # 根据波动率生成当根K线的波动范围
            intrabar_volatility = volatility * close
            high = close + abs(np.random.normal(0, intrabar_volatility))
            low = close - abs(np.random.normal(0, intrabar_volatility))

            # 确保低点不会低于0
            low = max(low, 0.1 * close)

            # 如果是第一根K线，开盘价等于初始价格
            # 否则开盘价等于上一根K线的收盘价
            if i == 0:
                open_price = initial_price
            else:
                open_price = prices[i-1]

            # 生成成交量（与价格变化幅度相关）
            volume = np.random.normal(1000, 500) * (1 + abs(price_changes[i]) * 10)
            volume = max(volume, 100)  # 确保成交量为正

            # 添加到数据列表
            data.append({
                'timestamp': int(timestamps[i].timestamp() * 1000),
                'datetime': timestamps[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

        # 创建DataFrame
        ohlcv_data = pd.DataFrame(data)
        ohlcv_data.set_index('datetime', inplace=True)

        log.info(f"成功生成 {len(ohlcv_data)} 根模拟 K线数据")

        # 加载数据到回测引擎
        engine.load_data(
            data=ohlcv_data,
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date
        )
    except Exception as e:
        log.error(f"加载数据失败: {str(e)}")
        import traceback
        log.error(traceback.format_exc())
        return

    # 运行回测
    log.info(f"开始回测 RSI和支撑阻力日内交易策略 | 交易对: {args.symbol} | 周期: {args.timeframe}")
    log.info(f"回测时间范围: {args.start_date} 至 {args.end_date}")
    log.info(f"策略参数: RSI周期={args.rsi_period}, 超卖={args.rsi_oversold}, 超买={args.rsi_overbought}")

    try:
        log.info("开始运行回测...")
        results = engine.run_backtest()
    except Exception as e:
        log.error(f"回测运行失败: {str(e)}")
        import traceback
        log.error(traceback.format_exc())
        return

    # 输出回测结果
    if results:
        log.info("回测完成，结果如下:")
        log.info(f"初始资金: {args.initial_balance:.2f} USDT")
        log.info(f"最终资金: {results.get('final_balance', 0):.2f} USDT")
        log.info(f"总收益率: {results.get('total_return', 0):.2f}%")
        log.info(f"年化收益率: {results.get('annual_return', 0):.2f}%")
        log.info(f"最大回撤: {results.get('max_drawdown', 0):.2f}%")
        log.info(f"夏普比率: {results.get('sharpe_ratio', 0):.2f}")
        log.info(f"总交易次数: {results.get('total_trades', 0)}")
        log.info(f"胜率: {results.get('win_rate', 0):.2f}%")
        log.info(f"盈亏比: {results.get('profit_factor', 0):.2f}")

        # 保存回测结果
        result_dir = 'results'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = f"{result_dir}/rsi_sr_{args.symbol}_{args.timeframe}_{timestamp}.json"

        with open(result_file, 'w', encoding='utf-8') as f:
            import json
            # 处理JSON序列化问题
            def json_serial(obj):
                """处理无法序列化的对象"""
                if isinstance(obj, (datetime, pd.Timestamp)):
                    return obj.isoformat()
                raise TypeError(f"Type {type(obj)} not serializable")

            json.dump(results, f, indent=4, default=json_serial)

        log.info(f"回测结果已保存至: {result_file}")

        # 绘制权益曲线
        if 'equity_curve' in results:
            try:
                import matplotlib.pyplot as plt
                import pandas as pd

                # 转换为DataFrame
                equity_df = pd.DataFrame(results['equity_curve'])
                equity_df['datetime'] = pd.to_datetime(equity_df['timestamp'], unit='ms')
                equity_df.set_index('datetime', inplace=True)

                # 绘制权益曲线
                plt.figure(figsize=(12, 6))
                plt.plot(equity_df.index, equity_df['equity'])
                plt.title(f'RSI和支撑阻力策略权益曲线 - {args.symbol} ({args.timeframe})')
                plt.xlabel('日期')
                plt.ylabel('权益 (USDT)')
                plt.grid(True)

                # 添加买入卖出标记
                if 'trades' in results:
                    trades_df = pd.DataFrame(results['trades'])
                    if not trades_df.empty and 'timestamp' in trades_df.columns:
                        trades_df['datetime'] = pd.to_datetime(trades_df['timestamp'], unit='ms')

                        # 买入点
                        buy_trades = trades_df[trades_df['side'] == 'buy']
                        if not buy_trades.empty:
                            plt.scatter(buy_trades['datetime'], buy_trades['equity_before'],
                                       marker='^', color='green', s=50, label='买入')

                        # 卖出点
                        sell_trades = trades_df[trades_df['side'] == 'sell']
                        if not sell_trades.empty:
                            plt.scatter(sell_trades['datetime'], sell_trades['equity_before'],
                                       marker='v', color='red', s=50, label='卖出')

                        plt.legend()

                # 保存图表
                chart_file = f"{result_dir}/rsi_sr_{args.symbol}_{args.timeframe}_{timestamp}.png"
                plt.savefig(chart_file)
                log.info(f"权益曲线已保存至: {chart_file}")

                # 显示图表
                plt.show()

            except Exception as e:
                log.error(f"绘制权益曲线失败: {str(e)}")
    else:
        log.error("回测失败，未返回结果")


if __name__ == '__main__':
    main()
