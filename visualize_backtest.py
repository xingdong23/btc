#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测结果可视化工具

用于将回测结果可视化，显示价格走势和买卖点位
"""
import os
import sys
import json
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_backtest_results(results_file):
    """
    加载回测结果文件
    
    Args:
        results_file: 回测结果文件路径
        
    Returns:
        dict: 回测结果数据
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results

def load_price_data(symbol, start_date, end_date, timeframe='1h'):
    """
    加载价格数据
    
    Args:
        symbol: 交易对
        start_date: 开始日期
        end_date: 结束日期
        timeframe: 时间周期
        
    Returns:
        pd.DataFrame: 价格数据
    """
    from main import load_data
    from exchange.gateio import GateIOExchange
    
    exchange = GateIOExchange()
    data = load_data(exchange, symbol, timeframe, start_date, end_date)
    return data

def format_price(x, pos):
    """格式化价格，用于y轴"""
    return f'${x:,.0f}'

def visualize_backtest(results_file, output_file=None):
    """
    可视化回测结果
    
    Args:
        results_file: 回测结果文件路径
        output_file: 输出图片文件路径，如果为None则显示图表
    """
    # 加载回测结果
    results = load_backtest_results(results_file)
    
    # 提取交易记录
    trades = results.get('trades', [])
    if not trades:
        print("没有找到交易记录")
        return
    
    # 提取回测参数
    params = results.get('params', {})
    symbol = params.get('symbol', 'BTC/USDT')
    start_date = params.get('start_date')
    end_date = params.get('end_date')
    strategy_name = params.get('strategy', 'Unknown')
    
    # 加载价格数据
    price_data = load_price_data(symbol, start_date, end_date)
    if price_data is None:
        print("无法加载价格数据")
        return
    
    # 转换交易记录为DataFrame
    trades_df = pd.DataFrame(trades)
    trades_df['datetime'] = pd.to_datetime(trades_df['datetime'])
    
    # 分离买入和卖出交易
    buy_trades = trades_df[trades_df['side'] == 'buy']
    sell_trades = trades_df[trades_df['side'] == 'sell']
    
    # 创建图表
    plt.figure(figsize=(16, 10))
    
    # 设置样式
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 绘制价格走势
    ax = plt.gca()
    price_data['close'].plot(ax=ax, color='#1f77b4', linewidth=1.5, label='价格')
    
    # 绘制买入点
    if not buy_trades.empty:
        ax.scatter(buy_trades['datetime'], buy_trades['price'], 
                  marker='^', color='green', s=100, label='买入', zorder=5)
    
    # 绘制卖出点
    if not sell_trades.empty:
        ax.scatter(sell_trades['datetime'], sell_trades['price'], 
                  marker='v', color='red', s=100, label='卖出', zorder=5)
    
    # 设置图表格式
    ax.yaxis.set_major_formatter(FuncFormatter(format_price))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.xticks(rotation=45)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 添加标题和标签
    plt.title(f'{symbol} {strategy_name} 策略回测结果', fontsize=16)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('价格 (USDT)', fontsize=12)
    
    # 添加图例
    plt.legend(loc='best', fontsize=12)
    
    # 添加回测结果摘要
    summary = (
        f"初始资金: ${results.get('initial_balance', 0):,.2f}\n"
        f"最终资金: ${results.get('final_balance', 0):,.2f}\n"
        f"总收益率: {results.get('total_return_percent', 0):.2f}%\n"
        f"最大回撤: {results.get('max_drawdown_percent', 0):.2f}%\n"
        f"夏普比率: {results.get('sharpe_ratio', 0):.2f}\n"
        f"总交易次数: {results.get('total_trades', 0)}"
    )
    
    # 在图表右上角添加文本框
    plt.figtext(0.92, 0.85, summary, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
               fontsize=10, ha='right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示图表
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图表已保存到 {output_file}")
    else:
        plt.show()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='回测结果可视化工具')
    parser.add_argument('--results', type=str, required=True,
                       help='回测结果文件路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出图片文件路径，如果不指定则显示图表')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    visualize_backtest(args.results, args.output)
