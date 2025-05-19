"""
回测引擎模块

提供完整的回测功能，包括数据加载、策略执行、交易模拟和绩效统计。
"""
import os
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta

from strategies.base import StrategyBase
from utils.logger import log
from utils.helpers import (
    timestamp_to_datetime,
    datetime_to_timestamp,
    calculate_sharpe_ratio,
    calculate_max_drawdown
)
from exchange.gateio import GateIOExchange


class BacktestEngine:
    """回测引擎类"""

    def __init__(self, initial_balance: float = 10000.0, fee_rate: float = 0.001, slippage: float = 0.0005):
        """
        初始化回测引擎

        Args:
            initial_balance: 初始资金（USDT）
            fee_rate: 手续费率（默认0.1%）
            slippage: 滑点（默认0.05%）
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.positions = {}  # 持仓信息
        self.trades = []  # 交易记录
        self.equity_curve = []  # 权益曲线
        self.current_time = None  # 当前时间
        self.data = None  # 回测数据
        self.strategy = None  # 交易策略
        self.exchange = GateIOExchange()  # 交易所接口
        self.symbol = None  # 交易对
        self.timeframe = None  # 时间周期
        self.commission = 0.0  # 总手续费
        self.slippage_cost = 0.0  # 总滑点成本

    def set_parameters(
        self,
        initial_balance: float = None,
        fee_rate: float = None,
        slippage: float = None
    ) -> 'BacktestEngine':
        """
        设置回测参数

        Args:
            initial_balance: 初始资金
            fee_rate: 手续费率
            slippage: 滑点

        Returns:
            BacktestEngine: 返回self以支持链式调用
        """
        if initial_balance is not None:
            self.initial_balance = initial_balance
            self.balance = initial_balance

        if fee_rate is not None:
            self.fee_rate = fee_rate

        if slippage is not None:
            self.slippage = slippage

        return self

    def set_strategy(self, strategy: StrategyBase) -> 'BacktestEngine':
        """
        设置交易策略

        Args:
            strategy: 交易策略实例

        Returns:
            BacktestEngine: 返回self以支持链式调用
        """
        self.strategy = strategy
        return self

    def load_data(
        self,
        data: Union[str, pd.DataFrame],
        symbol: str = None,
        timeframe: str = '1h',
        start_date: str = None,
        end_date: str = None
    ) -> 'BacktestEngine':
        """
        加载回测数据

        Args:
            data: 数据文件路径或DataFrame
            symbol: 交易对，如'BTC/USDT'
            timeframe: 时间周期，如'1h', '4h', '1d'等
            start_date: 开始日期，格式'YYYYMMDD'或'YYYY-MM-DD'
            end_date: 结束日期，格式'YYYYMMDD'或'YYYY-MM-DD'

        Returns:
            BacktestEngine: 返回self以支持链式调用
        """
        self.symbol = symbol
        self.timeframe = timeframe

        # 如果data是文件路径，则加载数据
        if isinstance(data, str) and os.path.exists(data):
            log.info(f"Loading data from {data}")
            self.data = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            raise ValueError("data must be a file path or pandas DataFrame")

        # 确保数据包含必要的列
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Data must contain '{col}' column")

        # 转换时间戳为datetime
        if 'datetime' not in self.data.columns:
            self.data['datetime'] = pd.to_datetime(self.data['timestamp'], unit='ms')

        # 设置时间索引
        self.data.set_index('datetime', inplace=True)

        # 按时间排序
        self.data.sort_index(inplace=True)

        # 过滤日期范围
        if start_date:
            start_date = pd.to_datetime(start_date)
            self.data = self.data[self.data.index >= start_date]

        if end_date:
            end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)  # 包含结束日期
            self.data = self.data[self.data.index <= end_date]

        log.info(f"Loaded {len(self.data)} bars from {self.data.index[0]} to {self.data.index[-1]}")

        return self

    def run_backtest(self) -> Dict[str, Any]:
        """
        运行回测

        Returns:
            Dict: 回测结果，包含绩效指标和交易记录
        """
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data loaded for backtesting")

        if self.strategy is None:
            raise ValueError("No strategy set for backtesting")

        # 初始化策略
        self.strategy.init()

        log.info(f"Starting backtest from {self.data.index[0]} to {self.data.index[-1]}")
        log.info(f"Initial balance: {self.initial_balance:.2f} USDT")

        start_time = time.time()

        # 逐根K线回测
        for i, (timestamp, row) in enumerate(self.data.iterrows()):
            self.current_time = timestamp

            # 准备K线数据
            bar = {
                'symbol': self.symbol,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
                'timestamp': int(timestamp.timestamp() * 1000),
                'datetime': timestamp
            }

            # 更新当前价格
            current_price = row['close']

            # 执行策略
            signal = self.strategy.on_bar(bar)

            # 处理信号
            if signal and signal['signal'] in ['buy', 'sell']:
                self._process_signal(signal, current_price)

            # 更新持仓市值和权益曲线
            self._update_equity(current_price, timestamp)

        # 计算绩效指标
        results = self._calculate_performance()

        end_time = time.time()
        log.info(f"Backtest completed in {end_time - start_time:.2f} seconds")
        log.info(f"Final balance: {self.balance:.2f} USDT")
        log.info(f"Total return: {results['total_return']:.2f}%")
        log.info(f"Max drawdown: {results['max_drawdown']:.2f}%")
        log.info(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")

        return results

    def _process_signal(self, signal: Dict, current_price: float) -> None:
        """
        处理交易信号

        Args:
            signal: 交易信号
            current_price: 当前价格
        """
        symbol = signal.get('symbol', self.symbol)
        signal_type = signal['signal']
        signal_price = signal.get('price', current_price)
        size = signal.get('size')

        # 应用滑点
        if signal_type == 'buy':
            execution_price = signal_price * (1 + self.slippage)
        else:  # sell
            execution_price = signal_price * (1 - self.slippage)

        # 计算手续费
        fee = execution_price * abs(size) * self.fee_rate if size else 0

        # 记录交易
        trade = {
            'timestamp': int(self.current_time.timestamp() * 1000),
            'datetime': self.current_time,
            'symbol': symbol,
            'type': signal_type,
            'price': execution_price,
            'size': size,
            'fee': fee,
            'slippage': abs(execution_price - signal_price) * (size or 0),
            'balance_before': self.balance,
            'info': signal.get('info', {})
        }

        # 更新余额和持仓
        if signal_type == 'buy' and self.balance >= execution_price * (size or 0) + fee:
            # 买入
            cost = execution_price * size + fee
            self.balance -= cost
            self.positions[symbol] = self.positions.get(symbol, 0) + size
            trade['cost'] = cost
            trade['balance_after'] = self.balance
            self.trades.append(trade)
            self.commission += fee
            self.slippage_cost += abs(execution_price - signal_price) * size

            # 更新策略状态
            self.strategy.on_trade({
                'symbol': symbol,
                'side': 'buy',
                'price': execution_price,
                'size': size,
                'fee': fee,
                'datetime': self.current_time,
                'info': signal.get('info', {})
            })

        elif signal_type == 'sell' and symbol in self.positions and self.positions[symbol] >= (size or 0):
            # 卖出
            size_to_sell = size or self.positions[symbol]
            revenue = execution_price * size_to_sell - fee
            self.balance += revenue
            self.positions[symbol] -= size_to_sell

            # 如果持仓为0，从字典中移除
            if self.positions[symbol] == 0:
                del self.positions[symbol]

            trade['size'] = size_to_sell
            trade['revenue'] = revenue
            trade['balance_after'] = self.balance
            self.trades.append(trade)
            self.commission += fee
            self.slippage_cost += abs(execution_price - signal_price) * size_to_sell

            # 更新策略状态
            self.strategy.on_trade({
                'symbol': symbol,
                'side': 'sell',
                'price': execution_price,
                'size': size_to_sell,
                'fee': fee,
                'datetime': self.current_time,
                'info': signal.get('info', {})
            })

    def _update_equity(self, current_price: float, timestamp: datetime) -> None:
        """
        更新权益曲线

        Args:
            current_price: 当前价格
            timestamp: 当前时间戳
        """
        # 计算持仓市值
        position_value = sum(
            current_price * size
            for symbol, size in self.positions.items()
        )

        # 计算总权益
        equity = self.balance + position_value

        # 记录权益曲线
        self.equity_curve.append({
            'timestamp': int(timestamp.timestamp() * 1000),
            'datetime': timestamp,
            'equity': equity,
            'balance': self.balance,
            'position_value': position_value
        })

        # 更新策略的权益曲线
        self.strategy.update_equity(equity, int(timestamp.timestamp() * 1000))

    def _calculate_performance(self) -> Dict[str, Any]:
        """
        计算回测绩效指标

        Returns:
            Dict: 包含各种绩效指标的字典
        """
        if not self.equity_curve:
            return {}

        # 转换为DataFrame
        df = pd.DataFrame(self.equity_curve)

        # 计算收益率
        df['returns'] = df['equity'].pct_change().fillna(0)

        # 计算累计收益率
        df['cum_returns'] = (1 + df['returns']).cumprod() - 1

        # 计算最大回撤
        df['peak'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['peak']) / df['peak']
        max_drawdown = df['drawdown'].min() * 100  # 转换为百分比

        # 计算年化收益率
        days = (df['datetime'].iloc[-1] - df['datetime'].iloc[0]).days or 1
        years = days / 365.0
        total_return = (df['equity'].iloc[-1] / self.initial_balance - 1) * 100  # 百分比
        annual_return = (1 + total_return / 100) ** (1 / years) - 1 if years > 0 else 0

        # 计算夏普比率（假设无风险利率为0）
        sharpe_ratio = calculate_sharpe_ratio(df['returns'].tolist())

        # 计算交易统计
        trades = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()

        if not trades.empty:
            # 计算每笔交易的收益
            trades['pnl'] = trades.apply(
                lambda x: (x['revenue'] - x.get('cost', 0)) / x.get('cost', 1) * 100
                if x['type'] == 'sell' else 0,
                axis=1
            )

            # 计算胜率
            win_trades = trades[trades['pnl'] > 0]
            loss_trades = trades[trades['pnl'] <= 0]
            win_rate = len(win_trades) / len(trades) * 100 if len(trades) > 0 else 0

            # 计算平均盈亏
            avg_win = win_trades['pnl'].mean() if not win_trades.empty else 0
            avg_loss = loss_trades['pnl'].mean() if not loss_trades.empty else 0

            # 计算盈亏比
            profit_factor = abs(win_trades['pnl'].sum() / loss_trades['pnl'].sum()) \
                if not loss_trades.empty and loss_trades['pnl'].sum() != 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        # 返回绩效指标
        return {
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_return': total_return,
            'annual_return': annual_return * 100,  # 转换为百分比
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'commission': self.commission,
            'slippage_cost': self.slippage_cost,
            'equity_curve': self.equity_curve,
            'trades': self.trades,
            'start_date': df['datetime'].iloc[0],
            'end_date': df['datetime'].iloc[-1],
            'days': days
        }

    def get_trades(self) -> List[Dict]:
        """
        获取交易记录

        Returns:
            List[Dict]: 交易记录列表
        """
        return self.trades

    def get_equity_curve(self) -> List[Dict]:
        """
        获取权益曲线

        Returns:
            List[Dict]: 权益曲线数据
        """
        return self.equity_curve

    def save_results(self, output_dir: str = 'results') -> Dict[str, str]:
        """
        保存回测结果到文件

        Args:
            output_dir: 输出目录

        Returns:
            Dict: 包含保存的文件路径
        """
        import os
        import json
        from datetime import datetime

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存权益曲线
        equity_file = os.path.join(output_dir, f'equity_curve_{timestamp}.csv')
        pd.DataFrame(self.equity_curve).to_csv(equity_file, index=False)

        # 保存交易记录
        trades_file = os.path.join(output_dir, f'trades_{timestamp}.csv')
        if self.trades:
            pd.DataFrame(self.trades).to_csv(trades_file, index=False)

        # 保存绩效指标
        metrics = self._calculate_performance()
        metrics_file = os.path.join(output_dir, f'metrics_{timestamp}.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        return {
            'equity_curve': equity_file,
            'trades': trades_file if self.trades else None,
            'metrics': metrics_file
        }

    def get_balance(self) -> float:
        """
        获取当前账户余额

        Returns:
            float: 当前账户余额
        """
        return self.balance

    def get_position(self, symbol: str) -> float:
        """
        获取指定交易对的持仓数量

        Args:
            symbol: 交易对，如'BTC/USDT'

        Returns:
            float: 持仓数量
        """
        return self.positions.get(symbol, 0.0)
