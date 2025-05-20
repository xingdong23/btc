"""
RSI (Relative Strength Index) 超买超卖策略

当RSI低于超卖阈值时产生买入信号，当RSI高于超买阈值时产生卖出信号。
"""
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from .base import StrategyBase
from utils.logger import log


class RSIStrategy(StrategyBase):
    """
    RSI超买超卖策略

    参数:
        rsi_period (int): RSI计算周期，默认为14
        oversold (float): 超卖阈值，默认为30
        overbought (float): 超买阈值，默认为70
        use_signal_confirmation (bool): 是否使用信号确认，默认为True
        confirmation_period (int): 信号确认周期，默认为3
    """

    @classmethod
    def default_params(cls) -> Dict:
        return {
            'rsi_period': 14,
            'oversold': 30.0,
            'overbought': 70.0,
            'use_signal_confirmation': True,
            'confirmation_period': 3,
            'stop_loss': 0.05,  # 5% 止损
            'take_profit': 0.1  # 10% 止盈
        }

    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.prices = []
        self.rsi_values = []
        self.signals = []

    def init(self) -> None:
        """策略初始化"""
        super().init()
        log.info(f"{self.name} strategy initialized with parameters: {self.params}")

    def on_bar(self, bar: Dict) -> Dict:
        """
        处理K线数据

        Args:
            bar: K线数据字典

        Returns:
            Dict: 交易信号
        """
        close_price = bar['close']
        self.prices.append(close_price)

        # 确保有足够的数据计算RSI
        if len(self.prices) < self.params['rsi_period'] + 1:
            return self.generate_signal('hold', close_price)

        # 计算RSI
        rsi = self._calculate_rsi(self.prices, self.params['rsi_period'])
        self.rsi_values.append(rsi)

        # 确保有足够的数据判断信号
        if len(self.rsi_values) < 2:
            return self.generate_signal('hold', close_price)

        # 获取前一个RSI值
        prev_rsi = self.rsi_values[-2] if len(self.rsi_values) > 1 else None

        # 生成信号
        signal = None

        # 超卖区域，产生买入信号
        if rsi < self.params['oversold'] and (prev_rsi is None or prev_rsi >= self.params['oversold']):
            if not self.params['use_signal_confirmation'] or self._confirm_signal('buy'):
                # 计算仓位大小（使用可用资金的50%进行交易）
                balance = 10000  # 默认余额
                if hasattr(self, 'broker') and hasattr(self.broker, 'get_balance'):
                    balance = self.broker.get_balance()
                position_size = (balance * 0.5) / close_price

                signal = self.generate_signal(
                    'buy',
                    price=close_price,
                    size=position_size,
                    stop_loss=close_price * (1 - self.params['stop_loss']),
                    take_profit=close_price * (1 + self.params['take_profit']),
                    info={'rsi': rsi}
                )
        # 超买区域，产生卖出信号
        elif rsi > self.params['overbought'] and (prev_rsi is None or prev_rsi <= self.params['overbought']):
            if not self.params['use_signal_confirmation'] or self._confirm_signal('sell'):
                # 获取当前持仓数量
                position_size = 0
                if hasattr(self, 'broker') and hasattr(self.broker, 'get_position'):
                    position_size = self.broker.get_position(bar.get('symbol', 'BTC/USDT'))
                else:
                    # 假设我们有仓位，全部卖出
                    position_size = 0.1

                signal = self.generate_signal(
                    'sell',
                    price=close_price,
                    size=position_size,
                    stop_loss=close_price * (1 + self.params['stop_loss']),
                    take_profit=close_price * (1 - self.params['take_profit']),
                    info={'rsi': rsi}
                )

        # 记录信号
        if signal:
            self.signals.append(signal)
            return signal

        return self.generate_signal('hold', close_price, info={'rsi': rsi})

    def _calculate_rsi(self, prices: list, period: int) -> float:
        """
        计算相对强弱指数(RSI)

        Args:
            prices: 价格序列
            period: 计算周期

        Returns:
            float: RSI值
        """
        if len(prices) < period + 1:
            return 50.0  # 默认中性值

        # 使用numpy实现RSI计算
        deltas = np.diff(prices[-period-1:])
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up / down if down != 0 else 1.0
        rsi = 100.0 - (100.0 / (1.0 + rs))

        return rsi

    def _confirm_signal(self, signal_type: str) -> bool:
        """
        确认信号是否有效

        Args:
            signal_type: 信号类型，'buy' 或 'sell'

        Returns:
            bool: 信号是否有效
        """
        if not self.params['use_signal_confirmation'] or not self.signals:
            return True

        # 获取最近的信号
        recent_signals = self.signals[-self.params['confirmation_period']:]

        # 检查最近的信号是否与当前信号类型相同
        for signal in recent_signals:
            if signal['signal'] == signal_type:
                return False

        return True

    def get_indicators(self) -> Dict[str, list]:
        """
        获取指标数据

        Returns:
            Dict: 包含指标数据的字典
        """
        return {
            'prices': self.prices,
            'rsi': self.rsi_values,
            'overbought': [self.params['overbought']] * len(self.prices) if self.prices else [],
            'oversold': [self.params['oversold']] * len(self.prices) if self.prices else []
        }
