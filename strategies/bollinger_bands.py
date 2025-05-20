"""
布林带策略 (Bollinger Bands Strategy)

当价格触及布林带下轨时产生买入信号，当价格触及布林带上轨时产生卖出信号。
也可以选择当价格突破上轨时买入，跌破下轨时卖出。
"""
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from .base import StrategyBase
from utils.logger import log


class BollingerBandsStrategy(StrategyBase):
    """
    布林带策略

    参数:
        period (int): 布林带计算周期，默认为20
        std_dev (float): 标准差倍数，默认为2.0
        use_breakout (bool): 是否使用突破策略，默认为False
                            如果为True，则价格突破上轨买入，跌破下轨卖出
                            如果为False，则价格触及下轨买入，触及上轨卖出
        use_middle_band (bool): 是否使用中轨作为退出信号，默认为True
        stop_loss (float): 止损比例，默认为0.05 (5%)
        take_profit (float): 止盈比例，默认为0.1 (10%)
    """

    @classmethod
    def default_params(cls) -> Dict:
        return {
            'period': 20,
            'std_dev': 2.0,
            'use_breakout': False,
            'use_middle_band': True,
            'stop_loss': 0.05,  # 5% 止损
            'take_profit': 0.1  # 10% 止盈
        }

    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.prices = []
        self.upper_band = []
        self.middle_band = []
        self.lower_band = []
        self.positions = []

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

        # 确保有足够的数据计算布林带
        if len(self.prices) < self.params['period']:
            return self.generate_signal('hold', close_price)

        # 计算布林带
        upper, middle, lower = self._calculate_bollinger_bands(
            self.prices,
            self.params['period'],
            self.params['std_dev']
        )

        # 保存指标值
        self.upper_band.append(upper)
        self.middle_band.append(middle)
        self.lower_band.append(lower)

        # 确保有足够的数据判断信号
        if len(self.prices) < 2 or len(self.upper_band) < 2:
            return self.generate_signal('hold', close_price)

        # 获取前一个价格和布林带值
        prev_price = self.prices[-2]
        prev_upper = self.upper_band[-2]
        prev_lower = self.lower_band[-2]

        # 生成信号
        signal = None

        if self.params['use_breakout']:
            # 突破策略：突破上轨买入，跌破下轨卖出
            if prev_price <= prev_upper and close_price > upper:
                signal = self._generate_buy_signal(close_price)
            elif prev_price >= prev_lower and close_price < lower:
                signal = self._generate_sell_signal(close_price)
        else:
            # 回归策略：触及下轨买入，触及上轨卖出
            if prev_price > prev_lower and close_price <= lower:
                signal = self._generate_buy_signal(close_price)
            elif prev_price < prev_upper and close_price >= upper:
                signal = self._generate_sell_signal(close_price)

        # 检查中轨退出信号
        if self.params['use_middle_band'] and self.position != 0:
            if (self.position > 0 and close_price <= middle) or \
               (self.position < 0 and close_price >= middle):
                signal = self.generate_signal(
                    'sell' if self.position > 0 else 'buy',
                    price=close_price,
                    info={'reason': 'middle_band_exit'}
                )

        # 记录持仓状态
        self.positions.append(self.position)

        return signal if signal else self.generate_signal('hold', close_price)

    def _calculate_bollinger_bands(self, prices: list, period: int, std_dev: float) -> Tuple[float, float, float]:
        """
        计算布林带

        Args:
            prices: 价格序列
            period: 计算周期
            std_dev: 标准差倍数

        Returns:
            Tuple[float, float, float]: (上轨, 中轨, 下轨)
        """
        if len(prices) < period:
            return 0.0, 0.0, 0.0

        # 使用numpy实现布林带计算
        slice_prices = prices[-period:]
        middle = np.mean(slice_prices)
        std = np.std(slice_prices, ddof=1)
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    def _generate_buy_signal(self, price: float) -> Dict:
        """生成买入信号"""
        # 计算仓位大小（使用可用资金的50%进行交易）
        balance = 10000  # 默认余额
        if hasattr(self, 'broker') and hasattr(self.broker, 'get_balance'):
            balance = self.broker.get_balance()
        position_size = (balance * 0.5) / price

        return self.generate_signal(
            'buy',
            price=price,
            size=position_size,
            stop_loss=price * (1 - self.params['stop_loss']),
            take_profit=price * (1 + self.params['take_profit']),
            info={
                'strategy': 'bollinger_bands',
                'type': 'breakout' if self.params['use_breakout'] else 'reversion'
            }
        )

    def _generate_sell_signal(self, price: float) -> Dict:
        """生成卖出信号"""
        # 获取当前持仓数量
        position_size = 0
        if hasattr(self, 'broker') and hasattr(self.broker, 'get_position'):
            position_size = self.broker.get_position('BTC/USDT')
        else:
            # 假设我们有仓位，全部卖出
            position_size = 0.1

        return self.generate_signal(
            'sell',
            price=price,
            size=position_size,
            stop_loss=price * (1 + self.params['stop_loss']),
            take_profit=price * (1 - self.params['take_profit']),
            info={
                'strategy': 'bollinger_bands',
                'type': 'breakout' if self.params['use_breakout'] else 'reversion'
            }
        )

    def get_indicators(self) -> Dict[str, list]:
        """
        获取指标数据

        Returns:
            Dict: 包含指标数据的字典
        """
        return {
            'prices': self.prices,
            'upper_band': self.upper_band,
            'middle_band': self.middle_band,
            'lower_band': self.lower_band,
            'positions': self.positions
        }
