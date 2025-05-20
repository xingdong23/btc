"""
双均线金叉死叉策略 (Dual Moving Average Crossover Strategy)

当短期均线上穿长期均线时产生买入信号，当短期均线下穿长期均线时产生卖出信号。
"""
import pandas as pd
from typing import Dict, Any, Optional

from .base import StrategyBase
from utils.logger import log


class MovingAverageCrossover(StrategyBase):
    """
    双均线金叉死叉策略

    参数:
        short_ma (int): 短期均线周期，默认为10
        long_ma (int): 长期均线周期，默认为30
        use_sma (bool): 是否使用SMA（简单移动平均），默认为True，False表示使用EMA
    """

    @classmethod
    def default_params(cls) -> Dict:
        return {
            'short_ma': 10,
            'long_ma': 30,
            'use_sma': True,  # True使用SMA，False使用EMA
            'stop_loss': 0.05,  # 5% 止损
            'take_profit': 0.1  # 10% 止盈
        }

    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.short_ma_values = []
        self.long_ma_values = []
        self.prices = []

    def init(self) -> None:
        """策略初始化"""
        super().init()
        log.info(f"{self.name} strategy initialized with parameters: {self.params}")

    def get_position_size(self, price: float, risk_percentage: float = 0.1) -> float:
        """
        计算仓位大小

        Args:
            price: 当前价格
            risk_percentage: 风险比例（0-1之间）

        Returns:
            float: 仓位大小（以基础货币计）
        """
        # 获取账户余额
        balance = self.broker.get_balance() if hasattr(self, 'broker') else 10000  # 默认10000 USDT

        # 计算可以用于交易的金额
        risk_amount = balance * risk_percentage

        # 计算可以购买的数量（以基础货币计）
        position_size = risk_amount / price

        return position_size

    def get_position(self, symbol: str, default: float = 0.0) -> float:
        """
        获取当前持仓数量

        Args:
            symbol: 交易对，如'BTC/USDT'
            default: 默认持仓数量（当无法获取持仓信息时返回）

        Returns:
            float: 当前持仓数量
        """
        if hasattr(self, 'broker') and hasattr(self.broker, 'get_position'):
            return self.broker.get_position(symbol)
        return default

    def on_bar(self, bar: Dict) -> Dict:
        """
        处理K线数据

        Args:
            bar: K线数据字典

        Returns:
            Dict: 交易信号
        """
        # 添加价格到序列
        close_price = bar['close']
        self.prices.append(close_price)

        # 确保有足够的数据计算均线
        if len(self.prices) < max(self.params['short_ma'], self.params['long_ma']) + 1:
            return self.generate_signal('hold', close_price)

        # 计算均线
        if self.params['use_sma']:
            # 简单移动平均
            short_ma = sum(self.prices[-self.params['short_ma']:]) / self.params['short_ma']
            long_ma = sum(self.prices[-self.params['long_ma']:]) / self.params['long_ma']
        else:
            # 指数移动平均
            short_ma = self._calculate_ema(self.prices, self.params['short_ma'])
            long_ma = self._calculate_ema(self.prices, self.params['long_ma'])

        # 保存均线值
        self.short_ma_values.append(short_ma)
        self.long_ma_values.append(long_ma)

        # 确保有足够的数据判断交叉
        if len(self.short_ma_values) < 2 or len(self.long_ma_values) < 2:
            return self.generate_signal('hold', close_price)

        # 获取前一根K线的均线值
        prev_short_ma = self.short_ma_values[-2]
        prev_long_ma = self.long_ma_values[-2]

        # 判断金叉死叉
        if prev_short_ma <= prev_long_ma and short_ma > long_ma:
            # 金叉，产生买入信号
            # 计算仓位大小（使用可用资金的50%进行交易）
            position_size = self.get_position_size(close_price, 0.5)  # 使用50%的可用资金
            return self.generate_signal(
                'buy',
                price=close_price,
                size=position_size,
                stop_loss=close_price * (1 - self.params['stop_loss']),
                take_profit=close_price * (1 + self.params['take_profit'])
            )
        elif prev_short_ma >= prev_long_ma and short_ma < long_ma:
            # 死叉，产生卖出信号
            # 卖出当前持有的全部仓位
            position_size = self.get_position(bar['symbol'], 0)  # 获取当前持仓数量
            return self.generate_signal(
                'sell',
                price=close_price,
                size=position_size,  # 卖出全部持仓
                stop_loss=close_price * (1 + self.params['stop_loss']),
                take_profit=close_price * (1 - self.params['take_profit'])
            )
        else:
            # 无信号
            return self.generate_signal('hold', close_price)

    def _calculate_ema(self, prices: list, period: int) -> float:
        """
        计算指数移动平均(EMA)

        Args:
            prices: 价格序列
            period: 计算周期

        Returns:
            float: EMA值
        """
        if len(prices) < period:
            return 0.0

        # 使用简单实现计算EMA
        multiplier = 2 / (period + 1)

        # 首先使用SMA作为第一个EMA值
        ema = sum(prices[-period:]) / period

        # 然后计算剩余的EMA
        for price in prices[-period+1:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def get_indicators(self) -> Dict[str, list]:
        """
        获取指标数据

        Returns:
            Dict: 包含指标数据的字典
        """
        return {
            'short_ma': self.short_ma_values,
            'long_ma': self.long_ma_values,
            'prices': self.prices
        }
