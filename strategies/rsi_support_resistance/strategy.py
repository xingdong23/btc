"""
RSI和支撑阻力日内交易策略

结合RSI指标和支撑阻力水平进行日内交易，捕捉短期价格波动。
"""
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from strategies.base import StrategyBase
from utils.logger import log
from utils.helpers import round_to_tick, calculate_position_size
from strategies.rsi_support_resistance.support_resistance import SupportResistanceDetector


class RSISupportResistanceStrategy(StrategyBase):
    """
    RSI和支撑阻力日内交易策略

    策略逻辑：
    1. 使用RSI指标识别超买超卖区域
    2. 结合支撑阻力水平确认交易信号
    3. 在支撑位附近RSI超卖时买入
    4. 在阻力位附近RSI超买时卖出
    5. 设置动态止损和止盈
    """

    @classmethod
    def default_params(cls) -> Dict:
        """
        返回策略默认参数

        Returns:
            Dict: 参数字典
        """
        return {
            'symbol': 'BTC_USDT',
            'rsi_period': 14,  # RSI计算周期
            'rsi_oversold': 30,  # RSI超卖阈值
            'rsi_overbought': 70,  # RSI超买阈值
            'rsi_exit_oversold': 50,  # RSI退出超卖区域阈值
            'rsi_exit_overbought': 50,  # RSI退出超买区域阈值
            'sr_lookback_period': 100,  # 支撑阻力回溯周期
            'sr_price_threshold': 0.02,  # 支撑阻力价格阈值
            'sr_touch_count': 2,  # 支撑阻力触及次数
            'sr_zone_merge_threshold': 0.01,  # 支撑阻力区域合并阈值
            'risk_per_trade': 0.02,  # 每笔交易风险比例 (2%)
            'reward_risk_ratio': 2.0,  # 盈亏比
            'max_position_ratio': 0.8,  # 最大仓位比例 (80%)
            'max_trades_per_day': 5,  # 每日最大交易次数
            'min_trade_interval': 3600,  # 最小交易间隔 (秒)
            'use_dynamic_exit': True,  # 是否使用动态止盈止损
            'trailing_stop_pct': 0.02,  # 追踪止损百分比 (2%)
            'confirmation_period': 3,  # 信号确认周期
            'time_in_market_limit': 24,  # 最长持仓时间 (小时)
            'data_dir': 'data'  # 数据存储目录
        }

    def __init__(self, params: Dict = None):
        """
        初始化策略

        Args:
            params: 策略参数字典
        """
        super().__init__(params)

        # 初始化支撑阻力检测器
        self.sr_detector = SupportResistanceDetector(
            lookback_period=self.params['sr_lookback_period'],
            price_threshold=self.params['sr_price_threshold'],
            touch_count=self.params['sr_touch_count'],
            zone_merge_threshold=self.params['sr_zone_merge_threshold']
        )

        # 价格和指标数据
        self.price_data = []  # 价格历史数据
        self.rsi_values = []  # RSI值历史
        self.current_price = None  # 当前价格
        self.current_rsi = None  # 当前RSI值

        # 交易状态
        self.in_position = False  # 是否持仓
        self.position_size = 0.0  # 持仓数量
        self.entry_price = 0.0  # 入场价格
        self.entry_time = None  # 入场时间
        self.stop_loss = 0.0  # 止损价格
        self.take_profit = 0.0  # 止盈价格
        self.trailing_stop = 0.0  # 追踪止损价格
        self.highest_since_entry = 0.0  # 入场后最高价
        self.lowest_since_entry = float('inf')  # 入场后最低价

        # 信号确认
        self.buy_signals = 0  # 连续买入信号计数
        self.sell_signals = 0  # 连续卖出信号计数

        # 交易限制
        self.trades_today = 0  # 今日交易次数
        self.last_trade_time = 0  # 上次交易时间
        self.last_trade_date = None  # 上次交易日期

        # 精度设置
        self.price_precision = 2  # 价格精度
        self.amount_precision = 6  # 数量精度

    def init(self) -> None:
        """策略初始化"""
        super().init()
        log.info(f"RSI和支撑阻力日内交易策略初始化 | 参数: {self.params}")

    def on_bar(self, bar: Dict) -> Dict:
        """
        处理K线数据

        Args:
            bar: K线数据字典

        Returns:
            Dict: 交易信号
        """
        # 更新当前价格
        self.current_price = bar['close']

        # 更新价格历史
        self.price_data.append({
            'timestamp': bar['timestamp'],
            'open': bar['open'],
            'high': bar['high'],
            'low': bar['low'],
            'close': bar['close'],
            'volume': bar['volume']
        })

        # 保持价格历史在合理范围内
        max_history = max(self.params['sr_lookback_period'], self.params['rsi_period'] * 3)
        if len(self.price_data) > max_history:
            self.price_data = self.price_data[-max_history:]

        # 更新支撑阻力水平
        # 如果数据足够，则更新支撑阻力水平
        if len(self.price_data) >= 20:  # 至少需要一定数量的数据
            self.sr_detector.update(self.price_data)

            # 如果是第一次更新，输出日志
            if len(self.sr_detector.support_levels) > 0 or len(self.sr_detector.resistance_levels) > 0:
                log.info(f"支撑阻力水平更新 | 支撑: {len(self.sr_detector.support_levels)} 个 | 阻力: {len(self.sr_detector.resistance_levels)} 个")

        # 计算RSI
        self._calculate_rsi()

        # 检查是否是新的交易日
        current_date = datetime.fromtimestamp(bar['timestamp'] / 1000).date()
        if self.last_trade_date is None or current_date != self.last_trade_date:
            self.trades_today = 0
            self.last_trade_date = current_date

        # 检查持仓状态
        if self.in_position:
            # 更新追踪止损
            self._update_trailing_stop()

            # 检查止损和止盈
            exit_signal = self._check_exit_conditions(bar)
            if exit_signal:
                return exit_signal
        else:
            # 检查入场信号
            entry_signal = self._check_entry_conditions(bar)
            if entry_signal:
                return entry_signal

        # 默认返回持有信号
        return self.generate_signal('hold', self.current_price)

    def _calculate_rsi(self) -> None:
        """计算RSI指标"""
        if len(self.price_data) < self.params['rsi_period'] + 1:
            return

        # 提取收盘价
        closes = [data['close'] for data in self.price_data]

        # 计算价格变化
        deltas = np.diff(closes)

        # 分离上涨和下跌
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # 初始平均值
        avg_gain = np.mean(gains[:self.params['rsi_period']])
        avg_loss = np.mean(losses[:self.params['rsi_period']])

        # 计算后续值
        for i in range(self.params['rsi_period'], len(deltas)):
            avg_gain = (avg_gain * (self.params['rsi_period'] - 1) + gains[i]) / self.params['rsi_period']
            avg_loss = (avg_loss * (self.params['rsi_period'] - 1) + losses[i]) / self.params['rsi_period']

        # 计算相对强度
        if avg_loss == 0:
            rs = float('inf')
        else:
            rs = avg_gain / avg_loss

        # 计算RSI
        rsi = 100 - (100 / (1 + rs))

        # 更新当前RSI值
        self.current_rsi = rsi

        # 更新RSI历史
        self.rsi_values.append(rsi)

        # 保持RSI历史在合理范围内
        if len(self.rsi_values) > max(50, self.params['confirmation_period'] * 2):
            self.rsi_values = self.rsi_values[-50:]

    def _check_entry_conditions(self, bar: Dict) -> Optional[Dict]:
        """
        检查入场条件

        Args:
            bar: K线数据

        Returns:
            Optional[Dict]: 入场信号或None
        """
        # 检查交易次数限制
        if self.trades_today >= self.params['max_trades_per_day']:
            return None

        # 检查交易间隔
        current_time = bar['timestamp'] / 1000
        if current_time - self.last_trade_time < self.params['min_trade_interval']:
            return None

        # 检查RSI和支撑阻力
        if self.current_rsi is None:
            return None

        # 确保有足够的数据进行交易决策
        if len(self.price_data) < 30 or len(self.rsi_values) < 5:
            return None

        # 获取支撑阻力水平
        nearest_support, nearest_resistance = self.sr_detector.get_nearest_levels(self.current_price)

        # 如果没有支撑阻力水平，则不交易
        if not nearest_support and not nearest_resistance:
            return None

        # 买入条件：RSI超卖且接近支撑位
        if (self.current_rsi <= self.params['rsi_oversold'] and
            self.sr_detector.is_near_support(self.current_price)):

            # 增加买入信号计数
            self.buy_signals += 1

            # 检查信号确认
            if self.buy_signals >= self.params['confirmation_period']:
                # 计算仓位大小
                account_balance = 10000.0  # 假设账户余额，实盘中应从交易所获取

                # 计算止损价格 (支撑位下方)
                if nearest_support:
                    stop_loss = nearest_support * 0.98  # 支撑位下方2%
                else:
                    stop_loss = self.current_price * (1 - self.params['trailing_stop_pct'])

                # 计算头寸大小
                position_size, risk_amount = calculate_position_size(
                    account_balance=account_balance,
                    risk_per_trade=self.params['risk_per_trade'],
                    entry_price=self.current_price,
                    stop_loss_price=stop_loss
                ), account_balance * self.params['risk_per_trade']

                # 计算止盈价格
                if nearest_resistance:
                    # 使用最近的阻力位作为止盈
                    take_profit = nearest_resistance
                else:
                    # 使用盈亏比计算止盈
                    price_risk = self.current_price - stop_loss
                    take_profit = self.current_price + price_risk * self.params['reward_risk_ratio']

                # 更新交易状态
                self.in_position = True
                self.position_size = position_size
                self.entry_price = self.current_price
                self.entry_time = datetime.fromtimestamp(bar['timestamp'] / 1000)
                self.stop_loss = stop_loss
                self.take_profit = take_profit
                self.trailing_stop = stop_loss
                self.highest_since_entry = self.current_price
                self.lowest_since_entry = self.current_price

                # 更新交易限制
                self.trades_today += 1
                self.last_trade_time = current_time

                # 重置信号计数
                self.buy_signals = 0
                self.sell_signals = 0

                # 生成买入信号
                log.info(f"买入信号 | 价格: {self.current_price:.2f} | RSI: {float(self.current_rsi):.1f} | "
                         f"支撑: {nearest_support:.2f if nearest_support is not None else 'N/A'} | "
                         f"止损: {stop_loss:.2f} | 止盈: {take_profit:.2f}")

                return self.generate_signal(
                    'buy',
                    self.current_price,
                    self.position_size,
                    self.stop_loss,
                    self.take_profit,
                    {
                        'rsi': self.current_rsi,
                        'support': nearest_support,
                        'resistance': nearest_resistance
                    }
                )
        else:
            # 重置买入信号计数
            self.buy_signals = 0

        return None

    def _check_exit_conditions(self, bar: Dict) -> Optional[Dict]:
        """
        检查出场条件

        Args:
            bar: K线数据

        Returns:
            Optional[Dict]: 出场信号或None
        """
        if not self.in_position:
            return None

        # 确保有足够的数据进行交易决策
        if self.current_rsi is None or len(self.price_data) < 20:
            return None

        # 更新入场后的最高最低价
        self.highest_since_entry = max(self.highest_since_entry, bar['high'])
        self.lowest_since_entry = min(self.lowest_since_entry, bar['low'])

        # 获取支撑阻力水平
        nearest_support, nearest_resistance = self.sr_detector.get_nearest_levels(self.current_price)

        # 检查止损
        if self.current_price <= self.trailing_stop:
            # 触发止损
            log.info(f"触发止损 | 价格: {float(self.current_price):.2f} | 止损价: {float(self.trailing_stop):.2f} | "
                     f"入场价: {float(self.entry_price):.2f} | 盈亏: {(self.current_price/self.entry_price-1)*100:.2f}%")

            # 更新交易状态
            self.in_position = False

            # 生成卖出信号
            return self.generate_signal(
                'sell',
                self.current_price,
                self.position_size,
                None,
                None,
                {
                    'exit_reason': 'stop_loss',
                    'entry_price': self.entry_price,
                    'profit_pct': (self.current_price / self.entry_price - 1) * 100
                }
            )

        # 检查止盈
        if self.current_price >= self.take_profit:
            # 触发止盈
            log.info(f"触发止盈 | 价格: {float(self.current_price):.2f} | 止盈价: {float(self.take_profit):.2f} | "
                     f"入场价: {float(self.entry_price):.2f} | 盈亏: {(self.current_price/self.entry_price-1)*100:.2f}%")

            # 更新交易状态
            self.in_position = False

            # 生成卖出信号
            return self.generate_signal(
                'sell',
                self.current_price,
                self.position_size,
                None,
                None,
                {
                    'exit_reason': 'take_profit',
                    'entry_price': self.entry_price,
                    'profit_pct': (self.current_price / self.entry_price - 1) * 100
                }
            )

        # 检查RSI超买且接近阻力位
        if (self.current_rsi >= self.params['rsi_overbought'] and
            self.sr_detector.is_near_resistance(self.current_price)):

            # 增加卖出信号计数
            self.sell_signals += 1

            # 检查信号确认
            if self.sell_signals >= self.params['confirmation_period']:
                # 触发RSI超买卖出
                log.info(f"RSI超买卖出 | 价格: {self.current_price:.2f} | RSI: {float(self.current_rsi):.1f} | "
                         f"阻力: {nearest_resistance:.2f if nearest_resistance is not None else 'N/A'} | "
                         f"入场价: {self.entry_price:.2f} | 盈亏: {(self.current_price/self.entry_price-1)*100:.2f}%")

                # 更新交易状态
                self.in_position = False

                # 重置信号计数
                self.buy_signals = 0
                self.sell_signals = 0

                # 生成卖出信号
                return self.generate_signal(
                    'sell',
                    self.current_price,
                    self.position_size,
                    None,
                    None,
                    {
                        'exit_reason': 'rsi_overbought',
                        'rsi': self.current_rsi,
                        'entry_price': self.entry_price,
                        'profit_pct': (self.current_price / self.entry_price - 1) * 100
                    }
                )
        else:
            # 重置卖出信号计数
            self.sell_signals = 0

        # 检查持仓时间限制
        if self.entry_time:
            hours_in_position = (datetime.now() - self.entry_time).total_seconds() / 3600
            if hours_in_position >= self.params['time_in_market_limit']:
                # 触发持仓时间限制
                log.info(f"持仓时间限制卖出 | 价格: {float(self.current_price):.2f} | "
                         f"持仓时间: {float(hours_in_position):.1f}小时 | "
                         f"入场价: {float(self.entry_price):.2f} | 盈亏: {(self.current_price/self.entry_price-1)*100:.2f}%")

                # 更新交易状态
                self.in_position = False

                # 生成卖出信号
                return self.generate_signal(
                    'sell',
                    self.current_price,
                    self.position_size,
                    None,
                    None,
                    {
                        'exit_reason': 'time_limit',
                        'hours_in_position': hours_in_position,
                        'entry_price': self.entry_price,
                        'profit_pct': (self.current_price / self.entry_price - 1) * 100
                    }
                )

        return None

    def _update_trailing_stop(self) -> None:
        """更新追踪止损"""
        if not self.in_position or not self.params['use_dynamic_exit']:
            return

        # 计算新的追踪止损
        new_stop = self.highest_since_entry * (1 - self.params['trailing_stop_pct'])

        # 仅上移止损，不下移
        if new_stop > self.trailing_stop:
            self.trailing_stop = new_stop
            log.info(f"更新追踪止损 | 新止损: {float(self.trailing_stop):.2f} | "
                     f"当前价: {float(self.current_price):.2f} | 最高价: {float(self.highest_since_entry):.2f}")

    def get_strategy_state(self) -> Dict:
        """
        获取策略状态

        Returns:
            Dict: 策略状态字典
        """
        return {
            'in_position': self.in_position,
            'position_size': self.position_size,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'current_rsi': self.current_rsi,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'trailing_stop': self.trailing_stop,
            'highest_since_entry': self.highest_since_entry,
            'lowest_since_entry': self.lowest_since_entry,
            'trades_today': self.trades_today,
            'support_levels': self.sr_detector.support_levels,
            'resistance_levels': self.sr_detector.resistance_levels
        }
