"""
Gate.io 现货网格交易策略

在预设的价格区间内，通过在多个网格点上自动进行低买高卖，捕捉小幅价格波动以获取利润。
"""
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import os
import json

from strategies.base import StrategyBase
from utils.logger import log
from utils.helpers import round_to_tick
from strategies.grid_trading.risk_manager import RiskManager
from strategies.grid_trading.order_tracker import OrderTracker


class GateioGridTrading(StrategyBase):
    """
    Gate.io 现货网格交易策略

    参数:
        symbol (str): 交易对，例如 "BTC_USDT" (Gate.io 现货格式)
        upper_price (float): 网格区间的价格上限
        lower_price (float): 网格区间的价格下限
        grid_num (int): 网格线的数量
        order_amount_base (float): 每个网格订单希望交易的基础货币数量
        max_initial_orders_side (int): 策略启动时，在当前市场价格的单侧最多预先挂出的网格订单数量
        check_interval_seconds (int): 策略主循环的执行间隔时间（秒）
        global_stop_loss_price (float, optional): 全局止损价格
        global_take_profit_price (float, optional): 全局止盈价格
    """

    @classmethod
    def default_params(cls) -> Dict:
        return {
            'symbol': 'BTC_USDT',
            'upper_price': 70000.0,
            'lower_price': 60000.0,
            'grid_num': 10,
            'order_amount_base': 0.001,
            'max_initial_orders_side': 5,
            'check_interval_seconds': 10,
            'global_stop_loss_price': None,
            'global_take_profit_price': None,
            'grid_size_pct': 2.0,  # 网格大小百分比
            'flip_threshold_pct': 0.4,  # 触发阈值百分比 (网格大小的1/5)
            'max_position_ratio': 0.9,  # 最大仓位比例 (90%)
            'min_position_ratio': 0.1,  # 最小仓位比例 (10%)
            'max_drawdown': 0.15,  # 最大回撤限制 (15%)
            'daily_loss_limit': 0.05,  # 每日亏损限制 (5%)
            'dynamic_grid': True,  # 是否启用动态网格
            'data_dir': 'data'  # 数据存储目录
        }

    def __init__(self, params: Dict = None):
        super().__init__(params)

        # 网格交易特有的状态变量
        self.grid_prices = []  # 网格价格点列表
        self.price_diff_per_grid = 0.0  # 每个网格的价格间距
        self.active_buy_orders = {}  # 活跃的买单 {order_id: {'price': price, 'amount': amount}}
        self.active_sell_orders = {}  # 活跃的卖单 {order_id: {'price': price, 'amount': amount}}
        self.processed_orders = set()  # 已处理的订单ID集合
        self.last_check_time = 0  # 上次检查时间
        self.price_precision = 8  # 价格精度，默认值
        self.amount_precision = 8  # 数量精度，默认值
        self.is_running = False  # 策略运行状态

        # 创建数据目录
        self.data_dir = self.params.get('data_dir', 'data')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # 初始化风控管理器和订单跟踪器
        self.risk_manager = RiskManager(self)
        self.order_tracker = OrderTracker(self.data_dir)

        # 价格相关变量
        self.current_price = None  # 当前价格
        self.highest_price = None  # 最高价格
        self.lowest_price = None  # 最低价格
        self.price_history = []  # 价格历史
        self.grid_size_pct = self.params.get('grid_size_pct', 2.0)  # 网格大小百分比
        self.flip_threshold_pct = self.params.get('flip_threshold_pct', 0.4)  # 触发阈值百分比

        # 交易相关变量
        self.last_trade_time = None  # 上次交易时间
        self.last_trade_price = None  # 上次交易价格
        self.last_trade_side = None  # 上次交易方向
        self.buying_or_selling = False  # 是否正在买入或卖出监测

        # 验证参数
        self._validate_params()

    def _validate_params(self) -> None:
        """验证策略参数"""
        if self.params['upper_price'] <= self.params['lower_price']:
            raise ValueError("upper_price必须大于lower_price")

        if self.params['grid_num'] < 2:
            raise ValueError("grid_num必须大于或等于2")

        if self.params['order_amount_base'] <= 0:
            raise ValueError("order_amount_base必须大于0")

        if self.params['max_initial_orders_side'] < 0:
            raise ValueError("max_initial_orders_side必须大于或等于0")

    def init(self) -> None:
        """策略初始化"""
        super().init()

        # 初始化风控管理器
        self.risk_manager.initialize()

        # 计算网格价格点
        self._calculate_grid_prices()

        # 获取交易对精度信息
        self._get_symbol_precision()

        # 获取当前市场价格
        if hasattr(self, 'broker') and hasattr(self.broker, 'exchange'):
            try:
                ccxt_symbol = self.params['symbol'].replace('_', '/')
                ticker = self.broker.exchange.fetch_ticker(ccxt_symbol)
                self.current_price = ticker['last']
                log.info(f"当前市场价格: {self.current_price}")
            except Exception as e:
                log.error(f"获取市场价格失败: {str(e)}")

        log.info(f"网格交易策略初始化完成，共{len(self.grid_prices)}个网格点")
        log.info(f"网格价格区间: {self.params['lower_price']} - {self.params['upper_price']}")
        log.info(f"每个网格价格间距: {self.price_diff_per_grid}")
        log.info(f"网格大小: {self.grid_size_pct}% | 触发阈值: {self.flip_threshold_pct}%")
        log.info(f"风控参数: 最大回撤={self.params['max_drawdown']*100}% | 日亏损限制={self.params['daily_loss_limit']*100}% | 最大仓位={self.params['max_position_ratio']*100}%")

    def _calculate_grid_prices(self) -> None:
        """计算网格价格点"""
        upper_price = self.params['upper_price']
        lower_price = self.params['lower_price']
        grid_num = self.params['grid_num']

        # 计算每个网格的价格间距
        self.price_diff_per_grid = (upper_price - lower_price) / (grid_num - 1)

        # 生成网格价格点
        self.grid_prices = [lower_price + i * self.price_diff_per_grid for i in range(grid_num)]

    def _get_symbol_precision(self) -> None:
        """获取交易对精度信息"""
        if hasattr(self, 'broker') and hasattr(self.broker, 'exchange'):
            try:
                # 将Gate.io格式的交易对转换为CCXT格式
                ccxt_symbol = self.params['symbol'].replace('_', '/')
                precision = self.broker.exchange.get_precision(ccxt_symbol)
                self.price_precision = precision['price']
                self.amount_precision = precision['amount']
                log.info(f"获取到交易对{ccxt_symbol}的精度信息: 价格精度={self.price_precision}, 数量精度={self.amount_precision}")
            except Exception as e:
                log.error(f"获取交易对精度信息失败: {str(e)}")
                log.info("使用默认精度: 价格精度=8, 数量精度=8")

    async def on_bar(self, bar: Dict) -> Dict:
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
        self.price_history.append(self.current_price)
        if len(self.price_history) > 100:  # 保留最近100个价格点
            self.price_history = self.price_history[-100:]

        # 网格交易策略不依赖K线数据生成信号，而是通过检查订单状态和市场价格来操作

        # 在回测环境中，每个K线都触发一次检查
        # 在实盘环境中，根据时间间隔检查
        is_backtest = not (hasattr(self, 'broker') and hasattr(self.broker, 'exchange'))
        current_time = int(time.time())

        if is_backtest or (current_time - self.last_check_time >= self.params['check_interval_seconds']):
            self.last_check_time = current_time

            # 如果策略尚未运行，则启动策略
            if not self.is_running:
                # 在回测中，根据当前价格范围调整网格上下限
                if is_backtest:
                    # 如果当前价格超出了网格范围，调整网格范围
                    if self.current_price > self.params['upper_price'] * 1.5 or self.current_price < self.params['lower_price'] * 0.5:
                        # 调整网格范围，使当前价格在网格范围内
                        price_range = self.params['upper_price'] - self.params['lower_price']
                        self.params['lower_price'] = self.current_price * 0.8  # 当前价格的8折作为下限
                        self.params['upper_price'] = self.params['lower_price'] + price_range  # 保持原来的价格范围大小

                        # 重新计算网格价格点
                        self._calculate_grid_prices()

                        log.info(f"调整网格范围为: {self.params['lower_price']} - {self.params['upper_price']}")

                self._start_grid_trading(self.current_price)
                self.is_running = True
            else:
                # 检查买入和卖出信号
                if await self._check_buy_signal():
                    await self._place_dynamic_buy_order()
                elif await self._check_sell_signal():
                    await self._place_dynamic_sell_order()
                else:
                    # 检查已成交订单并处理
                    if is_backtest:
                        # 在回测中，我们需要考虑K线的高低点来模拟订单成交
                        self._simulate_order_execution_with_range(bar)
                    else:
                        self._check_filled_orders(self.current_price)

                    # 检查风控
                    if await self.risk_manager.check_risk(self.current_price):
                        log.warning("触发风控保护，停止策略")
                        self._stop_strategy()
                    else:
                        # 检查全局止损/止盈
                        self._check_global_stop_loss_take_profit(self.current_price)

                        # 如果启用了动态网格，检查是否需要调整网格大小
                        if self.params.get('dynamic_grid', True):
                            self._adjust_grid_size()

        # 返回hold信号，因为网格交易不通过on_bar生成买卖信号
        return self.generate_signal('hold', self.current_price)

    def _start_grid_trading(self, current_price: float) -> None:
        """
        启动网格交易

        Args:
            current_price: 当前市场价格
        """
        log.info(f"启动网格交易，当前价格: {current_price}")

        # 取消所有未成交的订单（可选）
        if hasattr(self, 'broker') and hasattr(self.broker, 'exchange'):
            try:
                ccxt_symbol = self.params['symbol'].replace('_', '/')
                open_orders = self.broker.exchange.get_open_orders(ccxt_symbol)
                for order in open_orders:
                    self.broker.exchange.cancel_order(order['id'], ccxt_symbol)
                    log.info(f"取消订单: {order['id']}")
            except Exception as e:
                log.error(f"取消订单失败: {str(e)}")

        # 根据当前价格和max_initial_orders_side挂初始网格单
        self._place_initial_grid_orders(current_price)

    def _place_initial_grid_orders(self, current_price: float) -> None:
        """
        挂初始网格单

        Args:
            current_price: 当前市场价格
        """
        max_orders = self.params['max_initial_orders_side']

        # 找出低于当前价格的网格点
        buy_grid_prices = [p for p in self.grid_prices if p < current_price]
        buy_grid_prices.sort(reverse=True)  # 从高到低排序

        # 找出高于当前价格的网格点
        sell_grid_prices = [p for p in self.grid_prices if p > current_price]
        sell_grid_prices.sort()  # 从低到高排序

        # 挂买单
        for i, price in enumerate(buy_grid_prices):
            if i >= max_orders:
                break

            if price >= self.params['lower_price']:
                self._place_buy_order(price)

        # 挂卖单
        for i, price in enumerate(sell_grid_prices):
            if i >= max_orders:
                break

            if price <= self.params['upper_price']:
                self._place_sell_order(price)

    def _place_buy_order(self, price: float) -> None:
        """
        挂买单

        Args:
            price: 买单价格
        """
        # 格式化价格和数量
        formatted_price = round_to_tick(price, 10**(-self.price_precision))
        formatted_amount = round_to_tick(self.params['order_amount_base'], 10**(-self.amount_precision))

        # 检查是否已经在此价格点有活跃买单
        for order_info in self.active_buy_orders.values():
            if abs(order_info['price'] - formatted_price) < 0.0000001:
                log.info(f"价格{formatted_price}已有活跃买单，跳过")
                return

        # 生成唯一的订单ID
        order_id = f"buy_{int(time.time() * 1000)}_{formatted_price}"

        # 判断是否在回测环境中
        if hasattr(self, 'broker') and hasattr(self.broker, 'exchange'):
            try:
                # 实盘环境，使用交易所API创建订单
                ccxt_symbol = self.params['symbol'].replace('_', '/')
                order = self.broker.exchange.create_order(
                    symbol=ccxt_symbol,
                    order_type='limit',
                    side='buy',
                    amount=formatted_amount,
                    price=formatted_price
                )
                order_id = order['id']
                log.info(f"实盘挂买单成功: 价格={formatted_price}, 数量={formatted_amount}, 订单ID={order_id}")
            except Exception as e:
                log.error(f"实盘挂买单失败: {str(e)}")
                return
        else:
            # 回测环境，模拟订单创建
            log.info(f"回测模式挂买单: 价格={formatted_price}, 数量={formatted_amount}, 订单ID={order_id}")

        # 记录订单
        self.active_buy_orders[order_id] = {
            'price': formatted_price,
            'amount': formatted_amount,
            'timestamp': int(time.time() * 1000),
            'id': order_id,
            'side': 'buy',
            'status': 'open'
        }

    def _place_sell_order(self, price: float) -> None:
        """
        挂卖单

        Args:
            price: 卖单价格
        """
        # 格式化价格和数量
        formatted_price = round_to_tick(price, 10**(-self.price_precision))
        formatted_amount = round_to_tick(self.params['order_amount_base'], 10**(-self.amount_precision))

        # 检查是否已经在此价格点有活跃卖单
        for order_info in self.active_sell_orders.values():
            if abs(order_info['price'] - formatted_price) < 0.0000001:
                log.info(f"价格{formatted_price}已有活跃卖单，跳过")
                return

        # 生成唯一的订单ID
        order_id = f"sell_{int(time.time() * 1000)}_{formatted_price}"

        # 判断是否在回测环境中
        if hasattr(self, 'broker') and hasattr(self.broker, 'exchange'):
            try:
                # 实盘环境，使用交易所API创建订单
                ccxt_symbol = self.params['symbol'].replace('_', '/')
                order = self.broker.exchange.create_order(
                    symbol=ccxt_symbol,
                    order_type='limit',
                    side='sell',
                    amount=formatted_amount,
                    price=formatted_price
                )
                order_id = order['id']
                log.info(f"实盘挂卖单成功: 价格={formatted_price}, 数量={formatted_amount}, 订单ID={order_id}")
            except Exception as e:
                log.error(f"实盘挂卖单失败: {str(e)}")
                return
        else:
            # 回测环境，模拟订单创建
            log.info(f"回测模式挂卖单: 价格={formatted_price}, 数量={formatted_amount}, 订单ID={order_id}")

        # 记录订单
        self.active_sell_orders[order_id] = {
            'price': formatted_price,
            'amount': formatted_amount,
            'timestamp': int(time.time() * 1000),
            'id': order_id,
            'side': 'sell',
            'status': 'open'
        }

    async def _check_buy_signal(self) -> bool:
        """
        检查买入信号

        当价格低于网格下轨时，进入买入监测模式。
        当价格从最低点反弹超过指定阈值时，触发买入信号。

        Returns:
            bool: 是否触发买入信号
        """
        if not self.current_price:
            return False

        # 计算网格下轨
        lower_band = self._get_lower_band()

        if self.current_price <= lower_band:
            # 进入买入监测模式
            self.buying_or_selling = True

            # 更新最低价格
            if self.lowest_price is None or self.current_price < self.lowest_price:
                self.lowest_price = self.current_price
                log.info(
                    f"买入监测 | "
                    f"当前价: {self.current_price:.2f} | "
                    f"触发价: {lower_band:.2f} | "
                    f"最低价: {self.lowest_price:.2f} | "
                    f"反弹阈值: {self.flip_threshold_pct:.2f}%"
                )

            # 计算反弹阈值
            threshold = self.flip_threshold_pct / 100.0

            # 如果价格从最低点反弹超过阈值，触发买入信号
            if self.lowest_price and self.current_price >= self.lowest_price * (1 + threshold):
                self.buying_or_selling = False  # 重置监测状态
                log.info(f"触发买入信号 | 当前价: {self.current_price:.2f} | 已反弹: {(self.current_price/self.lowest_price-1)*100:.2f}%")

                # 重置最低价格
                self.lowest_price = None

                return True
        else:
            # 价格高于网格下轨，退出买入监测模式
            if self.buying_or_selling:
                self.buying_or_selling = False
                self.lowest_price = None

        return False

    async def _check_sell_signal(self) -> bool:
        """
        检查卖出信号

        当价格高于网格上轨时，进入卖出监测模式。
        当价格从最高点回调超过指定阈值时，触发卖出信号。

        Returns:
            bool: 是否触发卖出信号
        """
        if not self.current_price:
            return False

        # 计算网格上轨
        upper_band = self._get_upper_band()

        if self.current_price >= upper_band:
            # 进入卖出监测模式
            self.buying_or_selling = True

            # 更新最高价格
            if self.highest_price is None or self.current_price > self.highest_price:
                self.highest_price = self.current_price
                log.info(
                    f"卖出监测 | "
                    f"当前价: {self.current_price:.2f} | "
                    f"触发价: {upper_band:.2f} | "
                    f"最高价: {self.highest_price:.2f} | "
                    f"回调阈值: {self.flip_threshold_pct:.2f}%"
                )

            # 计算回调阈值
            threshold = self.flip_threshold_pct / 100.0

            # 如果价格从最高点回调超过阈值，触发卖出信号
            if self.highest_price and self.current_price <= self.highest_price * (1 - threshold):
                self.buying_or_selling = False  # 重置监测状态
                log.info(f"触发卖出信号 | 当前价: {self.current_price:.2f} | 已回调: {(1-self.current_price/self.highest_price)*100:.2f}%")

                # 重置最高价格
                self.highest_price = None

                return True
        else:
            # 价格低于网格上轨，退出卖出监测模式
            if self.buying_or_selling:
                self.buying_or_selling = False
                self.highest_price = None

        return False

    def _get_upper_band(self) -> float:
        """获取网格上轨价格"""
        return self.params['upper_price'] * (1 - self.grid_size_pct / 100.0)

    def _get_lower_band(self) -> float:
        """获取网格下轨价格"""
        return self.params['lower_price'] * (1 + self.grid_size_pct / 100.0)