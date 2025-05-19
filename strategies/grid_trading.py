"""
Gate.io 现货网格交易策略

在预设的价格区间内，通过在多个网格点上自动进行低买高卖，捕捉小幅价格波动以获取利润。
"""
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from .base import StrategyBase
from utils.logger import log
from utils.helpers import round_to_tick


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
            'global_take_profit_price': None
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

        # 计算网格价格点
        self._calculate_grid_prices()

        # 获取交易对精度信息
        self._get_symbol_precision()

        log.info(f"网格交易策略初始化完成，共{len(self.grid_prices)}个网格点")
        log.info(f"网格价格区间: {self.params['lower_price']} - {self.params['upper_price']}")
        log.info(f"每个网格价格间距: {self.price_diff_per_grid}")
        log.info(f"网格价格点: {self.grid_prices}")

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

    def on_bar(self, bar: Dict) -> Dict:
        """
        处理K线数据

        Args:
            bar: K线数据字典

        Returns:
            Dict: 交易信号
        """
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
                    # 获取当前价格
                    current_price = bar['close']

                    # 如果当前价格超出了网格范围，调整网格范围
                    if current_price > self.params['upper_price'] * 1.5 or current_price < self.params['lower_price'] * 0.5:
                        # 调整网格范围，使当前价格在网格范围内
                        price_range = self.params['upper_price'] - self.params['lower_price']
                        self.params['lower_price'] = current_price * 0.8  # 当前价格的8折作为下限
                        self.params['upper_price'] = self.params['lower_price'] + price_range  # 保持原来的价格范围大小

                        # 重新计算网格价格点
                        self._calculate_grid_prices()

                        log.info(f"调整网格范围为: {self.params['lower_price']} - {self.params['upper_price']}")

                self._start_grid_trading(bar['close'])
                self.is_running = True
            else:
                # 检查已成交订单并处理
                if is_backtest:
                    # 在回测中，我们需要考虑K线的高低点来模拟订单成交
                    self._simulate_order_execution_with_range(bar)
                else:
                    self._check_filled_orders(bar['close'])

                # 可选：检查全局止损/止盈
                self._check_global_stop_loss_take_profit(bar['close'])

        # 返回hold信号，因为网格交易不通过on_bar生成买卖信号
        return self.generate_signal('hold', bar['close'])

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

    def _check_filled_orders(self, current_price: float) -> None:
        """
        检查已成交的订单并处理

        Args:
            current_price: 当前市场价格
        """
        # 判断是否在回测环境中
        if hasattr(self, 'broker') and hasattr(self.broker, 'exchange'):
            # 实盘环境，使用交易所API获取订单状态
            try:
                ccxt_symbol = self.params['symbol'].replace('_', '/')

                # 获取所有订单
                orders = []

                # 获取活跃买单的状态
                for order_id in list(self.active_buy_orders.keys()):
                    try:
                        order = self.broker.exchange.get_order(order_id, ccxt_symbol)
                        orders.append(order)
                    except Exception as e:
                        log.error(f"获取买单{order_id}状态失败: {str(e)}")
                        # 如果订单不存在，从活跃订单列表中移除
                        self.active_buy_orders.pop(order_id, None)

                # 获取活跃卖单的状态
                for order_id in list(self.active_sell_orders.keys()):
                    try:
                        order = self.broker.exchange.get_order(order_id, ccxt_symbol)
                        orders.append(order)
                    except Exception as e:
                        log.error(f"获取卖单{order_id}状态失败: {str(e)}")
                        # 如果订单不存在，从活跃订单列表中移除
                        self.active_sell_orders.pop(order_id, None)

                # 处理已成交的订单
                for order in orders:
                    if order['id'] in self.processed_orders:
                        continue

                    if order['status'] == 'closed':
                        self._handle_filled_order(order)
                        self.processed_orders.add(order['id'])

            except Exception as e:
                log.error(f"检查订单状态失败: {str(e)}")
        else:
            # 回测环境，模拟订单成交
            self._simulate_order_execution(current_price)

    def _simulate_order_execution(self, current_price: float) -> None:
        """
        模拟订单成交

        Args:
            current_price: 当前市场价格
        """
        # 检查买单是否成交
        for order_id, order_info in list(self.active_buy_orders.items()):
            if order_id in self.processed_orders:
                continue

            # 如果当前价格低于或等于买单价格，则认为买单成交
            # 注意：在回测中，我们需要检查当前价格是否在这根K线的价格范围内触及了买单价格
            if current_price <= order_info['price']:
                # 创建模拟成交订单
                filled_order = {
                    'id': order_id,
                    'side': 'buy',
                    'price': order_info['price'],
                    'amount': order_info['amount'],
                    'status': 'closed',
                    'timestamp': int(time.time() * 1000),
                    'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'cost': order_info['price'] * order_info['amount'],
                    'fee': order_info['price'] * order_info['amount'] * 0.001  # 假设0.1%手续费
                }

                # 处理成交订单
                self._handle_filled_order(filled_order)
                self.processed_orders.add(order_id)

                # 记录交易
                trade = {
                    'datetime': datetime.now(),
                    'timestamp': int(time.time() * 1000),
                    'symbol': self.params['symbol'],
                    'type': 'buy',
                    'side': 'buy',
                    'price': order_info['price'],
                    'amount': order_info['amount'],
                    'cost': order_info['price'] * order_info['amount'],
                    'fee': order_info['price'] * order_info['amount'] * 0.001
                }
                self.trades.append(trade)

                log.info(f"回测模式买单成交: {order_id}, 价格={order_info['price']}, 数量={order_info['amount']}")

        # 检查卖单是否成交
        for order_id, order_info in list(self.active_sell_orders.items()):
            if order_id in self.processed_orders:
                continue

            # 如果当前价格高于或等于卖单价格，则认为卖单成交
            # 注意：在回测中，我们需要检查当前价格是否在这根K线的价格范围内触及了卖单价格
            if current_price >= order_info['price']:
                # 创建模拟成交订单
                filled_order = {
                    'id': order_id,
                    'side': 'sell',
                    'price': order_info['price'],
                    'amount': order_info['amount'],
                    'status': 'closed',
                    'timestamp': int(time.time() * 1000),
                    'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'cost': order_info['price'] * order_info['amount'],
                    'fee': order_info['price'] * order_info['amount'] * 0.001  # 假设0.1%手续费
                }

                # 处理成交订单
                self._handle_filled_order(filled_order)
                self.processed_orders.add(order_id)

                # 记录交易
                trade = {
                    'datetime': datetime.now(),
                    'timestamp': int(time.time() * 1000),
                    'symbol': self.params['symbol'],
                    'type': 'sell',
                    'side': 'sell',
                    'price': order_info['price'],
                    'amount': order_info['amount'],
                    'cost': order_info['price'] * order_info['amount'],
                    'fee': order_info['price'] * order_info['amount'] * 0.001
                }
                self.trades.append(trade)

                log.info(f"回测模式卖单成交: {order_id}, 价格={order_info['price']}, 数量={order_info['amount']}")

    def _handle_filled_order(self, order: Dict) -> None:
        """
        处理已成交的订单

        Args:
            order: 订单信息
        """
        order_id = order['id']
        side = order['side']
        price = order['price']
        amount = order['amount']

        log.info(f"订单{order_id}已成交: {side} {amount} @ {price}")

        # 根据成交的订单类型处理
        if side == 'buy':
            # 买单成交，从活跃买单列表中移除
            if order_id in self.active_buy_orders:
                self.active_buy_orders.pop(order_id)

                # 在上一个网格点挂卖单
                target_sell_price = price + self.price_diff_per_grid

                if target_sell_price <= self.params['upper_price']:
                    self._place_sell_order(target_sell_price)
                else:
                    log.info(f"目标卖出价格{target_sell_price}超过上限{self.params['upper_price']}，不挂卖单")

        elif side == 'sell':
            # 卖单成交，从活跃卖单列表中移除
            if order_id in self.active_sell_orders:
                self.active_sell_orders.pop(order_id)

                # 在下一个网格点挂买单
                target_buy_price = price - self.price_diff_per_grid

                if target_buy_price >= self.params['lower_price']:
                    self._place_buy_order(target_buy_price)
                else:
                    log.info(f"目标买入价格{target_buy_price}低于下限{self.params['lower_price']}，不挂买单")

    def _check_global_stop_loss_take_profit(self, current_price: float) -> None:
        """
        检查全局止损/止盈

        Args:
            current_price: 当前市场价格
        """
        # 检查全局止损
        if (self.params['global_stop_loss_price'] is not None and
            current_price <= self.params['global_stop_loss_price']):
            log.warning(f"触发全局止损: 当前价格{current_price} <= 止损价格{self.params['global_stop_loss_price']}")
            self._stop_strategy()
            return

        # 检查全局止盈
        if (self.params['global_take_profit_price'] is not None and
            current_price >= self.params['global_take_profit_price']):
            log.warning(f"触发全局止盈: 当前价格{current_price} >= 止盈价格{self.params['global_take_profit_price']}")
            self._stop_strategy()
            return

    def _stop_strategy(self) -> None:
        """停止策略，取消所有订单并平仓"""
        # 判断是否在回测环境中
        is_backtest = not (hasattr(self, 'broker') and hasattr(self.broker, 'exchange'))

        if is_backtest:
            # 回测环境，直接清空订单列表
            self.active_buy_orders.clear()
            self.active_sell_orders.clear()
            log.warning("回测模式停止策略，清空所有订单")
            self.is_running = False
            return

        # 实盘环境
        try:
            ccxt_symbol = self.params['symbol'].replace('_', '/')

            # 取消所有活跃订单
            for order_id in list(self.active_buy_orders.keys()):
                try:
                    self.broker.exchange.cancel_order(order_id, ccxt_symbol)
                    log.info(f"取消买单: {order_id}")
                except Exception as e:
                    log.error(f"取消买单{order_id}失败: {str(e)}")
                self.active_buy_orders.pop(order_id, None)

            for order_id in list(self.active_sell_orders.keys()):
                try:
                    self.broker.exchange.cancel_order(order_id, ccxt_symbol)
                    log.info(f"取消卖单: {order_id}")
                except Exception as e:
                    log.error(f"取消卖单{order_id}失败: {str(e)}")
                self.active_sell_orders.pop(order_id, None)

            # 市价卖出所有持仓
            base_currency = ccxt_symbol.split('/')[0]
            balance = self.broker.exchange.get_balance(base_currency)

            if balance > 0:
                self.broker.exchange.create_order(
                    symbol=ccxt_symbol,
                    order_type='market',
                    side='sell',
                    amount=balance
                )
                log.info(f"市价卖出所有{base_currency}持仓: {balance}")

            self.is_running = False
            log.warning("策略已停止")

        except Exception as e:
            log.error(f"停止策略失败: {str(e)}")

    def get_grid_status(self) -> Dict:
        """
        获取网格状态

        Returns:
            Dict: 网格状态信息
        """
        return {
            'grid_prices': self.grid_prices,
            'price_diff_per_grid': self.price_diff_per_grid,
            'active_buy_orders': self.active_buy_orders,
            'active_sell_orders': self.active_sell_orders,
            'is_running': self.is_running
        }
