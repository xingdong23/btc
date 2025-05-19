"""
实盘交易引擎模块

提供实盘交易的核心功能，包括订单管理、仓位管理、风险控制等。
"""
import time
import threading
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import traceback

from strategies.base import StrategyBase
from exchange.gateio import GateIOExchange
from risk_management.manager import RiskManager
from utils.logger import log
from utils.helpers import (
    timestamp_to_datetime,
    datetime_to_timestamp,
    calculate_pnl,
    parse_timeframe
)
from utils.notification import notification_manager


class LiveTradingEngine:
    """实盘交易引擎类"""

    def __init__(
        self,
        exchange: GateIOExchange,
        strategy: StrategyBase,
        symbol: str,
        timeframe: str,
        risk_manager: Optional[RiskManager] = None,
        test_mode: bool = False
    ):
        """
        初始化实盘交易引擎

        Args:
            exchange: 交易所接口
            strategy: 交易策略
            symbol: 交易对，如'BTC/USDT'
            timeframe: 时间周期，如'1h', '4h', '1d'等
            risk_manager: 风险管理器，如果为None则创建默认风险管理器
            test_mode: 测试模式，不执行实际交易，仅记录信号
        """
        self.exchange = exchange
        self.strategy = strategy
        self.symbol = symbol
        self.timeframe = timeframe
        self.test_mode = test_mode

        # 设置策略的broker属性为自身，以便策略可以调用交易引擎的方法
        self.strategy.broker = self

        # 初始化风险管理器
        if risk_manager is None:
            # 获取账户余额
            try:
                balance_info = self.exchange.get_balance()
                base_currency = self.symbol.split('/')[1]  # 如BTC/USDT中的USDT
                balance = balance_info.get(base_currency, {}).get('free', 0.0)
                self.risk_manager = RiskManager(account_balance=balance)
            except Exception as e:
                log.error(f"获取账户余额失败，使用默认值: {str(e)}")
                self.risk_manager = RiskManager()
        else:
            self.risk_manager = risk_manager

        # 交易状态
        self.running = False
        self.last_update_time = None
        self.orders = {}  # 订单管理
        self.positions = {}  # 持仓管理
        self.trades = []  # 交易记录
        self.equity_curve = []  # 权益曲线

        # 初始化市场数据
        self.market_info = self.exchange.get_market_info(self.symbol)

        # 线程锁，用于线程安全
        self.lock = threading.Lock()

        log.info(f"实盘交易引擎初始化完成: {self.symbol} {self.timeframe}")
        if self.test_mode:
            log.warning("当前为测试模式，不会执行实际交易")

    def start(self) -> None:
        """启动交易引擎"""
        if self.running:
            log.warning("交易引擎已经在运行中")
            return

        self.running = True
        log.info(f"启动实盘交易: {self.symbol} {self.timeframe}")

        # 初始化策略
        self._initialize_strategy()

        # 更新账户信息
        self._update_account_info()

        # 主交易循环
        try:
            while self.running:
                current_time = datetime.now()

                # 记录循环开始时间
                loop_start_time = time.time()

                try:
                    # 获取最新市场数据
                    self._update_market_data()

                    # 更新账户信息
                    self._update_account_info()

                    # 检查未完成订单
                    self._check_pending_orders()

                    # 更新权益曲线
                    self._update_equity()

                    # 计算等待时间
                    sleep_time = self._calculate_sleep_time()

                    # 记录状态
                    self._log_status()

                    # 等待下一个周期
                    log.info(f"等待 {sleep_time} 秒后更新...")
                    time.sleep(sleep_time)

                except Exception as e:
                    log.error(f"交易循环发生错误: {str(e)}")
                    log.error(traceback.format_exc())
                    time.sleep(10)  # 发生错误时等待10秒后重试

        except KeyboardInterrupt:
            log.info("收到中断信号，停止交易")
        finally:
            self.stop()

    def stop(self) -> None:
        """停止交易引擎"""
        if not self.running:
            return

        self.running = False
        log.info("停止实盘交易")

        # 显示最终账户状态
        self._log_final_status()

        # 发送停止交易通知
        notification_manager.send_alert(
            "交易系统已停止",
            f"交易对: {self.symbol}\n交易策略: {self.strategy.name}\n总交易次数: {len(self.trades)}",
            "info"
        )

    def _initialize_strategy(self) -> None:
        """初始化策略"""
        log.info("初始化策略...")

        # 获取历史数据用于初始化策略
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)  # 获取30天历史数据

        try:
            # 获取历史K线数据
            data = self.exchange.get_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                since=int(start_time.timestamp() * 1000),
                limit=1000
            )

            if data is None or data.empty:
                log.error("无法获取历史数据，策略初始化失败")
                return

            log.info(f"成功加载 {len(data)} 根K线数据，时间范围: {data.index[0]} 到 {data.index[-1]}")

            # 初始化策略
            self.strategy.init()

            # 逐根K线更新策略状态
            for _, row in data.iterrows():
                bar = {
                    'symbol': self.symbol,
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume'],
                    'datetime': row.name.to_pydatetime()
                }
                self.strategy.on_bar(bar)

            log.info("策略初始化完成")

            # 发送系统启动通知
            notification_manager.send_alert(
                "交易系统已启动",
                f"交易对: {self.symbol}\n交易策略: {self.strategy.name}\n模式: {'测试模式' if self.test_mode else '实盘模式'}",
                "info"
            )

        except Exception as e:
            log.error(f"策略初始化失败: {str(e)}")
            log.error(traceback.format_exc())

    def _update_market_data(self) -> Dict:
        """
        获取最新市场数据并更新策略

        Returns:
            Dict: 最新的K线数据
        """
        current_time = datetime.now()
        log.info(f"\n=== 实盘交易时间: {current_time} ===")

        try:
            # 获取最新K线数据
            ohlcv = self.exchange.get_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                since=int((current_time - timedelta(hours=1)).timestamp() * 1000),
                limit=1
            )

            if ohlcv is None or ohlcv.empty:
                log.warning("无法获取最新K线数据")
                return None

            latest_bar = ohlcv.iloc[-1]
            bar = {
                'symbol': self.symbol,
                'open': latest_bar['open'],
                'high': latest_bar['high'],
                'low': latest_bar['low'],
                'close': latest_bar['close'],
                'volume': latest_bar['volume'],
                'datetime': latest_bar.name.to_pydatetime()
            }

            # 更新策略状态
            signal = self.strategy.on_bar(bar)

            # 处理交易信号
            if signal and signal.get('signal') in ['buy', 'sell']:
                self._process_signal(signal)

            self.last_update_time = current_time
            return bar

        except Exception as e:
            log.error(f"更新市场数据失败: {str(e)}")
            log.error(traceback.format_exc())
            return None

    def _process_signal(self, signal: Dict) -> None:
        """
        处理交易信号

        Args:
            signal: 交易信号
        """
        if not signal or signal.get('signal') == 'hold':
            return

        signal_type = signal['signal']
        price = signal['price']
        size = signal.get('size', 0)

        log.info(f"收到交易信号: {signal_type.upper()} {self.symbol} @ {price} 数量: {size}")

        # 风险检查
        if not self._check_risk(signal):
            log.warning("风险检查未通过，取消交易")
            # 发送风险警报
            notification_manager.send_alert(
                "风险检查未通过",
                f"交易信号: {signal_type.upper()} {self.symbol} @ {price} 数量: {size}",
                "warning"
            )
            return

        # 测试模式下不执行实际交易
        if self.test_mode:
            log.info(f"[测试模式] 模拟{signal_type} {self.symbol} 数量: {size} 价格: {price}")
            return

        # 执行交易
        try:
            if signal_type == 'buy':
                order = self.exchange.create_order(
                    symbol=self.symbol,
                    order_type='limit',
                    side='buy',
                    amount=size,
                    price=price * 1.005  # 略高于当前价格，确保成交
                )
                log.info(f"买入订单已提交: {order['id']}")
                self.orders[order['id']] = order

            elif signal_type == 'sell':
                order = self.exchange.create_order(
                    symbol=self.symbol,
                    order_type='limit',
                    side='sell',
                    amount=size,
                    price=price * 0.995  # 略低于当前价格，确保成交
                )
                log.info(f"卖出订单已提交: {order['id']}")
                self.orders[order['id']] = order

        except Exception as e:
            log.error(f"提交订单失败: {str(e)}")
            log.error(traceback.format_exc())

    def _check_risk(self, signal: Dict) -> bool:
        """
        风险检查

        Args:
            signal: 交易信号

        Returns:
            bool: 是否通过风险检查
        """
        signal_type = signal['signal']
        price = signal['price']
        size = signal.get('size', 0)

        # 检查账户余额
        if signal_type == 'buy':
            base_currency = self.symbol.split('/')[1]  # 如BTC/USDT中的USDT
            balance = self.get_balance(base_currency)
            cost = price * size

            if balance < cost:
                log.warning(f"余额不足: {balance} {base_currency} < {cost} {base_currency}")
                return False

        # 检查持仓
        if signal_type == 'sell':
            quote_currency = self.symbol.split('/')[0]  # 如BTC/USDT中的BTC
            position = self.get_position(self.symbol)

            if position < size:
                log.warning(f"持仓不足: {position} {quote_currency} < {size} {quote_currency}")
                return False

        # 检查风险限制
        if not self.risk_manager.check_position_risk(self.symbol, size, price):
            log.warning("超出风险限制")
            return False

        # 检查最大回撤
        equity = self._calculate_equity()
        if self.risk_manager.check_max_drawdown(equity):
            log.warning("达到最大回撤限制")
            return False

        # 检查单日亏损限制
        if self.risk_manager.check_daily_loss_limit(equity):
            log.warning("达到单日亏损限制")
            return False

        return True

    def _check_pending_orders(self) -> None:
        """检查未完成订单状态"""
        if not self.orders:
            return

        for order_id, order in list(self.orders.items()):
            try:
                updated_order = self.exchange.get_order(order_id, self.symbol)

                # 更新订单状态
                self.orders[order_id] = updated_order

                # 处理已完成订单
                if updated_order['status'] in ['closed', 'filled']:
                    log.info(f"订单已成交: {order_id}")

                    # 更新持仓
                    self._update_position(updated_order)

                    # 记录交易
                    self._record_trade(updated_order)

                    # 从订单列表中移除
                    del self.orders[order_id]

                # 处理已取消订单
                elif updated_order['status'] in ['canceled', 'expired', 'rejected']:
                    log.info(f"订单已取消: {order_id}")
                    del self.orders[order_id]

            except Exception as e:
                log.error(f"检查订单状态失败: {str(e)}")

    def _update_position(self, order: Dict) -> None:
        """
        更新持仓信息

        Args:
            order: 订单信息
        """
        if order['status'] != 'closed' and order['status'] != 'filled':
            return

        symbol = order['symbol']
        side = order['side']
        amount = order['amount']
        price = order['price']

        with self.lock:
            if side == 'buy':
                # 买入增加持仓
                if symbol not in self.positions:
                    self.positions[symbol] = {'amount': 0, 'avg_price': 0}

                current = self.positions[symbol]
                total_cost = current['amount'] * current['avg_price'] + amount * price
                total_amount = current['amount'] + amount

                if total_amount > 0:
                    avg_price = total_cost / total_amount
                else:
                    avg_price = 0

                self.positions[symbol] = {'amount': total_amount, 'avg_price': avg_price}

            elif side == 'sell':
                # 卖出减少持仓
                if symbol in self.positions:
                    current = self.positions[symbol]
                    current['amount'] -= amount

                    if current['amount'] <= 0:
                        # 清仓
                        del self.positions[symbol]

    def _record_trade(self, order: Dict) -> None:
        """
        记录交易

        Args:
            order: 订单信息
        """
        trade = {
            'id': order['id'],
            'symbol': order['symbol'],
            'side': order['side'],
            'price': order['price'],
            'amount': order['amount'],
            'cost': order['cost'],
            'fee': order.get('fee', {}),
            'datetime': order['datetime'],
            'timestamp': order['timestamp']
        }

        self.trades.append(trade)
        log.info(f"记录交易: {trade['side']} {trade['symbol']} @ {trade['price']} 数量: {trade['amount']}")

        # 发送交易通知
        notification_manager.send_trade_notification(trade)

        # 通知策略
        self.strategy.on_trade(trade)

    def _update_account_info(self) -> None:
        """更新账户信息"""
        try:
            # 获取账户余额
            balance_info = self.exchange.get_balance()

            # 获取持仓信息
            for symbol in self.positions.keys():
                quote_currency = symbol.split('/')[0]  # 如BTC/USDT中的BTC
                if quote_currency in balance_info:
                    self.positions[symbol]['amount'] = balance_info[quote_currency]['total']

            # 更新风险管理器的账户余额
            base_currency = self.symbol.split('/')[1]  # 如BTC/USDT中的USDT
            if base_currency in balance_info:
                self.risk_manager.update_account_balance(balance_info[base_currency]['free'])

        except Exception as e:
            log.error(f"更新账户信息失败: {str(e)}")

    def _update_equity(self) -> None:
        """更新权益曲线"""
        equity = self._calculate_equity()

        self.equity_curve.append({
            'timestamp': int(datetime.now().timestamp() * 1000),
            'datetime': datetime.now(),
            'equity': equity
        })

        # 更新风险管理器的权益曲线
        self.risk_manager.update_equity(equity)

    def _calculate_equity(self) -> float:
        """
        计算当前总权益

        Returns:
            float: 总权益
        """
        try:
            # 获取账户余额
            balance_info = self.exchange.get_balance()

            # 计算基础货币总额
            base_currency = self.symbol.split('/')[1]  # 如BTC/USDT中的USDT
            base_balance = balance_info.get(base_currency, {}).get('total', 0.0)

            # 获取当前价格
            ticker = self.exchange.get_ticker(self.symbol)
            current_price = ticker['last'] if ticker else 0

            # 计算持仓市值
            position_value = 0.0
            for symbol, position in self.positions.items():
                if symbol == self.symbol:
                    position_value += position['amount'] * current_price
                else:
                    # 对于其他交易对，需要获取其价格
                    other_ticker = self.exchange.get_ticker(symbol)
                    if other_ticker:
                        position_value += position['amount'] * other_ticker['last']

            # 总权益 = 基础货币余额 + 持仓市值
            return base_balance + position_value

        except Exception as e:
            log.error(f"计算总权益失败: {str(e)}")
            return 0.0

    def _calculate_sleep_time(self) -> int:
        """
        计算下一次更新前的等待时间

        Returns:
            int: 等待时间（秒）
        """
        timeframe_seconds = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }

        # 默认等待时间
        default_sleep = 60  # 1分钟

        # 获取时间周期对应的秒数
        seconds = timeframe_seconds.get(self.timeframe.lower(), default_sleep)

        # 计算下一个周期的开始时间
        now = datetime.now()

        if self.timeframe.lower() == '1d':
            # 对于日线，下一个周期是明天的00:00
            next_time = datetime(now.year, now.month, now.day) + timedelta(days=1)
        elif self.timeframe.lower() == '1h':
            # 对于小时线，下一个周期是下一个整点
            next_time = datetime(now.year, now.month, now.day, now.hour) + timedelta(hours=1)
        elif self.timeframe.lower() == '1m':
            # 对于分钟线，下一个周期是下一分钟的开始
            next_time = datetime(now.year, now.month, now.day, now.hour, now.minute) + timedelta(minutes=1)
        else:
            # 其他周期，简单地等待对应的秒数
            return seconds

        # 计算等待时间
        wait_seconds = (next_time - now).total_seconds()

        # 确保等待时间为正数
        if wait_seconds <= 0:
            wait_seconds = default_sleep

        return int(wait_seconds)

    def _log_status(self) -> None:
        """记录当前状态"""
        try:
            # 获取账户余额
            balance_info = self.exchange.get_balance()

            # 获取基础货币余额
            base_currency = self.symbol.split('/')[1]  # 如BTC/USDT中的USDT
            base_balance = balance_info.get(base_currency, {}).get('free', 0.0)

            # 获取持仓
            quote_currency = self.symbol.split('/')[0]  # 如BTC/USDT中的BTC
            position = self.get_position(self.symbol)

            # 获取当前价格
            ticker = self.exchange.get_ticker(self.symbol)
            current_price = ticker['last'] if ticker else 0

            # 计算总权益
            equity = self._calculate_equity()

            log.info(f"账户状态 - 余额: {base_balance:.2f} {base_currency} | 持仓: {position:.6f} {quote_currency} | 当前价格: {current_price:.2f} | 总权益: {equity:.2f} {base_currency}")

            # 记录未完成订单
            if self.orders:
                log.info(f"未完成订单: {len(self.orders)}个")
                for order_id, order in self.orders.items():
                    log.info(f"  - {order_id}: {order['side']} {order['symbol']} @ {order['price']} 数量: {order['amount']} 状态: {order['status']}")

        except Exception as e:
            log.error(f"记录状态失败: {str(e)}")

    def _log_final_status(self) -> None:
        """记录最终状态"""
        try:
            # 获取账户余额
            balance_info = self.exchange.get_balance()

            # 获取基础货币余额
            base_currency = self.symbol.split('/')[1]  # 如BTC/USDT中的USDT
            base_balance = balance_info.get(base_currency, {}).get('total', 0.0)

            # 获取持仓
            positions_str = ""
            for symbol, position in self.positions.items():
                quote_currency = symbol.split('/')[0]
                positions_str += f"{position['amount']:.6f} {quote_currency}, "

            # 计算总权益
            equity = self._calculate_equity()

            log.info("\n=== 实盘交易结束 ===")
            log.info(f"最终账户状态:")
            log.info(f"- 余额: {base_balance:.2f} {base_currency}")
            log.info(f"- 持仓: {positions_str[:-2] if positions_str else '无'}")
            log.info(f"- 总权益: {equity:.2f} {base_currency}")
            log.info(f"- 总交易次数: {len(self.trades)}")

        except Exception as e:
            log.error(f"记录最终状态失败: {str(e)}")

    def get_balance(self, currency: str = None) -> float:
        """
        获取账户余额

        Args:
            currency: 货币代码，如'USDT'，如果为None则返回交易对的基础货币余额

        Returns:
            float: 账户余额
        """
        try:
            balance_info = self.exchange.get_balance()

            if currency is None:
                currency = self.symbol.split('/')[1]  # 如BTC/USDT中的USDT

            return balance_info.get(currency, {}).get('free', 0.0)

        except Exception as e:
            log.error(f"获取账户余额失败: {str(e)}")
            return 0.0

    def get_position(self, symbol: str) -> float:
        """
        获取持仓数量

        Args:
            symbol: 交易对，如'BTC/USDT'

        Returns:
            float: 持仓数量
        """
        try:
            if symbol in self.positions:
                return self.positions[symbol]['amount']

            # 如果持仓信息中没有，尝试从交易所获取
            quote_currency = symbol.split('/')[0]  # 如BTC/USDT中的BTC
            balance_info = self.exchange.get_balance()

            if quote_currency in balance_info:
                return balance_info[quote_currency]['total']

            return 0.0

        except Exception as e:
            log.error(f"获取持仓失败: {str(e)}")
            return 0.0
