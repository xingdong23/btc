"""
交易监控模块

提供实时监控交易状态、性能指标和资产变化的功能。
"""
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from utils.logger import log


class TradingMonitor:
    """
    交易监控器
    
    监控交易状态、性能指标和资产变化，提供实时反馈和警报。
    """
    
    def __init__(self, strategy):
        """
        初始化交易监控器
        
        Args:
            strategy: 策略实例，用于访问交易所接口和账户信息
        """
        self.strategy = strategy
        self.start_time = time.time()
        self.start_balance = None
        self.peak_balance = None
        self.last_balance = None
        self.balance_history = []  # [(timestamp, balance)]
        self.price_history = []  # [(timestamp, price)]
        self.order_history = []  # [(timestamp, order_id, side, price, amount, status)]
        self.performance_metrics = {}
        self.alerts = []
        self.last_check_time = 0
        self.check_interval = 60  # 监控检查间隔（秒）
        
    async def initialize(self) -> bool:
        """
        初始化监控器
        
        Returns:
            bool: 是否成功初始化
        """
        try:
            # 获取初始资产
            self.start_balance = await self._get_total_assets()
            self.peak_balance = self.start_balance
            self.last_balance = self.start_balance
            
            # 记录初始状态
            timestamp = int(time.time())
            self.balance_history.append((timestamp, self.start_balance))
            
            # 获取当前价格
            current_price = await self._get_current_price()
            self.price_history.append((timestamp, current_price))
            
            log.info(f"交易监控器初始化 | 初始资产: {self.start_balance:.2f} USDT | 当前价格: {current_price:.2f}")
            return True
        except Exception as e:
            log.error(f"交易监控器初始化失败: {str(e)}")
            return False
    
    async def check_status(self) -> Dict:
        """
        检查交易状态
        
        Returns:
            Dict: 状态信息
        """
        current_time = time.time()
        
        # 控制检查频率
        if current_time - self.last_check_time < self.check_interval:
            return {}
            
        self.last_check_time = current_time
        
        try:
            # 获取当前资产
            current_balance = await self._get_total_assets()
            
            # 获取当前价格
            current_price = await self._get_current_price()
            
            # 更新历史记录
            timestamp = int(current_time)
            self.balance_history.append((timestamp, current_balance))
            self.price_history.append((timestamp, current_price))
            
            # 限制历史记录长度
            max_history = 1000
            if len(self.balance_history) > max_history:
                self.balance_history = self.balance_history[-max_history:]
            if len(self.price_history) > max_history:
                self.price_history = self.price_history[-max_history:]
            
            # 更新峰值
            if current_balance > self.peak_balance:
                self.peak_balance = current_balance
            
            # 计算变化
            balance_change = current_balance - self.last_balance
            balance_change_pct = balance_change / self.last_balance if self.last_balance else 0
            
            # 计算回撤
            drawdown = (self.peak_balance - current_balance) / self.peak_balance if self.peak_balance else 0
            
            # 计算总收益
            total_profit = current_balance - self.start_balance
            total_profit_pct = total_profit / self.start_balance if self.start_balance else 0
            
            # 更新上次余额
            self.last_balance = current_balance
            
            # 获取活跃订单
            active_orders = await self._get_active_orders()
            
            # 获取交易统计
            trade_stats = self.strategy.order_tracker.get_statistics() if hasattr(self.strategy, 'order_tracker') else {}
            
            # 检查是否需要发出警报
            self._check_alerts(current_balance, drawdown)
            
            # 更新性能指标
            self.performance_metrics = {
                'current_balance': current_balance,
                'peak_balance': self.peak_balance,
                'balance_change': balance_change,
                'balance_change_pct': balance_change_pct,
                'drawdown': drawdown,
                'total_profit': total_profit,
                'total_profit_pct': total_profit_pct,
                'runtime': int(current_time - self.start_time),
                'active_orders': len(active_orders),
                'current_price': current_price,
                'trade_stats': trade_stats
            }
            
            # 记录状态
            log.info(
                f"交易状态 | "
                f"余额: {current_balance:.2f} USDT | "
                f"变化: {balance_change_pct*100:+.2f}% | "
                f"回撤: {drawdown*100:.2f}% | "
                f"总收益: {total_profit_pct*100:+.2f}% | "
                f"价格: {current_price:.2f} | "
                f"活跃订单: {len(active_orders)}"
            )
            
            return self.performance_metrics
            
        except Exception as e:
            log.error(f"检查交易状态失败: {str(e)}")
            return {}
    
    def _check_alerts(self, current_balance: float, drawdown: float) -> None:
        """
        检查是否需要发出警报
        
        Args:
            current_balance: 当前余额
            drawdown: 当前回撤
        """
        # 检查余额变化
        if self.last_balance and current_balance < self.last_balance * 0.95:
            # 余额下降超过5%
            alert = {
                'type': 'balance_drop',
                'message': f"余额下降超过5%: {self.last_balance:.2f} -> {current_balance:.2f} USDT",
                'timestamp': int(time.time())
            }
            self.alerts.append(alert)
            log.warning(f"警报: {alert['message']}")
        
        # 检查回撤
        if drawdown > 0.1:  # 回撤超过10%
            alert = {
                'type': 'high_drawdown',
                'message': f"回撤超过10%: {drawdown*100:.2f}%",
                'timestamp': int(time.time())
            }
            self.alerts.append(alert)
            log.warning(f"警报: {alert['message']}")
        
        # 限制警报数量
        max_alerts = 100
        if len(self.alerts) > max_alerts:
            self.alerts = self.alerts[-max_alerts:]
    
    async def _get_total_assets(self) -> float:
        """获取总资产价值 (USDT)"""
        try:
            if hasattr(self.strategy, 'broker') and hasattr(self.strategy.broker, 'exchange'):
                # 获取交易对信息
                symbol = self.strategy.params['symbol'].replace('_', '/')
                base_currency = symbol.split('/')[0]  # 基础货币 (如 BTC)
                quote_currency = symbol.split('/')[1]  # 计价货币 (如 USDT)
                
                # 获取余额
                balance = self.strategy.broker.get_balance()
                
                # 获取基础货币余额和计价货币余额
                base_balance = balance.get(base_currency, 0)
                quote_balance = balance.get(quote_currency, 0)
                
                # 获取当前价格
                current_price = await self._get_current_price()
                
                # 计算总资产 (USDT)
                total_assets = quote_balance + base_balance * current_price
                
                return total_assets
            else:
                # 回测环境，使用模拟数据
                return 10000.0
        except Exception as e:
            log.error(f"获取总资产失败: {str(e)}")
            return 0.0
    
    async def _get_current_price(self) -> float:
        """获取当前市场价格"""
        try:
            if hasattr(self.strategy, 'broker') and hasattr(self.strategy.broker, 'exchange'):
                symbol = self.strategy.params['symbol'].replace('_', '/')
                ticker = self.strategy.broker.exchange.fetch_ticker(symbol)
                return ticker['last']
            else:
                # 回测环境，使用策略中的当前价格
                return self.strategy.current_price
        except Exception as e:
            log.error(f"获取当前价格失败: {str(e)}")
            return 0.0
    
    async def _get_active_orders(self) -> List[Dict]:
        """获取活跃订单"""
        try:
            if hasattr(self.strategy, 'broker') and hasattr(self.strategy.broker, 'exchange'):
                symbol = self.strategy.params['symbol'].replace('_', '/')
                orders = self.strategy.broker.exchange.get_open_orders(symbol)
                
                # 记录订单历史
                timestamp = int(time.time())
                for order in orders:
                    self.order_history.append((
                        timestamp,
                        order['id'],
                        order['side'],
                        order['price'],
                        order['amount'],
                        order['status']
                    ))
                
                return orders
            else:
                # 回测环境，使用策略中的活跃订单
                active_orders = []
                active_orders.extend(self.strategy.active_buy_orders.values())
                active_orders.extend(self.strategy.active_sell_orders.values())
                return active_orders
        except Exception as e:
            log.error(f"获取活跃订单失败: {str(e)}")
            return []
    
    def get_performance_summary(self) -> Dict:
        """
        获取性能摘要
        
        Returns:
            Dict: 性能摘要
        """
        return {
            'start_time': self.start_time,
            'runtime': int(time.time() - self.start_time),
            'start_balance': self.start_balance,
            'current_balance': self.last_balance,
            'peak_balance': self.peak_balance,
            'total_profit': self.last_balance - self.start_balance if self.last_balance and self.start_balance else 0,
            'total_profit_pct': (self.last_balance / self.start_balance - 1) * 100 if self.last_balance and self.start_balance else 0,
            'max_drawdown': (self.peak_balance - self.last_balance) / self.peak_balance * 100 if self.peak_balance and self.last_balance else 0,
            'alerts': len(self.alerts),
            'recent_alerts': self.alerts[-5:] if self.alerts else []
        }
    
    def get_balance_history(self) -> List[Tuple[int, float]]:
        """
        获取余额历史
        
        Returns:
            List[Tuple[int, float]]: 余额历史记录 [(timestamp, balance)]
        """
        return self.balance_history
    
    def get_price_history(self) -> List[Tuple[int, float]]:
        """
        获取价格历史
        
        Returns:
            List[Tuple[int, float]]: 价格历史记录 [(timestamp, price)]
        """
        return self.price_history
    
    def get_alerts(self, limit: int = 10) -> List[Dict]:
        """
        获取警报历史
        
        Args:
            limit: 返回的警报数量
            
        Returns:
            List[Dict]: 警报历史
        """
        return self.alerts[-limit:] if self.alerts else []
