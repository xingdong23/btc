"""
仓位控制模块

提供基于市场高低点的动态仓位控制策略。
"""
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np

from utils.logger import log


class PositionController:
    """
    仓位控制器
    
    基于市场高低点和波动率动态调整仓位大小，优化资金利用率和风险控制。
    """
    
    def __init__(self, strategy):
        """
        初始化仓位控制器
        
        Args:
            strategy: 策略实例，用于访问交易所接口和账户信息
        """
        self.strategy = strategy
        self.lookback_days = 52  # 回顾天数
        self.daily_high = None  # 每日高点
        self.daily_low = None  # 每日低点
        self.sell_target_pct = 0.50  # 高点卖出目标仓位
        self.buy_target_pct = 0.70  # 低点买入目标仓位
        self.last_adjustment_time = 0
        self.adjustment_interval = 3600  # 调整间隔（秒）
        self.price_history = []  # [(timestamp, price)]
        self.position_history = []  # [(timestamp, position_ratio)]
        
    async def initialize(self) -> bool:
        """
        初始化仓位控制器
        
        Returns:
            bool: 是否成功初始化
        """
        try:
            # 获取当前价格
            current_price = await self._get_current_price()
            
            # 初始化每日高低点
            self.daily_high = current_price * 1.1  # 初始高点设为当前价格的110%
            self.daily_low = current_price * 0.9  # 初始低点设为当前价格的90%
            
            # 记录初始状态
            timestamp = int(time.time())
            self.price_history.append((timestamp, current_price))
            
            # 获取当前仓位比例
            position_ratio = await self._get_position_ratio()
            self.position_history.append((timestamp, position_ratio))
            
            log.info(f"仓位控制器初始化 | 当前价格: {current_price:.2f} | 当前仓位: {position_ratio*100:.1f}%")
            log.info(f"初始高低点 | 高点: {self.daily_high:.2f} | 低点: {self.daily_low:.2f}")
            
            return True
        except Exception as e:
            log.error(f"仓位控制器初始化失败: {str(e)}")
            return False
    
    async def check_position(self) -> Dict:
        """
        检查仓位并根据需要调整
        
        Returns:
            Dict: 调整结果
        """
        current_time = time.time()
        
        # 控制调整频率
        if current_time - self.last_adjustment_time < self.adjustment_interval:
            return {}
            
        self.last_adjustment_time = current_time
        
        try:
            # 获取当前价格
            current_price = await self._get_current_price()
            
            # 更新价格历史
            timestamp = int(current_time)
            self.price_history.append((timestamp, current_price))
            
            # 限制历史记录长度
            max_history = 1000
            if len(self.price_history) > max_history:
                self.price_history = self.price_history[-max_history:]
            
            # 更新每日高低点
            self._update_daily_high_low(current_price)
            
            # 获取当前仓位比例
            position_ratio = await self._get_position_ratio()
            self.position_history.append((timestamp, position_ratio))
            
            # 检查是否需要调整仓位
            adjustment_result = await self._check_position_adjustment(current_price, position_ratio)
            
            # 记录状态
            log.info(
                f"仓位检查 | "
                f"当前价格: {current_price:.2f} | "
                f"当前仓位: {position_ratio*100:.1f}% | "
                f"高点: {self.daily_high:.2f} | "
                f"低点: {self.daily_low:.2f}"
            )
            
            if adjustment_result:
                log.info(f"仓位调整 | {adjustment_result['action']} | 目标仓位: {adjustment_result['target_position']*100:.1f}%")
            
            return adjustment_result
            
        except Exception as e:
            log.error(f"检查仓位失败: {str(e)}")
            return {}
    
    def _update_daily_high_low(self, current_price: float) -> None:
        """
        更新每日高低点
        
        Args:
            current_price: 当前价格
        """
        # 更新高点
        if current_price > self.daily_high:
            self.daily_high = current_price
            log.info(f"更新每日高点: {self.daily_high:.2f}")
        
        # 更新低点
        if current_price < self.daily_low:
            self.daily_low = current_price
            log.info(f"更新每日低点: {self.daily_low:.2f}")
        
        # 每天重置高低点
        now = datetime.now()
        if now.hour == 0 and now.minute < 10:  # 每天0点重置
            # 保留一定的范围
            self.daily_high = current_price * 1.05
            self.daily_low = current_price * 0.95
            log.info(f"重置每日高低点 | 高点: {self.daily_high:.2f} | 低点: {self.daily_low:.2f}")
    
    async def _check_position_adjustment(self, current_price: float, position_ratio: float) -> Dict:
        """
        检查是否需要调整仓位
        
        Args:
            current_price: 当前价格
            position_ratio: 当前仓位比例
            
        Returns:
            Dict: 调整结果
        """
        # 计算价格相对于高低点的位置
        price_range = self.daily_high - self.daily_low
        if price_range <= 0:
            return {}
            
        relative_position = (current_price - self.daily_low) / price_range
        
        # 计算目标仓位
        target_position = 0.0
        action = 'none'
        
        if current_price > self.daily_high * 0.95:  # 接近高点
            # 接近高点，减仓
            target_position = self.sell_target_pct
            action = 'sell'
        elif current_price < self.daily_low * 1.05:  # 接近低点
            # 接近低点，加仓
            target_position = self.buy_target_pct
            action = 'buy'
        else:
            # 在中间区域，线性调整仓位
            target_position = self.buy_target_pct - relative_position * (self.buy_target_pct - self.sell_target_pct)
            action = 'adjust'
        
        # 如果目标仓位与当前仓位相差不大，不调整
        if abs(target_position - position_ratio) < 0.05:
            return {}
        
        # 计算需要调整的数量
        adjustment_needed = await self._calculate_adjustment(target_position, position_ratio)
        
        if abs(adjustment_needed['amount']) < 0.0001:  # 最小调整阈值
            return {}
        
        # 执行调整
        result = await self._execute_adjustment(adjustment_needed['side'], adjustment_needed['amount'])
        
        if result:
            return {
                'action': action,
                'target_position': target_position,
                'current_position': position_ratio,
                'adjustment': adjustment_needed,
                'result': result
            }
        
        return {}
    
    async def _calculate_adjustment(self, target_position: float, current_position: float) -> Dict:
        """
        计算需要调整的数量
        
        Args:
            target_position: 目标仓位比例
            current_position: 当前仓位比例
            
        Returns:
            Dict: 调整信息
        """
        try:
            # 获取总资产
            total_assets = await self._get_total_assets()
            
            # 获取当前价格
            current_price = await self._get_current_price()
            
            # 计算当前持仓价值
            current_position_value = total_assets * current_position
            
            # 计算目标持仓价值
            target_position_value = total_assets * target_position
            
            # 计算需要调整的价值
            adjustment_value = target_position_value - current_position_value
            
            # 计算需要调整的数量
            adjustment_amount = adjustment_value / current_price
            
            # 确定调整方向
            side = 'buy' if adjustment_amount > 0 else 'sell'
            
            return {
                'side': side,
                'amount': abs(adjustment_amount),
                'value': abs(adjustment_value)
            }
        except Exception as e:
            log.error(f"计算仓位调整失败: {str(e)}")
            return {'side': 'none', 'amount': 0, 'value': 0}
    
    async def _execute_adjustment(self, side: str, amount: float) -> Dict:
        """
        执行仓位调整
        
        Args:
            side: 交易方向 ('buy' 或 'sell')
            amount: 交易数量
            
        Returns:
            Dict: 执行结果
        """
        try:
            if hasattr(self.strategy, 'broker') and hasattr(self.strategy.broker, 'exchange'):
                # 实盘环境
                symbol = self.strategy.params['symbol'].replace('_', '/')
                
                # 检查余额是否足够
                if side == 'buy':
                    # 检查USDT余额
                    quote_currency = symbol.split('/')[1]  # 计价货币 (如 USDT)
                    balance = self.strategy.broker.get_balance()
                    available_balance = balance.get(quote_currency, 0)
                    
                    # 获取当前价格
                    current_price = await self._get_current_price()
                    
                    # 计算所需金额
                    required_amount = amount * current_price
                    
                    if required_amount > available_balance:
                        log.warning(f"余额不足，无法执行买入: 需要 {required_amount:.2f} {quote_currency}，可用 {available_balance:.2f} {quote_currency}")
                        return {}
                else:  # sell
                    # 检查基础货币余额
                    base_currency = symbol.split('/')[0]  # 基础货币 (如 BTC)
                    balance = self.strategy.broker.get_balance()
                    available_balance = balance.get(base_currency, 0)
                    
                    if amount > available_balance:
                        log.warning(f"余额不足，无法执行卖出: 需要 {amount:.8f} {base_currency}，可用 {available_balance:.8f} {base_currency}")
                        return {}
                
                # 执行交易
                order = self.strategy.broker.exchange.create_order(
                    symbol=symbol,
                    order_type='market',
                    side=side,
                    amount=amount
                )
                
                log.info(f"执行仓位调整: {side} {amount:.8f} @ market")
                
                return {
                    'order_id': order['id'],
                    'side': side,
                    'amount': amount,
                    'type': 'market',
                    'status': order['status']
                }
            else:
                # 回测环境
                log.info(f"回测模式仓位调整: {side} {amount:.8f} @ market")
                
                return {
                    'order_id': f"{side}_{int(time.time())}",
                    'side': side,
                    'amount': amount,
                    'type': 'market',
                    'status': 'closed'
                }
        except Exception as e:
            log.error(f"执行仓位调整失败: {str(e)}")
            return {}
    
    async def _get_position_ratio(self) -> float:
        """获取当前仓位占总资产比例"""
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
                
                # 计算总资产和仓位比例
                base_value = base_balance * current_price
                total_assets = base_value + quote_balance
                
                if total_assets <= 0:
                    return 0.0
                
                return base_value / total_assets
            else:
                # 回测环境，使用策略中的仓位信息
                return 0.5  # 默认50%仓位
        except Exception as e:
            log.error(f"获取仓位比例失败: {str(e)}")
            return 0.0
    
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
    
    def get_position_history(self) -> List[Tuple[int, float]]:
        """
        获取仓位历史
        
        Returns:
            List[Tuple[int, float]]: 仓位历史记录 [(timestamp, position_ratio)]
        """
        return self.position_history
