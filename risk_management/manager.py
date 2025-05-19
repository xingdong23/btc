"""
风险管理模块

提供风险管理功能，包括头寸管理、止损止盈、风险控制等。
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from decimal import Decimal, getcontext

from utils.logger import log
from config.config import config

# 设置Decimal精度
getcontext().prec = 8

class RiskManager:
    """风险管理类"""
    
    def __init__(self, account_balance: float = 10000.0, risk_per_trade: float = 0.01):
        """
        初始化风险管理器
        
        Args:
            account_balance: 账户余额
            risk_per_trade: 每笔交易风险比例（默认1%）
        """
        self.account_balance = Decimal(str(account_balance))
        self.risk_per_trade = Decimal(str(risk_per_trade))
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.max_drawdown = Decimal('0.0')
        self.max_drawdown_limit = Decimal('0.2')  # 最大回撤限制20%
        self.daily_loss_limit = Decimal('0.05')  # 单日最大亏损5%
        self.position_size_limit = Decimal('0.1')  # 单个头寸最大比例10%
        self.leverage = Decimal('1.0')  # 默认不使用杠杆
        
        # 加载配置
        self._load_config()
    
    def _load_config(self) -> None:
        """从配置文件加载风险参数"""
        try:
            risk_params = config.risk_params
            
            if 'max_drawdown' in risk_params:
                self.max_drawdown_limit = Decimal(str(risk_params['max_drawdown']))
            
            if 'position_size' in risk_params:
                self.position_size_limit = Decimal(str(risk_params['position_size']))
            
            if 'daily_loss_limit' in risk_params:
                self.daily_loss_limit = Decimal(str(risk_params['daily_loss_limit']))
            
            if 'leverage' in risk_params:
                self.leverage = Decimal(str(risk_params['leverage']))
                
        except Exception as e:
            log.error(f"Failed to load risk config: {str(e)}")
    
    def update_balance(self, balance: float) -> None:
        """
        更新账户余额
        
        Args:
            balance: 新的账户余额
        """
        self.account_balance = Decimal(str(balance))
    
    def calculate_position_size(
        self, 
        entry_price: float, 
        stop_loss_price: float, 
        risk_percentage: float = None
    ) -> Tuple[float, float]:
        """
        计算头寸大小
        
        Args:
            entry_price: 入场价格
            stop_loss_price: 止损价格
            risk_percentage: 风险比例，如果为None则使用实例变量中的值
            
        Returns:
            Tuple[float, float]: (头寸大小, 风险金额)
        """
        try:
            entry = Decimal(str(entry_price))
            stop_loss = Decimal(str(stop_loss_price))
            
            if entry <= Decimal('0') or stop_loss <= Decimal('0'):
                return 0.0, 0.0
                
            # 计算风险金额
            risk_amount = self.account_balance * (Decimal(str(risk_percentage)) if risk_percentage else self.risk_per_trade)
            
            # 计算价格风险（绝对差值）
            price_risk = abs(entry - stop_loss)
            
            if price_risk == Decimal('0'):
                return 0.0, 0.0
                
            # 计算头寸大小（考虑杠杆）
            position_size = (risk_amount * self.leverage) / price_risk
            
            # 应用头寸大小限制
            max_position = self.account_balance * self.position_size_limit * self.leverage / entry
            position_size = min(position_size, max_position)
            
            return float(position_size), float(risk_amount)
            
        except Exception as e:
            log.error(f"Error calculating position size: {str(e)}")
            return 0.0, 0.0
    
    def calculate_stop_loss(
        self, 
        entry_price: float, 
        position_size: float, 
        risk_amount: float,
        is_long: bool = True
    ) -> float:
        """
        计算止损价格
        
        Args:
            entry_price: 入场价格
            position_size: 头寸大小
            risk_amount: 风险金额
            is_long: 是否是多头头寸
            
        Returns:
            float: 止损价格
        """
        try:
            if position_size <= 0:
                return 0.0
                
            entry = Decimal(str(entry_price))
            size = Decimal(str(position_size))
            risk = Decimal(str(risk_amount))
            
            # 计算价格风险
            price_risk = risk / size
            
            if is_long:
                stop_loss = entry - price_risk
            else:
                stop_loss = entry + price_risk
                
            return float(stop_loss)
            
        except Exception as e:
            log.error(f"Error calculating stop loss: {str(e)}")
            return 0.0
    
    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss_price: float,
        reward_risk_ratio: float = 2.0,
        is_long: bool = True
    ) -> float:
        """
        计算止盈价格
        
        Args:
            entry_price: 入场价格
            stop_loss_price: 止损价格
            reward_risk_ratio: 盈亏比，默认为2.0
            is_long: 是否是多头头寸
            
        Returns:
            float: 止盈价格
        """
        try:
            entry = Decimal(str(entry_price))
            stop_loss = Decimal(str(stop_loss_price))
            ratio = Decimal(str(reward_risk_ratio))
            
            risk = abs(entry - stop_loss)
            reward = risk * ratio
            
            if is_long:
                take_profit = entry + reward
            else:
                take_profit = entry - reward
                
            return float(take_profit)
            
        except Exception as e:
            log.error(f"Error calculating take profit: {str(e)}")
            return 0.0
    
    def check_position_risk(
        self, 
        symbol: str, 
        position_size: float, 
        current_price: float
    ) -> bool:
        """
        检查头寸风险
        
        Args:
            symbol: 交易对
            position_size: 头寸大小
            current_price: 当前价格
            
        Returns:
            bool: 是否通过风险检查
        """
        try:
            # 计算头寸价值
            position_value = Decimal(str(position_size)) * Decimal(str(current_price))
            
            # 检查单个头寸大小限制
            if position_value > self.account_balance * self.position_size_limit * self.leverage:
                log.warning(f"Position size {position_value:.2f} exceeds limit for {symbol}")
                return False
                
            # 检查总风险敞口
            total_exposure = sum(
                Decimal(str(size)) * Decimal(str(price)) 
                for _, (size, price) in self.positions.items()
            )
            
            if total_exposure + position_value > self.account_balance * self.leverage:
                log.warning(f"Total exposure {total_exposure + position_value:.2f} exceeds account balance {self.account_balance:.2f}")
                return False
                
            return True
            
        except Exception as e:
            log.error(f"Error checking position risk: {str(e)}")
            return False
    
    def update_position(
        self, 
        symbol: str, 
        position_size: float, 
        entry_price: float, 
        is_long: bool = True
    ) -> None:
        """
        更新头寸信息
        
        Args:
            symbol: 交易对
            position_size: 头寸大小
            entry_price: 入场价格
            is_long: 是否是多头头寸
        """
        try:
            size = Decimal(str(position_size))
            price = Decimal(str(entry_price))
            
            if symbol in self.positions:
                # 更新现有头寸
                current_size, current_price = self.positions[symbol]
                total_size = current_size + (size if is_long else -size)
                
                if total_size == Decimal('0'):
                    # 平仓
                    del self.positions[symbol]
                else:
                    # 更新头寸
                    if (size > Decimal('0') and current_size > Decimal('0')) or \
                       (size < Decimal('0') and current_size < Decimal('0')):
                        # 同向加仓，计算平均成本
                        total_value = (current_size * current_price) + (size * price)
                        avg_price = total_value / total_size
                        self.positions[symbol] = (total_size, avg_price)
                    else:
                        # 反向减仓，先进先出
                        self.positions[symbol] = (total_size, current_price)
            else:
                # 新开仓
                if size != Decimal('0'):
                    self.positions[symbol] = (size, price)
                    
        except Exception as e:
            log.error(f"Error updating position: {str(e)}")
    
    def calculate_pnl(
        self, 
        symbol: str, 
        exit_price: float, 
        position_size: float = None,
        is_long: bool = True
    ) -> Tuple[float, float]:
        """
        计算盈亏
        
        Args:
            symbol: 交易对
            exit_price: 退出价格
            position_size: 头寸大小，如果为None则使用当前持仓
            is_long: 是否是多头头寸
            
        Returns:
            Tuple[float, float]: (盈亏金额, 盈亏百分比)
        """
        try:
            if symbol not in self.positions and position_size is None:
                return 0.0, 0.0
                
            size = Decimal(str(position_size)) if position_size is not None else self.positions[symbol][0]
            entry_price = self.positions[symbol][1] if symbol in self.positions else Decimal('0')
            exit_price_dec = Decimal(str(exit_price))
            
            if entry_price == Decimal('0') or size == Decimal('0'):
                return 0.0, 0.0
            
            if is_long:
                pnl = (exit_price_dec - entry_price) * abs(size)
            else:
                pnl = (entry_price - exit_price_dec) * abs(size)
                
            pnl_pct = (pnl / (entry_price * abs(size))) * Decimal('100')
            
            return float(pnl), float(pnl_pct)
            
        except Exception as e:
            log.error(f"Error calculating P&L: {str(e)}")
            return 0.0, 0.0
    
    def check_daily_loss_limit(self, current_equity: float) -> bool:
        """
        检查是否达到单日亏损限制
        
        Args:
            current_equity: 当前权益
            
        Returns:
            bool: 是否达到单日亏损限制
        """
        try:
            # 获取当天的权益曲线
            today = pd.Timestamp.now().date()
            today_equity = [
                Decimal(str(eq['equity'])) 
                for eq in self.equity_curve 
                if pd.Timestamp(eq['timestamp']/1000).date() == today
            ]
            
            if not today_equity:
                return False
                
            # 计算当日最大回撤
            peak = max(today_equity)
            current = Decimal(str(current_equity))
            drawdown = (peak - current) / peak if peak > Decimal('0') else Decimal('0')
            
            return drawdown >= self.daily_loss_limit
            
        except Exception as e:
            log.error(f"Error checking daily loss limit: {str(e)}")
            return False
    
    def update_equity_curve(self, equity: float, timestamp: int) -> None:
        """
        更新权益曲线
        
        Args:
            equity: 当前权益
            timestamp: 时间戳（毫秒）
        """
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity
        })
        
        # 更新最大回撤
        if len(self.equity_curve) > 1:
            peak = max(eq['equity'] for eq in self.equity_curve)
            current = equity
            drawdown = (peak - current) / peak if peak > 0 else 0
            self.max_drawdown = max(self.max_drawdown, Decimal(str(drawdown)))
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """
        获取风险指标
        
        Returns:
            Dict: 包含风险指标的字典
        """
        return {
            'account_balance': float(self.account_balance),
            'max_drawdown': float(self.max_drawdown * Decimal('100')),  # 转换为百分比
            'position_count': len(self.positions),
            'total_exposure': float(sum(
                Decimal(str(size)) * Decimal(str(price)) 
                for _, (size, price) in self.positions.items()
            )),
            'leverage': float(self.leverage)
        }
