"""
策略基类模块

定义策略的基类接口，所有交易策略都应继承自此类。
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from utils.logger import log


class StrategyBase(ABC):
    """策略基类"""
    
    def __init__(self, params: Dict = None):
        """
        初始化策略
        
        Args:
            params: 策略参数字典
        """
        self.params = params or {}
        self.name = self.__class__.__name__
        self.initialized = False
        self.position = 0  # 当前持仓方向：1 表示多头，-1 表示空头，0 表示空仓
        self.entry_price = 0.0  # 入场价格
        self.entry_time = None  # 入场时间
        self.stop_loss = 0.0  # 止损价格
        self.take_profit = 0.0  # 止盈价格
        self.trades = []  # 交易记录
        self.equity_curve = []  # 权益曲线
        self.signals = []  # 信号记录
        
        # 初始化策略参数
        self._init_params()
    
    def _init_params(self) -> None:
        """初始化策略参数"""
        # 设置默认参数
        default_params = self.default_params()
        
        # 更新用户自定义参数
        for key, value in default_params.items():
            if key not in self.params:
                self.params[key] = value
    
    @classmethod
    @abstractmethod
    def default_params(cls) -> Dict:
        """
        返回策略的默认参数
        
        Returns:
            Dict: 参数字典
        """
        return {}
    
    def init(self) -> None:
        """
        策略初始化，在回测或实盘开始前调用一次
        可以在这里进行数据预处理、指标计算等操作
        """
        self.initialized = True
        log.info(f"Strategy {self.name} initialized with params: {self.params}")
    
    @abstractmethod
    def on_bar(self, bar: Dict) -> Dict:
        """
        处理一根K线数据
        
        Args:
            bar: 包含K线数据的字典，至少包含以下字段：
                - symbol: 交易对，如'BTC/USDT'
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - volume: 成交量
                - timestamp: 时间戳（毫秒）
                - datetime: 日期时间对象
                - 其他自定义字段
                
        Returns:
            Dict: 交易信号，包含以下字段：
                - signal: 信号类型，'buy'、'sell' 或 'hold'
                - price: 信号价格
                - size: 交易数量（可选）
                - stop_loss: 止损价格（可选）
                - take_profit: 止盈价格（可选）
                - info: 其他信息（可选）
        """
        pass
    
    def on_tick(self, tick: Dict) -> Optional[Dict]:
        """
        处理一个tick数据（可选）
        
        Args:
            tick: 包含tick数据的字典
                - symbol: 交易对，如'BTC/USDT'
                - bid: 买一价
                - ask: 卖一价
                - last: 最新成交价
                - volume: 成交量
                - timestamp: 时间戳（毫秒）
                - datetime: 日期时间对象
                - 其他自定义字段
                
        Returns:
            Optional[Dict]: 交易信号，格式同on_bar方法，如果没有信号则返回None
        """
        return None
    
    def on_order(self, order: Dict) -> None:
        """
        订单状态更新回调
        
        Args:
            order: 订单信息
        """
        log.info(f"Order update: {order}")
    
    def on_trade(self, trade: Dict) -> None:
        """
        成交回调
        
        Args:
            trade: 成交信息
        """
        log.info(f"Trade executed: {trade}")
        self.trades.append(trade)
        
        # 更新持仓
        if trade['side'] == 'buy':
            self.position += trade['size']
        else:
            self.position -= trade['size']
            
        # 如果是开仓，设置止损止盈
        if self.position != 0 and self.entry_price == 0:
            self.entry_price = trade['price']
            self.entry_time = trade['datetime']
            self._update_stop_loss_take_profit()
        # 如果是平仓，重置状态
        elif self.position == 0:
            self.entry_price = 0.0
            self.entry_time = None
            self.stop_loss = 0.0
            self.take_profit = 0.0
    
    def _update_stop_loss_take_profit(self) -> None:
        """更新止损止盈价格"""
        if self.entry_price == 0 or 'stop_loss' not in self.params or 'take_profit' not in self.params:
            return
            
        if self.position > 0:  # 多头
            self.stop_loss = self.entry_price * (1 - self.params['stop_loss'])
            self.take_profit = self.entry_price * (1 + self.params['take_profit'])
        elif self.position < 0:  # 空头
            self.stop_loss = self.entry_price * (1 + self.params['stop_loss'])
            self.take_profit = self.entry_price * (1 - self.params['take_profit'])
    
    def update_equity(self, equity: float, timestamp: int) -> None:
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
    
    def get_performance_metrics(self) -> Dict:
        """
        计算策略性能指标
        
        Returns:
            Dict: 包含各种性能指标的字典
        """
        if not self.equity_curve:
            return {}
            
        # 计算收益率序列
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['datetime'] = pd.to_datetime(equity_df['timestamp'], unit='ms')
        equity_df.set_index('datetime', inplace=True)
        equity_df['returns'] = equity_df['equity'].pct_change().fillna(0)
        
        if len(equity_df) < 2:
            return {}
        
        # 计算基本指标
        total_return = (equity_df['equity'].iloc[-1] / equity_df['equity'].iloc[0] - 1) * 100
        annual_return = (1 + total_return/100) ** (365 / ((equity_df.index[-1] - equity_df.index[0]).days or 1)) - 1
        
        # 计算最大回撤
        cum_returns = (1 + equity_df['returns']).cumprod()
        peak = cum_returns.expanding(min_periods=1).max()
        drawdown = (cum_returns - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        # 计算夏普比率（假设无风险利率为0）
        sharpe_ratio = np.sqrt(252) * equity_df['returns'].mean() / (equity_df['returns'].std() or 1)
        
        # 计算交易统计
        win_trades = [t for t in self.trades if t.get('profit', 0) > 0]
        loss_trades = [t for t in self.trades if t.get('profit', 0) <= 0]
        
        win_rate = len(win_trades) / len(self.trades) * 100 if self.trades else 0
        avg_win = np.mean([t.get('profit', 0) for t in win_trades]) if win_trades else 0
        avg_loss = abs(np.mean([t.get('profit', 0) for t in loss_trades])) if loss_trades else 0
        profit_factor = (len(win_trades) * avg_win) / (len(loss_trades) * avg_loss) if loss_trades else float('inf')
        
        return {
            'total_return_percent': total_return,
            'annual_return_percent': annual_return * 100,
            'max_drawdown_percent': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate_percent': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(self.trades),
            'win_trades': len(win_trades),
            'loss_trades': len(loss_trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade_return_percent': (total_return / len(self.trades)) if self.trades else 0,
            'start_date': equity_df.index[0],
            'end_date': equity_df.index[-1],
            'final_equity': equity_df['equity'].iloc[-1]
        }
    
    def plot_equity_curve(self, save_path: str = None) -> None:
        """
        绘制权益曲线图
        
        Args:
            save_path: 图片保存路径，如果为None则显示图片
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.equity_curve:
                log.warning("No equity data to plot")
                return
                
            df = pd.DataFrame(self.equity_curve)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            plt.figure(figsize=(12, 6))
            plt.plot(df['datetime'], df['equity'], label='Equity Curve')
            
            # 标记交易点
            if self.trades:
                trades_df = pd.DataFrame(self.trades)
                buys = trades_df[trades_df['side'] == 'buy']
                sells = trades_df[trades_df['side'] == 'sell']
                
                if not buys.empty:
                    plt.scatter(
                        pd.to_datetime(buys['datetime']),
                        [df[df['timestamp'] <= t].iloc[-1]['equity'] for t in buys['timestamp']],
                        color='green', marker='^', label='Buy', alpha=0.7, s=100
                    )
                
                if not sells.empty:
                    plt.scatter(
                        pd.to_datetime(sells['datetime']),
                        [df[df['timestamp'] <= t].iloc[-1]['equity'] for t in sells['timestamp']],
                        color='red', marker='v', label='Sell', alpha=0.7, s=100
                    )
            
            plt.title(f'Equity Curve - {self.name}')
            plt.xlabel('Date')
            plt.ylabel('Equity')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
                log.info(f"Equity curve saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            log.error("Matplotlib is not installed. Please install it with 'pip install matplotlib'")
        except Exception as e:
            log.error(f"Failed to plot equity curve: {str(e)}")
    
    def generate_signal(
        self,
        signal_type: str,
        price: float,
        size: float = None,
        stop_loss: float = None,
        take_profit: float = None,
        info: Dict = None
    ) -> Dict:
        """
        生成交易信号
        
        Args:
            signal_type: 信号类型，'buy'、'sell' 或 'hold'
            price: 信号价格
            size: 交易数量（可选）
            stop_loss: 止损价格（可选）
            take_profit: 止盈价格（可选）
            info: 其他信息（可选）
            
        Returns:
            Dict: 交易信号
        """
        signal = {
            'signal': signal_type,
            'price': price,
            'size': size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'info': info or {},
            'timestamp': int(pd.Timestamp.now().timestamp() * 1000),
            'datetime': pd.Timestamp.now()
        }
        
        self.signals.append(signal)
        return signal
    
    def get_signals(self) -> List[Dict]:
        """
        获取所有交易信号
        
        Returns:
            List[Dict]: 交易信号列表
        """
        return self.signals
    
    def get_last_signal(self) -> Optional[Dict]:
        """
        获取最后一个交易信号
        
        Returns:
            Optional[Dict]: 最后一个交易信号，如果没有信号则返回None
        """
        return self.signals[-1] if self.signals else None
