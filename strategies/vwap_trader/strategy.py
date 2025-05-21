"""
VWAPu65e5u5185u4ea4u6613u7b56u7565

u7ed3u5408VWAPuff08u6210u4ea4u91cfu52a0u6743u5e73u5747u4ef7u683cuff09u6307u6807u8fdbu884cu4ea4u6613uff0cu5229u7528u4ef7u683cu76f8u5bf9u4e8eVWAPu7684u504fu79bbu8fdbu884cu4ea4u6613u3002
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from strategies.base import StrategyBase
from utils.logger import log
from utils.helpers import round_to_tick, calculate_position_size


class VWAPTraderStrategy(StrategyBase):
    """
    VWAPu65e5u5185u4ea4u6613u7b56u7565
    
    u7b56u7565u903bu8f91uff1a
    1. u8ba1u7b97u65e5u5185VWAPu503c
    2. u5f53u4ef7u683cu8dccu81f3VWAPu4e0bu65b9u4e00u5b9au6bd4u4f8bu65f6uff0cu8003u8651u4e70u5165uff08u4ef7u683cu88abu4f4eu4f30uff09
    3. u5f53u4ef7u683cu6da8u81f3VWAPu4e0au65b9u4e00u5b9au6bd4u4f8bu65f6uff0cu8003u8651u5356u51fauff08u4ef7u683cu88abu9ad8u4f30uff09
    4. u7ed3u5408u4ea4u6613u91cfu53d8u5316u786eu8ba4u4fe1u53f7
    5. u8bbeu7f6eu81eau9002u5e94u7684u6b62u635fu4e0eu6b62u76c8
    """
    
    @classmethod
    def default_params(cls) -> Dict:
        """
        u8fd4u56deu7b56u7565u9ed8u8ba4u53c2u6570
        
        Returns:
            Dict: u53c2u6570u5b57u5178
        """
        return {
            'symbol': 'BTC_USDT',
            'vwap_lookback': 120,         # VWAPu8ba1u7b97u7684Ku7ebfu6570u91cfuff08u65e5u5185uff09
            'buy_threshold': 0.01,        # u4e70u5165u9608u503cuff08u4ef7u683cu4f4eu4e8eVWAPu7684u6bd4u4f8buff09
            'sell_threshold': 0.01,       # u5356u51fau9608u503cuff08u4ef7u683cu9ad8u4e8eVWAPu7684u6bd4u4f8buff09
            'volume_factor': 1.5,         # u4ea4u6613u91cfu786eu8ba4u56e0u5b50uff08u76f8u5bf9u4e8eu5e73u5747u4ea4u6613u91cfuff09
            'risk_per_trade': 0.01,       # u6bcfu7b14u4ea4u6613u98ceu9669u6bd4u4f8b (1%)
            'reward_risk_ratio': 2.0,     # u76c8u4e8fu6bd4
            'max_position_ratio': 0.8,    # u6700u5927u4ed3u4f4du6bd4u4f8b (80%)
            'max_trades_per_day': 10,     # u6bcfu65e5u6700u5927u4ea4u6613u6b21u6570
            'min_trade_interval': 900,    # u6700u5c0fu4ea4u6613u95f4u9694 (u79d2)
            'trailing_stop_pct': 0.005,   # u8ffdu8e2au6b62u635fu767eu5206u6bd4 (0.5%)
            'session_start_hour': 0,      # u4ea4u6613u65f6u6bb5u5f00u59cbu5c0fu65f6uff08UTCuff09
            'session_end_hour': 23,       # u4ea4u6613u65f6u6bb5u7ed3u675fu5c0fu65f6uff08UTCuff09
            'reset_vwap_daily': True,     # u662fu5426u6bcfu65e5u91cdu7f6eVWAPu8ba1u7b97
            'use_trend_filter': True,     # u662fu5426u4f7fu7528u8d8bu52bfu8fc7u6ee4u5668
            'trend_ma_short': 20,         # u77edu671fu79fbu52a8u5e73u5747u7ebfu5468u671f
            'trend_ma_long': 50,          # u957fu671fu79fbu52a8u5e73u5747u7ebfu5468u671f
            'trend_slope_period': 10,     # u8d8bu52bfu659cu7387u8ba1u7b97u5468u671f
            'trend_slope_threshold': 0.001, # u8d8bu52bfu659cu7387u9608u503c
            'use_rsi_filter': True,       # 是否使用RSI过滤器
            'rsi_period': 14,             # RSI计算周期
            'rsi_buy_threshold': 30,      # RSI买入阈值（低于此值视为超卖）
            'rsi_sell_threshold': 70      # RSI卖出阈值（高于此值视为超买）
        }
    
    def __init__(self, params: Dict = None):
        """
        u521du59cbu5316u7b56u7565
        
        Args:
            params: u7b56u7565u53c2u6570u5b57u5178
        """
        super().__init__(params)
        
        # u4ef7u683cu548cu6307u6807u6570u636e
        self.price_data = []             # u4ef7u683cu5386u53f2u6570u636e
        self.current_price = None         # u5f53u524du4ef7u683c
        self.current_vwap = None         # u5f53u524d VWAP u503c
        self.vwap_values = []            # VWAPu503cu5386u53f2
        self.volume_ma = None            # u6210u4ea4u91cfu79fbu52a8u5e73u5747
        self.rsi_values = []             # RSIu503cu5386u53f2
        self.current_rsi = None          # u5f53u524dRSIu503c
        
        # u4ea4u6613u72b6u6001
        self.in_position = False          # u662fu5426u6301u4ed3
        self.position_size = 0.0          # u6301u4ed3u6570u91cf
        self.entry_price = 0.0            # u5165u573au4ef7u683c
        self.entry_time = None            # u5165u573au65f6u95f4
        self.stop_loss = 0.0              # u6b62u635fu4ef7u683c
        self.take_profit = 0.0            # u6b62u76c8u4ef7u683c
        self.trailing_stop = 0.0          # u8ffdu8e2au6b62u635fu4ef7u683c
        self.highest_since_entry = 0.0    # u5165u573au540eu6700u9ad8u4ef7
        self.lowest_since_entry = float('inf')  # u5165u573au540eu6700u4f4eu4ef7
        
        # u4ea4u6613u9650u5236
        self.trades_today = 0             # u4ecau65e5u4ea4u6613u6b21u6570
        self.last_trade_time = 0          # u4e0au6b21u4ea4u6613u65f6u95f4
        self.last_trade_date = None       # u4e0au6b21u4ea4u6613u65e5u671f
        self.vwap_reset_date = None       # VWAPu91cdu7f6eu65e5u671f
        
        # u7cbeu5ea6u8bbeu7f6e
        self.price_precision = 2          # u4ef7u683cu7cbeu5ea6
        self.amount_precision = 6         # u6570u91cfu7cbeu5ea6
    
    def init(self) -> None:
        """u7b56u7565u521du59cbu5316"""
        super().init()
        log.info(f"VWAPu65e5u5185u4ea4u6613u7b56u7565u521du59cbu5316 | u53c2u6570: {self.params}")
    
    def on_bar(self, bar: Dict) -> Dict:
        """
        u5904u7406Ku7ebfu6570u636e
        
        Args:
            bar: Ku7ebfu6570u636eu5b57u5178
            
        Returns:
            Dict: u4ea4u6613u4fe1u53f7
        """
        # u66f4u65b0u5f53u524du4ef7u683c
        self.current_price = bar['close']
        
        # u66f4u65b0u4ef7u683cu5386u53f2
        self.price_data.append({
            'timestamp': bar['timestamp'],
            'datetime': bar['datetime'] if isinstance(bar['datetime'], datetime) else 
                       datetime.fromtimestamp(bar['timestamp'] / 1000),
            'open': bar['open'],
            'high': bar['high'],
            'low': bar['low'],
            'close': bar['close'],
            'volume': bar['volume']
        })
        
        # u68c0u67e5u662fu5426u9700u8981u91cdu7f6eVWAP
        current_date = self.price_data[-1]['datetime'].date()
        if self.params['reset_vwap_daily'] and (self.vwap_reset_date is None or current_date > self.vwap_reset_date):
            log.info(f"u91cdu7f6eVWAPu8ba1u7b97 | u65e5u671f: {current_date}")
            # u4fddu7559u4ecau65e5u6570u636e
            today_data = [d for d in self.price_data if d['datetime'].date() == current_date]
            self.price_data = today_data
            self.vwap_reset_date = current_date
        
        # u4fddu6301u4ef7u683cu5386u53f2u5728u5408u7406u8303u56f4u5185
        max_history = max(self.params['vwap_lookback'] * 2, 300)  # u4fddu7559u8db3u591fu7684u5386u53f2u6570u636e
        if len(self.price_data) > max_history:
            self.price_data = self.price_data[-max_history:]
        
        # u68c0u67e5u662fu5426u662fu65b0u7684u4ea4u6613u65e5
        if self.last_trade_date is None or current_date != self.last_trade_date:
            self.trades_today = 0
            self.last_trade_date = current_date
        
        # u68c0u67e5u662fu5426u5728u4ea4u6613u65f6u6bb5u5185
        current_hour = self.price_data[-1]['datetime'].hour
        if not (self.params['session_start_hour'] <= current_hour < self.params['session_end_hour']):
            log.debug(f"u975eu4ea4u6613u65f6u6bb5 | u5f53u524du65f6u95f4: {self.price_data[-1]['datetime']}")
            return self.generate_signal('hold', self.current_price)
        
        # u8ba1u7b97VWAP
        self._calculate_vwap()
        
        # u8ba1u7b97u6210u4ea4u91cfu79fbu52a8u5e73u5747
        self._calculate_volume_ma()
        
        # u8ba1u7b97RSIu6307u6807
        self._calculate_rsi()
        
        # u68c0u67e5u6301u4ed3u72b6u6001
        if self.in_position:
            # u66f4u65b0u8ffdu8e2au6b62u635f
            self._update_trailing_stop()
            
            # u68c0u67e5u51fau573au6761u4ef6
            exit_signal = self._check_exit_conditions(bar)
            if exit_signal:
                return exit_signal
        else:
            # u68c0u67e5u5165u573au4fe1u53f7
            entry_signal = self._check_entry_conditions(bar)
            if entry_signal:
                return entry_signal
        
        # u9ed8u8ba4u8fd4u56deu6301u6709u4fe1u53f7
        return self.generate_signal('hold', self.current_price)
    
    def _calculate_vwap(self) -> None:
        """u8ba1u7b97VWAPu6307u6807"""
        if len(self.price_data) < 2:
            return
        
        # u5224u65adu662fu5426u6709u8db3u591fu7684u6570u636e
        lookback = min(self.params['vwap_lookback'], len(self.price_data))
        data = self.price_data[-lookback:]
        
        # u8ba1u7b97VWAP
        total_pv = 0.0  # price * volume
        total_volume = 0.0
        
        for bar in data:
            # u4f7fu7528u5178u578bu7684VWAPu8ba1u7b97u516cu5f0f: (u9ad8+u4f4e+u6536)/3 * u6210u4ea4u91cf
            typical_price = (bar['high'] + bar['low'] + bar['close']) / 3
            pv = typical_price * bar['volume']
            total_pv += pv
            total_volume += bar['volume']
        
        if total_volume > 0:
            self.current_vwap = total_pv / total_volume
            self.vwap_values.append(self.current_vwap)
            
            # u4fddu6301VWAPu5386u53f2u5728u5408u7406u8303u56f4u5185
            if len(self.vwap_values) > 100:
                self.vwap_values = self.vwap_values[-100:]
    
    def _calculate_volume_ma(self) -> None:
        """u8ba1u7b97u6210u4ea4u91cfu79fbu52a8u5e73u5747"""
        if len(self.price_data) < 10:  # u81f3u5c11u9700u8981u51e0u4e2aKu7ebfu7684u6570u636e
            return
        
        # u4f7fu7528u6700u8fd120u4e2aKu7ebfu7684u6210u4ea4u91cfu8ba1u7b97u79fbu52a8u5e73u5747
        lookback = min(20, len(self.price_data))
        volumes = [bar['volume'] for bar in self.price_data[-lookback:]]
        self.volume_ma = sum(volumes) / len(volumes)
    
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
        if len(self.rsi_values) > 100:
            self.rsi_values = self.rsi_values[-100:]
    
    def _detect_higher_timeframe_trend(self) -> int:
        """
        u8bc6u522bu66f4u9ad8u65f6u95f4u7ea7u522bu7684u8d8bu52bf
        
        Returns:
            int: 1(u4e0au5347u8d8bu52bf), -1(u4e0bu964du8d8bu52bf), 0(u6a2au76d8)
        """
        if not self.params.get('use_trend_filter', False) or len(self.price_data) < self.params['trend_ma_long']:
            return 0  # 默认横盘/不过滤
        
        # 提取收盘价
        closes = [data['close'] for data in self.price_data]
        
        # 计算短期和长期移动平均线
        ma_short = sum(closes[-self.params['trend_ma_short']:]) / self.params['trend_ma_short']
        ma_long = sum(closes[-self.params['trend_ma_long']:]) / self.params['trend_ma_long']
        
        # 如果数据量足够，计算移动平均线的斜率
        if len(closes) > self.params['trend_slope_period']:
            slope_period = self.params['trend_slope_period']
            ma_short_slope = (closes[-1] - closes[-slope_period]) / closes[-slope_period]
        else:
            ma_short_slope = 0
        
        # 确定趋势
        if ma_short > ma_long and ma_short_slope > self.params['trend_slope_threshold']:
            return 1  # 上升趋势
        elif ma_short < ma_long and ma_short_slope < -self.params['trend_slope_threshold']:
            return -1  # 下降趋势
        else:
            return 0  # 横盘
    
    def _check_entry_conditions(self, bar: Dict) -> Optional[Dict]:
        """
        u68c0u67e5u5165u573au6761u4ef6
        
        Args:
            bar: Ku7ebfu6570u636e
            
        Returns:
            Optional[Dict]: u5165u573au4fe1u53f7u6216None
        """
        if not self.current_vwap or not self.volume_ma or self.current_rsi is None:
            return None
        
        # 检查高时间级别趋势
        if self.params.get('use_trend_filter', False):
            trend = self._detect_higher_timeframe_trend()
            # 在下降趋势中不做多
            if trend == -1:
                return None
        
        # 检查RSI过滤器
        if self.params.get('use_rsi_filter', False):
            # 买入时需要RSI低于超卖阈值
            if self.current_rsi > self.params['rsi_buy_threshold']:
                return None
        
        # u68c0u67e5u4ea4u6613u6b21u6570u9650u5236
        if self.trades_today >= self.params['max_trades_per_day']:
            return None
        
        # u68c0u67e5u4ea4u6613u95f4u9694
        current_time = bar['timestamp'] / 1000
        if current_time - self.last_trade_time < self.params['min_trade_interval']:
            return None
        
        # u8ba1u7b97u5f53u524du4ef7u683cu4e0eVWAPu7684u504fu79bbu767eu5206u6bd4
        price_to_vwap_ratio = (self.current_price - self.current_vwap) / self.current_vwap
        
        # u5f53u524dKu7ebfu6210u4ea4u91cfu76f8u5bf9u4e8eu79fbu52a8u5e73u5747u7684u500du6570
        volume_ratio = bar['volume'] / self.volume_ma if self.volume_ma > 0 else 0
        
        # u4e70u5165u6761u4ef6uff1au4ef7u683cu4f4eu4e8eVWAPu4e00u5b9au6bd4u4f8buff0cu4e14u6210u4ea4u91cfu653eu5927
        if (price_to_vwap_ratio <= -self.params['buy_threshold'] and 
            volume_ratio >= self.params['volume_factor']):
            
            # u8ba1u7b97u4ed3u4f4du5927u5c0f
            account_balance = 10000.0  # u5047u8bbeu8d26u6237u4f59u989duff0cu5b9eu76d8u4e2du5e94u4eceu4ea4u6613u6240u83b7u53d6
            
            # u8ba1u7b97u6b62u635fu4ef7u683cuff08u5f53u524du4ef7u683cu4e0bu65b9u7684u4e00u5b9au6bd4u4f8buff09
            stop_loss = self.current_price * (1 - self.params['trailing_stop_pct'] * 2)  # u521du59cbu6b62u635fu8bbeu7f6eu5f97u5927u4e00u4e9b
            
            # u8ba1u7b97u5934u5bffu5927u5c0f
            position_size, risk_amount = calculate_position_size(
                account_balance=account_balance,
                risk_per_trade=self.params['risk_per_trade'],
                entry_price=self.current_price,
                stop_loss_price=stop_loss
            ), account_balance * self.params['risk_per_trade']
            
            # u8ba1u7b97u6b62u76c8u4ef7u683cuff08u4f7fu7528VWAPu4f5cu4e3au6b62u76c8u76eeu6807uff09
            take_profit = self.current_vwap * (1 + self.params['sell_threshold'])  # u4f7fu7528VWAPu7684u4e0au65b9u504fu79bbu4f5cu4e3au6b62u76c8
            
            # u68c0u67e5u6b62u76c8u7684u76c8u4e8fu6bd4
            risk = self.current_price - stop_loss
            reward = take_profit - self.current_price
            if reward / risk < self.params['reward_risk_ratio']:
                # u6839u636eu76c8u4e8fu6bd4u8c03u6574u6b62u76c8
                take_profit = self.current_price + risk * self.params['reward_risk_ratio']
            
            # u66f4u65b0u4ea4u6613u72b6u6001
            self.in_position = True
            self.position_size = position_size
            self.entry_price = self.current_price
            self.entry_time = bar['datetime'] if isinstance(bar['datetime'], datetime) else \
                             datetime.fromtimestamp(bar['timestamp'] / 1000)
            self.stop_loss = stop_loss
            self.take_profit = take_profit
            self.trailing_stop = stop_loss
            self.highest_since_entry = self.current_price
            self.lowest_since_entry = self.current_price
            
            # u66f4u65b0u4ea4u6613u9650u5236
            self.trades_today += 1
            self.last_trade_time = current_time
            
            # 获取当前趋势信息
            trend = self._detect_higher_timeframe_trend() if self.params.get('use_trend_filter', False) else 0
            trend_desc = {1: "上升", 0: "横盘", -1: "下降"}[trend]
            
            # 获取RSI信息
            rsi_value = self.current_rsi
            
            # u751fu6210u4e70u5165u4fe1u53f7
            vwap_diff_percent = price_to_vwap_ratio * 100
            log.info(f"u4e70u5165u4fe1u53f7 | u4ef7u683c: {self.current_price:.2f} | VWAP: {self.current_vwap:.2f} | "
                 f"u504fu79bb: {vwap_diff_percent:.2f}% | u6210u4ea4u91cfu6bd4: {volume_ratio:.2f}x | "
                 f"u8d8bu52bf: {trend_desc} | RSI: {rsi_value:.1f} | u6b62u635f: {stop_loss:.2f} | u6b62u76c8: {take_profit:.2f}")
            
            return self.generate_signal(
            'buy',
            self.current_price,
            self.position_size,
            self.stop_loss,
            self.take_profit,
            {
            'vwap': self.current_vwap,
            'vwap_diff_percent': vwap_diff_percent,
            'volume_ratio': volume_ratio,
            'trend': trend_desc,
                'rsi': rsi_value
                    }
                )
        
        return None
    
    def _check_exit_conditions(self, bar: Dict) -> Optional[Dict]:
        """
        u68c0u67e5u51fau573au6761u4ef6
        
        Args:
            bar: Ku7ebfu6570u636e
            
        Returns:
            Optional[Dict]: u51fau573au4fe1u53f7u6216None
        """
        if not self.in_position or not self.current_vwap:
            return None
        
        # u66f4u65b0u5165u573au540eu7684u6700u9ad8u6700u4f4eu4ef7
        self.highest_since_entry = max(self.highest_since_entry, bar['high'])
        self.lowest_since_entry = min(self.lowest_since_entry, bar['low'])
        
        # u8ba1u7b97u5f53u524du4ef7u683cu4e0eVWAPu7684u504fu79bbu767eu5206u6bd4
        price_to_vwap_ratio = (self.current_price - self.current_vwap) / self.current_vwap
        
        # u5f53u524dKu7ebfu6210u4ea4u91cfu76f8u5bf9u4e8eu79fbu52a8u5e73u5747u7684u500du6570
        volume_ratio = bar['volume'] / self.volume_ma if self.volume_ma > 0 else 0
        
        # u68c0u67e5u6b62u635f
        if self.current_price <= self.trailing_stop:
            # u89e6u53d1u6b62u635f
            log.info(f"u89e6u53d1u6b62u635f | u4ef7u683c: {self.current_price:.2f} | u6b62u635fu4ef7: {self.trailing_stop:.2f} | "
                     f"u5165u573au4ef7: {self.entry_price:.2f} | u76c8u4e8f: {(self.current_price/self.entry_price-1)*100:.2f}%")
            
            # u66f4u65b0u4ea4u6613u72b6u6001
            self.in_position = False
            
            # u751fu6210u5356u51fau4fe1u53f7
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
        
        # u68c0u67e5u6b62u76c8
        if self.current_price >= self.take_profit:
            # u89e6u53d1u6b62u76c8
            log.info(f"u89e6u53d1u6b62u76c8 | u4ef7u683c: {self.current_price:.2f} | u6b62u76c8u4ef7: {self.take_profit:.2f} | "
                     f"u5165u573au4ef7: {self.entry_price:.2f} | u76c8u4e8f: {(self.current_price/self.entry_price-1)*100:.2f}%")
            
            # u66f4u65b0u4ea4u6613u72b6u6001
            self.in_position = False
            
            # u751fu6210u5356u51fau4fe1u53f7
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
        
        # u68c0u67e5u662fu5426u5e94u8be5u57fau4e8eVWAPu76f8u5bf9u4f4du7f6eu5356u51fa
        # u5f53u4ef7u683cu8d85u8fc7VWAPu4e00u5b9au6bd4u4f8bu4e14u6709u5145u8db3u7684u6210u4ea4u91cf
        if ((price_to_vwap_ratio >= self.params['sell_threshold'] and 
            volume_ratio >= self.params['volume_factor']) or
            (self.params.get('use_rsi_filter', False) and self.current_rsi is not None and 
             self.current_rsi >= self.params['rsi_sell_threshold'])):
            
            vwap_diff_percent = price_to_vwap_ratio * 100
            exit_reason = 'vwap_deviation'
            
            # u786eu5b9au9000u51fau539fu56e0
            if self.params.get('use_rsi_filter', False) and self.current_rsi is not None and \
               self.current_rsi >= self.params['rsi_sell_threshold']:
                exit_reason = 'rsi_overbought'
                log.info(f"RSIu8d85u4e70u5356u51fa | u4ef7u683c: {self.current_price:.2f} | RSI: {self.current_rsi:.2f} | "
                         f"VWAP: {self.current_vwap:.2f} | u504fu79bb: {vwap_diff_percent:.2f}% | "
                         f"u5165u573au4ef7: {self.entry_price:.2f} | u76c8u4e8f: {(self.current_price/self.entry_price-1)*100:.2f}%")
            else:
                log.info(f"VWAPu76f8u5bf9u4f4du7f6eu5356u51fa | u4ef7u683c: {self.current_price:.2f} | VWAP: {self.current_vwap:.2f} | "
                         f"u504fu79bb: {vwap_diff_percent:.2f}% | u6210u4ea4u91cfu6bd4: {volume_ratio:.2f}x | "
                         f"u5165u573au4ef7: {self.entry_price:.2f} | u76c8u4e8f: {(self.current_price/self.entry_price-1)*100:.2f}%")
            
            # u66f4u65b0u4ea4u6613u72b6u6001
            self.in_position = False
            
            # u751fu6210u5356u51fau4fe1u53f7
            return self.generate_signal(
                'sell',
                self.current_price,
                self.position_size,
                None,
                None,
                {
                    'exit_reason': exit_reason,
                    'vwap': self.current_vwap,
                    'vwap_diff_percent': vwap_diff_percent,
                    'volume_ratio': volume_ratio,
                    'rsi': self.current_rsi,
                    'entry_price': self.entry_price,
                    'profit_pct': (self.current_price / self.entry_price - 1) * 100
                }
            )
        
        # u68c0u67e5u6301u4ed3u65f6u95f4u9650u5236 (u5982u679cu6301u4ed3u65f6u95f4u8d85u8fc74u5c0fu65f6u4e14u4ef7u683cu63a5u8fd1VWAPu5219u5356u51fa)
        if self.entry_time:
            hours_in_position = (datetime.now() - self.entry_time).total_seconds() / 3600
            # u5982u679cu6301u4ed3u65f6u95f4u8d85u8fc74u5c0fu65f6u4e14u4ef7u683cu63a5u8fd1u6216u8d85u8fc7VWAPuff0cu8003u8651u5356u51fa
            if hours_in_position >= 4 and self.current_price >= self.current_vwap * 0.998:
                log.info(f"u6301u4ed3u65f6u95f4u9650u5236u5356u51fa | u4ef7u683c: {self.current_price:.2f} | VWAP: {self.current_vwap:.2f} | "
                         f"u6301u4ed3u65f6u95f4: {hours_in_position:.1f}u5c0fu65f6 | "
                         f"u5165u573au4ef7: {self.entry_price:.2f} | u76c8u4e8f: {(self.current_price/self.entry_price-1)*100:.2f}%")
                
                # u66f4u65b0u4ea4u6613u72b6u6001
                self.in_position = False
                
                # u751fu6210u5356u51fau4fe1u53f7
                return self.generate_signal(
                    'sell',
                    self.current_price,
                    self.position_size,
                    None,
                    None,
                    {
                        'exit_reason': 'time_limit',
                        'hours_in_position': hours_in_position,
                        'vwap': self.current_vwap,
                        'entry_price': self.entry_price,
                        'profit_pct': (self.current_price / self.entry_price - 1) * 100
                    }
                )
        
        return None
    
    def _update_trailing_stop(self) -> None:
        """u66f4u65b0u8ffdu8e2au6b62u635f"""
        if not self.in_position:
            return
        
        # u8ba1u7b97u65b0u7684u8ffdu8e2au6b62u635f
        new_stop = self.highest_since_entry * (1 - self.params['trailing_stop_pct'])
        
        # u4ec5u4e0au79fbu6b62u635fuff0cu4e0du4e0bu79fb
        if new_stop > self.trailing_stop:
            # u5982u679cu7b56u7565u5df2u76c8u5229uff0cu786eu4fddu6b62u635fu4e0du4f4eu4e8eu5165u573au4ef7u683c(u6b62u76c8u51cfu4e2d)
            if self.highest_since_entry > self.entry_price * 1.01 and new_stop < self.entry_price:
                new_stop = self.entry_price
            
            self.trailing_stop = new_stop
            log.info(f"u66f4u65b0u8ffdu8e2au6b62u635f | u65b0u6b62u635f: {self.trailing_stop:.2f} | "
                     f"u5f53u524du4ef7: {self.current_price:.2f} | u6700u9ad8u4ef7: {self.highest_since_entry:.2f}")
    
    def get_strategy_state(self) -> Dict:
        """
        u83b7u53d6u7b56u7565u72b6u6001
        
        Returns:
            Dict: u7b56u7565u72b6u6001u5b57u5178
        """
        trend = self._detect_higher_timeframe_trend() if self.params.get('use_trend_filter', False) else 0
        trend_desc = {1: "上升", 0: "横盘", -1: "下降"}[trend]
        
        return {
            'in_position': self.in_position,
            'position_size': self.position_size,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'current_vwap': self.current_vwap,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'trailing_stop': self.trailing_stop,
            'highest_since_entry': self.highest_since_entry,
            'lowest_since_entry': self.lowest_since_entry,
            'trades_today': self.trades_today,
            'volume_ma': self.volume_ma,
            'trend': trend_desc,
            'current_rsi': self.current_rsi,
            'rsi_buy_threshold': self.params.get('rsi_buy_threshold', 30),
            'rsi_sell_threshold': self.params.get('rsi_sell_threshold', 70),
            'ma_short': sum([data['close'] for data in self.price_data[-self.params['trend_ma_short']:]]) / self.params['trend_ma_short'] if len(self.price_data) >= self.params['trend_ma_short'] else None,
            'ma_long': sum([data['close'] for data in self.price_data[-self.params['trend_ma_long']:]]) / self.params['trend_ma_long'] if len(self.price_data) >= self.params['trend_ma_long'] else None
        }