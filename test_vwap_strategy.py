import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategies.vwap_trader import VWAPTraderStrategy

class TestVWAPStrategy(unittest.TestCase):
    """
    VWAPu7b56u7565u6d4bu8bd5u7c7b
    """
    
    def setUp(self):
        """u521du59cbu5316u6d4bu8bd5u73afu5883"""
        # u521bu5efau7b56u7565u5bf9u8c61
        self.strategy = VWAPTraderStrategy({
            'vwap_lookback': 20,        # u4f7fu7528u8f83u77edu7684u56deu770bu5468u671fu65b9u4fbfu6d4bu8bd5
            'buy_threshold': 0.01,     # 1%u7684u8d2du4e70u9608u503c
            'sell_threshold': 0.01,    # 1%u7684u5356u51fau9608u503c
            'volume_factor': 1.2,      # u6210u4ea4u91cfu786eu8ba4u56e0u5b50
            'reset_vwap_daily': False  # u6d4bu8bd5u4e2du4e0du91cdu7f6eVWAP
        })
        
        # u751fu6210u6d4bu8bd5u6570u636e
        self.test_data = self._generate_test_data()
    
    def _generate_test_data(self):
        """u751fu6210u6d4bu8bd5u7528u7684Ku7ebfu6570u636e"""
        # u521bu5efa50u6839u865au62dfu7684Ku7ebfu6570u636e
        base_price = 30000.0  # u57fau7840u4ef7u683c
        bars = []
        
        # u8bbeu7f6eu57fau51c6u65f6u95f4
        base_time = datetime.now() - timedelta(hours=50)
        
        # u751fu6210u903bu8f91uff1au521du59cbu4ef7u683cu57fau4e8eu57fau7840u4ef7u683cuff0cu6bcfu4e2aKu7ebfu6ce2u52a8u4e00u5b9au767eu5206u6bd4
        # u524d20u6839Ku7ebfu518du57fau7840u4ef7u683cu9644u8fd1u6ce2u52a8uff0cu7136u540eu4ef7u683cu5f00u59cbu4e0bu8dccu518du53cdu5f39
        for i in range(50):
            timestamp = int((base_time + timedelta(hours=i)).timestamp() * 1000)
            dt = base_time + timedelta(hours=i)
            
            # u521bu5efau4ef7u683cu6a21u5f0fuff1au5148u7a33u5b9auff0cu7136u540eu4e0bu8dccuff0cu518du53cdu5f39
            if i < 20:
                # u57fau672cu7a33u5b9au671f
                change_pct = np.random.uniform(-0.005, 0.005)  # u5c0fu6ce2u52a8
                price_factor = 1 + change_pct
            elif i < 30:
                # u4e0bu8dccu671f
                change_pct = np.random.uniform(-0.018, -0.008)  # u8f83u5927u8dccu5e45
                price_factor = 1 + change_pct
            else:
                # u53cdu5f39u671f
                change_pct = np.random.uniform(0.005, 0.015)  # u53cdu5f39
                price_factor = 1 + change_pct
            
            # u751fu6210u5f53u524dKu7ebfu4ef7u683c
            if i == 0:
                close_price = base_price
            else:
                close_price = bars[-1]['close'] * price_factor
            
            # u751fu6210u9ad8u4f4eu5f00u6536
            open_price = close_price * np.random.uniform(0.997, 1.003)
            high_price = max(close_price, open_price) * np.random.uniform(1.001, 1.005)
            low_price = min(close_price, open_price) * np.random.uniform(0.995, 0.999)
            
            # u751fu6210u6210u4ea4u91cf (u7279u610fu5728u67d0u4e9bKu7ebfu8bbeu7f6eu8f83u5927u6210u4ea4u91cf)
            if i == 25 or i == 35:  # u5728u4e0bu8dccu548cu53cdu5f39u65f6u8bbeu7f6eu5927u6210u4ea4u91cf
                volume = np.random.uniform(200, 500)  # u8f83u5927u6210u4ea4u91cf
            else:
                volume = np.random.uniform(50, 150)  # u666eu901au6210u4ea4u91cf
            
            # u6dfbu52a0Ku7ebfu6570u636e
            bar = {
                'timestamp': timestamp,
                'datetime': dt,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            }
            bars.append(bar)
        
        return bars
    
    def test_vwap_calculation(self):
        """u6d4bu8bd5VWAPu8ba1u7b97"""
        # u5904u7406u591au4e2aKu7ebfu8ba9u7b56u7565u8ba1u7b97VWAP
        for i in range(25):  # u524d25u6839Ku7ebf
            self.strategy.on_bar(self.test_data[i])
        
        # u68c0u67e5VWAPu662fu5426u5df2u7ecfu8ba1u7b97
        self.assertIsNotNone(self.strategy.current_vwap)
        self.assertTrue(len(self.strategy.vwap_values) > 0)
        
        # u68c0u67e5VWAPu5e94u8be5u63a5u8fd1u5f53u524du4ef7u683c
        latest_price = self.test_data[24]['close']
        vwap = self.strategy.current_vwap
        # VWAPu5e94u8be5u5728u4ef7u683cu7684u5408u7406u8303u56f4u5185
        self.assertTrue(0.9 * latest_price < vwap < 1.1 * latest_price)
    
    def test_buy_signal_generation(self):
        """u6d4bu8bd5u4e70u5165u4fe1u53f7u751fu6210"""
        # u5148u5904u7406u524d25u6839Ku7ebfu521du59cbu5316VWAP
        for i in range(25):  # u524d25u6839Ku7ebf
            self.strategy.on_bar(self.test_data[i])
        
        # u6536u96c6u4fe1u53f7
        signals = []
        for i in range(25, 35):  # u6d4bu8bd5u4e0bu8dccu671fu95f4u7684u4fe1u53f7
            signal = self.strategy.on_bar(self.test_data[i])
            signals.append(signal)
        
        # u5e94u8be5u81f3u5c11u6709u4e00u4e2au4e70u5165u4fe1u53f7
        buy_signals = [s for s in signals if s['signal'] == 'buy']
        self.assertTrue(len(buy_signals) > 0)
        
        # u68c0u67e5u4e70u5165u4fe1u53f7u7684u5408u7406u6027
        if buy_signals:
            # u4e70u5165u65f6u4ef7u683cu5e94u5f53u4f4eu4e8eVWAP
            for signal in buy_signals:
                self.assertTrue(signal['price'] < self.strategy.current_vwap * (1 - 0.005))  # u81f3u5c11u4f4eu4e8eVWAP 0.5%
    
    def test_sell_signal_generation(self):
        """u6d4bu8bd5u5356u51fau4fe1u53f7u751fu6210"""
        # u5148u5904u7406u524d35u6839Ku7ebfuff0cu5e76u4ebau4e3au8bbeu7f6eu4e00u4e2au6301u4ed3u72b6u6001
        for i in range(35):
            self.strategy.on_bar(self.test_data[i])
        
        # u624bu52a8u6a21u62dfu4e00u4e2au4e70u5165u72b6u6001
        self.strategy.in_position = True
        self.strategy.position_size = 0.1  # 0.1 BTC
        self.strategy.entry_price = self.test_data[34]['close'] * 0.98  # u5047u8bbeu5728u7a0du5faeu4f4eu4e8eu5f53u524du4ef7u683cu5165u573a
        self.strategy.entry_time = self.test_data[34]['datetime']
        self.strategy.stop_loss = self.strategy.entry_price * 0.95  # 5%u6b62u635f
        self.strategy.take_profit = self.strategy.entry_price * 1.05  # 5%u6b62u76c8
        self.strategy.trailing_stop = self.strategy.stop_loss
        self.strategy.highest_since_entry = self.strategy.entry_price
        
        # u6536u96c6u4fe1u53f7
        signals = []
        for i in range(35, 50):  # u6d4bu8bd5u53cdu5f39u671fu95f4u7684u4fe1u53f7
            signal = self.strategy.on_bar(self.test_data[i])
            signals.append(signal)
            if signal['signal'] == 'sell':
                break  # u4e00u65e6u5356u51fau5c31u7ed3u675f
        
        # u5e94u8be5u81f3u5c11u6709u4e00u4e2au5356u51fau4fe1u53f7
        sell_signals = [s for s in signals if s['signal'] == 'sell']
        self.assertTrue(len(sell_signals) > 0)
        
        # u68c0u67e5u5356u51fau4fe1u53f7u7684u5408u7406u6027
        if sell_signals:
            # u9a8cu8bc1u5356u51fau4ef7u683cu4e0eu6b62u76c8/u6b62u635f/VWAPu7684u5173u7cfb
            first_sell = sell_signals[0]
            self.assertTrue(
                first_sell['price'] >= self.strategy.entry_price or  # u76c8u5229u5356u51fa
                'exit_reason' in first_sell['info']  # u6709u51fau573au539fu56e0
            )

if __name__ == '__main__':
    unittest.main()