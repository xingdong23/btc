"""
VWAP日内交易策略包

结合VWAP（成交量加权平均价格）指标进行日内交易，利用价格围绕VWAP的波动特性。
"""

from .strategy import VWAPTraderStrategy

__all__ = [
    'VWAPTraderStrategy'
]