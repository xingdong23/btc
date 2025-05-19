"""
Gate.io 交易所接口实现

提供与Gate.io交易所交互的功能，包括获取市场数据、账户信息和执行交易。
"""
import time
import ccxt
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any

from config.config import config
from utils.logger import log
from utils.helpers import (
    timestamp_to_datetime,
    datetime_to_timestamp,
    parse_timeframe,
    round_to_tick
)


class GateIOExchange:
    """Gate.io交易所接口类"""
    
    def __init__(self, api_key: str = None, secret: str = None, testnet: bool = False):
        """
        初始化Gate.io交易所接口
        
        Args:
            api_key: API密钥，如果为None则从配置中读取
            secret: API密钥的secret，如果为None则从配置中读取
            testnet: 是否使用测试网络
        """
        self.api_key = api_key or config.gateio_api_key
        self.secret = secret or config.gateio_secret
        self.testnet = testnet
        
        # 初始化ccxt客户端
        exchange_class = getattr(ccxt, 'gateio')
        exchange_params = {
            'apiKey': self.api_key,
            'secret': self.secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # 现货交易
                'adjustForTimeDifference': True,
            },
            'timeout': 30000,  # 30秒超时
        }
        
        if testnet:
            exchange_params['urls'] = {
                'api': {
                    'public': 'https://fx-api-testnet.gateio.ws/api/v4',
                    'private': 'https://fx-api-testnet.gateio.ws/api/v4',
                }
            }
        
        self.exchange = exchange_class(exchange_params)
        self._markets = None
        self._symbols = []
        self._precision_cache = {}
    
    def load_markets(self, reload: bool = False) -> Dict:
        """
        加载市场数据
        
        Args:
            reload: 是否强制重新加载
            
        Returns:
            Dict: 市场数据
        """
        if self._markets is None or reload:
            try:
                self._markets = self.exchange.load_markets()
                self._symbols = list(self._markets.keys())
                log.info(f"Loaded {len(self._symbols)} trading pairs from Gate.io")
            except Exception as e:
                log.error(f"Failed to load markets: {str(e)}")
                raise
        
        return self._markets
    
    def get_market_info(self, symbol: str) -> Dict:
        """
        获取交易对的市场信息
        
        Args:
            symbol: 交易对，如'BTC/USDT'
            
        Returns:
            Dict: 市场信息
        """
        self.load_markets()
        return self._markets.get(symbol, {})
    
    def get_precision(self, symbol: str) -> Dict[str, int]:
        """
        获取交易对的精度信息
        
        Args:
            symbol: 交易对，如'BTC/USDT'
            
        Returns:
            Dict: 包含价格精度和数量精度的字典
        """
        if symbol in self._precision_cache:
            return self._precision_cache[symbol]
        
        market = self.get_market_info(symbol)
        if not market:
            return {'price': 8, 'amount': 8}  # 默认精度
        
        precision = {
            'price': market['precision']['price'],
            'amount': market['precision']['amount']
        }
        self._precision_cache[symbol] = precision
        return precision
    
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        since: Optional[int] = None,
        limit: int = 1000,
        params: Dict = None
    ) -> pd.DataFrame:
        """
        获取K线数据
        
        Args:
            symbol: 交易对，如'BTC/USDT'
            timeframe: K线周期，如'1m', '5m', '15m', '1h', '4h', '1d'等
            since: 开始时间戳（毫秒）
            limit: 获取的K线数量
            params: 其他参数
            
        Returns:
            pd.DataFrame: 包含OHLCV数据的DataFrame
        """
        params = params or {}
        
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit,
                params=params
            )
            
            if not ohlcv:
                return pd.DataFrame()
                
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # 转换时间戳为datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
                
            return df
            
        except Exception as e:
            log.error(f"Failed to fetch OHLCV data for {symbol}: {str(e)}")
            raise
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        获取指定交易对的行情数据
        
        Args:
            symbol: 交易对，例如 'BTC/USDT'
            
        Returns:
            dict: 包含行情数据的字典，如果获取失败则返回None
        """
        max_retries = 3
        retry_delay = 1  # 初始重试延迟1秒
        
        for attempt in range(max_retries):
            try:
                log.info(f"Fetching ticker for {symbol} (attempt {attempt + 1}/{max_retries})")
                ticker = self.exchange.fetch_ticker(symbol)
                if ticker and 'last' in ticker and ticker['last'] is not None:
                    log.info(f"Successfully fetched ticker for {symbol}")
                    return ticker
                else:
                    log.warning(f"Received invalid ticker data for {symbol}")
                    
            except ccxt.NetworkError as e:
                log.warning(f"Network error fetching ticker for {symbol}: {str(e)}")
            except ccxt.ExchangeError as e:
                log.error(f"Exchange error fetching ticker for {symbol}: {str(e)}")
                break  # 交易所错误时不重试
            except Exception as e:
                log.error(f"Unexpected error fetching ticker for {symbol}: {str(e)}")
            
            # 如果不是最后一次尝试，则等待后重试
            if attempt < max_retries - 1:
                log.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
        
        log.error(f"Failed to fetch ticker for {symbol} after {max_retries} attempts")
        return None
    
    def get_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """
        获取订单簿
        
        Args:
            symbol: 交易对，如'BTC/USDT'
            limit: 订单簿深度
            
        Returns:
            Dict: 包含买单和卖单的订单簿
        """
        try:
            return self.exchange.fetch_order_book(symbol, limit=limit)
        except Exception as e:
            log.error(f"Failed to fetch order book for {symbol}: {str(e)}")
            raise
    
    def get_balance(self) -> Dict[str, float]:
        """
        获取账户余额
        
        Returns:
            Dict: 包含各币种余额的字典
        """
        try:
            balance = self.exchange.fetch_balance()
            return {
                currency: {
                    'free': float(balance[currency]['free']) if currency in balance else 0.0,
                    'used': float(balance[currency]['used']) if currency in balance else 0.0,
                    'total': float(balance[currency]['total']) if currency in balance else 0.0
                }
                for currency in balance
                if currency and not currency.startswith('')
            }
        except Exception as e:
            log.error(f"Failed to fetch balance: {str(e)}")
            raise
    
    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Dict = None
    ) -> Dict:
        """
        创建订单
        
        Args:
            symbol: 交易对，如'BTC/USDT'
            order_type: 订单类型，'limit'或'market'
            side: 买卖方向，'buy'或'sell'
            amount: 交易数量
            price: 价格，市价单可忽略
            params: 其他参数
            
        Returns:
            Dict: 订单信息
        """
        params = params or {}
        
        # 获取精度信息并调整数量
        precision = self.get_precision(symbol)
        amount = round(amount, precision['amount'])
        
        if price is not None and order_type == 'limit':
            price = round(price, precision['price'])
        
        try:
            log.info(f"Creating {order_type} {side} order for {amount} {symbol} at {price}")
            order = self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params
            )
            log.info(f"Order created: {order['id']}")
            return order
            
        except Exception as e:
            log.error(f"Failed to create {order_type} {side} order for {amount} {symbol}: {str(e)}")
            raise
    
    def get_order(self, order_id: str, symbol: str = None) -> Dict:
        """
        获取订单信息
        
        Args:
            order_id: 订单ID
            symbol: 交易对，如'BTC/USDT'
            
        Returns:
            Dict: 订单信息
        """
        try:
            return self.exchange.fetch_order(order_id, symbol)
        except Exception as e:
            log.error(f"Failed to fetch order {order_id}: {str(e)}")
            raise
    
    def cancel_order(self, order_id: str, symbol: str = None) -> Dict:
        """
        取消订单
        
        Args:
            order_id: 订单ID
            symbol: 交易对，如'BTC/USDT'
            
        Returns:
            Dict: 取消订单的结果
        """
        try:
            log.info(f"Cancelling order {order_id}")
            result = self.exchange.cancel_order(order_id, symbol)
            log.info(f"Order {order_id} cancelled")
            return result
        except Exception as e:
            log.error(f"Failed to cancel order {order_id}: {str(e)}")
            raise
    
    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """
        获取未成交订单
        
        Args:
            symbol: 交易对，如'BTC/USDT'，如果为None则返回所有交易对的未成交订单
            
        Returns:
            List[Dict]: 未成交订单列表
        """
        try:
            return self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            log.error(f"Failed to fetch open orders: {str(e)}")
            raise
    
    def get_my_trades(
        self,
        symbol: str = None,
        since: int = None,
        limit: int = None,
        params: Dict = None
    ) -> List[Dict]:
        """
        获取个人成交记录
        
        Args:
            symbol: 交易对，如'BTC/USDT'，如果为None则返回所有交易对的成交记录
            since: 开始时间戳（毫秒）
            limit: 返回的成交记录数量
            params: 其他参数
            
        Returns:
            List[Dict]: 成交记录列表
        """
        params = params or {}
        
        try:
            return self.exchange.fetch_my_trades(
                symbol=symbol,
                since=since,
                limit=limit,
                params=params
            )
        except Exception as e:
            log.error(f"Failed to fetch my trades: {str(e)}")
            raise


# 全局交易所实例
gateio = GateIOExchange()
