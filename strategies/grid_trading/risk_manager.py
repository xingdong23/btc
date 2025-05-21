"""
风险管理模块

提供高级风险管理功能，包括回撤控制、仓位管理和动态风险调整。
"""
import time
import numpy as np
from datetime import datetime

from utils.logger import log


class RiskManager:
    """
    风险管理器

    提供高级风险管理功能，包括回撤控制、仓位管理和动态风险调整。
    """

    def __init__(self, strategy):
        """
        初始化风险管理器

        Args:
            strategy: 策略实例，用于访问交易所接口和账户信息
        """
        self.strategy = strategy
        self.max_drawdown = 0.15  # 最大回撤限制 (15%)
        self.daily_loss_limit = 0.05  # 每日亏损限制 (5%)
        self.max_position_ratio = 0.9  # 最大仓位比例 (90%)
        self.min_position_ratio = 0.1  # 最小仓位比例 (10%)
        self.risk_check_interval = 300  # 风控检查间隔 (秒)
        self.last_check_time = 0
        self.initial_balance = None  # 初始资产
        self.peak_balance = None  # 峰值资产
        self.daily_high = None  # 当日最高资产
        self.daily_start_balance = None  # 当日起始资产
        self.last_day = None  # 上次检查的日期
        self.price_history = []  # 价格历史
        self.volatility = None  # 市场波动率
        self.volatility_window = 20  # 波动率计算窗口
        self.position_value = 0  # 当前持仓价值
        self.total_assets = 0  # 总资产

    async def initialize(self):
        """初始化风险管理器"""
        try:
            # 获取当前总资产
            self.total_assets = await self._get_total_assets()
            self.initial_balance = self.total_assets
            self.peak_balance = self.total_assets
            self.daily_high = self.total_assets
            self.daily_start_balance = self.total_assets
            self.last_day = datetime.now().date()

            log.info(f"风险管理器初始化 | 初始资产: {self.initial_balance:.2f} USDT")
            log.info(f"风险参数 | 最大回撤: {self.max_drawdown*100:.0f}% | 日亏损限制: {self.daily_loss_limit*100:.0f}% | 最大仓位: {self.max_position_ratio*100:.0f}%")

            return True
        except Exception as e:
            log.error(f"风险管理器初始化失败: {str(e)}")
            return False

    async def check_risk(self, current_price: float) -> bool:
        """
        执行风险检查

        Args:
            current_price: 当前市场价格

        Returns:
            bool: 如果触发风控措施返回True，否则返回False
        """
        current_time = time.time()

        # 控制检查频率
        if current_time - self.last_check_time < self.risk_check_interval:
            return False

        self.last_check_time = current_time

        try:
            # 更新资产信息
            self.total_assets = await self._get_total_assets()
            self.position_value = await self._get_position_value(current_price)

            # 更新峰值
            if self.total_assets > self.peak_balance:
                self.peak_balance = self.total_assets

            # 检查日期变更
            today = datetime.now().date()
            if today != self.last_day:
                # 新的一天，重置日内指标
                log.info(f"日期变更: {self.last_day} -> {today}")
                self.daily_start_balance = self.total_assets
                self.daily_high = self.total_assets
                self.last_day = today

            # 更新日内最高值
            if self.total_assets > self.daily_high:
                self.daily_high = self.total_assets

            # 计算回撤
            drawdown = 0
            if self.peak_balance > 0:
                drawdown = (self.peak_balance - self.total_assets) / self.peak_balance

            # 计算日内亏损
            daily_loss = 0
            if self.daily_start_balance > 0:
                daily_loss = (self.daily_start_balance - self.total_assets) / self.daily_start_balance

            # 计算仓位比例
            position_ratio = 0
            if self.total_assets > 0:
                position_ratio = self.position_value / self.total_assets

            # 记录风控检查日志
            log.info(
                f"风控检查 | "
                f"总资产: {self.total_assets:.2f} USDT | "
                f"持仓价值: {self.position_value:.2f} USDT | "
                f"仓位比例: {position_ratio*100:.1f}% | "
                f"回撤: {drawdown*100:.2f}% | "
                f"日内亏损: {daily_loss*100:.2f}%"
            )

            # 多层风控检查
            # 1. 基本风控检查
            if await self._check_basic_risk(drawdown, daily_loss, position_ratio):
                return True

            # 2. 仓位比例检查
            if await self._check_position_ratio(position_ratio):
                return True

            # 3. 市场情绪检查
            if await self._check_market_sentiment(current_price):
                return True

            # 4. 价格异常检查
            if await self._check_price_anomaly(current_price):
                return True

            # 更新波动率
            await self._update_volatility(current_price)

            return False

        except Exception as e:
            log.error(f"风控检查失败: {str(e)}")
            return False

    async def _check_basic_risk(self, drawdown: float, daily_loss: float, position_ratio: float) -> bool:
        """
        基本风控检查

        Args:
            drawdown: 回撤率
            daily_loss: 日内亏损率
            position_ratio: 仓位比例

        Returns:
            bool: 是否触发风控
        """
        # 检查最大回撤
        if drawdown >= self.max_drawdown:
            log.warning(f"触发最大回撤保护: {drawdown*100:.2f}% >= {self.max_drawdown*100:.0f}%")
            return True

        # 检查日内亏损
        if daily_loss >= self.daily_loss_limit:
            log.warning(f"触发日内亏损限制: {daily_loss*100:.2f}% >= {self.daily_loss_limit*100:.0f}%")
            return True

        # 检查最大仓位
        if position_ratio > self.max_position_ratio:
            log.warning(f"触发最大仓位限制: {position_ratio*100:.1f}% > {self.max_position_ratio*100:.0f}%")
            return True

        return False

    async def _check_position_ratio(self, position_ratio: float) -> bool:
        """
        仓位比例检查

        Args:
            position_ratio: 仓位比例

        Returns:
            bool: 是否触发风控
        """
        # 检查最小仓位
        if position_ratio < self.min_position_ratio:
            log.warning(f"触发最小仓位保护: {position_ratio*100:.1f}% < {self.min_position_ratio*100:.0f}%")
            return True

        # 检查仓位变化速度
        if hasattr(self, 'last_position_ratio') and self.last_position_ratio is not None:
            position_change = abs(position_ratio - self.last_position_ratio)
            if position_change > 0.2:  # 仓位变化超过20%
                log.warning(f"触发仓位变化速度限制: {position_change*100:.1f}% > 20%")
                return True

        # 更新上次仓位比例
        self.last_position_ratio = position_ratio

        return False

    async def _check_market_sentiment(self, current_price: float) -> bool:
        """
        市场情绪检查

        Args:
            current_price: 当前价格

        Returns:
            bool: 是否触发风控
        """
        # 检查波动率
        if self.volatility is not None and self.volatility > 1.0:  # 波动率超过100%
            log.warning(f"触发波动率限制: {self.volatility*100:.1f}% > 100%")
            return True

        # 检查价格趋势
        if len(self.price_history) >= 10:
            # 计算短期趋势
            short_trend = (current_price / self.price_history[-10]) - 1

            # 如果短期价格下跌超过20%
            if short_trend < -0.2:
                log.warning(f"触发短期下跌限制: {short_trend*100:.1f}% < -20%")
                return True

        return False

    async def _check_price_anomaly(self, current_price: float) -> bool:
        """
        价格异常检查

        Args:
            current_price: 当前价格

        Returns:
            bool: 是否触发风控
        """
        # 检查价格突变
        if len(self.price_history) >= 2:
            last_price = self.price_history[-1]
            price_change = abs(current_price / last_price - 1)

            # 如果价格瞬间变化超过10%
            if price_change > 0.1:
                log.warning(f"触发价格突变限制: {price_change*100:.1f}% > 10%")
                return True

        return False

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

    async def _get_position_value(self, current_price: float) -> float:
        """获取当前持仓价值 (USDT)"""
        try:
            if hasattr(self.strategy, 'broker') and hasattr(self.strategy.broker, 'exchange'):
                # 获取交易对信息
                symbol = self.strategy.params['symbol'].replace('_', '/')
                base_currency = symbol.split('/')[0]  # 基础货币 (如 BTC)

                # 获取余额
                balance = self.strategy.broker.get_balance()

                # 获取基础货币余额
                base_balance = balance.get(base_currency, 0)

                # 计算持仓价值 (USDT)
                position_value = base_balance * current_price

                return position_value
            else:
                # 回测环境，使用模拟数据
                return 5000.0
        except Exception as e:
            log.error(f"获取持仓价值失败: {str(e)}")
            return 0.0

    async def _get_current_price(self) -> float:
        """获取当前市场价格"""
        try:
            if hasattr(self.strategy, 'broker') and hasattr(self.strategy.broker, 'exchange'):
                symbol = self.strategy.params['symbol'].replace('_', '/')
                ticker = self.strategy.broker.exchange.fetch_ticker(symbol)
                return ticker['last']
            else:
                # 回测环境，使用最后一个价格
                return self.strategy.current_price
        except Exception as e:
            log.error(f"获取当前价格失败: {str(e)}")
            return 0.0

    async def _update_volatility(self, current_price: float) -> None:
        """更新市场波动率"""
        try:
            # 添加当前价格到历史记录
            self.price_history.append(current_price)

            # 保持固定窗口大小
            if len(self.price_history) > self.volatility_window:
                self.price_history = self.price_history[-self.volatility_window:]

            # 至少需要10个数据点才能计算波动率
            if len(self.price_history) >= 10:
                # 计算收益率
                returns = np.diff(self.price_history) / self.price_history[:-1]

                # 计算波动率 (标准差 * sqrt(交易日))
                self.volatility = np.std(returns) * np.sqrt(365)

                # 每10次检查记录一次波动率
                if int(time.time()) % (self.risk_check_interval * 10) < self.risk_check_interval:
                    log.info(f"市场波动率: {self.volatility*100:.2f}%")
        except Exception as e:
            log.error(f"更新波动率失败: {str(e)}")

    def get_dynamic_grid_size(self) -> float:
        """
        根据市场波动率获取动态网格大小

        Returns:
            float: 建议的网格大小百分比
        """
        # 默认网格大小
        default_grid_size = self.strategy.params.get('grid_size_pct', 2.0)

        # 如果没有波动率数据，返回默认值
        if self.volatility is None or len(self.price_history) < 10:
            return default_grid_size

        # 方法1: 映射表方式
        volatility_grid_map = [
            (0.20, 1.0),  # 波动率 0-20%，网格1.0%
            (0.40, 1.5),  # 波动率 20-40%，网格1.5%
            (0.60, 2.0),  # 波动率 40-60%，网格2.0%
            (0.80, 2.5),  # 波动率 60-80%，网格2.5%
            (1.00, 3.0),  # 波动率 80-100%，网格3.0%
            (1.20, 3.5),  # 波动率 100-120%，网格3.5%
            (999, 4.0)    # 波动率 >120%，网格4.0%
        ]

        # 方法2: 线性插值方式
        # 计算波动率比例
        standard_volatility = 0.5  # 标准波动率为50%
        volatility_ratio = self.volatility / standard_volatility

        # 调整网格大小，但设置上下限
        min_grid_size = default_grid_size * 0.5  # 最小为默认值的1/2
        max_grid_size = default_grid_size * 2.0  # 最大为默认值的2倍

        # 计算动态网格大小
        dynamic_grid_size = default_grid_size * volatility_ratio

        # 限制在上下限范围内
        dynamic_grid_size = max(min_grid_size, min(dynamic_grid_size, max_grid_size))

        # 根据策略参数决定使用哪种方式
        use_linear = self.strategy.params.get('use_linear_grid', True)

        if use_linear:
            # 使用线性插值方式
            result = dynamic_grid_size
        else:
            # 使用映射表方式
            for threshold, grid_size in volatility_grid_map:
                if self.volatility <= threshold:
                    result = grid_size
                    break
            else:
                # 默认返回最大网格大小
                result = volatility_grid_map[-1][1]

        log.info(f"动态网格大小: {result:.2f}% | 波动率: {self.volatility*100:.2f}%")
        return result

    def _get_position_ratio(self) -> float:
        """
        获取当前仓位比例

        Returns:
            float: 仓位比例 (0-1)
        """
        if self.total_assets <= 0:
            return 0.0

        return self.position_value / self.total_assets
        
    async def get_position_ratio(self) -> float:
        """
        异步获取当前仓位比例

        Returns:
            float: 仓位比例 (0-1)
        """
        return self._get_position_ratio()
        
    async def get_position_size(self) -> float:
        """
        获取当前持仓数量

        Returns:
            float: 持仓数量（基础货币数量）
        """
        try:
            if hasattr(self.strategy, 'broker') and hasattr(self.strategy.broker, 'exchange'):
                # 实盘环境，从交易所获取持仓信息
                ccxt_symbol = self.strategy.params['symbol'].replace('_', '/')
                # 获取基础货币名称（如BTC_USDT中的BTC）
                base_currency = ccxt_symbol.split('/')[0]
                
                try:
                    balance = self.strategy.broker.exchange.fetch_balance()
                    if base_currency in balance and 'free' in balance[base_currency]:
                        return float(balance[base_currency]['free'])
                except Exception as e:
                    log.error(f"获取持仓数量失败: {str(e)}")
                    return 0.0
            
            # 回测环境或获取失败，估算持仓
            if self.strategy.current_price and self.strategy.current_price > 0:
                return self.position_value / self.strategy.current_price
                
            return 0.0
        except Exception as e:
            log.error(f"计算持仓数量时出错: {str(e)}")
            return 0.0
