"""
支撑阻力识别模块

提供基于价格历史的支撑阻力水平识别功能。
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

from utils.logger import log


class SupportResistanceDetector:
    """
    支撑阻力水平检测器

    使用多种方法识别价格图表中的支撑和阻力水平。
    """

    def __init__(self, lookback_period: int = 100, price_threshold: float = 0.02,
                 touch_count: int = 2, zone_merge_threshold: float = 0.01):
        """
        初始化支撑阻力检测器

        Args:
            lookback_period: 回溯周期，用于计算支撑阻力的K线数量
            price_threshold: 价格阈值，用于确定价格是否接近支撑阻力水平 (百分比)
            touch_count: 形成支撑阻力所需的最小触及次数
            zone_merge_threshold: 合并相近支撑阻力区域的阈值 (百分比)
        """
        self.lookback_period = lookback_period
        self.price_threshold = price_threshold
        self.touch_count = touch_count
        self.zone_merge_threshold = zone_merge_threshold
        self.price_history = []  # 价格历史
        self.support_levels = []  # 支撑水平
        self.resistance_levels = []  # 阻力水平
        self.last_update_time = None  # 上次更新时间
        self.update_interval = 4 * 3600  # 更新间隔 (秒)，默认4小时

    def update(self, ohlcv_data: List[Dict]) -> None:
        """
        更新支撑阻力水平

        Args:
            ohlcv_data: OHLCV数据列表，每个元素是包含'timestamp', 'open', 'high', 'low', 'close', 'volume'的字典
        """
        # 检查是否需要更新
        current_time = datetime.now().timestamp()
        if (self.last_update_time is not None and
            current_time - self.last_update_time < self.update_interval):
            return

        # 更新时间
        self.last_update_time = current_time

        # 确保数据足够
        if len(ohlcv_data) < self.lookback_period:
            log.warning(f"数据不足，需要至少 {self.lookback_period} 根K线，当前仅有 {len(ohlcv_data)} 根")
            # 如果数据不足，使用所有可用数据并生成一些基本的支撑阻力水平
            if len(ohlcv_data) >= 20:  # 至少需要一定数量的数据
                # 使用简化的方法生成支撑阻力水平
                self._generate_simple_levels(ohlcv_data)
                log.info(f"使用简化方法生成支撑阻力水平 | 支撑: {len(self.support_levels)} 个 | 阻力: {len(self.resistance_levels)} 个")
                return
            return

        # 取最近的数据
        recent_data = ohlcv_data[-self.lookback_period:]

        # 转换为DataFrame
        df = pd.DataFrame(recent_data)

        # 识别支撑阻力水平
        self._identify_support_resistance(df)

        # 记录日志
        log.info(f"更新支撑阻力水平 | 支撑: {len(self.support_levels)} 个 | 阻力: {len(self.resistance_levels)} 个")

    def _identify_support_resistance(self, df: pd.DataFrame) -> None:
        """
        识别支撑阻力水平

        使用多种方法识别支撑阻力：
        1. 峰谷法：寻找局部高低点
        2. 价格聚集区：寻找价格频繁触及的区域

        Args:
            df: 包含OHLCV数据的DataFrame
        """
        # 方法1: 峰谷法
        self._identify_peaks_and_troughs(df)

        # 方法2: 价格聚集区
        self._identify_price_clusters(df)

        # 合并相近的支撑阻力水平
        self._merge_levels()

        # 按价格排序
        self.support_levels.sort()
        self.resistance_levels.sort()

    def _identify_peaks_and_troughs(self, df: pd.DataFrame) -> None:
        """
        使用峰谷法识别支撑阻力

        Args:
            df: 包含OHLCV数据的DataFrame
        """
        # 获取高低价
        highs = df['high'].values
        lows = df['low'].values

        # 临时存储识别到的水平
        temp_support = []
        temp_resistance = []

        # 窗口大小 (左右各看n个点)
        window = 5

        # 识别峰和谷
        for i in range(window, len(highs) - window):
            # 检查是否是局部高点 (峰)
            if highs[i] == max(highs[i-window:i+window+1]):
                temp_resistance.append(highs[i])

            # 检查是否是局部低点 (谷)
            if lows[i] == min(lows[i-window:i+window+1]):
                temp_support.append(lows[i])

        # 更新支撑阻力水平
        self.support_levels.extend(temp_support)
        self.resistance_levels.extend(temp_resistance)

    def _identify_price_clusters(self, df: pd.DataFrame) -> None:
        """
        使用价格聚集区识别支撑阻力

        Args:
            df: 包含OHLCV数据的DataFrame
        """
        # 创建价格区间
        price_range = np.linspace(df['low'].min() * 0.95, df['high'].max() * 1.05, 100)
        bin_size = (price_range[-1] - price_range[0]) / 100

        # 计算每个价格区间的触及次数
        touch_counts = np.zeros(len(price_range) - 1)

        for _, row in df.iterrows():
            # 检查每个价格区间是否被触及
            for i in range(len(price_range) - 1):
                lower = price_range[i]
                upper = price_range[i + 1]

                # 如果K线的范围与价格区间有重叠，则认为触及
                if not (row['high'] < lower or row['low'] > upper):
                    touch_counts[i] += 1

        # 识别触及次数较多的区域作为支撑阻力
        threshold = np.percentile(touch_counts, 80)  # 取前20%的区域

        for i in range(len(touch_counts)):
            if touch_counts[i] >= threshold:
                # 计算区域中心价格
                center_price = (price_range[i] + price_range[i + 1]) / 2

                # 根据位置判断是支撑还是阻力
                if i < len(touch_counts) / 2:  # 下半部分为支撑
                    self.support_levels.append(center_price)
                else:  # 上半部分为阻力
                    self.resistance_levels.append(center_price)

    def _merge_levels(self) -> None:
        """合并相近的支撑阻力水平"""
        # 合并支撑水平
        self.support_levels = self._merge_nearby_levels(self.support_levels)

        # 合并阻力水平
        self.resistance_levels = self._merge_nearby_levels(self.resistance_levels)

    def _generate_simple_levels(self, ohlcv_data: List[Dict]) -> None:
        """
        使用简化的方法生成支撑阻力水平

        当数据不足时使用这个方法生成基本的支撑阻力水平

        Args:
            ohlcv_data: OHLCV数据列表
        """
        # 提取高低点
        highs = [data['high'] for data in ohlcv_data]
        lows = [data['low'] for data in ohlcv_data]
        closes = [data['close'] for data in ohlcv_data]

        # 计算价格范围
        min_price = min(lows)
        max_price = max(highs)
        current_price = closes[-1]
        price_range = max_price - min_price

        # 生成简单的支撑阻力水平
        # 1. 使用最高价和最低价
        self.resistance_levels = [max_price]
        self.support_levels = [min_price]

        # 2. 添加中间水平
        mid_price = (max_price + min_price) / 2
        if abs(mid_price - current_price) / current_price > 0.02:  # 如果中间价格与当前价格相差超过2%
            if mid_price < current_price:
                self.support_levels.append(mid_price)
            else:
                self.resistance_levels.append(mid_price)

        # 3. 添加四分位水平
        quarter_price = min_price + price_range * 0.25
        three_quarter_price = min_price + price_range * 0.75

        if abs(quarter_price - current_price) / current_price > 0.02:
            if quarter_price < current_price:
                self.support_levels.append(quarter_price)
            else:
                self.resistance_levels.append(quarter_price)

        if abs(three_quarter_price - current_price) / current_price > 0.02:
            if three_quarter_price < current_price:
                self.support_levels.append(three_quarter_price)
            else:
                self.resistance_levels.append(three_quarter_price)

        # 4. 添加近期的局部高低点
        if len(ohlcv_data) >= 10:
            recent_data = ohlcv_data[-10:]
            recent_highs = [data['high'] for data in recent_data]
            recent_lows = [data['low'] for data in recent_data]

            # 找到最近的高点和低点
            recent_max = max(recent_highs)
            recent_min = min(recent_lows)

            # 如果这些点与已有的水平不重复，则添加
            if recent_max not in self.resistance_levels and abs(recent_max - current_price) / current_price > 0.01:
                self.resistance_levels.append(recent_max)

            if recent_min not in self.support_levels and abs(recent_min - current_price) / current_price > 0.01:
                self.support_levels.append(recent_min)

        # 排序支撑阻力水平
        self.support_levels.sort()
        self.resistance_levels.sort()

    def _merge_nearby_levels(self, levels: List[float]) -> List[float]:
        """
        合并相近的价格水平

        Args:
            levels: 价格水平列表

        Returns:
            List[float]: 合并后的价格水平列表
        """
        if not levels:
            return []

        # 排序
        sorted_levels = sorted(levels)

        # 合并结果
        merged = []
        current_group = [sorted_levels[0]]

        for i in range(1, len(sorted_levels)):
            # 计算当前价格与组内平均价格的差异
            current_price = sorted_levels[i]
            group_avg = sum(current_group) / len(current_group)

            # 如果差异小于阈值，则合并
            if abs(current_price - group_avg) / group_avg <= self.zone_merge_threshold:
                current_group.append(current_price)
            else:
                # 添加当前组的平均值到结果
                merged.append(sum(current_group) / len(current_group))
                # 开始新的组
                current_group = [current_price]

        # 添加最后一组
        if current_group:
            merged.append(sum(current_group) / len(current_group))

        return merged

    def get_nearest_levels(self, price: float) -> Tuple[Optional[float], Optional[float]]:
        """
        获取距离当前价格最近的支撑和阻力水平

        Args:
            price: 当前价格

        Returns:
            Tuple[Optional[float], Optional[float]]: (最近支撑, 最近阻力)
        """
        nearest_support = None
        nearest_resistance = None

        # 寻找最近的支撑
        supports_below = [s for s in self.support_levels if s < price]
        if supports_below:
            nearest_support = max(supports_below)

        # 寻找最近的阻力
        resistances_above = [r for r in self.resistance_levels if r > price]
        if resistances_above:
            nearest_resistance = min(resistances_above)

        return nearest_support, nearest_resistance

    def is_near_support(self, price: float) -> bool:
        """
        检查价格是否接近支撑位

        Args:
            price: 当前价格

        Returns:
            bool: 是否接近支撑位
        """
        for support in self.support_levels:
            if abs(price - support) / support <= self.price_threshold:
                return True
        return False

    def is_near_resistance(self, price: float) -> bool:
        """
        检查价格是否接近阻力位

        Args:
            price: 当前价格

        Returns:
            bool: 是否接近阻力位
        """
        for resistance in self.resistance_levels:
            if abs(price - resistance) / resistance <= self.price_threshold:
                return True
        return False

    def get_support_strength(self, price: float) -> float:
        """
        获取支撑强度 (0-1)

        Args:
            price: 当前价格

        Returns:
            float: 支撑强度，0表示无支撑，1表示强支撑
        """
        if not self.support_levels:
            return 0.0

        # 找到最近的支撑位
        nearest_support = None
        min_distance = float('inf')

        for support in self.support_levels:
            if support < price:
                distance = price - support
                if distance < min_distance:
                    min_distance = distance
                    nearest_support = support

        if nearest_support is None:
            return 0.0

        # 计算支撑强度 (距离越近，强度越大)
        strength = 1.0 - min(min_distance / (price * self.price_threshold), 1.0)
        return strength

    def get_resistance_strength(self, price: float) -> float:
        """
        获取阻力强度 (0-1)

        Args:
            price: 当前价格

        Returns:
            float: 阻力强度，0表示无阻力，1表示强阻力
        """
        if not self.resistance_levels:
            return 0.0

        # 找到最近的阻力位
        nearest_resistance = None
        min_distance = float('inf')

        for resistance in self.resistance_levels:
            if resistance > price:
                distance = resistance - price
                if distance < min_distance:
                    min_distance = distance
                    nearest_resistance = resistance

        if nearest_resistance is None:
            return 0.0

        # 计算阻力强度 (距离越近，强度越大)
        strength = 1.0 - min(min_distance / (price * self.price_threshold), 1.0)
        return strength

    def get_all_levels(self) -> Dict[str, List[float]]:
        """
        获取所有支撑阻力水平

        Returns:
            Dict[str, List[float]]: 包含支撑和阻力水平的字典
        """
        return {
            'support': self.support_levels,
            'resistance': self.resistance_levels
        }
