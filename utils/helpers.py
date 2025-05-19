"""
辅助函数模块

包含各种工具函数，用于数据处理、时间转换、数值计算等。
"""
import os
import time
import hashlib
import hmac
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np


def timestamp_to_datetime(timestamp: int, ms: bool = True) -> datetime:
    """
    将时间戳转换为datetime对象
    
    Args:
        timestamp: 时间戳
        ms: 是否为毫秒级时间戳，默认为True
        
    Returns:
        datetime: 转换后的datetime对象
    """
    if ms:
        return datetime.fromtimestamp(timestamp / 1000.0)
    return datetime.fromtimestamp(timestamp)


def datetime_to_timestamp(dt: datetime, ms: bool = True) -> int:
    """
    将datetime对象转换为时间戳
    
    Args:
        dt: datetime对象
        ms: 是否返回毫秒级时间戳，默认为True
        
    Returns:
        int: 时间戳
    """
    timestamp = int(dt.timestamp())
    return timestamp * 1000 if ms else timestamp


def parse_timeframe(timeframe: str) -> int:
    """
    将时间周期字符串转换为秒数
    
    Args:
        timeframe: 时间周期字符串，如'1m', '5m', '1h', '4h', '1d'等
        
    Returns:
        int: 对应的秒数
        
    Raises:
        ValueError: 当时间周期格式无效时
    """
    units = {
        's': 1,
        'm': 60,
        'h': 60 * 60,
        'd': 24 * 60 * 60,
        'w': 7 * 24 * 60 * 60,
        'M': 30 * 24 * 60 * 60,  # 近似值
        'y': 365 * 24 * 60 * 60   # 近似值
    }
    
    try:
        num = int(timeframe[:-1])
        unit = timeframe[-1].lower()
    except (ValueError, IndexError):
        raise ValueError(f"Invalid timeframe format: {timeframe}")
    
    if unit not in units:
        raise ValueError(f"Unsupported timeframe unit: {unit}")
    
    return num * units[unit]


def round_to_tick(price: float, tick_size: float) -> float:
    """
    将价格四舍五入到最小价格单位
    
    Args:
        price: 原始价格
        tick_size: 最小价格单位
        
    Returns:
        float: 四舍五入后的价格
    """
    if tick_size <= 0:
        return price
    
    precision = len(str(tick_size).rstrip('0').split('.')[-1]) if '.' in str(tick_size) else 0
    return round(round(price / tick_size) * tick_size, precision)


def calculate_position_size(
    account_balance: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss_price: float,
    leverage: float = 1.0
) -> float:
    """
    计算头寸大小
    
    Args:
        account_balance: 账户余额
        risk_per_trade: 每笔交易风险比例（0.01表示1%）
        entry_price: 入场价格
        stop_loss_price: 止损价格
        leverage: 杠杆倍数，默认为1（无杠杆）
        
    Returns:
        float: 计算出的头寸大小（基础货币数量）
    """
    if entry_price <= 0 or stop_loss_price <= 0 or account_balance <= 0 or risk_per_trade <= 0:
        return 0.0
    
    risk_amount = account_balance * risk_per_trade
    price_diff = abs(entry_price - stop_loss_price)
    
    if price_diff == 0:
        return 0.0
    
    position_size = (risk_amount * leverage) / price_diff
    return position_size


def calculate_pnl(
    entry_price: float,
    exit_price: float,
    quantity: float,
    is_long: bool,
    fee_rate: float = 0.001,
    is_percentage_fee: bool = True
) -> Tuple[float, float, float]:
    """
    计算盈亏和收益率
    
    Args:
        entry_price: 入场价格
        exit_price: 出场价格
        quantity: 数量
        is_long: 是否做多
        fee_rate: 手续费率，默认为0.1%
        is_percentage_fee: 手续费是否为百分比，默认为True
        
    Returns:
        Tuple[float, float, float]: (毛盈亏, 手续费, 净盈亏)
    """
    if entry_price <= 0 or exit_price <= 0 or quantity <= 0:
        return 0.0, 0.0, 0.0
    
    # 计算毛盈亏
    if is_long:
        gross_pnl = (exit_price - entry_price) * quantity
    else:
        gross_pnl = (entry_price - exit_price) * quantity
    
    # 计算手续费
    if is_percentage_fee:
        entry_fee = entry_price * quantity * fee_rate
        exit_fee = exit_price * quantity * fee_rate
    else:
        entry_fee = exit_fee = fee_rate
    
    total_fee = entry_fee + exit_fee
    
    # 计算净盈亏
    net_pnl = gross_pnl - total_fee
    
    return gross_pnl, total_fee, net_pnl


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    计算夏普比率
    
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率，年化
        
    Returns:
        float: 夏普比率
    """
    if not returns:
        return 0.0
    
    returns = np.array(returns)
    excess_returns = returns - risk_free_rate / 252  # 假设日收益率，年化252个交易日
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # 年化


def calculate_max_drawdown(values: List[float]) -> float:
    """
    计算最大回撤
    
    Args:
        values: 账户净值序列
        
    Returns:
        float: 最大回撤（0-1之间）
    """
    if not values:
        return 0.0
    
    peak = values[0]
    max_drawdown = 0.0
    
    for value in values[1:]:
        if value > peak:
            peak = value
        else:
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
    
    return max_drawdown


def generate_signature(secret: str, data: str) -> str:
    """
    使用HMAC-SHA256生成签名
    
    Args:
        secret: API密钥的secret
        data: 需要签名的数据
        
    Returns:
        str: 签名结果（小写十六进制字符串）
    """
    return hmac.new(
        secret.encode('utf-8'),
        data.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()


def format_float(value: float, precision: int = 8) -> str:
    """
    格式化浮点数为字符串，避免科学计数法
    
    Args:
        value: 需要格式化的浮点数
        precision: 小数位数
        
    Returns:
        str: 格式化后的字符串
    """
    return f"{value:.{precision}f}".rstrip('0').rstrip('.')


def ensure_directory_exists(directory: str) -> None:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def load_json_file(file_path: str) -> Union[dict, list, None]:
    """
    加载JSON文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        Union[dict, list, None]: 解析后的JSON数据，如果文件不存在或格式错误则返回None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_json_file(data: Union[dict, list], file_path: str, indent: int = 2) -> bool:
    """
    保存数据到JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
        indent: 缩进空格数
        
    Returns:
        bool: 是否保存成功
    """
    try:
        ensure_directory_exists(os.path.dirname(file_path))
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        return True
    except (IOError, TypeError):
        return False
