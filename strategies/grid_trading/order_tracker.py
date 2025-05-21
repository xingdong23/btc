"""
订单跟踪模块

提供订单跟踪、交易历史记录和统计分析功能。
"""
import os
import json
import time
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from utils.logger import log


class OrderThrottler:
    """
    订单限流器

    限制单位时间内的订单数量，防止过度交易。
    """

    def __init__(self, limit: int = 10, interval: int = 60):
        """
        初始化订单限流器

        Args:
            limit: 单位时间内的最大订单数量
            interval: 时间窗口（秒）
        """
        self.order_timestamps = []
        self.limit = limit
        self.interval = interval

    def check_rate(self) -> bool:
        """
        检查是否可以下新订单

        Returns:
            bool: 如果可以下新订单返回true，否则返回false
        """
        current_time = time.time()

        # 清理过期的时间戳
        self.order_timestamps = [t for t in self.order_timestamps if current_time - t < self.interval]

        # 检查是否超过限制
        if len(self.order_timestamps) >= self.limit:
            log.warning(f"订单频率超过限制: {self.limit}/{self.interval}s")
            return False

        # 添加新的时间戳
        self.order_timestamps.append(current_time)
        return True


class OrderTracker:
    """
    订单跟踪器

    跟踪订单状态、记录交易历史并提供统计分析功能。
    """

    def __init__(self, data_dir: str = 'data'):
        """
        初始化订单跟踪器

        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = data_dir

        # 创建数据目录
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # 文件路径
        self.history_file = os.path.join(self.data_dir, 'trade_history.json')
        self.backup_file = os.path.join(self.data_dir, 'trade_history.backup.json')

        # 创建子目录
        self.archive_dir = os.path.join(self.data_dir, 'archives')
        self.export_dir = os.path.join(self.data_dir, 'exports')
        for directory in [self.archive_dir, self.export_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        # 状态变量
        self.orders = {}  # 订单状态 {order_id: order_info}
        self.trade_history = []  # 交易历史
        self.trade_count = 0  # 交易计数
        self.total_profit = 0.0  # 总利润
        self.win_count = 0  # 盈利交易数
        self.loss_count = 0  # 亏损交易数
        self.max_archive_months = 12  # 归档保存最大月数
        self.consecutive_wins = 0  # 连续盈利次数
        self.consecutive_losses = 0  # 连续亏损次数
        self.max_consecutive_wins = 0  # 最大连续盈利次数
        self.max_consecutive_losses = 0  # 最大连续亏损次数
        self.last_trade_profit = None  # 上次交易盈亏

        # 加载历史数据
        self._load_trade_history()

        # 清理过期归档
        self.clean_old_archives()

    def _load_trade_history(self) -> None:
        """加载交易历史"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.trade_history = json.load(f)
                    self.trade_count = len(self.trade_history)

                    # 计算统计数据
                    self.total_profit = sum(trade.get('profit', 0) for trade in self.trade_history)
                    self.win_count = sum(1 for trade in self.trade_history if trade.get('profit', 0) > 0)
                    self.loss_count = sum(1 for trade in self.trade_history if trade.get('profit', 0) < 0)

                    log.info(f"加载交易历史: {self.trade_count}条记录")
        except Exception as e:
            log.error(f"加载交易历史失败: {str(e)}")
            self.trade_history = []

    def _save_trade_history(self) -> None:
        """保存交易历史"""
        try:
            # 备份当前文件
            if os.path.exists(self.history_file):
                import shutil
                shutil.copy2(self.history_file, self.backup_file)

            # 保存新文件
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.trade_history, f, ensure_ascii=False, indent=2)

            log.info(f"保存交易历史: {len(self.trade_history)}条记录")
        except Exception as e:
            log.error(f"保存交易历史失败: {str(e)}")

    def add_order(self, order: Dict) -> None:
        """
        添加订单

        Args:
            order: 订单信息
        """
        try:
            order_id = order['id']

            # 记录订单信息
            self.orders[order_id] = {
                'id': order_id,
                'side': order['side'],
                'price': float(order['price']),
                'amount': float(order['amount']),
                'status': order['status'],
                'timestamp': int(time.time() * 1000),
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            log.info(f"添加订单: {order_id} | {order['side']} {order['amount']} @ {order['price']}")
        except Exception as e:
            log.error(f"添加订单失败: {str(e)}")

    def update_order(self, order_id: str, status: str) -> None:
        """
        更新订单状态

        Args:
            order_id: 订单ID
            status: 订单状态
        """
        if order_id in self.orders:
            self.orders[order_id]['status'] = status
            log.info(f"更新订单状态: {order_id} -> {status}")

    def add_trade(self, trade: Dict) -> None:
        """
        添加交易记录

        Args:
            trade: 交易信息，包含以下字段:
                - side: 交易方向 ('buy' 或 'sell')
                - price: 成交价格
                - amount: 成交数量
                - timestamp: 成交时间戳 (可选)
                - order_id: 订单ID (可选)
                - profit: 利润 (可选)
        """
        try:
            # 验证必要字段
            required_fields = ['side', 'price', 'amount']
            for field in required_fields:
                if field not in trade:
                    log.error(f"交易记录缺少必要字段: {field}")
                    return

            # 添加时间戳
            if 'timestamp' not in trade:
                trade['timestamp'] = int(time.time() * 1000)

            # 添加日期时间
            trade['datetime'] = datetime.fromtimestamp(
                trade['timestamp'] / 1000
            ).strftime('%Y-%m-%d %H:%M:%S')

            # 计算交易成本
            trade['cost'] = float(trade['price']) * float(trade['amount'])

            # 计算手续费 (假设0.1%)
            trade['fee'] = trade['cost'] * 0.001

            # 添加到历史记录
            self.trade_history.append(trade)
            self.trade_count += 1

            # 更新统计数据
            if 'profit' in trade:
                profit = trade['profit']
                self.total_profit += profit

                # 更新盈亏计数
                if profit > 0:
                    self.win_count += 1

                    # 更新连续盈利计数
                    if self.last_trade_profit is None or self.last_trade_profit <= 0:
                        # 新的盈利序列开始
                        self.consecutive_wins = 1
                        self.consecutive_losses = 0
                    else:
                        # 继续盈利序列
                        self.consecutive_wins += 1

                    # 更新最大连续盈利计数
                    self.max_consecutive_wins = max(self.max_consecutive_wins, self.consecutive_wins)

                elif profit < 0:
                    self.loss_count += 1

                    # 更新连续亏损计数
                    if self.last_trade_profit is None or self.last_trade_profit >= 0:
                        # 新的亏损序列开始
                        self.consecutive_losses = 1
                        self.consecutive_wins = 0
                    else:
                        # 继续亏损序列
                        self.consecutive_losses += 1

                    # 更新最大连续亏损计数
                    self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)

                # 记录当前交易盈亏
                self.last_trade_profit = profit

            # 保存历史记录
            self._save_trade_history()

            log.info(
                f"添加交易记录: {trade['side']} {trade['amount']} @ {trade['price']} | "
                f"成本: {trade['cost']:.2f} | "
                f"手续费: {trade['fee']:.2f}" +
                (f" | 利润: {trade['profit']:.2f}" if 'profit' in trade else "")
            )
        except Exception as e:
            log.error(f"添加交易记录失败: {str(e)}")

    def get_statistics(self) -> Dict:
        """
        获取交易统计信息

        Returns:
            Dict: 统计信息
        """
        try:
            if not self.trade_history:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'total_profit': 0,
                    'avg_profit': 0,
                    'max_profit': 0,
                    'max_loss': 0,
                    'profit_factor': 0,
                    'consecutive_wins': 0,
                    'consecutive_losses': 0,
                    'max_consecutive_wins': 0,
                    'max_consecutive_losses': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'win_loss_ratio': 0,
                    'expectancy': 0
                }

            # 计算统计数据
            total_trades = self.trade_count
            win_rate = self.win_count / total_trades if total_trades > 0 else 0
            avg_profit = self.total_profit / total_trades if total_trades > 0 else 0

            # 计算最大盈利和最大亏损
            profits = [trade.get('profit', 0) for trade in self.trade_history if 'profit' in trade]
            max_profit = max(profits) if profits else 0
            max_loss = min(profits) if profits else 0

            # 计算盈亏比
            winning_trades = [p for p in profits if p > 0]
            losing_trades = [p for p in profits if p < 0]
            total_wins = sum(winning_trades)
            total_losses = abs(sum(losing_trades))
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

            # 计算平均盈利和平均亏损
            avg_win = total_wins / len(winning_trades) if winning_trades else 0
            avg_loss = total_losses / len(losing_trades) if losing_trades else 0

            # 计算盈亏比率
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

            # 计算期望值
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss) if total_trades > 0 else 0

            # 计算最大连续盈利和亏损
            current_streak = 1
            max_win_streak = 0
            max_loss_streak = 0

            for i in range(1, len(profits)):
                if (profits[i] > 0 and profits[i-1] > 0) or (profits[i] < 0 and profits[i-1] < 0):
                    current_streak += 1
                else:
                    if profits[i-1] > 0:
                        max_win_streak = max(max_win_streak, current_streak)
                    else:
                        max_loss_streak = max(max_loss_streak, current_streak)
                    current_streak = 1

            # 处理最后一个连续序列
            if len(profits) > 0:
                if profits[-1] > 0:
                    max_win_streak = max(max_win_streak, current_streak)
                else:
                    max_loss_streak = max(max_loss_streak, current_streak)

            # 计算交易频率
            if len(self.trade_history) >= 2:
                first_trade_time = self.trade_history[0].get('timestamp', 0) / 1000
                last_trade_time = self.trade_history[-1].get('timestamp', 0) / 1000
                trading_days = (last_trade_time - first_trade_time) / (24 * 3600)
                trades_per_day = total_trades / trading_days if trading_days > 0 else 0
            else:
                trades_per_day = 0

            return {
                'total_trades': total_trades,
                'win_count': self.win_count,
                'loss_count': self.loss_count,
                'win_rate': win_rate,
                'total_profit': self.total_profit,
                'avg_profit': avg_profit,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'profit_factor': profit_factor,
                'consecutive_wins': self.consecutive_wins,
                'consecutive_losses': self.consecutive_losses,
                'max_consecutive_wins': max(max_win_streak, self.max_consecutive_wins),
                'max_consecutive_losses': max(max_loss_streak, self.max_consecutive_losses),
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'win_loss_ratio': win_loss_ratio,
                'expectancy': expectancy,
                'trades_per_day': trades_per_day
            }
        except Exception as e:
            log.error(f"获取统计信息失败: {str(e)}")
            return {}

    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """
        获取最近的交易记录

        Args:
            limit: 返回的记录数量

        Returns:
            List[Dict]: 交易记录列表
        """
        return self.trade_history[-limit:] if self.trade_history else []

    def analyze_daily_performance(self, days: int = 30) -> Dict:
        """
        分析每日交易表现

        Args:
            days: 分析的天数

        Returns:
            Dict: 每日表现数据
        """
        try:
            if not self.trade_history:
                return {}

            # 转换为DataFrame
            df = pd.DataFrame(self.trade_history)

            # 确保有timestamp字段
            if 'timestamp' not in df.columns:
                return {}

            # 转换时间戳为日期
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date

            # 过滤最近N天的数据
            cutoff_date = (datetime.now() - timedelta(days=days)).date()
            recent_df = df[df['date'] >= cutoff_date]

            if recent_df.empty:
                return {}

            # 按日期分组
            daily_stats = recent_df.groupby('date').agg({
                'profit': ['sum', 'count'],
                'cost': 'sum'
            })

            # 重置列名
            daily_stats.columns = ['profit', 'trades', 'volume']

            # 转换为字典
            result = {}
            for date, row in daily_stats.iterrows():
                result[str(date)] = {
                    'profit': float(row['profit']) if 'profit' in row else 0,
                    'trades': int(row['trades']),
                    'volume': float(row['volume'])
                }

            return result
        except Exception as e:
            log.error(f"分析每日表现失败: {str(e)}")
            return {}

    def export_trades(self, format: str = 'csv') -> bool:
        """
        导出交易记录

        Args:
            format: 导出格式 ('csv' 或 'json')

        Returns:
            bool: 是否成功导出
        """
        try:
            if not self.trade_history:
                log.warning("没有交易记录可导出")
                return False

            # 确保导出目录存在
            if not os.path.exists(self.export_dir):
                os.makedirs(self.export_dir)

            # 生成文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            if format.lower() == 'csv':
                # 导出为CSV
                export_file = os.path.join(self.export_dir, f'trades_export_{timestamp}.csv')

                # 确定列名
                fieldnames = ['datetime', 'side', 'price', 'amount', 'cost', 'fee', 'profit']

                with open(export_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

                    for trade in self.trade_history:
                        # 创建一个新字典，只包含我们需要的字段
                        row = {}
                        for field in fieldnames:
                            row[field] = trade.get(field, '')
                        writer.writerow(row)
            else:
                # 导出为JSON
                export_file = os.path.join(self.export_dir, f'trades_export_{timestamp}.json')
                with open(export_file, 'w', encoding='utf-8') as f:
                    json.dump(self.trade_history, f, ensure_ascii=False, indent=2)

            log.info(f"交易记录已导出到: {export_file}")
            return True

        except Exception as e:
            log.error(f"导出交易记录失败: {str(e)}")
            return False

    def archive_old_trades(self) -> None:
        """归档旧的交易记录"""
        try:
            if len(self.trade_history) <= 100:
                return

            # 获取当前月份作为归档文件名
            current_month = datetime.now().strftime('%Y%m')
            archive_file = os.path.join(self.archive_dir, f'trades_{current_month}.json')

            # 将旧记录移动到归档
            old_trades = self.trade_history[:-100]

            # 如果归档文件存在，先读取并合并
            if os.path.exists(archive_file):
                with open(archive_file, 'r', encoding='utf-8') as f:
                    archived_trades = json.load(f)
                    old_trades = archived_trades + old_trades

            # 保存归档
            with open(archive_file, 'w', encoding='utf-8') as f:
                json.dump(old_trades, f, ensure_ascii=False, indent=2)

            # 更新当前交易历史
            self.trade_history = self.trade_history[-100:]
            log.info(f"已归档 {len(old_trades)} 条交易记录到 {archive_file}")
        except Exception as e:
            log.error(f"归档交易记录失败: {str(e)}")

    def clean_old_archives(self) -> None:
        """清理过期的归档文件"""
        try:
            if not os.path.exists(self.archive_dir):
                return

            archive_files = [f for f in os.listdir(self.archive_dir) if f.startswith('trades_') and f.endswith('.json')]
            archive_files.sort(reverse=True)  # 按时间倒序排列

            # 保留最近N个月的归档
            if len(archive_files) > self.max_archive_months:
                for old_file in archive_files[self.max_archive_months:]:
                    file_path = os.path.join(self.archive_dir, old_file)
                    os.remove(file_path)
                    log.info(f"已删除过期归档: {old_file}")
        except Exception as e:
            log.error(f"清理归档失败: {str(e)}")

    def analyze_trades(self, days: int = 30) -> Dict:
        """
        分析最近交易表现

        Args:
            days: 分析的天数

        Returns:
            Dict: 分析结果
        """
        try:
            if not self.trade_history:
                return None

            # 计算时间范围
            now = time.time()
            start_time = now - (days * 24 * 3600)

            # 筛选时间范围内的交易
            recent_trades = [t for t in self.trade_history if t.get('timestamp', 0) > start_time * 1000]

            if not recent_trades:
                return None

            # 按天统计
            daily_stats = {}
            for trade in recent_trades:
                trade_date = datetime.fromtimestamp(trade.get('timestamp', 0) / 1000).strftime('%Y-%m-%d')

                if trade_date not in daily_stats:
                    daily_stats[trade_date] = {
                        'trades': 0,
                        'profit': 0,
                        'volume': 0
                    }

                daily_stats[trade_date]['trades'] += 1
                daily_stats[trade_date]['profit'] += trade.get('profit', 0)
                daily_stats[trade_date]['volume'] += trade.get('price', 0) * trade.get('amount', 0)

            # 计算最佳和最差交易日
            best_day = max(daily_stats.items(), key=lambda x: x[1]['profit']) if daily_stats else None
            worst_day = min(daily_stats.items(), key=lambda x: x[1]['profit']) if daily_stats else None

            return {
                'period': f'最近{days}天',
                'total_days': len(daily_stats),
                'active_days': len([d for d in daily_stats.values() if d['trades'] > 0]),
                'daily_stats': daily_stats,
                'avg_daily_trades': sum(d['trades'] for d in daily_stats.values()) / len(daily_stats) if daily_stats else 0,
                'avg_daily_profit': sum(d['profit'] for d in daily_stats.values()) / len(daily_stats) if daily_stats else 0,
                'best_day': best_day,
                'worst_day': worst_day
            }
        except Exception as e:
            log.error(f"分析交易失败: {str(e)}")
            return None

    def reset(self) -> None:
        """重置订单跟踪器"""
        # 先备份当前交易历史
        self._save_trade_history()

        # 清空状态
        self.orders.clear()
        self.trade_count = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.last_trade_profit = None

        log.info("订单跟踪器已重置")
