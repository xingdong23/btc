"""
日志管理模块

提供统一的日志记录功能，支持控制台和文件输出，支持不同日志级别。
"""
import os
import logging
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger
from pathlib import Path
from typing import Optional

from config.config import config


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """自定义JSON日志格式化器"""
    
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['timestamp'] = self.formatTime(record, self.datefmt)
        if record.exc_info:
            log_record['exc_info'] = self.formatException(record.exc_info)


def setup_logger(
    name: str,
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    设置并返回一个配置好的日志记录器
    
    Args:
        name: 日志记录器名称
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径，如果为None则只输出到控制台
        max_bytes: 单个日志文件最大大小（字节）
        backup_count: 保留的备份文件数量
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 获取日志级别
    level = getattr(logging, log_level or config.get('logging', 'level', 'INFO').upper(), logging.INFO)
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        
        # 使用JSON格式记录到文件
        json_formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)
    
    return logger


# 全局日志记录器
log = setup_logger(
    'trading_system',
    log_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'trading.log')
)
