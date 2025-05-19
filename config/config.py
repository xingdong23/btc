"""
配置管理模块

负责加载和管理系统配置，包括交易所API密钥、交易参数、策略参数等。
"""
import os
import configparser
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """配置管理类"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, config_file: str = None):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_file: str = None):
        if not self._initialized:
            self._config = configparser.ConfigParser()
            self._config_file = config_file or os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'config.ini'
            )
            self._load_config()
            self._initialized = True
    
    def _load_config(self) -> None:
        """加载配置文件"""
        if not os.path.exists(self._config_file):
            raise FileNotFoundError(
                f"配置文件 {self._config_file} 不存在。请复制 config.example.ini 为 config.ini 并修改配置。"
            )
        self._config.read(self._config_file, encoding='utf-8')
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """获取配置项"""
        try:
            return self._config.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default
    
    def get_section(self, section: str) -> Dict[str, str]:
        """获取整个配置节"""
        try:
            return dict(self._config.items(section))
        except configparser.NoSectionError:
            return {}
    
    def get_float(self, section: str, key: str, default: float = 0.0) -> float:
        """获取浮点型配置项"""
        try:
            return self._config.getfloat(section, key)
        except (ValueError, configparser.NoSectionError, configparser.NoOptionError):
            return default
    
    def get_int(self, section: str, key: str, default: int = 0) -> int:
        """获取整型配置项"""
        try:
            return self._config.getint(section, key)
        except (ValueError, configparser.NoSectionError, configparser.NoOptionError):
            return default
    
    def get_boolean(self, section: str, key: str, default: bool = False) -> bool:
        """获取布尔型配置项"""
        try:
            return self._config.getboolean(section, key)
        except (ValueError, configparser.NoSectionError, configparser.NoOptionError):
            return default
    
    def set(self, section: str, key: str, value: Any) -> None:
        """设置配置项"""
        if not self._config.has_section(section):
            self._config.add_section(section)
        self._config.set(section, key, str(value))
    
    def save(self) -> None:
        """保存配置到文件"""
        with open(self._config_file, 'w', encoding='utf-8') as f:
            self._config.write(f)
    
    @property
    def gateio_api_key(self) -> str:
        """获取Gate.io API Key"""
        return self.get('gateio', 'api_key', '')
    
    @property
    def gateio_secret(self) -> str:
        """获取Gate.io Secret Key"""
        return self.get('gateio', 'secret', '')
    
    @property
    def trading_symbols(self) -> list:
        """获取交易对列表"""
        symbols = self.get('trading', 'symbols', '')
        return [s.strip() for s in symbols.split(',') if s.strip()]
    
    @property
    def risk_params(self) -> dict:
        """获取风险管理参数"""
        return {
            'max_position_size': self.get_float('risk_management', 'max_position_size', 0.1),
            'stop_loss': self.get_float('risk_management', 'stop_loss', 0.05),
            'take_profit': self.get_float('risk_management', 'take_profit', 0.1),
            'max_drawdown': self.get_float('risk_management', 'max_drawdown', 0.2)
        }
    
    @property
    def backtest_params(self) -> dict:
        """获取回测参数"""
        return {
            'initial_balance': self.get_float('backtest', 'initial_balance', 10000.0),
            'fee': self.get_float('backtest', 'fee', 0.001),
            'slippage': self.get_float('backtest', 'slippage', 0.0005)
        }
    
    @property
    def data_params(self) -> dict:
        """获取数据参数"""
        return {
            'timeframe': self.get('data', 'timeframe', '1h'),
            'start_date': self.get('data', 'start_date', '20230101'),
            'data_source': self.get('data', 'data_source', 'gateio')
        }


# 全局配置实例
config = Config()
