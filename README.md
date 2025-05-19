# Gate.io 量化交易系统

这是一个基于Python的Gate.io量化交易系统框架，支持回测、模拟交易和实盘交易。

## 功能特点

- 多策略支持（均线交叉、RSI、布林带等）
- 完整的回测框架
- 模拟交易和实盘交易支持
- 风险管理功能
- 实时行情监控
- 详细的日志记录

## 快速开始

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 复制示例配置文件：
   ```bash
   cp config.example.ini config.ini
   ```

3. 编辑`config.ini`文件，填入您的Gate.io API密钥

4. 运行回测：
   ```bash
   python main.py --mode backtest --strategy ma_crossover --pair BTC/USDT --start_date 20240101 --end_date 20240519
   ```

5. 运行模拟交易：
   ```bash
   python main.py --mode simulate --strategy rsi --pair ETH/USDT
   ```

6. 运行实盘交易（谨慎使用）：
   ```bash
   python main.py --mode live --strategy bollinger_bands --pair XRP/USDT
   ```

## 项目结构

```
.
├── config/                  # 配置文件目录
│   ├── __init__.py
│   └── config.py           # 配置管理
├── data/                   # 数据存储
│   └── __init__.py
├── exchange/               # 交易所接口
│   ├── __init__.py
│   └── gateio.py           # Gate.io接口实现
├── strategies/             # 交易策略
│   ├── __init__.py
│   ├── base.py             # 策略基类
│   ├── ma_crossover.py     # 均线交叉策略
│   ├── rsi.py              # RSI策略
│   └── bollinger_bands.py  # 布林带策略
├── backtest/               # 回测引擎
│   ├── __init__.py
│   └── engine.py
├── risk_management/        # 风险管理
│   ├── __init__.py
│   └── manager.py
├── utils/                  # 工具函数
│   ├── __init__.py
│   ├── logger.py           # 日志工具
│   └── helpers.py          # 辅助函数
├── main.py                 # 主程序入口
├── requirements.txt        # 项目依赖
└── README.md              # 项目说明
```

## 注意事项

- 实盘交易前请确保充分测试
- 请妥善保管API密钥
- 建议先使用模拟交易测试策略表现
- 系统默认使用UTC时间，请注意时区转换

## 许可证

MIT License
# btc
