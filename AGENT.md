# BTCTrader Coding Guide for AI Agents

## Commands
- Run application: `python main.py --mode [backtest|simulate|live] --strategy [strategy_name] --symbol [COIN/USDT]`
- Run tests: `python test_trading_system.py`
- Run single test: `python -m unittest discover -p "test_*.py" -k [test_name]`
- Test config: `python test_config.py`
- Run UI: `bash run_ui.sh`

## Code Style
- Use type hints with `from typing import Dict, Any, Optional, List, Type`
- Follow PEP 8 standards for formatting
- Docstrings use triple quotes with Args/Returns sections
- Error handling with try/except blocks and logging to `log`
- Import organization: standard library first, then third-party, then local imports
- Log errors with `log.error(f"Error message: {str(e)}", exc_info=True)`
- Class naming: CamelCase, functions/variables: snake_case
- Constants in UPPER_CASE

## Project Structure
- Trading strategies go in strategies/ directory
- Config management in config/
- Exchange API calls through exchange/