import os
import sys
import json
import configparser
from config.config import config

def test_config():
    """Test configuration loading"""
    print("=" * 50)
    print("Testing configuration...")
    print("=" * 50)
    
    # Check if config file exists
    if not os.path.exists('config.ini'):
        print("Error: config.ini not found")
        return False
    
    # Load config file directly to see all sections
    config_parser = configparser.ConfigParser()
    config_parser.read('config.ini', encoding='utf-8')
    
    # Print all sections
    print("\nConfiguration Sections:")
    for section in config_parser.sections():
        print(f"\n[{section}]")
        for key, value in config_parser.items(section):
            # Hide sensitive information
            if 'key' in key.lower() or 'secret' in key.lower() or 'password' in key.lower():
                value = '***HIDDEN***' if value.strip() else 'Not set'
            print(f"{key} = {value}")
    
    # Test API keys
    print("\nAPI Configuration:")
    print(f"API Key: {'Set' if config.gateio_api_key and config.gateio_api_key != 'your_api_key_here' else 'Not set or using default'}")
    print(f"Secret Key: {'Set' if config.gateio_secret and config.gateio_secret != 'your_secret_key_here' else 'Not set or using default'}")
    
    # Test trading symbols
    print("\nTrading Symbols:")
    symbols = config.trading_symbols
    print(f"Trading Pairs: {symbols if symbols else 'No trading pairs configured'}")
    
    # Test risk management
    print("\nRisk Management:")
    try:
        risk_config = config.get_section('risk_management')
        if risk_config:
            for key, value in risk_config.items():
                print(f"{key}: {value}")
        else:
            print("No risk management configuration found")
    except Exception as e:
        print(f"Error reading risk management config: {e}")
    
    # Test backtest settings
    print("\nBacktest Settings:")
    try:
        backtest_config = config.get_section('backtest')
        if backtest_config:
            for key, value in backtest_config.items():
                print(f"{key}: {value}")
        else:
            print("No backtest configuration found")
    except Exception as e:
        print(f"Error reading backtest config: {e}")
    
    # Test data settings
    print("\nData Settings:")
    try:
        data_config = config.get_section('data')
        if data_config:
            for key, value in data_config.items():
                print(f"{key}: {value}")
        else:
            print("No data configuration found")
    except Exception as e:
        print(f"Error reading data config: {e}")
    
    print("\n" + "=" * 50)
    print("Configuration test completed")
    print("=" * 50)
    
    return True

def test_exchange_connection():
    """Test connection to Gate.io exchange"""
    print("\n" + "=" * 50)
    print("Testing Gate.io connection...")
    print("=" * 50)
    
    from exchange.gateio import GateIOExchange
    
    try:
        exchange = GateIOExchange()
        
        # Test public API (no authentication required)
        print("\nFetching ticker for BTC/USDT...")
        ticker = exchange.get_ticker('BTC/USDT')
        print(f"Current BTC/USDT price: {ticker.get('last') if ticker else 'N/A'}")
        
        # Test authenticated API (if credentials are set)
        if config.gateio_api_key and config.gateio_secret and \
           config.gateio_api_key != 'your_api_key_here' and \
           config.gateio_secret != 'your_secret_key_here':
            print("\nTesting authenticated API...")
            try:
                balance = exchange.get_balance()
                if balance:
                    print("Account balances:")
                    for currency, amount in balance.items():
                        if float(amount['free']) > 0 or float(amount['used']) > 0:
                            print(f"{currency}: Free={amount['free']}, Used={amount['used']}")
                else:
                    print("No balance information available")
            except Exception as e:
                print(f"Error fetching balance: {e}")
                print("This might be due to invalid API keys or insufficient permissions")
        else:
            print("\nSkipping authenticated API test - please set valid API keys in config.ini")
            
    except Exception as e:
        print(f"Error connecting to Gate.io: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("Connection test completed")
    print("=" * 50)
    return True

if __name__ == "__main__":
    test_config()
    test_exchange_connection()
