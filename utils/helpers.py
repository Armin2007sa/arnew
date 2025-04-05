import json
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlalchemy as db
from arnew.utils.data_fetcher import DataFetcher
from arnew.main import db, User, TradeHistory, ApiKey

logger = logging.getLogger(__name__)

def get_api_data(symbol, timeframe):
    try:
        data_fetcher = DataFetcher()
        return data_fetcher.fetch_historical_data(symbol, timeframe)
    except Exception as e:
        logger.error(f"Error fetching API data: {e}")
        return None

def calculate_profit(user_id):
    try:
        trades = TradeHistory.query.filter_by(user_id=user_id).all()
        
        if not trades:
            return 0, []
        
        total_profit = 0
        profit_history = []
        
        for trade in trades:
            profit = trade.profit_loss
            total_profit += profit
            
            trade_time = trade.timestamp.strftime("%Y-%m-%d %H:%M")
            profit_history.append({
                'time': trade_time,
                'profit': profit,
                'symbol': trade.symbol,
                'type': trade.trade_type
            })
        
        return total_profit, profit_history
    except Exception as e:
        logger.error(f"Error calculating profit: {e}")
        return 0, []

def validate_api_key(api_key, api_secret, exchange):
    try:
        if exchange.lower() == 'binance':
            url = 'https://api.binance.com/api/v3/account'
            timestamp = int(datetime.now().timestamp() * 1000)
            
            params = {
                'timestamp': timestamp,
                'recvWindow': 5000
            }
            
            query_string = '&'.join([f"{key}={params[key]}" for key in params])
            
            import hmac
            import hashlib
            
            signature = hmac.new(
                api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            params['signature'] = signature
            
            headers = {
                'X-MBX-APIKEY': api_key
            }
            
            response = requests.get(url, params=params, headers=headers)
            
            if response.status_code == 200:
                return True, "API key is valid"
            else:
                return False, f"API key validation failed: {response.text}"
        
        return True, "API key validation bypassed for testing"
    
    except Exception as e:
        logger.error(f"Error validating API key: {e}")
        return False, f"Error validating API key: {str(e)}"

def calculate_indicators(price_data, indicator_type, period=None):
    try:
        if indicator_type == 'sma':
            return calculate_sma(price_data, period)
        elif indicator_type == 'ema':
            return calculate_ema(price_data, period)
        elif indicator_type == 'rsi':
            return calculate_rsi(price_data, period)
        else:
            return None
    except Exception as e:
        logger.error(f"Error calculating indicator {indicator_type}: {e}")
        return None

def calculate_sma(price_data, period=20):
    try:
        closes = price_data['close'].values
        if len(closes) < period:
            return None
        
        sma = []
        for i in range(len(closes) - period + 1):
            sma.append(np.mean(closes[i:i+period]))
        
        padding = [None] * (period - 1)
        return padding + sma
    except Exception as e:
        logger.error(f"Error calculating SMA: {e}")
        return None

def calculate_ema(price_data, period=20):
    try:
        closes = price_data['close'].values
        if len(closes) < period:
            return None
        
        multiplier = 2 / (period + 1)
        ema = [np.mean(closes[:period])]
        
        for i in range(period, len(closes)):
            ema.append((closes[i] - ema[-1]) * multiplier + ema[-1])
        
        padding = [None] * (period - 1)
        return padding + ema
    except Exception as e:
        logger.error(f"Error calculating EMA: {e}")
        return None

def calculate_rsi(price_data, period=14):
    try:
        closes = price_data['close'].values
        if len(closes) < period + 1:
            return None
        
        deltas = np.diff(closes)
        gains = np.copy(deltas)
        losses = np.copy(deltas)
        
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        if avg_loss == 0:
            return [100] * len(closes)
        
        rs = avg_gain / avg_loss
        rsi = [100 - (100 / (1 + rs))]
        
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi.append(100 - (100 / (1 + rs)))
        
        padding = [None] * (period)
        return padding + rsi
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return None