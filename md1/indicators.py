import pandas as pd
import numpy as np
import pandas_ta as ta
import logging

def calculate_rsi(dataframe, period=14, column='close'):
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        dataframe (pd.DataFrame): Price data
        period (int): RSI period
        column (str): Column to use for calculation
        
    Returns:
        pd.DataFrame: DataFrame with RSI values
    """
    try:
        df = dataframe.copy()
        df['rsi'] = ta.rsi(df[column], length=period)
        
        # Add RSI conditions
        df['rsi_overbought'] = df['rsi'] > 70
        df['rsi_oversold'] = df['rsi'] < 30
        
        # Detect RSI divergences
        df['price_higher_high'] = (
            (df['close'] > df['close'].shift(1)) & 
            (df['close'].shift(1) > df['close'].shift(2))
        )
        df['rsi_lower_high'] = (
            (df['rsi'] < df['rsi'].shift(1)) & 
            (df['rsi'].shift(1) > df['rsi'].shift(2))
        )
        
        df['price_lower_low'] = (
            (df['close'] < df['close'].shift(1)) & 
            (df['close'].shift(1) < df['close'].shift(2))
        )
        df['rsi_higher_low'] = (
            (df['rsi'] > df['rsi'].shift(1)) & 
            (df['rsi'].shift(1) < df['rsi'].shift(2))
        )
        
        df['bearish_divergence'] = df['price_higher_high'] & df['rsi_lower_high']
        df['bullish_divergence'] = df['price_lower_low'] & df['rsi_higher_low']
        
        logging.debug("RSI calculation completed")
        return df
        
    except Exception as e:
        logging.error(f"Error calculating RSI: {str(e)}")
        raise

def add_all_indicators(dataframe):
    """
    Add all technical indicators needed for the strategy
    
    Args:
        dataframe (pd.DataFrame): Price data
        
    Returns:
        pd.DataFrame: DataFrame with indicators
    """
    try:
        df = dataframe.copy()
        
        # Add RSI
        df = calculate_rsi(df)
        
        # Add moving averages
        df['ema_10'] = ta.ema(df['close'], length=10)
        df['ema_20'] = ta.ema(df['close'], length=20)
        df['ema_50'] = ta.ema(df['close'], length=50)
        
        # Add trend direction
        df['trend_up'] = df['ema_10'] > df['ema_20']
        df['trend_down'] = df['ema_10'] < df['ema_20']
        
        # Add Average True Range for volatility
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # Add Bollinger Bands
        bb = ta.bbands(df['close'], length=20, std=2.0)
        df = pd.concat([df, bb], axis=1)
        
        # Clean up NaN values (using bfill() instead of method='bfill' to avoid deprecation warning)
        df = df.bfill()
        
        logging.debug("All indicators added to dataframe")
        return df
        
    except Exception as e:
        logging.error(f"Error adding indicators: {str(e)}")
        raise
