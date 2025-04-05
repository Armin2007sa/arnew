import pandas as pd
import numpy as np
import logging

def identify_candle_patterns(dataframe):
    """
    Identify candlestick patterns in the price data
    
    Args:
        dataframe (pd.DataFrame): Price data with OHLC data
        
    Returns:
        pd.DataFrame: DataFrame with pattern indicators
    """
    try:
        df = dataframe.copy()
        
        # Calculate candle properties
        df['body_size'] = abs(df['close'] - df['open'])
        df['total_size'] = df['high'] - df['low']
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Calculate average body size (for relative comparisons)
        avg_body_size = df['body_size'].rolling(window=14).mean()
        df['rel_body_size'] = df['body_size'] / avg_body_size
        
        # Identify Doji (very small body)
        df['is_doji'] = df['body_size'] <= (0.1 * df['total_size'])
        
        # Identify Hammer (small body at top, long lower shadow)
        df['is_hammer'] = (
            (df['lower_shadow'] >= df['body_size'] * 2) & 
            (df['upper_shadow'] <= df['body_size'] * 0.3) &
            (df['body_size'] <= df['total_size'] * 0.3)
        )
        
        # Identify Shooting Star (small body at bottom, long upper shadow)
        df['is_shooting_star'] = (
            (df['upper_shadow'] >= df['body_size'] * 2) & 
            (df['lower_shadow'] <= df['body_size'] * 0.3) &
            (df['body_size'] <= df['total_size'] * 0.3)
        )
        
        # Identify Engulfing patterns
        df['bullish_engulfing'] = (
            (df['close'].shift(1) < df['open'].shift(1)) &  # Previous candle is bearish
            (df['close'] > df['open']) &                     # Current candle is bullish
            (df['open'] <= df['close'].shift(1)) &           # Open below previous close
            (df['close'] >= df['open'].shift(1))             # Close above previous open
        )
        
        df['bearish_engulfing'] = (
            (df['close'].shift(1) > df['open'].shift(1)) &  # Previous candle is bullish
            (df['close'] < df['open']) &                     # Current candle is bearish
            (df['open'] >= df['close'].shift(1)) &           # Open above previous close
            (df['close'] <= df['open'].shift(1))             # Close below previous open
        )
        
        # Identify Piercing Line pattern
        df['piercing_line'] = (
            (df['close'].shift(1) < df['open'].shift(1)) &  # Previous candle is bearish
            (df['close'] > df['open']) &                     # Current candle is bullish
            (df['open'] < df['low'].shift(1)) &              # Open below previous low
            (df['close'] > (df['open'].shift(1) + df['close'].shift(1)) / 2) &  # Close above 50% of previous candle
            (df['close'] < df['open'].shift(1))             # Close below previous open
        )
        
        # Identify Dark Cloud Cover pattern
        df['dark_cloud_cover'] = (
            (df['close'].shift(1) > df['open'].shift(1)) &  # Previous candle is bullish
            (df['close'] < df['open']) &                     # Current candle is bearish
            (df['open'] > df['high'].shift(1)) &             # Open above previous high
            (df['close'] < (df['open'].shift(1) + df['close'].shift(1)) / 2) &  # Close below 50% of previous candle
            (df['close'] > df['close'].shift(1))            # Close above previous close
        )
        
        # Identify Three White Soldiers
        df['three_white_soldiers'] = (
            (df['close'] > df['open']) &                     # Current candle is bullish
            (df['close'].shift(1) > df['open'].shift(1)) &  # Previous candle is bullish
            (df['close'].shift(2) > df['open'].shift(2)) &  # Candle before previous is bullish
            (df['open'] > df['open'].shift(1)) &             # Open higher than previous open
            (df['open'].shift(1) > df['open'].shift(2)) &    # Previous open higher than the one before
            (df['close'] > df['close'].shift(1)) &           # Close higher than previous close
            (df['close'].shift(1) > df['close'].shift(2))    # Previous close higher than the one before
        )
        
        # Identify Three Black Crows
        df['three_black_crows'] = (
            (df['close'] < df['open']) &                     # Current candle is bearish
            (df['close'].shift(1) < df['open'].shift(1)) &  # Previous candle is bearish
            (df['close'].shift(2) < df['open'].shift(2)) &  # Candle before previous is bearish
            (df['open'] < df['open'].shift(1)) &             # Open lower than previous open
            (df['open'].shift(1) < df['open'].shift(2)) &    # Previous open lower than the one before
            (df['close'] < df['close'].shift(1)) &           # Close lower than previous close
            (df['close'].shift(1) < df['close'].shift(2))    # Previous close lower than the one before
        )
        
        logging.debug("Candle patterns identified successfully")
        return df
        
    except Exception as e:
        logging.error(f"Error identifying candle patterns: {str(e)}")
        raise

def identify_order_blocks(dataframe):
    """
    Identify order blocks in the price data
    
    Args:
        dataframe (pd.DataFrame): Price data
        
    Returns:
        pd.DataFrame: DataFrame with order block indicators
    """
    try:
        df = dataframe.copy()
        
        # Calculate average true range for volatility reference
        df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
        
        # Identify bullish order blocks (potential support areas)
        df['price_drop'] = (df['close'].shift(3) - df['low'].shift(1)) / df['atr'].shift(1)
        df['is_bullish_order_block'] = (
            (df['price_drop'] > 0.5) &  # Significant drop
            (df['close'] > df['open']) &  # Bullish candle
            (df['close'].shift(1) < df['open'].shift(1))  # After bearish candle
        )
        
        # Identify bearish order blocks (potential resistance areas)
        df['price_rise'] = (df['high'].shift(1) - df['close'].shift(3)) / df['atr'].shift(1)
        df['is_bearish_order_block'] = (
            (df['price_rise'] > 0.5) &  # Significant rise
            (df['close'] < df['open']) &  # Bearish candle
            (df['close'].shift(1) > df['open'].shift(1))  # After bullish candle
        )
        
        # Track order block levels
        df['bullish_ob_level'] = np.nan
        df['bearish_ob_level'] = np.nan
        
        for i in range(1, len(df)):
            if df['is_bullish_order_block'].iloc[i]:
                level = min(df['low'].iloc[i], df['open'].iloc[i])
                # Mark this level for the next 10 candles
                for j in range(i, min(i+10, len(df))):
                    df.loc[j, 'bullish_ob_level'] = level
            
            if df['is_bearish_order_block'].iloc[i]:
                level = max(df['high'].iloc[i], df['open'].iloc[i])
                # Mark this level for the next 10 candles
                for j in range(i, min(i+10, len(df))):
                    df.loc[j, 'bearish_ob_level'] = level
        
        logging.debug("Order blocks identified successfully")
        return df
        
    except Exception as e:
        logging.error(f"Error identifying order blocks: {str(e)}")
        raise

def identify_pullbacks(dataframe):
    """
    Identify pullbacks in the price action
    
    Args:
        dataframe (pd.DataFrame): Price data
        
    Returns:
        pd.DataFrame: DataFrame with pullback indicators
    """
    try:
        df = dataframe.copy()
        
        # Calculate 3-period and 8-period EMAs for trend direction
        df['ema3'] = df['close'].ewm(span=3, adjust=False).mean()
        df['ema8'] = df['close'].ewm(span=8, adjust=False).mean()
        
        # Identify uptrend and downtrend
        df['uptrend'] = df['ema3'] > df['ema8']
        df['downtrend'] = df['ema3'] < df['ema8']
        
        # Identify pullbacks in uptrend
        df['pullback_in_uptrend'] = (
            df['uptrend'] &  # We're in an uptrend
            (df['close'] < df['close'].shift(1)) &  # Current close is lower than previous
            (df['close'].shift(1) < df['close'].shift(2)) &  # Previous close was also lower
            (df['close'] > df['ema8'])  # But still above the 8-period EMA
        )
        
        # Identify pullbacks in downtrend
        df['pullback_in_downtrend'] = (
            df['downtrend'] &  # We're in a downtrend
            (df['close'] > df['close'].shift(1)) &  # Current close is higher than previous
            (df['close'].shift(1) > df['close'].shift(2)) &  # Previous close was also higher
            (df['close'] < df['ema8'])  # But still below the 8-period EMA
        )
        
        logging.debug("Pullbacks identified successfully")
        return df
        
    except Exception as e:
        logging.error(f"Error identifying pullbacks: {str(e)}")
        raise

def identify_price_reactions(dataframe):
    """
    Identify price reactions to key levels
    
    Args:
        dataframe (pd.DataFrame): Price data
        
    Returns:
        pd.DataFrame: DataFrame with price reaction indicators
    """
    try:
        df = dataframe.copy()
        
        # Calculate pivot points
        df['pivot_high'] = df['high'].rolling(window=5, center=True).apply(
            lambda x: x[2] == max(x), raw=True
        )
        
        df['pivot_low'] = df['low'].rolling(window=5, center=True).apply(
            lambda x: x[2] == min(x), raw=True
        )
        
        # Extract pivot levels
        df['pivot_high_level'] = df['high'].where(df['pivot_high'])
        df['pivot_low_level'] = df['low'].where(df['pivot_low'])
        
        # Forward fill pivot levels
        df['last_pivot_high'] = df['pivot_high_level'].fillna(method='ffill')
        df['last_pivot_low'] = df['pivot_low_level'].fillna(method='ffill')
        
        # Replace NaN values with large/small numbers to avoid comparison issues
        df['last_pivot_high'] = df['last_pivot_high'].fillna(float('inf'))
        df['last_pivot_low'] = df['last_pivot_low'].fillna(0)
        
        # Identify price reactions
        df['reaction_to_resistance'] = (
            (df['high'] >= df['last_pivot_high'] * 0.995) &  # Price approached resistance
            (df['high'] <= df['last_pivot_high'] * 1.005) &  # But didn't break too far
            (df['close'] < df['open'])  # And closed lower
        )
        
        df['reaction_to_support'] = (
            (df['low'] <= df['last_pivot_low'] * 1.005) &  # Price approached support
            (df['low'] >= df['last_pivot_low'] * 0.995) &  # But didn't break too far
            (df['close'] > df['open'])  # And closed higher
        )
        
        logging.debug("Price reactions identified successfully")
        return df
        
    except Exception as e:
        logging.error(f"Error identifying price reactions: {str(e)}")
        # Instead of raising, return the dataframe without the reactions
        # Always create a new copy of the dataframe to avoid unbound df variable
        result_df = dataframe.copy()
        result_df['reaction_to_resistance'] = False
        result_df['reaction_to_support'] = False
        return result_df

def analyze_price_action(dataframe):
    """
    Complete price action analysis
    
    Args:
        dataframe (pd.DataFrame): OHLCV price data
        
    Returns:
        pd.DataFrame: DataFrame with all price action analysis
    """
    try:
        df = dataframe.copy()
        
        # Apply all price action analysis functions
        df = identify_candle_patterns(df)
        df = identify_order_blocks(df)
        df = identify_pullbacks(df)
        df = identify_price_reactions(df)
        
        # Clean up NaN values from calculations
        # Updated to use bfill() instead of method='bfill' to avoid deprecation warning
        df = df.bfill()
        
        logging.debug("Complete price action analysis completed")
        return df
        
    except Exception as e:
        logging.error(f"Error in price action analysis: {str(e)}")
        # Return original dataframe if analysis fails
        return dataframe.copy()
