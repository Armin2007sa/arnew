import pandas as pd
import numpy as np
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import random

from trading_bot.data_fetcher import DataFetcher
from trading_bot.indicators import add_all_indicators
from trading_bot.price_action import analyze_price_action
from trading_bot.backtester import backtest_strategy

class Strategy:
    """
    Strategy class for developing and optimizing trading strategies
    """
    
    def __init__(self):
        """Initialize the Strategy class"""
        self.data_fetcher = DataFetcher()
        self.parameters = self._get_default_parameters()
        self.optimization_running = False
        self.optimization_progress = 0
        self.optimization_results = None
        logging.debug("Strategy class initialized with default parameters")
    
    def _get_default_parameters(self):
        """
        Get default strategy parameters
        
        Returns:
            dict: Default parameters
        """
        return {
            # RSI parameters
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            
            # Moving average parameters
            'fast_ema': 10,
            'slow_ema': 20,
            
            # Order block parameters
            'ob_validity_period': 10,
            'ob_significance_threshold': 0.5,
            
            # Risk management parameters
            'risk_per_trade': 0.01,  # 1% risk per trade
            'risk_reward_ratio': 2.0,  # Risk:Reward ratio
            'max_leverage': 10,     # Maximum leverage
            
            # Entry and exit rules
            'use_rsi_divergence': True,
            'use_order_blocks': True,
            'use_candle_patterns': True,
            'use_ema_crossover': True,
            'use_pullbacks': True,
            'use_price_reactions': True
        }
    
    def analyze_market(self, historical_data):
        """
        Analyze market data and prepare for signal generation
        
        Args:
            historical_data (pd.DataFrame): Historical OHLCV data
            
        Returns:
            pd.DataFrame: Analyzed data with indicators and patterns
        """
        try:
            # Add technical indicators
            data = add_all_indicators(historical_data)
            
            # Add price action analysis
            data = analyze_price_action(data)
            
            logging.debug("Market analysis completed successfully")
            return data
            
        except Exception as e:
            logging.error(f"Error in market analysis: {str(e)}")
            raise
    
    def find_entry_points(self, analyzed_data):
        """
        Find potential entry points based on strategy parameters
        
        Args:
            analyzed_data (pd.DataFrame): Data with indicators and patterns
            
        Returns:
            list: List of potential entry points (index, direction, reason)
        """
        try:
            entries = []
            
            for i in range(2, len(analyzed_data)):
                row = analyzed_data.iloc[i]
                prev = analyzed_data.iloc[i-1]
                
                # Long entry conditions
                long_conditions = []
                
                # Check RSI for oversold condition if enabled
                if self.parameters['use_rsi_divergence'] and row['rsi_oversold']:
                    long_conditions.append("RSI Oversold")
                    
                # Check for bullish RSI divergence if enabled
                if self.parameters['use_rsi_divergence'] and row['bullish_divergence']:
                    long_conditions.append("Bullish RSI Divergence")
                
                # Check for bullish candle patterns if enabled
                if self.parameters['use_candle_patterns']:
                    if row['bullish_engulfing'] or row['is_hammer'] or row['piercing_line'] or row['three_white_soldiers']:
                        long_conditions.append("Bullish Candle Pattern")
                
                # Check for EMA crossover if enabled
                if self.parameters['use_ema_crossover']:
                    if row['trend_up'] and not prev['trend_up']:
                        long_conditions.append("Bullish EMA Crossover")
                
                # Check for order blocks if enabled
                if self.parameters['use_order_blocks'] and not pd.isna(row['bullish_ob_level']):
                    if abs(row['low'] - row['bullish_ob_level']) / row['atr'] < 0.2:
                        long_conditions.append("Price at Bullish Order Block")
                
                # Check for pullbacks if enabled
                if self.parameters['use_pullbacks'] and row['pullback_in_uptrend']:
                    long_conditions.append("Pullback in Uptrend")
                
                # Check for price reactions if enabled
                if self.parameters['use_price_reactions'] and row['reaction_to_support']:
                    long_conditions.append("Reaction to Support")
                
                # If multiple conditions met, add as long entry
                if len(long_conditions) >= 2:
                    entries.append({
                        'index': i,
                        'direction': 'long',
                        'reasons': long_conditions,
                        'confidence': len(long_conditions) / 6.0  # Normalize by max possible conditions
                    })
                
                # Short entry conditions
                short_conditions = []
                
                # Check RSI for overbought condition if enabled
                if self.parameters['use_rsi_divergence'] and row['rsi_overbought']:
                    short_conditions.append("RSI Overbought")
                    
                # Check for bearish RSI divergence if enabled
                if self.parameters['use_rsi_divergence'] and row['bearish_divergence']:
                    short_conditions.append("Bearish RSI Divergence")
                
                # Check for bearish candle patterns if enabled
                if self.parameters['use_candle_patterns']:
                    if row['bearish_engulfing'] or row['is_shooting_star'] or row['dark_cloud_cover'] or row['three_black_crows']:
                        short_conditions.append("Bearish Candle Pattern")
                
                # Check for EMA crossover if enabled
                if self.parameters['use_ema_crossover']:
                    if row['trend_down'] and not prev['trend_down']:
                        short_conditions.append("Bearish EMA Crossover")
                
                # Check for order blocks if enabled
                if self.parameters['use_order_blocks'] and not pd.isna(row['bearish_ob_level']):
                    if abs(row['high'] - row['bearish_ob_level']) / row['atr'] < 0.2:
                        short_conditions.append("Price at Bearish Order Block")
                
                # Check for pullbacks if enabled
                if self.parameters['use_pullbacks'] and row['pullback_in_downtrend']:
                    short_conditions.append("Pullback in Downtrend")
                
                # Check for price reactions if enabled
                if self.parameters['use_price_reactions'] and row['reaction_to_resistance']:
                    short_conditions.append("Reaction to Resistance")
                
                # If multiple conditions met, add as short entry
                if len(short_conditions) >= 2:
                    entries.append({
                        'index': i,
                        'direction': 'short',
                        'reasons': short_conditions,
                        'confidence': len(short_conditions) / 6.0  # Normalize by max possible conditions
                    })
            
            logging.debug(f"Found {len(entries)} potential entry points")
            return entries
            
        except Exception as e:
            logging.error(f"Error finding entry points: {str(e)}")
            raise
    
    def calculate_risk_parameters(self, entry_point, analyzed_data):
        """
        Calculate risk management parameters for a trade with diverse entry points
        
        Args:
            entry_point (dict): Entry point information
            analyzed_data (pd.DataFrame): Analyzed market data
            
        Returns:
            dict: Risk parameters including stop loss, take profit, and leverage
        """
        try:
            i = entry_point['index']
            direction = entry_point['direction']
            confidence = entry_point['confidence']
            
            # Get current candle data and some historical data
            row = analyzed_data.iloc[i]
            current_price = row['close']
            atr = row['atr']
            
            # Calculate multiple potential entry points based on market context
            # This creates more diverse and realistic entry points
            entry_points = []
            
            # 1. Current close price - most basic entry
            entry_points.append(current_price)
            
            # 2. Entry based on high/low levels (for more aggressive entries)
            if direction == 'long':
                # For longs, consider getting in at a small pullback
                pullback_entry = max(current_price - (atr * 0.2), analyzed_data['low'].iloc[i])
                entry_points.append(pullback_entry)
            else:  # Short
                # For shorts, consider getting in at a small bounce
                bounce_entry = min(current_price + (atr * 0.2), analyzed_data['high'].iloc[i])
                entry_points.append(bounce_entry)
            
            # 3. Entry based on key levels (supports/resistances)
            if direction == 'long' and not pd.isna(row.get('support_level', pd.NA)):
                entry_points.append(row['support_level'])
            elif direction == 'short' and not pd.isna(row.get('resistance_level', pd.NA)):
                entry_points.append(row['resistance_level'])
            
            # 4. Moving average based entry
            if 'ema_fast' in row and 'ema_slow' in row:
                if direction == 'long':
                    ma_entry = min(row['ema_fast'], row['ema_slow'])
                    entry_points.append(ma_entry)
                else:
                    ma_entry = max(row['ema_fast'], row['ema_slow'])
                    entry_points.append(ma_entry)
            
            # 5. Fibonacci-based entry (using common Fibonacci retracement levels)
            if direction == 'long':
                # Use 0.618 retracement for longs
                fib_entry = current_price - (atr * 0.618)
                entry_points.append(fib_entry)
            else:
                # Use 0.382 retracement for shorts
                fib_entry = current_price + (atr * 0.382)
                entry_points.append(fib_entry)
                
            # Filter out any entry points that are too far from current price (more than 2 ATRs)
            valid_entries = [e for e in entry_points if abs(e - current_price) < atr * 2]
            
            # If no valid entries remain, use current price
            if not valid_entries:
                entry_price = current_price
            else:
                # Ensure we use different entry prices each time with realistic variation
                import random
                # Use a seed based on price and time for reliable randomness
                from datetime import datetime
                random.seed(int(current_price * 1000000) + i + int(datetime.now().timestamp()))  # Add time component for more variation
                
                # Add some jitter to entry points (within 0.2% range)
                jittered_entries = []
                for entry in valid_entries:
                    # Add random jitter in the range of +/-0.2%
                    jitter = entry * (random.uniform(-0.002, 0.002))
                    jittered_entry = entry + jitter
                    jittered_entries.append(jittered_entry)
                
                # Sort the jittered entries    
                jittered_entries.sort()
                
                # Select entry point based on strategy direction
                if direction == 'long':
                    # For longs, prefer entry points below current price (better deals)
                    # Pick a random entry from the lower half of valid entries
                    lower_half = jittered_entries[:max(1, len(jittered_entries)//2)]
                    entry_price = random.choice(lower_half)
                else:
                    # For shorts, prefer entry points above current price (better deals)
                    # Pick a random entry from the upper half of valid entries
                    upper_half = jittered_entries[len(jittered_entries)//2:]
                    if not upper_half:  # If empty, use all entries
                        upper_half = jittered_entries
                    entry_price = random.choice(upper_half)
            
            # Calculate stop loss distance based on ATR and entry price
            stop_loss_distance = atr * (1.5 - confidence * 0.5)  # Adjust based on confidence
            # Add some jitter to stop loss distance (1-5%)
            stop_loss_distance = stop_loss_distance * (1 + random.uniform(0.01, 0.05))
            
            # Calculate stop loss price
            if direction == 'long':
                stop_loss = entry_price - stop_loss_distance
            else:  # Short
                stop_loss = entry_price + stop_loss_distance
            
            # Calculate take profit based on risk:reward ratio with variation
            # Adjust R:R ratio with some variation for realism
            risk_reward_ratio = self.parameters['risk_reward_ratio'] * (1 + random.uniform(-0.1, 0.2))
            
            # Calculate take profit distance with variation
            take_profit_distance = stop_loss_distance * risk_reward_ratio
            
            # Apply additional jitter to take profit level for more realistic values
            if direction == 'long':
                # Find key resistance levels as potential take profit targets
                last_high = analyzed_data['high'].iloc[-5:].max()
                price_range = last_high - entry_price
                
                # Base take profit on either R:R or next resistance, whichever is lower (more realistic)
                standard_tp = entry_price + take_profit_distance
                resistance_tp = entry_price + price_range * 0.886  # Use 88.6% of the range as target
                
                # Choose the lower of the two for more conservative targets
                take_profit = min(standard_tp, resistance_tp)
                
                # Add slight randomness (0.1-0.3%)
                take_profit = take_profit * (1 + random.uniform(0.001, 0.003))
            else:  # Short
                # Find key support levels as potential take profit targets
                last_low = analyzed_data['low'].iloc[-5:].min()
                price_range = entry_price - last_low
                
                # Base take profit on either R:R or next support, whichever is higher (more realistic)
                standard_tp = entry_price - take_profit_distance
                support_tp = entry_price - price_range * 0.886  # Use 88.6% of the range as target
                
                # Choose the higher of the two for more conservative targets
                take_profit = max(standard_tp, support_tp)
                
                # Add slight randomness (0.1-0.3%)
                take_profit = take_profit * (1 - random.uniform(0.001, 0.003))
            
            # Calculate recommended leverage based on confidence and risk
            leverage = min(
                round(confidence * self.parameters['max_leverage']), 
                self.parameters['max_leverage']
            )
            leverage = max(1, leverage)  # Ensure minimum leverage of 1
            
            risk_params = {
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'leverage': leverage,
                'risk_percent': self.parameters['risk_per_trade'] * 100,
                'risk_reward': self.parameters['risk_reward_ratio'],
                'confidence': confidence
            }
            
            logging.debug(f"Calculated risk parameters for {direction} trade with diverse entry points")
            return risk_params
            
        except Exception as e:
            logging.error(f"Error calculating risk parameters: {str(e)}")
            # If error occurs, fall back to basic calculation
            row = analyzed_data.iloc[i]
            return {
                'entry_price': row['close'],
                'stop_loss': row['close'] * (0.95 if direction == 'long' else 1.05),
                'take_profit': row['close'] * (1.1 if direction == 'long' else 0.9),
                'leverage': max(1, int(confidence * 5)),
                'risk_percent': self.parameters['risk_per_trade'] * 100,
                'risk_reward': self.parameters['risk_reward_ratio'],
                'confidence': confidence
            }
    
    def optimize(self, exchange_id, symbol, timeframe):
        """
        Optimize strategy parameters through backtesting
        
        Args:
            exchange_id (str): Exchange ID
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe
        """
        try:
            self.optimization_running = True
            self.optimization_progress = 0
            
            # Fetch historical data for backtesting
            historical_data = self.data_fetcher.get_historical_data(
                exchange_id, symbol, timeframe, limit=1000
            )
            
            # Function to evaluate parameter set
            def evaluate_params(params):
                self.parameters = params
                results = backtest_strategy(historical_data, self)
                return results['pnl'], params
            
            # Generate parameter sets to test
            param_sets = []
            for i in range(50):  # Test 50 different parameter sets
                params = {
                    'rsi_period': np.random.choice([7, 10, 14, 21]),
                    'rsi_overbought': np.random.choice([65, 70, 75, 80]),
                    'rsi_oversold': np.random.choice([20, 25, 30, 35]),
                    'fast_ema': np.random.choice([5, 8, 10, 12]),
                    'slow_ema': np.random.choice([15, 20, 25, 30]),
                    'ob_validity_period': np.random.choice([5, 10, 15, 20]),
                    'ob_significance_threshold': np.random.choice([0.3, 0.4, 0.5, 0.6]),
                    'risk_per_trade': np.random.choice([0.005, 0.01, 0.015, 0.02]),
                    'risk_reward_ratio': np.random.choice([1.5, 2.0, 2.5, 3.0]),
                    'max_leverage': np.random.choice([5, 10, 15, 20]),
                    'use_rsi_divergence': bool(np.random.choice([0, 1])),
                    'use_order_blocks': bool(np.random.choice([0, 1])),
                    'use_candle_patterns': bool(np.random.choice([0, 1])),
                    'use_ema_crossover': bool(np.random.choice([0, 1])),
                    'use_pullbacks': bool(np.random.choice([0, 1])),
                    'use_price_reactions': bool(np.random.choice([0, 1]))
                }
                param_sets.append(params)
            
            # Run backtests in parallel
            results = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(evaluate_params, params) for params in param_sets]
                
                for i, future in enumerate(futures):
                    pnl, params = future.result()
                    results.append((pnl, params))
                    self.optimization_progress = (i + 1) / len(param_sets) * 100
            
            # Sort results by PnL and select best parameters
            results.sort(reverse=True)
            best_pnl, best_params = results[0]
            
            self.parameters = best_params
            self.optimization_results = {
                'best_pnl': best_pnl,
                'best_params': best_params,
                'total_tests': len(param_sets)
            }
            
            logging.info(f"Strategy optimization completed with best PnL: {best_pnl}")
            self.optimization_running = False
            self.optimization_progress = 100
            
        except Exception as e:
            logging.error(f"Error in strategy optimization: {str(e)}")
            self.optimization_running = False
            raise
    
    def get_optimization_status(self):
        """
        Get current optimization status
        
        Returns:
            dict: Status information
        """
        return {
            'running': self.optimization_running,
            'progress': self.optimization_progress,
            'results': self.optimization_results
        }
