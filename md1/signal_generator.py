import pandas as pd
import numpy as np
import logging
from datetime import datetime

class SignalGenerator:
    """
    Class for generating trading signals
    """
    
    def __init__(self, strategy):
        """
        Initialize the SignalGenerator
        
        Args:
            strategy (Strategy): Strategy instance
        """
        self.strategy = strategy
        logging.debug("SignalGenerator initialized")
    
    def generate_signal(self, historical_data, symbol):
        """
        Generate trading signal for the given historical data
        
        Args:
            historical_data (pd.DataFrame): Historical OHLCV data
            symbol (str): Trading pair symbol
            
        Returns:
            dict: Trading signal with entry, stop loss, take profit, and leverage
        """
        try:
            # Analyze market data
            analyzed_data = self.strategy.analyze_market(historical_data)
            
            # Get the most recent data
            recent_data = analyzed_data.iloc[-20:]
            
            # Find potential entry points in recent data
            entry_points = self.strategy.find_entry_points(recent_data)
            
            # Filter for most recent entries (last 3 candles)
            recent_entries = [e for e in entry_points if e['index'] >= len(recent_data) - 3]
            
            # Sort by confidence level
            recent_entries.sort(key=lambda x: x['confidence'], reverse=True)
            
            # If no recent entries, return no signal
            if not recent_entries:
                return {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'symbol': symbol,
                    'signal': 'NO_SIGNAL',
                    'message': 'No valid trading signals detected in recent data'
                }
            
            # Get the highest confidence signal
            best_entry = recent_entries[0]
            
            # Calculate risk parameters
            risk_params = self.strategy.calculate_risk_parameters(best_entry, recent_data)
            
            # Format the reasons as a string
            reasons_str = ', '.join(best_entry['reasons'])
            
            # Generate signal
            signal = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': symbol,
                'signal': best_entry['direction'].upper(),
                'entry': round(risk_params['entry_price'], 8),
                'stop_loss': round(risk_params['stop_loss'], 8),
                'take_profit': round(risk_params['take_profit'], 8),
                'leverage': risk_params['leverage'],
                'risk_percent': risk_params['risk_percent'],
                'confidence': round(risk_params['confidence'] * 100, 2),
                'reasons': reasons_str
            }
            
            logging.debug(f"Generated {best_entry['direction']} signal for {symbol} with {risk_params['confidence'] * 100:.2f}% confidence")
            return signal
            
        except Exception as e:
            logging.error(f"Error generating signal: {str(e)}")
            return {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': symbol,
                'signal': 'ERROR',
                'message': f"Error generating signal: {str(e)}"
            }
