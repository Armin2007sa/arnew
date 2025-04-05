"""
Multi-Analysis System

This module implements a machine learning-based system for combining multiple analysis approaches
and generating optimal trading signals.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from trading_bot.strategy import Strategy

class MultiAnalysisSystem:
    """
    A system that combines multiple analysis approaches using machine learning
    to generate optimal trading signals.
    """
    
    def __init__(self):
        """Initialize the MultiAnalysisSystem"""
        self.strategies = {
            'aeai': Strategy(),
            'modern': Strategy(),
            'indicator': Strategy(),
            'supply_demand': Strategy(),
            'elliott': Strategy(), 
            'harmonic': Strategy(),
            'time': Strategy()
        }
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        logging.debug("MultiAnalysisSystem initialized with strategies: %s", list(self.strategies.keys()))
        
    def analyze_with_all_strategies(self, historical_data, symbol):
        """
        Analyze market data with all available strategies
        
        Args:
            historical_data (pd.DataFrame): Historical OHLCV data
            symbol (str): Trading pair symbol
            
        Returns:
            dict: Dictionary of signals from each strategy
        """
        signals = {}
        features = []
        
        for strategy_name, strategy in self.strategies.items():
            try:
                analyzed_data = strategy.analyze_market(historical_data)
                entry_points = strategy.find_entry_points(analyzed_data)
                
                if entry_points:
                    entry_point = entry_points[0]  # Take the first entry point
                    risk_params = strategy.calculate_risk_parameters(entry_point, analyzed_data)
                    
                    signal = {
                        'signal': entry_point['direction'],
                        'entry': float(risk_params['entry']),
                        'stop_loss': float(risk_params['stop_loss']),
                        'take_profit': float(risk_params['take_profit']),
                        'leverage': float(risk_params['leverage']),
                        'confidence': float(risk_params.get('confidence', 50.0)),
                        'strategy': strategy_name
                    }
                    
                    # Extract features for machine learning
                    features.append({
                        'strategy': strategy_name,
                        'signal': 1 if signal['signal'] == 'LONG' else (-1 if signal['signal'] == 'SHORT' else 0),
                        'risk_reward_ratio': (signal['take_profit'] - signal['entry']) / (signal['entry'] - signal['stop_loss']) if signal['signal'] == 'LONG' else (signal['entry'] - signal['take_profit']) / (signal['stop_loss'] - signal['entry']),
                        'confidence': signal['confidence'],
                        'leverage': signal['leverage']
                    })
                    
                    signals[strategy_name] = signal
                else:
                    signals[strategy_name] = {'signal': 'NO_SIGNAL', 'strategy': strategy_name}
                    
            except Exception as e:
                logging.error(f"Error analyzing with {strategy_name} strategy: {str(e)}")
                signals[strategy_name] = {'signal': 'ERROR', 'error': str(e), 'strategy': strategy_name}
                
        return signals, pd.DataFrame(features) if features else None
        
    def train_model(self, historical_data):
        """
        Train the machine learning model on historical data
        
        Args:
            historical_data (pd.DataFrame): Historical OHLCV data
            
        Returns:
            bool: True if training was successful
        """
        try:
            # Get signals from all strategies for training data
            all_signals = []
            all_outcomes = []
            
            # Split data into chunks for training
            chunk_size = 100
            for i in range(0, len(historical_data) - chunk_size, chunk_size // 2):
                chunk = historical_data.iloc[i:i+chunk_size].copy()
                next_chunk = historical_data.iloc[i+chunk_size:i+chunk_size+20].copy()
                
                signals, features_df = self.analyze_with_all_strategies(chunk, "TRAINING")
                
                if features_df is not None and not features_df.empty:
                    # Determine actual outcome (price went up or down in next period)
                    start_price = chunk.iloc[-1]['close']
                    end_price = next_chunk.iloc[-1]['close'] if len(next_chunk) > 0 else start_price
                    outcome = 1 if end_price > start_price else (-1 if end_price < start_price else 0)
                    
                    all_signals.append(features_df)
                    all_outcomes.extend([outcome] * len(features_df))
            
            if all_signals and all_outcomes:
                # Combine all feature dataframes
                X = pd.concat(all_signals, ignore_index=True)
                y = np.array(all_outcomes)
                
                # Convert categorical features
                X = pd.get_dummies(X, columns=['strategy'])
                
                # Train the model
                self.model.fit(X, y)
                self.is_trained = True
                logging.info("Machine learning model trained successfully")
                return True
            else:
                logging.warning("Not enough data to train the model")
                return False
                
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            return False
            
    def generate_combined_signal(self, historical_data, symbol):
        """
        Generate a combined trading signal using machine learning
        
        Args:
            historical_data (pd.DataFrame): Historical OHLCV data
            symbol (str): Trading pair symbol
            
        Returns:
            dict: Combined trading signal
        """
        try:
            # Get signals from all strategies
            signals, features_df = self.analyze_with_all_strategies(historical_data, symbol)
            
            # If no valid signals, return NO_SIGNAL
            if features_df is None or features_df.empty:
                return {'signal': 'NO_SIGNAL', 'strategy': 'multi'}
            
            # If the model is not trained, return the signal with highest confidence
            if not self.is_trained:
                logging.warning("Model not trained, using highest confidence signal")
                best_signal = None
                highest_confidence = -1
                
                for strategy_name, signal in signals.items():
                    if signal['signal'] in ['LONG', 'SHORT'] and signal.get('confidence', 0) > highest_confidence:
                        highest_confidence = signal.get('confidence', 0)
                        best_signal = signal
                
                if best_signal:
                    best_signal['strategy'] = 'multi (best of)'
                    return best_signal
                else:
                    return {'signal': 'NO_SIGNAL', 'strategy': 'multi'}
            
            # Use the trained model to predict
            X = pd.get_dummies(features_df, columns=['strategy'])
            
            # Only use these features for prediction
            keep_cols = [col for col in X.columns if col.startswith('strategy_') or col in ['signal', 'risk_reward_ratio', 'confidence', 'leverage']]
            X = X[keep_cols]
            
            # Add missing columns that might be in the training data
            for col in self.model.feature_names_in_:
                if col not in X.columns:
                    X[col] = 0
            
            # Keep only columns that were in the training data
            X = X[self.model.feature_names_in_]
            
            # Make prediction
            predictions = self.model.predict(X)
            
            # If at least one strategy predicts LONG and model agrees
            if 1 in predictions:
                # Find all LONG signals
                long_signals = [s for s, p in zip(signals.values(), predictions) if s['signal'] == 'LONG' and p == 1]
                
                if long_signals:
                    # Calculate average parameters
                    avg_entry = sum(s['entry'] for s in long_signals) / len(long_signals)
                    avg_stop = sum(s['stop_loss'] for s in long_signals) / len(long_signals)
                    avg_take = sum(s['take_profit'] for s in long_signals) / len(long_signals)
                    avg_leverage = sum(s['leverage'] for s in long_signals) / len(long_signals)
                    avg_confidence = sum(s['confidence'] for s in long_signals) / len(long_signals)
                    
                    # Compute weighted confidence
                    probas = self.model.predict_proba(X)
                    long_proba = float(np.mean([p[2] for p in probas])) * 100  # Class 2 corresponds to LONG (1)
                    
                    # Combined confidence (from strategies and model)
                    combined_confidence = (avg_confidence + long_proba) / 2
                    
                    return {
                        'signal': 'LONG',
                        'entry': float(avg_entry),
                        'stop_loss': float(avg_stop),
                        'take_profit': float(avg_take),
                        'leverage': float(avg_leverage),
                        'confidence': float(combined_confidence),
                        'strategy': 'multi (ML)',
                        'contributing_strategies': [s['strategy'] for s in long_signals]
                    }
            
            # If at least one strategy predicts SHORT and model agrees
            if -1 in predictions:
                # Find all SHORT signals
                short_signals = [s for s, p in zip(signals.values(), predictions) if s['signal'] == 'SHORT' and p == -1]
                
                if short_signals:
                    # Calculate average parameters
                    avg_entry = sum(s['entry'] for s in short_signals) / len(short_signals)
                    avg_stop = sum(s['stop_loss'] for s in short_signals) / len(short_signals)
                    avg_take = sum(s['take_profit'] for s in short_signals) / len(short_signals)
                    avg_leverage = sum(s['leverage'] for s in short_signals) / len(short_signals)
                    avg_confidence = sum(s['confidence'] for s in short_signals) / len(short_signals)
                    
                    # Compute weighted confidence
                    probas = self.model.predict_proba(X)
                    short_proba = float(np.mean([p[0] for p in probas])) * 100  # Class 0 corresponds to SHORT (-1)
                    
                    # Combined confidence (from strategies and model)
                    combined_confidence = (avg_confidence + short_proba) / 2
                    
                    return {
                        'signal': 'SHORT',
                        'entry': float(avg_entry),
                        'stop_loss': float(avg_stop),
                        'take_profit': float(avg_take),
                        'leverage': float(avg_leverage),
                        'confidence': float(combined_confidence),
                        'strategy': 'multi (ML)',
                        'contributing_strategies': [s['strategy'] for s in short_signals]
                    }
            
            return {'signal': 'NO_SIGNAL', 'strategy': 'multi (ML)'}
                
        except Exception as e:
            logging.error(f"Error generating combined signal: {str(e)}")
            return {'signal': 'ERROR', 'error': str(e), 'strategy': 'multi'}