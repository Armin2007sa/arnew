import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score

logger = logging.getLogger(__name__)

class AITradingModel:
    def __init__(self):
        self.lookback_period = 20
        self.prediction_horizon = 12
        self.confidence_threshold = 70
        
        self.model_long = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        self.model_short = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ])
        
        self.is_trained = False
        self.feature_columns = None
        self.performance_metrics = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0
        }
        
        logger.info("AI Trading Model initialized")
    
    def _extract_features(self, df):
        data = df.copy()
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns:
                data[col] = data[col].astype(float)
        
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        data['body_size'] = abs(data['close'] - data['open'])
        data['upper_shadow'] = data['high'] - data[['open', 'close']].max(axis=1)
        data['lower_shadow'] = data[['open', 'close']].min(axis=1) - data['low']
        data['range'] = data['high'] - data['low']
        data['body_to_range'] = data['body_size'] / data['range']
        data['is_bullish'] = (data['close'] > data['open']).astype(int)
        
        for period in [5, 10, 20, 50]:
            data[f'ma_{period}'] = data['close'].rolling(window=period).mean()
            data[f'std_{period}'] = data['close'].rolling(window=period).std()
            data[f'dist_from_ma_{period}'] = (data['close'] - data[f'ma_{period}']) / data[f'ma_{period}']
            
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            data[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        data['volume_ma_5'] = data['volume'].rolling(window=5).mean()
        data['volume_ma_20'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume'].shift(1)
        data['volume_ma_ratio'] = data['volume_ma_5'] / data['volume_ma_20']
        
        data['trend_5'] = (data['close'] > data['close'].shift(5)).astype(int)
        data['trend_10'] = (data['close'] > data['close'].shift(10)).astype(int)
        
        for lag in range(1, 6):
            data[f'close_lag_{lag}'] = data['close'].shift(lag)
            data[f'range_lag_{lag}'] = data['range'].shift(lag)
            data[f'returns_lag_{lag}'] = data['returns'].shift(lag)
            data[f'volume_lag_{lag}'] = data['volume'].shift(lag)
        
        future_close = data['close'].shift(-self.prediction_horizon)
        data['target_long'] = (future_close > data['close'] * 1.01).astype(int)
        data['target_short'] = (future_close < data['close'] * 0.99).astype(int)
        
        data = data.dropna()
        
        self.feature_columns = [col for col in data.columns 
                              if col not in ['open', 'high', 'low', 'close', 'volume', 
                                           'target_long', 'target_short', 'timestamp']]
        
        return data
    
    def train(self, historical_data, retrain=False):
        if self.is_trained and not retrain:
            logger.info("Model already trained, skipping training")
            return True
        
        try:
            logger.info("Extracting features for training")
            features_df = self._extract_features(historical_data)
            
            X = features_df[self.feature_columns]
            y_long = features_df['target_long']
            y_short = features_df['target_short']
            
            if len(X) < 100:
                logger.warning("Not enough data for training. Need at least 100 data points.")
                return False
            
            X_train, X_val, y_long_train, y_long_val = train_test_split(
                X, y_long, test_size=0.2, shuffle=False, random_state=42)
            _, _, y_short_train, y_short_val = train_test_split(
                X, y_short, test_size=0.2, shuffle=False, random_state=42)
            
            logger.info("Training long signal model")
            self.model_long.fit(X_train, y_long_train)
            
            logger.info("Training short signal model")
            self.model_short.fit(X_train, y_short_train)
            
            y_long_pred = self.model_long.predict(X_val)
            y_short_pred = self.model_short.predict(X_val)
            
            long_accuracy = accuracy_score(y_long_val, y_long_pred)
            long_precision = precision_score(y_long_val, y_long_pred, zero_division=0)
            
            short_accuracy = accuracy_score(y_short_val, y_short_pred)
            short_precision = precision_score(y_short_val, y_short_pred, zero_division=0)
            
            self.performance_metrics = {
                'accuracy': (long_accuracy + short_accuracy) / 2,
                'precision': (long_precision + short_precision) / 2,
                'long_accuracy': long_accuracy,
                'long_precision': long_precision,
                'short_accuracy': short_accuracy,
                'short_precision': short_precision
            }
            
            logger.info(f"Model trained with accuracy: {self.performance_metrics['accuracy']:.4f}, "
                      f"precision: {self.performance_metrics['precision']:.4f}")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False
    
    def predict(self, current_data):
        if not self.is_trained:
            logger.warning("Model not trained yet, cannot generate prediction")
            current_price = current_data['close'].iloc[-1]
            price_5ago = current_data['close'].iloc[-5] if len(current_data) >= 5 else current_price
            
            signal = 'LONG' if current_price > price_5ago else 'SHORT'
            confidence = 40
            
            if signal == 'LONG':
                entry = current_price
                stop_loss = current_price * 0.98
                take_profit = current_price * 1.05
            else:
                entry = current_price
                stop_loss = current_price * 1.02
                take_profit = current_price * 0.95
                
            return {
                'signal': signal,
                'confidence': confidence,
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'leverage': 2,
                'risk_percent': 1,
                'reasons': 'Basic price action analysis (AI model not trained yet)',
            }
        
        try:
            features_df = self._extract_features(current_data)
            
            latest_features = features_df[self.feature_columns].iloc[-1:].copy()
            
            long_prob = self.model_long.predict_proba(latest_features)[0][1] * 100
            short_prob = self.model_short.predict_proba(latest_features)[0][1] * 100
            
            logger.info(f"Long probability: {long_prob:.2f}%, Short probability: {short_prob:.2f}%")
            
            current_price = current_data['close'].iloc[-1]
            volatility = features_df['std_20'].iloc[-1]
            
            signal = 'NO_SIGNAL'
            confidence = 0
            entry = current_price
            stop_loss = current_price
            take_profit = current_price
            reasons = []
            
            if long_prob > self.confidence_threshold and long_prob > short_prob:
                signal = 'LONG'
                confidence = long_prob
                
                entry = current_price
                stop_loss = current_price * (1 - volatility / current_price * 2)
                take_profit = current_price * (1 + volatility / current_price * 3)
                
                reasons = [
                    f"Strong bullish probability ({long_prob:.1f}%)",
                    "Positive momentum indicators",
                    "Price above key moving averages"
                ]
                
                if features_df['rsi_14'].iloc[-1] < 70:
                    reasons.append("RSI not overbought")
                
                if features_df['volume_ratio'].iloc[-1] > 1.2:
                    reasons.append("Increasing volume")
            
            elif short_prob > self.confidence_threshold and short_prob > long_prob:
                signal = 'SHORT'
                confidence = short_prob
                
                entry = current_price
                stop_loss = current_price * (1 + volatility / current_price * 2)
                take_profit = current_price * (1 - volatility / current_price * 3)
                
                reasons = [
                    f"Strong bearish probability ({short_prob:.1f}%)",
                    "Negative momentum indicators",
                    "Price below key moving averages"
                ]
                
                if features_df['rsi_14'].iloc[-1] > 30:
                    reasons.append("RSI not oversold")
                
                if features_df['volume_ratio'].iloc[-1] > 1.2:
                    reasons.append("Increasing volume")
            
            else:
                reasons = ["No strong signal detected", "Market in consolidation phase"]
                if long_prob > short_prob:
                    reasons.append(f"Weak bullish bias ({long_prob:.1f}%)")
                else:
                    reasons.append(f"Weak bearish bias ({short_prob:.1f}%)")
            
            leverage = 1
            if confidence > 85:
                leverage = 5
            elif confidence > 75:
                leverage = 3
            elif confidence > 65:
                leverage = 2
            
            if stop_loss <= 0:
                stop_loss = current_price * 0.95
            if take_profit <= 0:
                take_profit = current_price * 1.05
            
            risk_percent = min(confidence / 100, 0.02) * 100
            
            trading_signal = {
                'signal': signal,
                'confidence': confidence,
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'leverage': leverage,
                'risk_percent': risk_percent,
                'reasons': ', '.join(reasons),
                'timeframe': '5m'
            }
            
            logger.info(f"Generated {signal} signal with {confidence:.2f}% confidence")
            return trading_signal
            
        except Exception as e:
            logger.error(f"Error generating prediction: {str(e)}")
            return {'signal': 'ERROR', 'confidence': 0, 'message': f"Error: {str(e)}"}
    
    def get_performance_metrics(self):
        return self.performance_metrics

ai_model = AITradingModel()

def generate_ai_signal(historical_data, symbol, timeframe='5m'):
    try:
        if not ai_model.is_trained:
            if len(historical_data) >= 200:
                ai_model.train(historical_data)
            else:
                logger.warning("Not enough historical data to train AI model")
        
        signal = ai_model.predict(historical_data)
        
        signal['symbol'] = symbol
        signal['timeframe'] = timeframe
        
        return signal
        
    except Exception as e:
        logger.error(f"Error in AI signal generation: {str(e)}")
        return {
            'signal': 'ERROR',
            'confidence': 0,
            'entry': historical_data['close'].iloc[-1],
            'stop_loss': historical_data['close'].iloc[-1] * 0.95,
            'take_profit': historical_data['close'].iloc[-1] * 1.05,
            'message': f"Error: {str(e)}"
        }