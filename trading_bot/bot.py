import json
import time
import threading
import hmac
import hashlib
import requests
import logging
from datetime import datetime
from arnew.main import db, TradingStrategy, User, ApiKey, TradeHistory
from arnew.utils.helpers import get_api_data, calculate_indicators

logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, strategy_id):
        self.strategy_id = strategy_id
        self.strategy = TradingStrategy.query.get(strategy_id)
        self.user = User.query.get(self.strategy.user_id)
        self.api_key = ApiKey.query.filter_by(user_id=self.user.id).first()
        self.parameters = json.loads(self.strategy.parameters)
        self.running = False
        self.thread = None
    
    def start(self):
        if self.thread and self.thread.is_alive():
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_bot)
        self.thread.daemon = True
        self.thread.start()
        
        self.strategy.last_run = datetime.utcnow()
        db.session.commit()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
    
    def _run_bot(self):
        while self.running:
            try:
                self._execute_strategy()
            except Exception as e:
                logger.error(f"Error in trading bot: {e}")
            
            time.sleep(60)
    
    def _execute_strategy(self):
        asset = self.parameters.get('asset', 'BTC/USDT')
        time_frame = self.parameters.get('time_frame', '1h')
        risk_level = self.parameters.get('risk_level', 'medium')
        
        price_data = get_api_data(asset, time_frame)
        
        if not price_data:
            return
        
        action = self._analyze_market(price_data, self.strategy.strategy_type)
        
        if action == 'buy':
            self._place_order(asset, 'buy')
        elif action == 'sell':
            self._place_order(asset, 'sell')
    
    def _analyze_market(self, price_data, strategy_type):
        if strategy_type == 'sma_crossover':
            return self._sma_crossover_strategy(price_data)
        elif strategy_type == 'rsi_strategy':
            return self._rsi_strategy(price_data)
        elif strategy_type == 'macd_strategy':
            return self._macd_strategy(price_data)
        return 'hold'
    
    def _sma_crossover_strategy(self, price_data):
        short_period = 10
        long_period = 30
        
        short_sma = calculate_indicators(price_data, 'sma', short_period)
        long_sma = calculate_indicators(price_data, 'sma', long_period)
        
        if not short_sma or not long_sma:
            return 'hold'
        
        if short_sma[-2] < long_sma[-2] and short_sma[-1] > long_sma[-1]:
            return 'buy'
        elif short_sma[-2] > long_sma[-2] and short_sma[-1] < long_sma[-1]:
            return 'sell'
        
        return 'hold'
    
    def _rsi_strategy(self, price_data):
        rsi = calculate_indicators(price_data, 'rsi')
        
        if not rsi:
            return 'hold'
        
        if rsi[-1] < 30:
            return 'buy'
        elif rsi[-1] > 70:
            return 'sell'
        
        return 'hold'
    
    def _macd_strategy(self, price_data):
        ema12 = calculate_indicators(price_data, 'ema', 12)
        ema26 = calculate_indicators(price_data, 'ema', 26)
        
        if not ema12 or not ema26:
            return 'hold'
        
        macd = [ema12[i] - ema26[i] for i in range(len(ema12)) if ema12[i] is not None and ema26[i] is not None]
        
        if len(macd) < 2:
            return 'hold'
        
        if macd[-2] < 0 and macd[-1] > 0:
            return 'buy'
        elif macd[-2] > 0 and macd[-1] < 0:
            return 'sell'
        
        return 'hold'
    
    def _place_order(self, symbol, order_type):
        if not self.api_key:
            logger.error("No API key configured")
            return
        
        if self.api_key.exchange.lower() == 'binance':
            self._place_binance_order(symbol, order_type)
        else:
            logger.error(f"Unsupported exchange: {self.api_key.exchange}")
    
    def _place_binance_order(self, symbol, order_type):
        url = 'https://api.binance.com/api/v3/order'
        timestamp = int(datetime.now().timestamp() * 1000)
        
        risk_level = self.parameters.get('risk_level', 'medium')
        
        if risk_level == 'low':
            quantity = 0.001
        elif risk_level == 'medium':
            quantity = 0.005
        else:
            quantity = 0.01
        
        symbol_formatted = symbol.replace("/", "")
        
        params = {
            'symbol': symbol_formatted,
            'side': order_type.upper(),
            'type': 'MARKET',
            'quantity': quantity,
            'timestamp': timestamp,
            'recvWindow': 5000
        }
        
        query_string = '&'.join([f"{key}={params[key]}" for key in params])
        signature = hmac.new(
            self.api_key.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        params['signature'] = signature
        
        headers = {
            'X-MBX-APIKEY': self.api_key.api_key
        }
        
        try:
            response = requests.post(url, params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                
                price = 0
                for fill in data.get('fills', []):
                    price += float(fill.get('price', 0))
                
                if data.get('fills'):
                    price /= len(data.get('fills'))
                
                trade = TradeHistory(
                    user_id=self.user.id,
                    strategy_id=self.strategy_id,
                    trade_type=order_type,
                    symbol=symbol,
                    amount=quantity,
                    price=price,
                    status='executed'
                )
                
                db.session.add(trade)
                db.session.commit()
                
                logger.info(f"Order placed: {order_type} {quantity} {symbol} at {price}")
            else:
                logger.error(f"Error placing order: {response.text}")
        except Exception as e:
            logger.error(f"Error placing order: {e}")