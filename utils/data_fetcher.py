import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random
import json

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, exchange='binance'):
        self.exchange = exchange.lower()
        self.base_urls = {
            'binance': 'https://api.binance.com',
            'kucoin': 'https://api.kucoin.com',
            'mexc': 'https://api.mexc.com'
        }
        self.timeframe_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '2h': 7200,
            '4h': 14400,
            '6h': 21600,
            '12h': 43200,
            '1d': 86400,
            '1w': 604800
        }
        
        self.cache = {}
        self.cache_ttl = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
    
    def fetch_historical_data(self, symbol, timeframe='5m', limit=200):
        try:
            cache_key = f"{symbol}_{timeframe}_{limit}"
            
            # Check if we have cached data and it's still valid
            if cache_key in self.cache:
                cache_entry = self.cache[cache_key]
                if time.time() - cache_entry['timestamp'] < self.cache_ttl.get(timeframe, 300):
                    logger.info(f"Using cached data for {symbol} ({timeframe})")
                    return cache_entry['data']
            
            # Fetch new data
            if self.exchange == 'binance':
                data = self._fetch_binance_data(symbol, timeframe, limit)
            elif self.exchange == 'kucoin':
                data = self._fetch_kucoin_data(symbol, timeframe, limit)
            elif self.exchange == 'mexc':
                data = self._fetch_mexc_data(symbol, timeframe, limit)
            else:
                logger.error(f"Unsupported exchange: {self.exchange}")
                return None
            
            # Process data to standard format
            df = self._process_data(data)
            
            # Cache the data
            if df is not None:
                self.cache[cache_key] = {
                    'data': df,
                    'timestamp': time.time()
                }
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return None
    
    def _fetch_binance_data(self, symbol, timeframe='5m', limit=200):
        url = f"{self.base_urls['binance']}/api/v3/klines"
        
        # Format symbol for Binance
        formatted_symbol = symbol.replace('/', '')
        
        params = {
            'symbol': formatted_symbol,
            'interval': timeframe,
            'limit': limit
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Binance API error: {response.text}")
                # Fallback to mock data for testing
                return self._generate_mock_data(limit)
        except Exception as e:
            logger.error(f"Error fetching from Binance: {str(e)}")
            return self._generate_mock_data(limit)
    
    def _fetch_kucoin_data(self, symbol, timeframe='5m', limit=200):
        # KuCoin uses different time formatting
        kucoin_timeframe = timeframe
        if timeframe == '1m':
            kucoin_timeframe = '1min'
        elif timeframe == '5m':
            kucoin_timeframe = '5min'
        elif timeframe == '15m':
            kucoin_timeframe = '15min'
        elif timeframe == '30m':
            kucoin_timeframe = '30min'
        elif timeframe == '1h':
            kucoin_timeframe = '1hour'
        elif timeframe == '4h':
            kucoin_timeframe = '4hour'
        elif timeframe == '1d':
            kucoin_timeframe = '1day'
        
        url = f"{self.base_urls['kucoin']}/api/v1/market/candles"
        
        # Format symbol for KuCoin
        formatted_symbol = symbol.replace('/', '-')
        
        end_time = int(time.time())
        start_time = end_time - (self.timeframe_map.get(timeframe, 300) * limit)
        
        params = {
            'symbol': formatted_symbol,
            'type': kucoin_timeframe,
            'startAt': start_time,
            'endAt': end_time
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200 and response.json().get('data'):
                return response.json().get('data')
            else:
                logger.error(f"KuCoin API error: {response.text}")
                # Fallback to mock data for testing
                return self._generate_mock_data(limit)
        except Exception as e:
            logger.error(f"Error fetching from KuCoin: {str(e)}")
            return self._generate_mock_data(limit)
    
    def _fetch_mexc_data(self, symbol, timeframe='5m', limit=200):
        url = f"{self.base_urls['mexc']}/api/v3/klines"
        
        # Format symbol for MEXC
        formatted_symbol = symbol.replace('/', '')
        
        params = {
            'symbol': formatted_symbol,
            'interval': timeframe,
            'limit': limit
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"MEXC API error: {response.text}")
                # Fallback to mock data for testing
                return self._generate_mock_data(limit)
        except Exception as e:
            logger.error(f"Error fetching from MEXC: {str(e)}")
            return self._generate_mock_data(limit)
    
    def _process_data(self, data):
        if not data:
            return None
        
        try:
            if self.exchange == 'binance' or self.exchange == 'mexc':
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                               'close_time', 'quote_asset_volume', 'number_of_trades',
                                               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                
                # Convert timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
            elif self.exchange == 'kucoin':
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'amount'])
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                
                # Convert timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Convert strings to float for price and volume data
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Sort by timestamp
            df.sort_index(inplace=True)
            
            # Add technical indicators
            df = self._add_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            return None
    
    def _add_indicators(self, df):
        # Add some basic indicators
        # 1. SMA
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        # 2. EMA
        df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # 3. MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # 4. RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # 5. Bollinger Bands
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (std * 2)
        
        return df
    
    def _generate_mock_data(self, limit):
        logger.warning("Using mock data for demonstration")
        now = datetime.now()
        data = []
        
        close_price = 50.0
        
        for i in range(limit):
            timestamp = int((now - timedelta(minutes=i)).timestamp() * 1000)
            
            # Random price movement
            price_change = (random.random() - 0.5) * 0.02 * close_price
            close_price += price_change
            
            # Generate candle
            high_price = close_price * (1 + random.random() * 0.01)
            low_price = close_price * (1 - random.random() * 0.01)
            open_price = low_price + random.random() * (high_price - low_price)
            
            volume = random.random() * 100 + 50
            
            candle = [
                timestamp,  # timestamp
                str(open_price),  # open
                str(high_price),  # high
                str(low_price),  # low
                str(close_price),  # close
                str(volume),  # volume
                timestamp + 60000,  # close_time
                str(volume * close_price),  # quote_asset_volume
                100,  # number_of_trades
                str(volume * 0.7),  # taker_buy_base_asset_volume
                str(volume * 0.7 * close_price),  # taker_buy_quote_asset_volume
                "0"  # ignore
            ]
            
            data.append(candle)
        
        # Sort by timestamp (oldest first)
        data.sort(key=lambda x: x[0])
        
        return data
    
    def get_current_price(self, symbol):
        try:
            # Format symbol
            formatted_symbol = symbol.replace('/', '')
            
            if self.exchange == 'binance':
                url = f"{self.base_urls['binance']}/api/v3/ticker/price"
                params = {'symbol': formatted_symbol}
                
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    return float(data['price'])
                    
            elif self.exchange == 'kucoin':
                formatted_symbol = symbol.replace('/', '-')
                url = f"{self.base_urls['kucoin']}/api/v1/market/orderbook/level1"
                params = {'symbol': formatted_symbol}
                
                response = requests.get(url, params=params)
                if response.status_code == 200 and response.json().get('data'):
                    data = response.json().get('data')
                    return float(data['price'])
                    
            elif self.exchange == 'mexc':
                url = f"{self.base_urls['mexc']}/api/v3/ticker/price"
                params = {'symbol': formatted_symbol}
                
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    return float(data['price'])
            
            # If we get here, something went wrong
            logger.error(f"Error getting current price from {self.exchange} for {symbol}")
            
            # For testing purposes, return a random price
            df = self.fetch_historical_data(symbol, timeframe='1m', limit=1)
            if df is not None and not df.empty:
                return df['close'].iloc[-1]
            else:
                return 50.0 * (1 + random.random() * 0.01)
                
        except Exception as e:
            logger.error(f"Error fetching current price: {str(e)}")
            return 50.0 * (1 + random.random() * 0.01)
    
    def get_supported_symbols(self, base_currency=None):
        try:
            if self.exchange == 'binance':
                url = f"{self.base_urls['binance']}/api/v3/exchangeInfo"
                
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    symbols = [s['symbol'] for s in data['symbols'] if s['status'] == 'TRADING']
                    
                    if base_currency:
                        symbols = [s for s in symbols if s.endswith(base_currency)]
                    
                    # Format symbols to standard format
                    formatted_symbols = []
                    for s in symbols:
                        if base_currency:
                            formatted_symbols.append(s[:-len(base_currency)] + '/' + base_currency)
                        else:
                            # Try to infer the base currency
                            for bc in ['USDT', 'BTC', 'ETH', 'BNB']:
                                if s.endswith(bc):
                                    formatted_symbols.append(s[:-len(bc)] + '/' + bc)
                                    break
                    
                    return formatted_symbols
                    
            elif self.exchange == 'kucoin':
                url = f"{self.base_urls['kucoin']}/api/v1/symbols"
                
                response = requests.get(url)
                if response.status_code == 200 and response.json().get('data'):
                    data = response.json().get('data')
                    symbols = [s['symbol'] for s in data if s['enableTrading']]
                    
                    if base_currency:
                        symbols = [s for s in symbols if s.split('-')[1] == base_currency]
                    
                    # Format symbols to standard format
                    formatted_symbols = [s.replace('-', '/') for s in symbols]
                    
                    return formatted_symbols
                    
            elif self.exchange == 'mexc':
                url = f"{self.base_urls['mexc']}/api/v3/exchangeInfo"
                
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    symbols = [s['symbol'] for s in data['symbols'] if s['status'] == 'ENABLED']
                    
                    if base_currency:
                        symbols = [s for s in symbols if s.endswith(base_currency)]
                    
                    # Format symbols to standard format
                    formatted_symbols = []
                    for s in symbols:
                        if base_currency:
                            formatted_symbols.append(s[:-len(base_currency)] + '/' + base_currency)
                        else:
                            # Try to infer the base currency
                            for bc in ['USDT', 'BTC', 'ETH', 'MX']:
                                if s.endswith(bc):
                                    formatted_symbols.append(s[:-len(bc)] + '/' + bc)
                                    break
                    
                    return formatted_symbols
                    
            # If we get here, something went wrong
            logger.error(f"Error getting supported symbols from {self.exchange}")
            
            # For testing purposes, return some common symbols
            return [
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT',
                'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'MATIC/USDT'
            ]
                
        except Exception as e:
            logger.error(f"Error fetching supported symbols: {str(e)}")
            return [
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT',
                'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'MATIC/USDT'
            ]

def get_crypto_news(symbol=None, limit=5):
    try:
        # Try to get news from major crypto news APIs
        news_items = []
        
        # Add some dummy news items for testing
        all_news = [
            {
                'title': 'Bitcoin Hits New All-Time High Amid Institutional Adoption',
                'description': 'Bitcoin has reached a new all-time high as institutional investors continue to enter the cryptocurrency market.',
                'url': 'https://example.com/bitcoin-ath',
                'sentiment': 'positive'
            },
            {
                'title': 'Ethereum 2.0 Upgrade on Track for Q3 Implementation',
                'description': 'The highly anticipated Ethereum 2.0 upgrade is proceeding as planned with a Q3 target date.',
                'url': 'https://example.com/ethereum-upgrade',
                'sentiment': 'positive'
            },
            {
                'title': 'SEC Delays Decision on Spot Bitcoin ETF Approval',
                'description': 'The SEC has once again delayed its decision on approving a spot Bitcoin ETF, citing market concerns.',
                'url': 'https://example.com/sec-delays-etf',
                'sentiment': 'negative'
            },
            {
                'title': 'Major Bank Announces Cryptocurrency Custody Services',
                'description': 'A major international bank has announced plans to offer cryptocurrency custody services to institutional clients.',
                'url': 'https://example.com/bank-crypto-custody',
                'sentiment': 'positive'
            },
            {
                'title': 'New Crypto Regulation Framework Proposed in EU',
                'description': 'The European Union has proposed a comprehensive regulatory framework for cryptocurrencies and digital assets.',
                'url': 'https://example.com/eu-crypto-regulation',
                'sentiment': 'neutral'
            },
            {
                'title': 'Leading DeFi Protocol Suffers Security Breach',
                'description': 'A popular DeFi protocol has reported a security breach resulting in the loss of user funds.',
                'url': 'https://example.com/defi-security-breach',
                'sentiment': 'negative'
            },
            {
                'title': 'Central Bank Digital Currencies Gain Momentum Globally',
                'description': 'More central banks are accelerating their CBDC development efforts as digital currencies become mainstream.',
                'url': 'https://example.com/cbdc-momentum',
                'sentiment': 'neutral'
            },
            {
                'title': 'NFT Market Rebounds After Three-Month Slump',
                'description': 'The NFT market is showing signs of recovery after a prolonged downturn in sales and floor prices.',
                'url': 'https://example.com/nft-rebound',
                'sentiment': 'positive'
            },
            {
                'title': 'Mining Difficulty Increases as Hash Rate Reaches Record High',
                'description': 'Bitcoin mining difficulty has adjusted upward following significant growth in network hash rate.',
                'url': 'https://example.com/mining-difficulty',
                'sentiment': 'neutral'
            }
        ]
        
        if symbol:
            symbol_news = [
                {
                    'title': f'{symbol.split("/")[0]} Price Analysis: Technical Indicators Point to Potential Breakout',
                    'description': f'Technical analysis of {symbol.split("/")[0]} shows bullish patterns forming on multiple timeframes.',
                    'url': f'https://example.com/{symbol.split("/")[0].lower()}-analysis',
                    'sentiment': 'positive'
                },
                {
                    'title': f'Development Update: {symbol.split("/")[0]} Network Upgrades Coming Soon',
                    'description': f'The {symbol.split("/")[0]} development team has announced significant protocol upgrades scheduled for next month.',
                    'url': f'https://example.com/{symbol.split("/")[0].lower()}-upgrades',
                    'sentiment': 'positive'
                },
                {
                    'title': f'Major Exchange Lists New {symbol.split("/")[0]} Trading Pairs',
                    'description': f'A top cryptocurrency exchange has added new trading pairs for {symbol.split("/")[0]}, increasing market accessibility.',
                    'url': f'https://example.com/{symbol.split("/")[0].lower()}-new-pairs',
                    'sentiment': 'positive'
                }
            ]
            news_items = symbol_news + all_news
        else:
            news_items = all_news
        
        # Shuffle and limit
        random.shuffle(news_items)
        return news_items[:limit]
        
    except Exception as e:
        logger.error(f"Error fetching crypto news: {str(e)}")
        return []

def get_top_traded_symbols(exchange_id, limit=10):
    try:
        # In a real application, this would fetch actual data from the exchange API
        # For this demo, we'll return random popular pairs
        
        top_symbols = [
            {'symbol': 'BTC/USDT', 'volume_24h': 1200000000, 'price_change_24h': 2.5},
            {'symbol': 'ETH/USDT', 'volume_24h': 800000000, 'price_change_24h': 3.1},
            {'symbol': 'SOL/USDT', 'volume_24h': 500000000, 'price_change_24h': 5.7},
            {'symbol': 'BNB/USDT', 'volume_24h': 300000000, 'price_change_24h': 1.8},
            {'symbol': 'XRP/USDT', 'volume_24h': 250000000, 'price_change_24h': -1.2},
            {'symbol': 'AVAX/USDT', 'volume_24h': 200000000, 'price_change_24h': 7.3},
            {'symbol': 'DOGE/USDT', 'volume_24h': 180000000, 'price_change_24h': 4.2},
            {'symbol': 'MATIC/USDT', 'volume_24h': 150000000, 'price_change_24h': -2.1},
            {'symbol': 'DOT/USDT', 'volume_24h': 120000000, 'price_change_24h': 0.8},
            {'symbol': 'LINK/USDT', 'volume_24h': 100000000, 'price_change_24h': 3.5},
            {'symbol': 'ADA/USDT', 'volume_24h': 90000000, 'price_change_24h': -0.5},
            {'symbol': 'UNI/USDT', 'volume_24h': 80000000, 'price_change_24h': 1.9},
            {'symbol': 'SHIB/USDT', 'volume_24h': 70000000, 'price_change_24h': 6.2},
            {'symbol': 'LTC/USDT', 'volume_24h': 60000000, 'price_change_24h': 0.3},
            {'symbol': 'ATOM/USDT', 'volume_24h': 50000000, 'price_change_24h': 2.7},
        ]
        
        # Add some randomness
        for symbol in top_symbols:
            symbol['volume_24h'] = symbol['volume_24h'] * (0.8 + random.random() * 0.4)
            symbol['price_change_24h'] = symbol['price_change_24h'] + (random.random() * 4 - 2)
        
        # Sort by volume
        top_symbols.sort(key=lambda x: x['volume_24h'], reverse=True)
        
        return top_symbols[:limit]
        
    except Exception as e:
        logger.error(f"Error fetching top traded symbols: {str(e)}")
        return []