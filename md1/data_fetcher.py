import ccxt
import pandas as pd
import numpy as np
import logging
from datetime import datetime

class DataFetcher:
    """
    Class to fetch market data from cryptocurrency exchanges using ccxt
    """
    
    def __init__(self):
        """Initialize the DataFetcher class"""
        self.exchanges = {}
        logging.debug("DataFetcher initialized")
    
    def get_available_exchanges(self):
        """Get list of available exchanges"""
        # Return a curated list of reliable exchanges that support OHLCV data
        recommended_exchanges = [
            'binance', 'kucoin', 'mexc', 'gate', 'okx', 'bybit', 
            'kraken', 'bitget', 'cryptocom', 'bitfinex'
        ]
        
        # Filter recommended exchanges that are in the ccxt supported exchanges list
        return [ex for ex in recommended_exchanges if ex in ccxt.exchanges]
    
    def _get_exchange(self, exchange_id):
        """
        Get exchange instance, create if not exists
        
        Args:
            exchange_id (str): ID of the exchange
            
        Returns:
            ccxt.Exchange: Exchange instance
        """
        if exchange_id not in self.exchanges:
            try:
                # Create exchange instance
                exchange_class = getattr(ccxt, exchange_id)
                # Use proxy for Binance to avoid regional restrictions
                config = {'enableRateLimit': True}
                
                # Specify different options for specific exchanges if needed
                if exchange_id == 'binance':
                    # Add options directly to config dictionary
                    config['options'] = {'defaultType': 'spot'}
                
                self.exchanges[exchange_id] = exchange_class(config)
                logging.debug(f"Created exchange instance for {exchange_id}")
            except Exception as e:
                logging.error(f"Failed to create exchange instance for {exchange_id}: {str(e)}")
                raise
        
        return self.exchanges[exchange_id]
    
    def get_available_symbols(self, exchange_id):
        """
        Get available symbols for a given exchange
        
        Args:
            exchange_id (str): ID of the exchange
            
        Returns:
            list: List of available symbols
        """
        # Try to get symbols from the specified exchange
        try:
            exchange = self._get_exchange(exchange_id)
            exchange.load_markets()
            # Filter for symbols with USDT as the quote currency, which are most common
            symbols = [symbol for symbol in exchange.symbols if '/USDT' in symbol]
            return sorted(symbols)
        except Exception as e:
            logging.error(f"Failed to get symbols for {exchange_id}: {str(e)}")
            
            # Try alternate exchanges if the primary one fails
            fallback_exchanges = self.get_available_exchanges()
            if exchange_id in fallback_exchanges:
                fallback_exchanges.remove(exchange_id)
            
            for fallback_exchange in fallback_exchanges:
                try:
                    logging.info(f"Trying fallback exchange {fallback_exchange} for symbols")
                    exchange = self._get_exchange(fallback_exchange)
                    exchange.load_markets()
                    symbols = [symbol for symbol in exchange.symbols if '/USDT' in symbol]
                    return sorted(symbols)
                except Exception as fallback_err:
                    logging.error(f"Fallback {fallback_exchange} also failed to get symbols: {str(fallback_err)}")
            
            # If all exchanges fail, return a default list of common symbols
            logging.warning("Using default symbols list as all exchanges failed")
            return [
                'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'ADA/USDT',
                'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'MATIC/USDT'
            ]
    
    def get_historical_data(self, exchange_id, symbol, timeframe='5m', limit=100):
        """
        Fetch historical OHLCV data
        
        Args:
            exchange_id (str): ID of the exchange
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe (default: 5m)
            limit (int): Number of candles to fetch
            
        Returns:
            pd.DataFrame: DataFrame with historical data
        """
        # Try to get data from the specified exchange
        try:
            return self._fetch_data_from_exchange(exchange_id, symbol, timeframe, limit)
        except Exception as e:
            logging.error(f"Failed to fetch data from {exchange_id}: {str(e)}")
            
            # Try alternate exchanges if the primary one fails
            fallback_exchanges = self.get_available_exchanges()
            if exchange_id in fallback_exchanges:
                fallback_exchanges.remove(exchange_id)
            
            for fallback_exchange in fallback_exchanges:
                try:
                    logging.info(f"Trying fallback exchange {fallback_exchange} for {symbol}")
                    return self._fetch_data_from_exchange(fallback_exchange, symbol, timeframe, limit)
                except Exception as fallback_err:
                    logging.error(f"Fallback {fallback_exchange} also failed: {str(fallback_err)}")
            
            # If all exchanges fail, raise the original error
            raise e
    
    def _fetch_data_from_exchange(self, exchange_id, symbol, timeframe='5m', limit=100):
        """
        Internal method to fetch data from a specific exchange
        
        Args:
            exchange_id (str): ID of the exchange
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe
            limit (int): Number of candles to fetch
            
        Returns:
            pd.DataFrame: DataFrame with historical data
        """
        exchange = self._get_exchange(exchange_id)
        
        # Check if the exchange supports OHLCV data
        if not exchange.has['fetchOHLCV']:
            raise Exception(f"Exchange {exchange_id} does not support OHLCV data")
        
        try:
            # Normalize the symbol format if needed
            if '/' not in symbol and symbol.endswith('USDT'):
                # Convert format like BTCUSDT to BTC/USDT
                symbol = symbol[:-4] + '/USDT'
            
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Store the actual symbol used for the query to prevent mismatches
            df['symbol'] = symbol
            
            # Add detailed logging for debugging symbol issues
            logging.info(f"Successfully fetched data for symbol '{symbol}' - symbol value stored in DataFrame: '{symbol}'")
            logging.debug(f"OHLCV data sample for symbol '{symbol}': {df.head(1).to_dict('records')}")
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Calculate additional data
            df['body_size'] = abs(df['close'] - df['open'])
            df['shadow_upper'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['shadow_lower'] = df[['open', 'close']].min(axis=1) - df['low']
            df['is_bullish'] = df['close'] > df['open']
            
            # Log successful data fetch
            logging.info(f"Successfully fetched data for {symbol} from {exchange_id}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching data for symbol {symbol} from {exchange_id}: {str(e)}")
            raise
        

