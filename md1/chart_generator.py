"""
Chart Generator Module

This module is responsible for generating professional-looking trading charts with technical analysis
elements including Fibonacci retracements, support/resistance lines, and other indicators.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChartGenerator:
    """
    Class to generate professional crypto trading charts with technical analysis
    using real-time data and AI-generated predictions.
    """
    
    def __init__(self):
        """Initialize the ChartGenerator"""
        # Define chart style and appearance
        self.chart_style = {
            'background': '#0e1117',  # Darker background for better contrast
            'text': '#e0e0e0',        # Off-white text for readability
            'grid': '#1e222d',
            'candle_up': '#26a69a',   # TradingView style green
            'candle_down': '#ef5350', # TradingView style red
            'volume_up': 'rgba(38, 166, 154, 0.6)',
            'volume_down': 'rgba(239, 83, 80, 0.6)',
            'ma_fast': '#ffeb3b',     # Yellow for fast MA
            'ma_slow': '#ff9800',     # Orange for slow MA
            'ma_long': '#2962ff',     # Blue for long-term MA
            'fib_lines': ['#ff9800', '#4caf50', '#2196f3', '#9c27b0', '#f44336'],
            'support': '#26a69a',     # Green for support
            'resistance': '#ef5350',  # Red for resistance
            'spine': '#2a2e39',       # Chart border
            'watermark': '#333333'    # Watermark
        }
        
        # Set up matplotlib defaults for consistent appearance
        plt.style.use('dark_background')
        plt.rcParams['figure.facecolor'] = self.chart_style['background']
        plt.rcParams['axes.facecolor'] = self.chart_style['background']
        plt.rcParams['axes.edgecolor'] = self.chart_style['spine']
        plt.rcParams['axes.labelcolor'] = self.chart_style['text']
        plt.rcParams['xtick.color'] = self.chart_style['text']
        plt.rcParams['ytick.color'] = self.chart_style['text']
        
        # Set additional chart style elements
        self.chart_style['entry'] = '#ffeb3b'         # Yellow for entry points
        self.chart_style['take_profit'] = '#00e676'   # Green for take profit
        self.chart_style['stop_loss'] = '#ff5252'     # Red for stop loss
        self.chart_style['trend_up'] = '#2ecc71'      # Green for uptrends
        self.chart_style['trend_down'] = '#e74c3c'    # Red for downtrends
        plt.rcParams['axes.edgecolor'] = self.chart_style['grid']
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.color'] = self.chart_style['grid']
        plt.rcParams['grid.alpha'] = 0.3
        
        logger.info("ChartGenerator initialized")
    
    def generate_chart(self, historical_data, symbol, signal=None, timeframe='5m'):
        """
        Generate a professional-looking chart with technical analysis
        
        Args:
            historical_data (pd.DataFrame): OHLCV data
            symbol (str): Trading pair symbol
            signal (dict): Trading signal with entry, take profit, stop loss
            timeframe (str): Chart timeframe
            
        Returns:
            str: Base64 encoded image
        """
        logger.info(f"Generating chart for {symbol} ({timeframe})")
        
        # Create a new figure
        fig = plt.figure(figsize=(12, 8), dpi=150, facecolor=self.chart_style['background'])
        
        # Main price subplot
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3, fig=fig)
        
        # Volume subplot
        ax2 = plt.subplot2grid((4, 1), (3, 0), fig=fig, sharex=ax1)
        
        # Convert dates if timestamp is in Unix format
        dates = historical_data.index if isinstance(historical_data.index, pd.DatetimeIndex) else pd.to_datetime(historical_data['timestamp'], unit='s')
        
        # Plot candlestick chart
        self._plot_candlesticks(ax1, historical_data, dates)
        
        # Add moving averages
        self._add_moving_averages(ax1, historical_data, dates)
        
        # Add Fibonacci levels if signal is provided
        if signal and signal.get('signal') != 'NOT_SUITABLE' and 'entry' in signal:
            self._add_fibonacci_levels(ax1, historical_data, signal, dates)
            
            # Add support and resistance levels
            self._add_support_resistance(ax1, historical_data, dates)
            
            # Add trading signal markers
            self._add_signal_markers(ax1, dates, signal)
            
            # Add trend lines
            self._add_trend_lines(ax1, historical_data, signal, dates)
        
        # Plot volume
        self._plot_volume(ax2, historical_data, dates)
        
        # Format x-axis
        self._format_time_axis(ax1, ax2, dates, timeframe)
        
        # Add chart title and other formatting
        plt.suptitle(f"{symbol} - {timeframe} Chart", fontsize=16, color=self.chart_style['text'])
        
        # Set background color for both axes
        ax1.set_facecolor(self.chart_style['background'])
        ax2.set_facecolor(self.chart_style['background'])
        
        # Add watermark
        self._add_watermark(fig)
        
        # Save chart to bytes buffer
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', facecolor=self.chart_style['background'])
        plt.close(fig)
        
        # Convert to base64 for embedding in HTML
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Save locally to static folder for serving
        chart_path = f"static/images/charts/{symbol.replace('/', '_')}_{timeframe}.png"
        os.makedirs(os.path.dirname(chart_path), exist_ok=True)
        
        with open(chart_path, 'wb') as f:
            buf.seek(0)
            f.write(buf.getvalue())
        
        logger.info(f"Chart generated and saved to {chart_path}")
        
        return chart_path, img_base64
    
    def _plot_candlesticks(self, ax, data, dates):
        """Plot candlestick chart"""
        width = 0.6  # Width of candlestick body
        
        for i in range(len(data)):
            # Extract OHLC
            op, hi, lo, cl = data.iloc[i][['open', 'high', 'low', 'close']]
            
            # Determine color based on price movement
            color = self.chart_style['candle_up'] if cl >= op else self.chart_style['candle_down']
            
            # Plot candlestick body
            body_height = abs(cl - op)
            y_bottom = min(cl, op)
            rect = plt.Rectangle((i - width/2, y_bottom), width, body_height, 
                                 fill=True, color=color, alpha=1.0, zorder=3)
            ax.add_patch(rect)
            
            # Plot upper and lower wicks
            ax.plot([i, i], [y_bottom + body_height, hi], color=color, linewidth=1.2, zorder=2)
            ax.plot([i, i], [y_bottom, lo], color=color, linewidth=1.2, zorder=2)
        
        # Set y-axis formatter for price
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.8f}'))
        
        # Add price labels
        ax.set_ylabel('Price', fontsize=12, color=self.chart_style['text'])
    
    def _plot_volume(self, ax, data, dates):
        """Plot volume bars"""
        for i in range(len(data)):
            op, cl = data.iloc[i][['open', 'close']]
            vol = data.iloc[i]['volume']
            
            # Determine color based on price movement
            color = self.chart_style['volume_up'] if cl >= op else self.chart_style['volume_down']
            
            # Plot volume bar
            ax.bar(i, vol, width=0.8, color=color, zorder=2)
        
        # Format volume axis
        ax.set_ylabel('Volume', fontsize=10, color=self.chart_style['text'])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}'))
    
    def _add_moving_averages(self, ax, data, dates):
        """Add moving average lines to the chart"""
        # Calculate and plot fast MA (20-period)
        ma_fast = data['close'].rolling(window=20).mean()
        ax.plot(range(len(ma_fast)), ma_fast, color=self.chart_style['ma_fast'], 
                linewidth=1.5, label='MA20', zorder=4)
        
        # Calculate and plot slow MA (50-period)
        ma_slow = data['close'].rolling(window=50).mean()
        ax.plot(range(len(ma_slow)), ma_slow, color=self.chart_style['ma_slow'], 
                linewidth=1.5, label='MA50', zorder=4)
        
        # Add legend
        ax.legend(loc='upper left', framealpha=0.7)
    
    def _add_fibonacci_levels(self, ax, data, signal, dates):
        """Add Fibonacci retracement levels with proper TradingView-style visualization"""
        # Find significant price swing for Fibonacci calculation
        try:
            closes = data['close'].values
            highs = data['high'].values
            lows = data['low'].values
            entry_price = signal.get('entry', closes[-1])
            
            # Find swing high and swing low for better Fibonacci analysis
            # Use more sophisticated detection of swing points for more accurate fib levels
            if signal.get('signal') == 'LONG':
                # For long signals, look for a recent swing high followed by a swing low
                lookback = min(50, len(closes) - 1)
                
                # Find the most significant swing high in recent price action
                # Look for higher highs followed by significant pullbacks
                high_idx = len(highs) - 1
                for i in range(len(highs) - 2, len(highs) - lookback, -1):
                    if highs[i] > highs[i+1] and highs[i] > highs[i-1]:
                        # Check if this high is followed by a significant pullback
                        if min(lows[i:]) < highs[i] * 0.98:  # At least 2% pullback
                            high_idx = i
                            break
                
                swing_high = highs[high_idx]
                
                # Find the swing low after the swing high
                low_idx = len(lows) - 1
                for i in range(high_idx, len(lows)):
                    if lows[i] < lows[i-1] and (i == len(lows)-1 or lows[i] < lows[i+1]):
                        low_idx = i
                        break
                
                swing_low = lows[low_idx]
                
                # For visualization, mark the swing points
                ax.scatter(high_idx, swing_high, s=120, marker='o', color=self.chart_style['text'], alpha=0.7, zorder=5)
                ax.scatter(low_idx, swing_low, s=120, marker='o', color=self.chart_style['text'], alpha=0.7, zorder=5)
                
                # Connect swing points with a line
                ax.plot([high_idx, low_idx], [swing_high, swing_low], 'w-', alpha=0.5)
                
                # Fib levels go from high to low for longs
                start_level = swing_high
                end_level = swing_low
                
            else:  # SHORT signal
                # For short signals, look for a recent swing low followed by a swing high
                lookback = min(50, len(closes) - 1)
                
                # Find the most significant swing low in recent price action
                low_idx = len(lows) - 1
                for i in range(len(lows) - 2, len(lows) - lookback, -1):
                    if lows[i] < lows[i+1] and lows[i] < lows[i-1]:
                        # Check if this low is followed by a significant bounce
                        if max(highs[i:]) > lows[i] * 1.02:  # At least 2% bounce
                            low_idx = i
                            break
                
                swing_low = lows[low_idx]
                
                # Find the swing high after the swing low
                high_idx = len(highs) - 1
                for i in range(low_idx, len(highs)):
                    if highs[i] > highs[i-1] and (i == len(highs)-1 or highs[i] > highs[i+1]):
                        high_idx = i
                        break
                
                swing_high = highs[high_idx]
                
                # For visualization, mark the swing points
                ax.scatter(low_idx, swing_low, s=120, marker='o', color=self.chart_style['text'], alpha=0.7, zorder=5)
                ax.scatter(high_idx, swing_high, s=120, marker='o', color=self.chart_style['text'], alpha=0.7, zorder=5)
                
                # Connect swing points with a line
                ax.plot([low_idx, high_idx], [swing_low, swing_high], 'w-', alpha=0.5)
                
                # Fib levels go from low to high for shorts
                start_level = swing_low
                end_level = swing_high
            
            # Fibonacci levels (standard): 0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0
            fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
            
            # Calculate price range for Fibonacci levels
            price_range = abs(end_level - start_level)
            
            # Plot Fibonacci levels with TradingView-style visuals
            for i, level in enumerate(fib_levels):
                # Calculate the price level for this Fibonacci ratio
                if signal.get('signal') == 'LONG':
                    price_level = swing_low + price_range * level
                else:
                    price_level = swing_high - price_range * level
                
                # Choose color and style based on significance of the level
                if level == 0 or level == 1.0:
                    color = '#f5f5f5'  # White
                    linestyle = '-'
                    alpha = 0.8
                    linewidth = 1.5
                    font_weight = 'bold'
                elif level == 0.5:
                    color = '#ff9800'  # Orange
                    linestyle = '--'
                    alpha = 0.8
                    linewidth = 1.5
                    font_weight = 'bold'
                elif level == 0.618:
                    color = '#4caf50'  # Green - Golden ratio
                    linestyle = '--'
                    alpha = 0.9
                    linewidth = 1.8
                    font_weight = 'bold'
                else:
                    color = self.chart_style['fib_lines'][i % len(self.chart_style['fib_lines'])]
                    linestyle = '--'
                    alpha = 0.6
                    linewidth = 1.0
                    font_weight = 'normal'
                
                # Draw the horizontal line for this Fibonacci level
                ax.axhline(y=price_level, color=color, linestyle=linestyle, 
                           linewidth=linewidth, alpha=alpha, zorder=4)
                
                # Add a label showing the Fibonacci ratio and price
                label = f'Fib {level:.3f} ({price_level:.8f})'
                ax.text(len(data) * 0.98, price_level, label, 
                        color=color, weight=font_weight, 
                        verticalalignment='center', horizontalalignment='right',
                        fontsize=8)
                
                # Highlight if the level is close to entry, stop loss, or take profit
                signal_levels = [
                    (signal.get('entry', None), 'Entry', '#4caf50'),
                    (signal.get('stop_loss', None), 'SL', '#ff3d00'),
                    (signal.get('take_profit', None), 'TP', '#00c853')
                ]
                
                for sig_level, sig_name, sig_color in signal_levels:
                    if sig_level and abs(price_level - sig_level) / price_level < 0.005:
                        ax.axhline(y=price_level, color=sig_color, linestyle='-', 
                                 linewidth=2.0, alpha=1.0, zorder=5)
                        ax.text(len(data) * 0.99, price_level, f' ✓ {sig_name}', 
                                color=sig_color, fontsize=9, weight='bold',
                                verticalalignment='center', horizontalalignment='right')
        
        except Exception as e:
            logger.error(f"Error adding Fibonacci levels: {str(e)}")
    
    def _add_support_resistance(self, ax, data, dates):
        """Add support and resistance levels using advanced detection algorithm"""
        try:
            # Get OHLC data
            highs = data['high'].values
            lows = data['low'].values
            closes = data['close'].values
            
            # Improved support/resistance detection algorithm
            # Uses both price and volume data with clustering to find significant levels
            
            # Parameters for detection
            window_size = 10
            max_levels = 3
            min_level_distance = 0.01  # Minimum 1% distance between levels
            
            # Find potential resistance levels using highs and price rejection
            resistance_points = []
            for i in range(window_size, len(highs) - window_size):
                # Look for price peaks (local maxima)
                if all(highs[i] >= highs[i-j] for j in range(1, window_size)) and \
                   all(highs[i] >= highs[i+j] for j in range(1, window_size)):
                    
                    # Check for strong price rejection (indicates stronger resistance)
                    if closes[i] < highs[i] * 0.995:  # Close at least 0.5% below high
                        strength = sum(1 for j in range(max(0, i-20), min(i+20, len(highs))) 
                                     if abs(highs[j] - highs[i]) / highs[i] < 0.005)
                        
                        resistance_points.append((highs[i], strength, i))
            
            # Find potential support levels using lows and price rejection
            support_points = []
            for i in range(window_size, len(lows) - window_size):
                # Look for price troughs (local minima)
                if all(lows[i] <= lows[i-j] for j in range(1, window_size)) and \
                   all(lows[i] <= lows[i+j] for j in range(1, window_size)):
                    
                    # Check for strong price rejection (indicates stronger support)
                    if closes[i] > lows[i] * 1.005:  # Close at least 0.5% above low
                        strength = sum(1 for j in range(max(0, i-20), min(i+20, len(lows))) 
                                     if abs(lows[j] - lows[i]) / lows[i] < 0.005)
                        
                        support_points.append((lows[i], strength, i))
            
            # Sort by strength (more touches = stronger level)
            resistance_points.sort(key=lambda x: x[1], reverse=True)
            support_points.sort(key=lambda x: x[1], reverse=True)
            
            # Choose top N levels but ensure they're not too close to each other
            def filter_nearby_levels(points, min_distance):
                if not points:
                    return []
                
                # Start with the strongest level
                filtered = [points[0]]
                
                # Add other levels only if they're not too close to existing ones
                for level, strength, idx in points[1:]:
                    if all(abs(level - existing[0]) / existing[0] > min_distance for existing in filtered):
                        filtered.append((level, strength, idx))
                    
                    # Stop once we have enough levels
                    if len(filtered) >= max_levels:
                        break
                
                return filtered
            
            # Get filtered levels
            resistance_levels = filter_nearby_levels(resistance_points, min_level_distance)
            support_levels = filter_nearby_levels(support_points, min_level_distance)
            
            # Plot resistance levels with TradingView-style visuals
            for level, strength, idx in resistance_levels:
                # Strength-based styling
                alpha = min(0.9, 0.5 + strength * 0.05)
                linewidth = min(2.0, 1.0 + strength * 0.1)
                
                # Draw the horizontal line
                ax.axhline(y=level, color=self.chart_style['resistance'], linestyle='-', 
                           linewidth=linewidth, alpha=alpha)
                
                # Mark the point where the level was detected
                ax.scatter(idx, level, color=self.chart_style['resistance'], marker='_', 
                          s=100, linewidth=2, alpha=0.8)
                
                # Add label with appropriate styling
                ax.text(len(data) * 0.99, level, f'R ({level:.8f})', 
                        color=self.chart_style['resistance'], fontsize=9, 
                        weight='bold' if strength > 3 else 'normal',
                        verticalalignment='bottom', horizontalalignment='right')
            
            # Plot support levels with TradingView-style visuals
            for level, strength, idx in support_levels:
                # Strength-based styling
                alpha = min(0.9, 0.5 + strength * 0.05)
                linewidth = min(2.0, 1.0 + strength * 0.1)
                
                # Draw the horizontal line
                ax.axhline(y=level, color=self.chart_style['support'], linestyle='-', 
                           linewidth=linewidth, alpha=alpha)
                
                # Mark the point where the level was detected
                ax.scatter(idx, level, color=self.chart_style['support'], marker='_', 
                          s=100, linewidth=2, alpha=0.8)
                
                # Add label with appropriate styling
                ax.text(len(data) * 0.99, level, f'S ({level:.8f})', 
                        color=self.chart_style['support'], fontsize=9,
                        weight='bold' if strength > 3 else 'normal',
                        verticalalignment='top', horizontalalignment='right')
            
            # Add level zones if we can detect multiple touches at similar prices
            def find_zones(points, window=0.005):
                if len(points) < 2:
                    return []
                
                # Sort by price level
                sorted_points = sorted(points, key=lambda x: x[0])
                
                zones = []
                current_zone = [sorted_points[0]]
                
                for i in range(1, len(sorted_points)):
                    level, strength, idx = sorted_points[i]
                    prev_level = current_zone[-1][0]
                    
                    # If this level is close to the previous one, add to current zone
                    if abs(level - prev_level) / prev_level < window:
                        current_zone.append((level, strength, idx))
                    else:
                        # If zone has multiple points, add it
                        if len(current_zone) > 1:
                            zones.append(current_zone)
                        # Start a new zone
                        current_zone = [(level, strength, idx)]
                
                # Add the last zone if it has multiple points
                if len(current_zone) > 1:
                    zones.append(current_zone)
                
                return zones
            
            # Find and draw support and resistance zones
            support_zones = find_zones(support_points)
            resistance_zones = find_zones(resistance_points)
            
            # Draw the zones as shaded areas
            for zone in resistance_zones:
                if len(zone) < 2:
                    continue
                    
                # Calculate zone boundaries
                upper = max(point[0] for point in zone)
                lower = min(point[0] for point in zone)
                
                # Draw zone if it's not too wide
                if (upper - lower) / lower < 0.02:  # Zone width < 2%
                    ax.axhspan(lower, upper, alpha=0.1, color=self.chart_style['resistance'])
            
            for zone in support_zones:
                if len(zone) < 2:
                    continue
                    
                # Calculate zone boundaries
                upper = max(point[0] for point in zone)
                lower = min(point[0] for point in zone)
                
                # Draw zone if it's not too wide
                if (upper - lower) / lower < 0.02:  # Zone width < 2%
                    ax.axhspan(lower, upper, alpha=0.1, color=self.chart_style['support'])
        
        except Exception as e:
            logger.error(f"Error adding support/resistance: {str(e)}")
    
    def _add_signal_markers(self, ax, dates, signal):
        """Add trading signal markers"""
        try:
            if signal and signal.get('signal') in ['LONG', 'SHORT']:
                # Get index for most recent date
                latest_idx = len(dates) - 1
                
                # Entry point
                entry_price = signal.get('entry')
                if entry_price:
                    marker = '^' if signal.get('signal') == 'LONG' else 'v'
                    color = 'lime' if signal.get('signal') == 'LONG' else 'red'
                    ax.scatter(latest_idx, entry_price, s=150, marker=marker, color=color, 
                              edgecolors='white', linewidth=1.5, zorder=5, 
                              label='Entry Point')
                    ax.text(latest_idx, entry_price * 1.003, 'Entry', fontsize=10, 
                           color=self.chart_style['text'], ha='center')
                
                # Stop Loss
                stop_loss = signal.get('stop_loss')
                if stop_loss:
                    ax.scatter(latest_idx, stop_loss, s=120, marker='x', color='red', 
                              linewidth=2, zorder=5, label='Stop Loss')
                    ax.text(latest_idx, stop_loss * 0.997, 'SL', fontsize=10, 
                           color='red', ha='center')
                
                # Take Profit
                take_profit = signal.get('take_profit')
                if take_profit:
                    ax.scatter(latest_idx, take_profit, s=120, marker='*', color='lime', 
                              linewidth=2, zorder=5, label='Take Profit')
                    ax.text(latest_idx, take_profit * 1.002, 'TP', fontsize=10, 
                           color='lime', ha='center')
                
                # Draw vertical line at entry point
                ax.axvline(x=latest_idx, color='gray', linestyle='--', alpha=0.7, linewidth=1)
                
                # Draw horizontal lines for stop loss and take profit
                if stop_loss:
                    ax.axhline(y=stop_loss, color='red', linestyle='--', alpha=0.7, linewidth=1)
                if take_profit:
                    ax.axhline(y=take_profit, color='lime', linestyle='--', alpha=0.7, linewidth=1)
        
        except Exception as e:
            logger.error(f"Error adding signal markers: {str(e)}")
    
    def _add_trend_lines(self, ax, data, signal, dates):
        """Add trend lines based on price action in TradingView style"""
        try:
            if signal and signal.get('signal') in ['LONG', 'SHORT']:
                # Get price data
                highs = data['high'].values
                lows = data['low'].values
                closes = data['close'].values
                
                # Find significant swing points for trend line anchoring
                if signal.get('signal') == 'LONG':
                    # For long signals, we want to identify an uptrend
                    # Look for significant lows that form an ascending pattern
                    
                    # Step 1: Find local minima (potential swing lows)
                    window_size = 5
                    swing_lows = []
                    
                    for i in range(window_size, len(lows) - window_size):
                        if all(lows[i] <= lows[i-j] for j in range(1, window_size)) and \
                           all(lows[i] <= lows[i+j] for j in range(1, window_size)):
                            swing_lows.append((i, lows[i]))
                    
                    # Step 2: If we have at least 2 swing lows, draw a trend line through them
                    if len(swing_lows) >= 2:
                        # Sort by time (index)
                        swing_lows.sort(key=lambda x: x[0])
                        
                        # Look for pairs of swing lows that form an ascending line
                        for i in range(len(swing_lows) - 1):
                            idx1, price1 = swing_lows[i]
                            
                            # Find the next swing low that is higher (for uptrend)
                            for j in range(i + 1, len(swing_lows)):
                                idx2, price2 = swing_lows[j]
                                
                                # For uptrend, we want later swing lows to be higher
                                if price2 >= price1 and idx2 - idx1 >= 10:  # Ensure some distance
                                    # Draw a line connecting these two swing points
                                    ax.plot([idx1, idx2], [price1, price2], 
                                          color='#2ecc71', linewidth=2, linestyle='-', alpha=0.8,
                                          label='Support Trend')
                                    
                                    # Calculate slope for extending the line
                                    if idx2 > idx1:  # Avoid division by zero
                                        slope = (price2 - price1) / (idx2 - idx1)
                                        
                                        # Extend the trend line to the right (future)
                                        future_idx = len(closes) - 1 + 10  # Extend 10 candles into future
                                        future_price = price2 + slope * (future_idx - idx2)
                                        
                                        ax.plot([idx2, future_idx], [price2, future_price], 
                                              color='#2ecc71', linewidth=2, linestyle='--', alpha=0.6)
                                        
                                        # Label the trend line
                                        ax.text(idx2, price2, 'Support Trend', 
                                               color='#2ecc71', fontsize=10, 
                                               verticalalignment='bottom', horizontalalignment='right')
                                    
                                    break  # Use the first valid pair found
                            
                            # If we've drawn a trend line, exit the loop
                            if j < len(swing_lows):
                                break
                    
                    # Add a supplementary trend line using regression
                    lookback = min(30, len(closes) - 1)
                    x = np.array(range(len(closes) - lookback, len(closes)))
                    y = closes[-lookback:]
                    
                    # Calculate trend line with linear regression
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    
                    # Only draw if the slope is positive (indicating uptrend)
                    if z[0] > 0:
                        # Draw trend line
                        ax.plot(x, p(x), color='#3498db', linewidth=1.5, 
                               linestyle='-', alpha=0.7, label='Regression Trend')
                        
                        # Extend trend line into the future
                        future_x = np.array(range(len(closes) - 1, len(closes) + 10))
                        ax.plot(future_x, p(future_x), color='#3498db', linewidth=1.5, 
                               linestyle='--', alpha=0.5)
                
                else:  # SHORT signal
                    # For short signals, we want to identify a downtrend
                    # Look for significant highs that form a descending pattern
                    
                    # Step 1: Find local maxima (potential swing highs)
                    window_size = 5
                    swing_highs = []
                    
                    for i in range(window_size, len(highs) - window_size):
                        if all(highs[i] >= highs[i-j] for j in range(1, window_size)) and \
                           all(highs[i] >= highs[i+j] for j in range(1, window_size)):
                            swing_highs.append((i, highs[i]))
                    
                    # Step 2: If we have at least 2 swing highs, draw a trend line through them
                    if len(swing_highs) >= 2:
                        # Sort by time (index)
                        swing_highs.sort(key=lambda x: x[0])
                        
                        # Look for pairs of swing highs that form a descending line
                        for i in range(len(swing_highs) - 1):
                            idx1, price1 = swing_highs[i]
                            
                            # Find the next swing high that is lower (for downtrend)
                            for j in range(i + 1, len(swing_highs)):
                                idx2, price2 = swing_highs[j]
                                
                                # For downtrend, we want later swing highs to be lower
                                if price2 <= price1 and idx2 - idx1 >= 10:  # Ensure some distance
                                    # Draw a line connecting these two swing points
                                    ax.plot([idx1, idx2], [price1, price2], 
                                          color='#e74c3c', linewidth=2, linestyle='-', alpha=0.8,
                                          label='Resistance Trend')
                                    
                                    # Calculate slope for extending the line
                                    if idx2 > idx1:  # Avoid division by zero
                                        slope = (price2 - price1) / (idx2 - idx1)
                                        
                                        # Extend the trend line to the right (future)
                                        future_idx = len(closes) - 1 + 10  # Extend 10 candles into future
                                        future_price = price2 + slope * (future_idx - idx2)
                                        
                                        ax.plot([idx2, future_idx], [price2, future_price], 
                                              color='#e74c3c', linewidth=2, linestyle='--', alpha=0.6)
                                        
                                        # Label the trend line
                                        ax.text(idx2, price2, 'Resistance Trend', 
                                               color='#e74c3c', fontsize=10, 
                                               verticalalignment='top', horizontalalignment='right')
                                    
                                    break  # Use the first valid pair found
                            
                            # If we've drawn a trend line, exit the loop
                            if j < len(swing_highs):
                                break
                    
                    # Add a supplementary trend line using regression
                    lookback = min(30, len(closes) - 1)
                    x = np.array(range(len(closes) - lookback, len(closes)))
                    y = closes[-lookback:]
                    
                    # Calculate trend line with linear regression
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    
                    # Only draw if the slope is negative (indicating downtrend)
                    if z[0] < 0:
                        # Draw trend line
                        ax.plot(x, p(x), color='#9b59b6', linewidth=1.5, 
                               linestyle='-', alpha=0.7, label='Regression Trend')
                        
                        # Extend trend line into the future
                        future_x = np.array(range(len(closes) - 1, len(closes) + 10))
                        ax.plot(future_x, p(future_x), color='#9b59b6', linewidth=1.5, 
                               linestyle='--', alpha=0.5)
                
                # Additionally, draw a channel if we can find appropriate swing points
                try:
                    if signal.get('signal') == 'LONG':
                        # For uptrend, draw a channel parallel to the support trend line
                        # through the highest high in the period
                        
                        # Find the highest high
                        highest_idx = np.argmax(highs)
                        highest_price = highs[highest_idx]
                        
                        # Find the last two valid swing lows that we used for the trend line
                        if len(swing_lows) >= 2:
                            idx1, price1 = swing_lows[-2]
                            idx2, price2 = swing_lows[-1]
                            
                            if idx2 > idx1 and price2 >= price1:  # Ensure valid trend
                                # Calculate slope of trend line
                                slope = (price2 - price1) / (idx2 - idx1)
                                
                                # Calculate vertical distance from the trend line to the highest high
                                trend_line_price_at_highest = price1 + slope * (highest_idx - idx1)
                                channel_height = highest_price - trend_line_price_at_highest
                                
                                if channel_height > 0:  # Ensure positive channel height
                                    # Draw upper channel line - parallel to trend line
                                    channel_start_price = price1 + channel_height
                                    channel_end_price = price2 + channel_height
                                    
                                    # Draw channel line
                                    ax.plot([idx1, idx2], [channel_start_price, channel_end_price], 
                                          color='#2ecc71', linewidth=1.5, linestyle='-.', alpha=0.6)
                                    
                                    # Extend channel line
                                    future_idx = len(closes) - 1 + 10
                                    future_price = channel_end_price + slope * (future_idx - idx2)
                                    
                                    ax.plot([idx2, future_idx], [channel_end_price, future_price], 
                                          color='#2ecc71', linewidth=1.5, linestyle='--', alpha=0.4)
                    
                    else:  # SHORT signal
                        # For downtrend, draw a channel parallel to the resistance trend line
                        # through the lowest low in the period
                        
                        # Find the lowest low
                        lowest_idx = np.argmin(lows)
                        lowest_price = lows[lowest_idx]
                        
                        # Find the last two valid swing highs that we used for the trend line
                        if len(swing_highs) >= 2:
                            idx1, price1 = swing_highs[-2]
                            idx2, price2 = swing_highs[-1]
                            
                            if idx2 > idx1 and price2 <= price1:  # Ensure valid trend
                                # Calculate slope of trend line
                                slope = (price2 - price1) / (idx2 - idx1)
                                
                                # Calculate vertical distance from the trend line to the lowest low
                                trend_line_price_at_lowest = price1 + slope * (lowest_idx - idx1)
                                channel_height = trend_line_price_at_lowest - lowest_price
                                
                                if channel_height > 0:  # Ensure positive channel height
                                    # Draw lower channel line - parallel to trend line
                                    channel_start_price = price1 - channel_height
                                    channel_end_price = price2 - channel_height
                                    
                                    # Draw channel line
                                    ax.plot([idx1, idx2], [channel_start_price, channel_end_price], 
                                          color='#e74c3c', linewidth=1.5, linestyle='-.', alpha=0.6)
                                    
                                    # Extend channel line
                                    future_idx = len(closes) - 1 + 10
                                    future_price = channel_end_price + slope * (future_idx - idx2)
                                    
                                    ax.plot([idx2, future_idx], [channel_end_price, future_price], 
                                          color='#e74c3c', linewidth=1.5, linestyle='--', alpha=0.4)
                
                except Exception as channel_e:
                    # If channel drawing fails, just log and continue
                    logger.info(f"Channel drawing skipped: {str(channel_e)}")
        
        except Exception as e:
            logger.error(f"Error adding trend lines: {str(e)}")
    
    def _format_time_axis(self, price_ax, volume_ax, dates, timeframe):
        """Format the time axis based on the timeframe"""
        # Set x-axis limits
        price_ax.set_xlim(-1, len(dates))
        
        # Hide x-axis labels on price chart
        price_ax.set_xticklabels([])
        
        # Format volume chart x-axis labels
        volume_ax.set_xlabel('Time', fontsize=10, color=self.chart_style['text'])
        
        # Only show a reasonable number of x-tick labels to avoid overcrowding
        max_ticks = min(10, len(dates))
        
        if len(dates) <= max_ticks:
            # If we have few data points, show all
            tick_locations = range(0, len(dates))
        else:
            # Otherwise, show evenly spaced ticks
            tick_stride = len(dates) // max_ticks
            tick_locations = range(0, len(dates), tick_stride)
        
        # Set x-ticks and format based on timeframe
        has_datetime = False
        if hasattr(dates, 'iloc'):
            has_datetime = len(dates) > 0 and isinstance(dates.iloc[0], (datetime, np.datetime64))
        else:
            has_datetime = len(dates) > 0 and isinstance(dates[0], (datetime, np.datetime64))
            
        if has_datetime:
            # If we have datetime objects, format them appropriately
            if timeframe in ['1m', '5m', '15m', '30m']:
                date_format = '%H:%M'
            elif timeframe in ['1h', '4h', '6h']:
                date_format = '%m-%d %H:%M'
            elif timeframe in ['1d', '3d', '1w']:
                date_format = '%Y-%m-%d'
            else:
                date_format = '%Y-%m-%d'
            
            # Format dates for display
            date_labels = [dates.iloc[i].strftime(date_format) if hasattr(dates, 'iloc') else dates[i].strftime(date_format) for i in tick_locations if i < len(dates)]
            
            # Set ticks and labels on volume axis
            volume_ax.set_xticks(list(tick_locations))
            volume_ax.set_xticklabels(date_labels, rotation=45, ha='right')
        else:
            # If not datetime objects, just use index numbers
            volume_ax.set_xticks(list(tick_locations))
            volume_ax.set_xticklabels([str(i) for i in tick_locations], rotation=45)
    
    def _add_watermark(self, fig):
        """Add watermark to the chart"""
        fig.text(0.5, 0.5, 'Decoded by Armin', fontsize=40, color='gray',
                ha='center', va='center', alpha=0.1, rotation=45)

def generate_tradingview_style_chart(historical_data, symbol, signal=None, timeframe='5m'):
    """
    Generate a TradingView-style chart for the given data and signal
    
    Args:
        historical_data (pd.DataFrame): Historical OHLCV data
        symbol (str): Trading pair symbol
        signal (dict): Trading signal information
        timeframe (str): Timeframe (default: 5m)
        
    Returns:
        str: Path to the saved chart image and base64 encoded image
    """
    try:
        # Ensure the chart directory exists
        os.makedirs('static/images/charts', exist_ok=True)
        
        # Create chart generator
        generator = ChartGenerator()
        
        # Override chart style with valid colors
        generator.chart_style.update({
            'background': '#1e222d',
            'text': '#e0e0e0',
            'grid': '#2a2e39',
            'spine': '#2a2e39',
            'candle_up': '#26a69a',
            'candle_down': '#ef5350',
            'volume_up': '#26a69a',
            'volume_down': '#ef5350',
            'ma_fast': '#2196f3',
            'ma_slow': '#ff9800',
            'fib_lines': ['#f44336', '#9c27b0', '#3f51b5', '#03a9f4', '#009688'],
            'support': '#2e7d32',
            'resistance': '#c62828',
            'annotation': '#ffeb3b'
        })
        
        # Generate chart
        chart_path, base64_img = generator.generate_chart(
            historical_data=historical_data,
            symbol=symbol,
            signal=signal,
            timeframe=timeframe
        )
        
        return chart_path, base64_img
    
    except Exception as e:
        logger.error(f"Error generating TradingView chart: {str(e)}")
        # Add detailed error logging
        import traceback
        logger.error(traceback.format_exc())
        # Return a default image path in case of error
        return 'static/images/sample_analysis.svg', ''