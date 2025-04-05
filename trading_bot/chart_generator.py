import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def generate_tradingview_style_chart(historical_data, symbol, signal=None, timeframe='5m'):
    try:
        os.makedirs('arnew/static/images/charts', exist_ok=True)
        
        data = historical_data.copy()
        if len(data) < 30:
            logger.error("Not enough data points for chart generation")
            return None
        
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        
        dates = data.index.astype('datetime64[ns]')
        
        candle_width = 0.8
        up_color = '#26A69A'
        down_color = '#EF5350'
        volume_up_color = '#26A69A55'
        volume_down_color = '#EF535055'
        ema_colors = ['#B388FF', '#EF6C00', '#42A5F5']
        
        data['body_size'] = abs(data['close'] - data['open'])
        data['shadow_size'] = data['high'] - data['low']
        data['body_percentage'] = data['body_size'] / data['shadow_size']
        
        for idx, (d, op, hi, lo, cl, vol) in enumerate(zip(
            dates, data['open'], data['high'], data['low'], data['close'], data['volume']
        )):
            color = up_color if cl >= op else down_color
            vol_color = volume_up_color if cl >= op else volume_down_color
            
            ax1.vlines(idx, lo, hi, color=color, linewidth=1)
            
            if data['body_percentage'].iloc[idx] > 0.001:
                rect = Rectangle(
                    (idx - candle_width/2, min(op, cl)),
                    candle_width,
                    abs(op - cl),
                    facecolor=color,
                    edgecolor=color,
                    alpha=1
                )
                ax1.add_patch(rect)
            
            ax2.bar(idx, vol, width=candle_width, color=vol_color, alpha=0.8)
        
        try:
            ema20 = data['close'].ewm(span=20, adjust=False).mean()
            ema50 = data['close'].ewm(span=50, adjust=False).mean()
            ema100 = data['close'].ewm(span=100, adjust=False).mean()
            
            ax1.plot(range(len(dates)), ema20, color=ema_colors[0], linewidth=1, label='EMA 20')
            ax1.plot(range(len(dates)), ema50, color=ema_colors[1], linewidth=1, label='EMA 50')
            ax1.plot(range(len(dates)), ema100, color=ema_colors[2], linewidth=1, label='EMA 100')
        except Exception as e:
            logger.error(f"Error calculating EMAs: {str(e)}")
        
        # Draw signal
        if signal and signal.get('signal') in ['LONG', 'SHORT']:
            entry = signal.get('entry')
            stop_loss = signal.get('stop_loss')
            take_profit = signal.get('take_profit')
            signal_type = signal.get('signal')
            confidence = signal.get('confidence', 0)
            
            # Draw horizontal lines for entry, SL and TP
            if signal_type == 'LONG':
                ax1.axhline(y=entry, color='white', linestyle='--', alpha=0.8, linewidth=1)
                ax1.axhline(y=stop_loss, color='red', linestyle='--', alpha=0.8, linewidth=1)
                ax1.axhline(y=take_profit, color='green', linestyle='--', alpha=0.8, linewidth=1)
                
                # Add labels
                ax1.text(len(dates) - 5, entry, f'Entry: {entry:.8f}', 
                        verticalalignment='bottom', horizontalalignment='right', color='white')
                ax1.text(len(dates) - 5, stop_loss, f'SL: {stop_loss:.8f}', 
                        verticalalignment='bottom', horizontalalignment='right', color='red')
                ax1.text(len(dates) - 5, take_profit, f'TP: {take_profit:.8f}', 
                        verticalalignment='bottom', horizontalalignment='right', color='green')
                
                # Add signal arrow
                ax1.annotate('BUY', xy=(len(dates) - 10, min(data['low'].iloc[-15:])), 
                            xytext=(len(dates) - 20, min(data['low'].iloc[-15:]) * 0.98),
                            arrowprops=dict(facecolor='green', shrink=0.05, width=2, headwidth=8),
                            color='white', fontsize=12, fontweight='bold')
                
            elif signal_type == 'SHORT':
                ax1.axhline(y=entry, color='white', linestyle='--', alpha=0.8, linewidth=1)
                ax1.axhline(y=stop_loss, color='red', linestyle='--', alpha=0.8, linewidth=1)
                ax1.axhline(y=take_profit, color='green', linestyle='--', alpha=0.8, linewidth=1)
                
                # Add labels
                ax1.text(len(dates) - 5, entry, f'Entry: {entry:.8f}', 
                        verticalalignment='top', horizontalalignment='right', color='white')
                ax1.text(len(dates) - 5, stop_loss, f'SL: {stop_loss:.8f}', 
                        verticalalignment='top', horizontalalignment='right', color='red')
                ax1.text(len(dates) - 5, take_profit, f'TP: {take_profit:.8f}', 
                        verticalalignment='top', horizontalalignment='right', color='green')
                
                # Add signal arrow
                ax1.annotate('SELL', xy=(len(dates) - 10, max(data['high'].iloc[-15:])), 
                            xytext=(len(dates) - 20, max(data['high'].iloc[-15:]) * 1.02),
                            arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
                            color='white', fontsize=12, fontweight='bold')
            
            # Add confidence indicator
            ax1.text(5, min(data['low']), f'Signal Confidence: {confidence:.1f}%', 
                    color='white', fontsize=9, alpha=0.8)
        
        # Set titles and labels
        last_price = data['close'].iloc[-1]
        first_price = data['close'].iloc[0]
        price_change = ((last_price - first_price) / first_price) * 100
        period_text = f"Last {len(data)} periods"
        if timeframe == '1m':
            period_text = f"Last {len(data)} minutes"
        elif timeframe == '5m':
            period_text = f"Last {len(data) * 5} minutes"
        elif timeframe == '15m':
            period_text = f"Last {len(data) * 15} minutes"
        elif timeframe == '1h':
            period_text = f"Last {len(data)} hours"
        elif timeframe == '4h':
            period_text = f"Last {len(data) * 4} hours"
        elif timeframe == '1d':
            period_text = f"Last {len(data)} days"
        
        color = 'green' if price_change >= 0 else 'red'
        fig.suptitle(f"{symbol} - {timeframe} - {period_text} - Price: {last_price:.8f} ({price_change:.2f}%)", 
                   color='white', fontsize=12, fontweight='bold')
        
        # Remove axes spines
        for spine in ['top', 'right']:
            ax1.spines[spine].set_visible(False)
            ax2.spines[spine].set_visible(False)
        
        # Set tick parameters
        ax1.tick_params(axis='both', colors='gray', labelsize=8)
        ax2.tick_params(axis='both', colors='gray', labelsize=8)
        
        # Set grid
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Set legend
        ax1.legend(loc='upper left', frameon=False, fontsize=8)
        
        # Set volume title
        ax2.set_title('Volume', fontsize=9, color='gray', pad=2)
        
        # Calculate reasonable tick size
        min_price = min(data['low']) * 0.9995
        max_price = max(data['high']) * 1.0005
        price_range = max_price - min_price
        
        # Set finer y-axis for better price resolution
        yticks = np.linspace(min_price, max_price, 10)
        ax1.set_yticks(yticks)
        
        # Format y-axis labels to show appropriate decimal places
        if last_price < 0.1:
            ax1.set_yticklabels([f'{p:.8f}' for p in yticks])
        elif last_price < 1:
            ax1.set_yticklabels([f'{p:.6f}' for p in yticks])
        elif last_price < 10:
            ax1.set_yticklabels([f'{p:.4f}' for p in yticks])
        elif last_price < 1000:
            ax1.set_yticklabels([f'{p:.2f}' for p in yticks])
        else:
            ax1.set_yticklabels([f'{p:.1f}' for p in yticks])
        
        # Set x-axis labels
        x_tick_positions = np.linspace(0, len(dates) - 1, 10, dtype=int)
        x_tick_labels = [dates[i].strftime('%m-%d %H:%M') if isinstance(dates[i], datetime) else str(dates[i]) for i in x_tick_positions]
        ax2.set_xticks(x_tick_positions)
        ax2.set_xticklabels(x_tick_labels, rotation=45, ha='right', fontsize=8)
        
        plt.tight_layout()
        
        chart_path = f'arnew/static/images/charts/{symbol.replace("/", "_")}_{timeframe}.png'
        plt.savefig(chart_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return chart_path.replace('arnew/', '')
        
    except Exception as e:
        logger.error(f"Error generating chart: {str(e)}")
        return None