import pandas as pd
import numpy as np
import logging

def backtest_strategy(historical_data, strategy):
    """
    Backtest a trading strategy on historical data
    
    Args:
        historical_data (pd.DataFrame): Historical OHLCV data
        strategy (Strategy): Strategy instance
        
    Returns:
        dict: Backtest results
    """
    try:
        # Analyze market data
        analyzed_data = strategy.analyze_market(historical_data)
        
        # Find entry points
        entry_points = strategy.find_entry_points(analyzed_data)
        
        # Initialize backtest variables
        initial_capital = 1000.0
        current_capital = initial_capital
        positions = []
        trades = []
        
        # Simulate trades
        for entry in entry_points:
            idx = entry['index']
            if idx >= len(analyzed_data) - 1:
                continue  # Skip if entry is at the last candle
            
            # Calculate risk parameters
            risk_params = strategy.calculate_risk_parameters(entry, analyzed_data)
            
            # Calculate position size based on risk percentage
            risk_amount = current_capital * (risk_params['risk_percent'] / 100)
            price_distance = abs(risk_params['entry_price'] - risk_params['stop_loss'])
            position_size = risk_amount / price_distance * risk_params['leverage']
            
            # Track entry
            entry_candle = analyzed_data.iloc[idx]
            entry_time = entry_candle.name  # Timestamp is the index
            entry_price = risk_params['entry_price']
            direction = entry['direction']
            
            # Simulate trade outcome
            stop_hit = False
            target_hit = False
            exit_idx = None
            exit_price = None
            
            # Look at future candles to see if stop loss or take profit was hit
            for future_idx in range(idx + 1, min(idx + 50, len(analyzed_data))):
                future_candle = analyzed_data.iloc[future_idx]
                
                if direction == 'long':
                    # Check if stop loss was hit
                    if future_candle['low'] <= risk_params['stop_loss']:
                        stop_hit = True
                        exit_idx = future_idx
                        exit_price = risk_params['stop_loss']
                        break
                    
                    # Check if take profit was hit
                    if future_candle['high'] >= risk_params['take_profit']:
                        target_hit = True
                        exit_idx = future_idx
                        exit_price = risk_params['take_profit']
                        break
                
                else:  # Short
                    # Check if stop loss was hit
                    if future_candle['high'] >= risk_params['stop_loss']:
                        stop_hit = True
                        exit_idx = future_idx
                        exit_price = risk_params['stop_loss']
                        break
                    
                    # Check if take profit was hit
                    if future_candle['low'] <= risk_params['take_profit']:
                        target_hit = True
                        exit_idx = future_idx
                        exit_price = risk_params['take_profit']
                        break
            
            # If neither stop nor target was hit, close at the last candle
            if not stop_hit and not target_hit:
                exit_idx = min(idx + 49, len(analyzed_data) - 1)
                exit_candle = analyzed_data.iloc[exit_idx]
                exit_price = exit_candle['close']
            
            # Calculate profit/loss
            if direction == 'long':
                pnl = (exit_price - entry_price) / entry_price * position_size * risk_params['leverage']
            else:  # Short
                pnl = (entry_price - exit_price) / entry_price * position_size * risk_params['leverage']
            
            # Update capital
            current_capital += pnl
            
            # Record trade
            trades.append({
                'entry_time': entry_time,
                'exit_time': analyzed_data.iloc[exit_idx].name,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position_size': position_size,
                'leverage': risk_params['leverage'],
                'pnl': pnl,
                'pnl_percent': (pnl / current_capital) * 100,
                'stop_hit': stop_hit,
                'target_hit': target_hit,
                'reasons': entry['reasons']
            })
        
        # Calculate backtest metrics
        if trades:
            wins = sum(1 for t in trades if t['pnl'] > 0)
            losses = sum(1 for t in trades if t['pnl'] <= 0)
            win_rate = wins / len(trades) if trades else 0
            
            total_gain = sum(t['pnl'] for t in trades if t['pnl'] > 0)
            total_loss = sum(t['pnl'] for t in trades if t['pnl'] <= 0)
            avg_win = total_gain / wins if wins else 0
            avg_loss = total_loss / losses if losses else 0
            
            profit_factor = abs(total_gain / total_loss) if total_loss != 0 else float('inf')
            
            # Create equity curve
            equity_curve = [initial_capital]
            for trade in trades:
                equity_curve.append(equity_curve[-1] + trade['pnl'])
            
            # Calculate drawdown
            rolling_max = pd.Series(equity_curve).cummax()
            drawdown = (pd.Series(equity_curve) - rolling_max) / rolling_max * 100
            max_drawdown = drawdown.min()
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            max_drawdown = 0
            equity_curve = [initial_capital]
        
        results = {
            'initial_capital': initial_capital,
            'final_capital': current_capital,
            'pnl': current_capital - initial_capital,
            'return_percent': (current_capital - initial_capital) / initial_capital * 100,
            'trades': trades,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'equity_curve': equity_curve
        }
        
        logging.debug(f"Backtest completed with {len(trades)} trades and {results['return_percent']:.2f}% return")
        return results
        
    except Exception as e:
        logging.error(f"Error in backtesting: {str(e)}")
        raise
