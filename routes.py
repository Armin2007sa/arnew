import os
import json
import logging
from flask import render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime
import pandas as pd
import numpy as np

from arnew.main import app, db
from arnew.utils.helpers import get_api_data, calculate_profit, validate_api_key
from arnew.utils.data_fetcher import DataFetcher, get_crypto_news, get_top_traded_symbols
from arnew.utils.ai.analysis_generator import generate_detailed_analysis
from arnew.trading_bot.bot import TradingBot
from arnew.trading_bot.ai_models.ai_model import generate_ai_signal
from arnew.trading_bot.chart_generator import generate_tradingview_style_chart
from arnew.main import User, TradingStrategy, TradeHistory, ApiKey

# Create data fetcher instance
data_fetcher = DataFetcher()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        
        existing_email = User.query.filter_by(email=email).first()
        if existing_email:
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))
        
        password_hash = generate_password_hash(password)
        
        new_user = User(username=username, email=email, password_hash=password_hash)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful, please log in', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    strategies = TradingStrategy.query.filter_by(user_id=current_user.id).all()
    api_keys = ApiKey.query.filter_by(user_id=current_user.id).all()
    
    profit, profit_history = calculate_profit(current_user.id)
    
    active_strategies = TradingStrategy.query.filter_by(user_id=current_user.id, is_active=True).count()
    completed_trades = TradeHistory.query.filter_by(user_id=current_user.id).count()
    
    exchange = "Not set"
    if api_keys:
        exchange = api_keys[0].exchange
    
    # Get recent news
    news = get_crypto_news(limit=3)
    
    # Get top trading pairs
    top_pairs = get_top_traded_symbols(exchange, limit=5)
    
    return render_template('dashboard.html', 
                          strategies=strategies,
                          api_configured=(len(api_keys) > 0),
                          profit=profit,
                          profit_history=json.dumps(profit_history),
                          active_strategies=active_strategies,
                          completed_trades=completed_trades,
                          exchange=exchange,
                          news=news,
                          top_pairs=top_pairs)

@app.route('/setup_api', methods=['GET', 'POST'])
@login_required
def setup_api():
    if request.method == 'POST':
        exchange = request.form.get('exchange')
        api_key = request.form.get('api_key')
        api_secret = request.form.get('api_secret')
        
        # Validate API key
        is_valid, message = validate_api_key(api_key, api_secret, exchange)
        
        if is_valid:
            # Check if user already has an API key for this exchange
            existing_key = ApiKey.query.filter_by(user_id=current_user.id, exchange=exchange).first()
            
            if existing_key:
                existing_key.api_key = api_key
                existing_key.api_secret = api_secret
                existing_key.created_at = datetime.utcnow()
            else:
                new_api_key = ApiKey(
                    user_id=current_user.id,
                    exchange=exchange,
                    api_key=api_key,
                    api_secret=api_secret
                )
                db.session.add(new_api_key)
            
            db.session.commit()
            flash(f'API key for {exchange} set successfully', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash(f'API key validation failed: {message}', 'danger')
    
    return render_template('setup_api.html')

@app.route('/create_strategy', methods=['GET', 'POST'])
@login_required
def create_strategy():
    if request.method == 'POST':
        name = request.form.get('name')
        strategy_type = request.form.get('strategy_type')
        asset = request.form.get('asset')
        time_frame = request.form.get('time_frame')
        risk_level = request.form.get('risk_level')
        
        parameters = {
            'asset': asset,
            'time_frame': time_frame,
            'risk_level': risk_level
        }
        
        new_strategy = TradingStrategy(
            user_id=current_user.id,
            name=name,
            strategy_type=strategy_type,
            parameters=json.dumps(parameters),
            is_active=False
        )
        
        db.session.add(new_strategy)
        db.session.commit()
        
        flash(f'Strategy "{name}" created successfully', 'success')
        return redirect(url_for('dashboard'))
    
    # Get available pairs from API
    try:
        symbols = data_fetcher.get_supported_symbols(base_currency='USDT')
        symbols = sorted(symbols)
    except:
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
    
    return render_template('create_strategy.html', symbols=symbols)

@app.route('/activate_strategy/<int:strategy_id>')
@login_required
def activate_strategy(strategy_id):
    strategy = TradingStrategy.query.get_or_404(strategy_id)
    
    # Ensure the strategy belongs to the current user
    if strategy.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard'))
    
    strategy.is_active = True
    db.session.commit()
    
    # Start the trading bot in a separate thread
    bot = TradingBot(strategy_id)
    bot.start()
    
    flash(f'Strategy "{strategy.name}" activated', 'success')
    return redirect(url_for('dashboard'))

@app.route('/deactivate_strategy/<int:strategy_id>')
@login_required
def deactivate_strategy(strategy_id):
    strategy = TradingStrategy.query.get_or_404(strategy_id)
    
    # Ensure the strategy belongs to the current user
    if strategy.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard'))
    
    strategy.is_active = False
    db.session.commit()
    
    # Stop the trading bot
    bot = TradingBot(strategy_id)
    bot.stop()
    
    flash(f'Strategy "{strategy.name}" deactivated', 'success')
    return redirect(url_for('dashboard'))

@app.route('/get_market_data')
@login_required
def get_market_data():
    symbol = request.args.get('symbol', 'BTC/USDT')
    timeframe = request.args.get('timeframe', '1h')
    
    try:
        data = get_api_data(symbol, timeframe)
        
        if data is None:
            return jsonify({'error': 'Failed to fetch market data'}), 400
        
        # Generate chart and save to static folder
        chart_path = generate_tradingview_style_chart(data, symbol, timeframe=timeframe)
        
        # Convert dataframe to dictionary
        data_dict = data.tail(20).reset_index().to_dict(orient='records')
        
        # Generate signal
        signal = generate_ai_signal(data, symbol, timeframe)
        
        # Generate analysis text
        analysis = generate_detailed_analysis(data, signal, symbol, strategy_type="aeai")
        
        return jsonify({
            'data': data_dict,
            'signal': signal,
            'chart_url': chart_path,
            'analysis': analysis
        })
        
    except Exception as e:
        logging.error(f"Error getting market data: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/analyze', methods=['GET', 'POST'])
@login_required
def analyze():
    if request.method == 'POST':
        symbol = request.form.get('symbol', 'BTC/USDT')
        timeframe = request.form.get('timeframe', '5m')
        analysis_type = request.form.get('analysis_type', 'classic')
        
        try:
            # Get historical data
            data = get_api_data(symbol, timeframe)
            
            if data is None:
                flash('Failed to fetch market data', 'danger')
                return redirect(url_for('analyze'))
            
            # Generate signal
            signal = generate_ai_signal(data, symbol, timeframe)
            
            # Generate chart and save to static folder
            chart_path = generate_tradingview_style_chart(data, symbol, signal=signal, timeframe=timeframe)
            
            # Generate detailed analysis
            analysis = generate_detailed_analysis(data, signal, symbol, strategy_type=analysis_type)
            
            # Get news related to the symbol
            news = get_crypto_news(symbol.split('/')[0], limit=3)
            
            session['last_analysis'] = {
                'symbol': symbol,
                'timeframe': timeframe,
                'signal': signal,
                'chart_path': chart_path,
                'analysis': analysis,
                'news': news,
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return redirect(url_for('results'))
            
        except Exception as e:
            logging.error(f"Error in analysis: {e}")
            flash(f'Error in analysis: {str(e)}', 'danger')
    
    # Get available pairs from API
    try:
        symbols = data_fetcher.get_supported_symbols(base_currency='USDT')
        symbols = sorted(symbols)
    except:
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
    
    timeframes = ['5m', '15m', '30m', '1h', '4h', '1d']
    analysis_types = [
        {'id': 'classic', 'name': 'Classic Technical Analysis'},
        {'id': 'aeai', 'name': 'AEai (Aladdin) Style Analysis'},
        {'id': 'modern', 'name': 'Modern Price Action Analysis'}
    ]
    
    return render_template('analyze.html', 
                         symbols=symbols, 
                         timeframes=timeframes, 
                         analysis_types=analysis_types)

@app.route('/results')
@login_required
def results():
    if 'last_analysis' not in session:
        flash('No analysis results available', 'warning')
        return redirect(url_for('analyze'))
    
    last_analysis = session['last_analysis']
    
    return render_template('results.html', analysis=last_analysis)

@app.route('/multi_analyze', methods=['GET', 'POST'])
@login_required
def multi_analyze():
    if request.method == 'POST':
        exchange = request.form.get('exchange', 'mexc')
        timeframe = request.form.get('timeframe', '5m')
        analysis_type = request.form.get('analysis_type', 'aeai')
        
        try:
            # Get top traded symbols
            top_symbols = get_top_traded_symbols(exchange, limit=6)
            symbol_names = [symbol['symbol'] for symbol in top_symbols]
            
            # Initialize data fetcher
            data_fetcher = DataFetcher(exchange=exchange)
            
            results = []
            
            for symbol in symbol_names:
                # Get historical data
                data = data_fetcher.fetch_historical_data(symbol, timeframe=timeframe)
                
                if data is None:
                    continue
                
                # Generate signal
                signal = generate_ai_signal(data, symbol, timeframe)
                
                # Only include signals with clear direction
                if signal.get('signal') in ['LONG', 'SHORT']:
                    # Generate chart
                    chart_path = generate_tradingview_style_chart(data, symbol, signal=signal, timeframe=timeframe)
                    
                    # Generate analysis
                    analysis = generate_detailed_analysis(data, signal, symbol, strategy_type=analysis_type)
                    
                    results.append({
                        'symbol': symbol,
                        'signal': signal,
                        'chart_path': chart_path,
                        'analysis': analysis
                    })
            
            if not results:
                flash('No trading signals found', 'warning')
                return redirect(url_for('multi_analyze'))
            
            session['multi_results'] = {
                'results': results,
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return redirect(url_for('multi_results'))
            
        except Exception as e:
            logging.error(f"Error in multi-analysis: {e}")
            flash(f'Error in analysis: {str(e)}', 'danger')
    
    exchanges = ['binance', 'mexc', 'kucoin']
    timeframes = ['5m', '15m', '30m', '1h', '4h', '1d']
    analysis_types = [
        {'id': 'aeai', 'name': 'AEai (Aladdin) Style Analysis'},
        {'id': 'classic', 'name': 'Classic Technical Analysis'},
        {'id': 'modern', 'name': 'Modern Price Action Analysis'}
    ]
    
    return render_template('multi_analyze.html',
                         exchanges=exchanges,
                         timeframes=timeframes,
                         analysis_types=analysis_types)

@app.route('/multi_results')
@login_required
def multi_results():
    if 'multi_results' not in session:
        flash('No multi-analysis results available', 'warning')
        return redirect(url_for('multi_analyze'))
    
    multi_results = session['multi_results']
    
    return render_template('multi_results.html', results=multi_results)