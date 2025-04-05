import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from datetime import datetime, timedelta
import requests
import json

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///trading_bot.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)
migrate = Migrate(app, db)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from models import User, TradingStrategy, TradeHistory, ApiKey
from utils.helpers import get_api_data, calculate_profit, validate_api_key
from trading_bot.bot import TradingBot

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

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
        
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password_hash=hashed_password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
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
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'success')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    user_strategies = TradingStrategy.query.filter_by(user_id=current_user.id).all()
    trade_history = TradeHistory.query.filter_by(user_id=current_user.id).order_by(TradeHistory.timestamp.desc()).limit(10).all()
    
    api_keys = ApiKey.query.filter_by(user_id=current_user.id).all()
    has_api_keys = len(api_keys) > 0
    
    total_profit = calculate_profit(current_user.id)
    
    return render_template(
        'dashboard.html',
        strategies=user_strategies,
        trade_history=trade_history,
        has_api_keys=has_api_keys,
        total_profit=total_profit
    )

@app.route('/api/setup', methods=['POST'])
@login_required
def setup_api():
    api_key = request.form.get('api_key')
    api_secret = request.form.get('api_secret')
    exchange = request.form.get('exchange')
    
    if not api_key or not api_secret or not exchange:
        flash('All fields are required', 'danger')
        return redirect(url_for('dashboard'))
    
    is_valid = validate_api_key(api_key, api_secret, exchange)
    
    if not is_valid:
        flash('Invalid API credentials', 'danger')
        return redirect(url_for('dashboard'))
    
    existing_key = ApiKey.query.filter_by(user_id=current_user.id, exchange=exchange).first()
    
    if existing_key:
        existing_key.api_key = api_key
        existing_key.api_secret = api_secret
        db.session.commit()
        flash('API key updated successfully', 'success')
    else:
        new_api_key = ApiKey(
            user_id=current_user.id,
            exchange=exchange,
            api_key=api_key,
            api_secret=api_secret
        )
        db.session.add(new_api_key)
        db.session.commit()
        flash('API key added successfully', 'success')
    
    return redirect(url_for('dashboard'))

@app.route('/api/strategy/create', methods=['POST'])
@login_required
def create_strategy():
    name = request.form.get('strategy_name')
    strategy_type = request.form.get('strategy_type')
    parameters = json.dumps({
        'asset': request.form.get('asset'),
        'time_frame': request.form.get('time_frame'),
        'risk_level': request.form.get('risk_level'),
        'initial_investment': float(request.form.get('initial_investment', 0))
    })
    
    new_strategy = TradingStrategy(
        user_id=current_user.id,
        name=name,
        strategy_type=strategy_type,
        parameters=parameters,
        is_active=False
    )
    
    db.session.add(new_strategy)
    db.session.commit()
    
    flash('Strategy created successfully', 'success')
    return redirect(url_for('dashboard'))

@app.route('/api/strategy/<int:strategy_id>/activate', methods=['POST'])
@login_required
def activate_strategy(strategy_id):
    strategy = TradingStrategy.query.filter_by(id=strategy_id, user_id=current_user.id).first()
    
    if not strategy:
        flash('Strategy not found', 'danger')
        return redirect(url_for('dashboard'))
    
    api_key = ApiKey.query.filter_by(user_id=current_user.id).first()
    
    if not api_key:
        flash('Please set up API keys first', 'danger')
        return redirect(url_for('dashboard'))
    
    strategy.is_active = True
    db.session.commit()
    
    bot = TradingBot(strategy_id)
    bot.start()
    
    flash('Strategy activated successfully', 'success')
    return redirect(url_for('dashboard'))

@app.route('/api/strategy/<int:strategy_id>/deactivate', methods=['POST'])
@login_required
def deactivate_strategy(strategy_id):
    strategy = TradingStrategy.query.filter_by(id=strategy_id, user_id=current_user.id).first()
    
    if not strategy:
        flash('Strategy not found', 'danger')
        return redirect(url_for('dashboard'))
    
    strategy.is_active = False
    db.session.commit()
    
    flash('Strategy deactivated successfully', 'success')
    return redirect(url_for('dashboard'))

@app.route('/api/market-data')
@login_required
def get_market_data():
    symbol = request.args.get('symbol', 'BTC/USDT')
    timeframe = request.args.get('timeframe', '1h')
    
    try:
        data = get_api_data(symbol, timeframe)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return jsonify({"error": str(e)}), 500

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
