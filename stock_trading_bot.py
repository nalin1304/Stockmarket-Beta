
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')
st.set_page_config(
    page_title="🚀 Smart Stock Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        text-align: center;
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        text-align: center;
        margin: 10px 0 0 0;
        font-size: 1.1rem;
    }

    .metric-card {
        background: linear-gradient(145deg, #1e1e30 0%, #2a2a40 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        color: #a0a0a0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stock-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #252540 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 15px;
        margin: 8px 0;
        transition: all 0.3s ease;
    }
    
    .stock-card:hover {
        border-color: #667eea;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.2);
    }

    .signal-buy {
        background: linear-gradient(90deg, #00b894, #00cec9);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .signal-sell {
        background: linear-gradient(90deg, #e74c3c, #c0392b);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .signal-hold {
        background: linear-gradient(90deg, #f39c12, #e67e22);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }

    .score-badge {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        padding: 8px 16px;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
    }

    .portfolio-header {
        background: linear-gradient(90deg, #00b894 0%, #00cec9 100%);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }

    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }

    .dataframe {
        background: #1a1a2e !important;
        border-radius: 10px;
    }
    
    div[data-testid="stDataFrame"] > div {
        background: linear-gradient(145deg, #1e1e30 0%, #2a2a40 100%) !important;
        border-radius: 10px;
        padding: 5px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    div[data-testid="stDataFrame"] table {
        color: white !important;
    }
    
    div[data-testid="stDataFrame"] th {
        background: rgba(102, 126, 234, 0.3) !important;
        color: white !important;
        font-weight: bold !important;
    }
    
    div[data-testid="stDataFrame"] td {
        color: #e0e0e0 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(145deg, #1e1e30, #2a2a40);
        border-radius: 10px;
        padding: 10px 20px;
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }

    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
    }

    input[type="number"] {
        background: #1e1e30 !important;
        color: white !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 5px !important;
    }

    div[data-baseweb="select"] > div {
        background: #1e1e30 !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
    }

    .stAlert {
        border-radius: 10px !important;
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .live-indicator {
        animation: pulse 2s infinite;
        color: #00b894;
    }

    .charges-table {
        background: linear-gradient(145deg, #1e1e30 0%, #2a2a40 100%);
        border-radius: 10px;
        padding: 15px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
</style>
""", unsafe_allow_html=True)

NIFTY_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS",
    "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS", "MPHASIS.NS",
    "AXISBANK.NS", "INDUSINDBK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS",
    "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS",
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS",
    "NTPC.NS", "POWERGRID.NS", "ADANIGREEN.NS", "TATAPOWER.NS", "ONGC.NS",
    "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS", "GODREJCP.NS", "COLPAL.NS",
    "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "COALINDIA.NS", "VEDL.NS",
    "TITAN.NS", "ASIANPAINT.NS", "ULTRACEMCO.NS", "GRASIM.NS", "ADANIPORTS.NS"
]

PORTFOLIO_FILE = "paper_trading_portfolio.json"

STT_BUY = 0.001
STT_SELL = 0.001
STAMP_DUTY = 0.00015
BROKERAGE_PERCENT = 0.0003
GST_RATE = 0.18
EXCHANGE_CHARGES = 0.0000297
SEBI_CHARGES = 0.000001
DP_CHARGES = 13.5

if 'portfolio' not in st.session_state:
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, 'r') as f:
            saved_data = json.load(f)
            st.session_state.portfolio = saved_data.get('holdings', {})
            st.session_state.cash = saved_data.get('cash', 100000)
            st.session_state.trade_history = saved_data.get('trade_history', [])
    else:
        st.session_state.portfolio = {}
        st.session_state.cash = 100000  
        st.session_state.trade_history = []

if 'stock_data' not in st.session_state:
    st.session_state.stock_data = {}

if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = None

if 'num_stocks' not in st.session_state:
    st.session_state.num_stocks = 25

if 'buy_qty_value' not in st.session_state:
    st.session_state.buy_qty_value = 10

st.markdown("""
<div class="main-header">
    <h1>🚀 Smart Stock Trading Bot</h1>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🎛️ Control Panel")
st.sidebar.markdown("---")

current_time = datetime.now()
market_open = current_time.hour >= 9 and current_time.hour < 16
if current_time.hour == 9 and current_time.minute < 15:
    market_open = False
if current_time.hour == 15 and current_time.minute > 30:
    market_open = False

if market_open:
    st.sidebar.markdown("### 🟢 Market Status: **OPEN**")
else:
    st.sidebar.markdown("### 🔴 Market Status: **CLOSED**")

st.sidebar.markdown(f"📅 {current_time.strftime('%d %B %Y')}")
st.sidebar.markdown(f"🕐 {current_time.strftime('%H:%M:%S')}")
st.sidebar.markdown("---")

if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.session_state.stock_data = {}
    st.session_state.last_refresh = datetime.now()
    st.rerun()

num_stocks = st.sidebar.slider("📊 Stocks to Analyze", 10, 50, st.session_state.num_stocks)

# If slider changed, clear cached data to refetch
if num_stocks != st.session_state.num_stocks:
    st.session_state.num_stocks = num_stocks
    st.session_state.stock_data = {}
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 Paper Trading Account")
st.sidebar.metric("Available Cash", f"₹{st.session_state.cash:,.2f}")
st.sidebar.metric("Holdings", f"{len(st.session_state.portfolio)} stocks")


tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Market Dashboard", 
    "🏆 Top Picks", 
    "💼 Portfolio", 
    "📈 Trade Now"
])

with st.spinner("🔄 Fetching live market data from NSE..."):
    if not st.session_state.stock_data:
        stock_data_temp = {}
        symbols_to_fetch = NIFTY_STOCKS[:num_stocks]
        try:
            # Batch download is MUCH faster than individual requests
            batch_data = yf.download(symbols_to_fetch, period="1y", group_by='ticker', progress=False, threads=True)
            for symbol in symbols_to_fetch:
                try:
                    if len(symbols_to_fetch) == 1:
                        hist = batch_data
                    else:
                        hist = batch_data[symbol].dropna()
                    if not hist.empty and len(hist) > 50:
                        stock_data_temp[symbol] = hist
                except:
                    pass
        except:
            # Fallback to individual fetch if batch fails
            for symbol in symbols_to_fetch:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1y")
                    if not hist.empty and len(hist) > 50:
                        stock_data_temp[symbol] = hist
                except:
                    pass
        st.session_state.stock_data = stock_data_temp
        st.session_state.last_refresh = datetime.now()

stock_data = st.session_state.stock_data

if not stock_data:
    st.error("❌ Unable to fetch stock data. Please check your internet connection.")
    st.stop()

stock_analysis = []

for symbol, df in stock_data.items():
    if len(df) < 50:
        continue
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    ma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
    ma_50 = df['Close'].rolling(window=50).mean().iloc[-1]
    current_price = df['Close'].iloc[-1]
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    macd_histogram = macd - signal_line
    current_macd = macd_histogram.iloc[-1]
    bb_middle = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    bb_upper = bb_middle + (2 * bb_std)
    bb_lower = bb_middle - (2 * bb_std)
    bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) if (bb_upper.iloc[-1] - bb_lower.iloc[-1]) != 0 else 0.5
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    stoch_k = ((df['Close'] - low_14) / (high_14 - low_14)) * 100
    stoch_d = stoch_k.rolling(window=3).mean()
    current_stoch_k = stoch_k.iloc[-1] if not pd.isna(stoch_k.iloc[-1]) else 50
    current_stoch_d = stoch_d.iloc[-1] if not pd.isna(stoch_d.iloc[-1]) else 50
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean().iloc[-1]
    atr_percent = (atr / current_price) * 100  
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm > 0), 0)
    atr_14 = true_range.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr_14.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr_14.replace(0, np.nan))
    di_sum = plus_di + minus_di
    dx = 100 * abs(plus_di - minus_di) / di_sum.replace(0, np.nan)
    adx = dx.rolling(window=14).mean().iloc[-1] if not pd.isna(dx.rolling(window=14).mean().iloc[-1]) else 25
    volatility = df['Close'].pct_change().rolling(window=20).std().iloc[-1] * np.sqrt(252) * 100
    avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]
    current_volume = df['Volume'].iloc[-1]
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
    price_change_1d = ((current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
    price_change_5d = ((current_price - df['Close'].iloc[-5]) / df['Close'].iloc[-5]) * 100 if len(df) >= 5 else 0
    price_change_1m = ((current_price - df['Close'].iloc[-20]) / df['Close'].iloc[-20]) * 100 if len(df) >= 20 else 0
    daily_returns = df['Close'].pct_change().dropna()
    mean_return = daily_returns.mean() * 252
    std_return = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = (mean_return - 0.06) / std_return if std_return > 0 else 0
    cumulative_returns = (1 + daily_returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    trend_strength = "Strong Up" if adx > 25 and current_price > ma_50 else \
                     "Strong Down" if adx > 25 and current_price < ma_50 else \
                     "Weak/Sideways"
    
    stock_analysis.append({
        'Symbol': symbol.replace('.NS', ''),
        'Price': current_price,
        'Change_1D': price_change_1d,
        'Change_5D': price_change_5d,
        'Change_1M': price_change_1m,
        'RSI': current_rsi,
        'MACD': current_macd,
        'MA_20': ma_20,
        'MA_50': ma_50,
        'Volatility': volatility,
        'Volume_Ratio': volume_ratio,
        'BB_Position': bb_position,
        'Stoch_K': current_stoch_k,
        'Stoch_D': current_stoch_d,
        'ATR': atr,
        'ATR_Percent': atr_percent,
        'ADX': adx,
        'Sharpe_Ratio': sharpe_ratio,
        'Max_Drawdown': max_drawdown,
        'Trend_Strength': trend_strength,
        'Data': df
    })

analysis_df = pd.DataFrame(stock_analysis)

ml_features = []
ml_targets = []

for stock in stock_analysis:
    df = stock['Data']
    if len(df) < 60:
        continue
    
    for i in range(50, len(df) - 5):
        delta = df['Close'].iloc[:i+1].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean().iloc[-1]
        avg_loss = loss.rolling(window=14).mean().iloc[-1]
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi_val = 100 - (100 / (1 + rs)) if rs != 0 else 50
        
        ma20 = df['Close'].iloc[:i+1].rolling(window=20).mean().iloc[-1]
        ma50 = df['Close'].iloc[:i+1].rolling(window=50).mean().iloc[-1]
        price = df['Close'].iloc[i]
        
        features = [
            rsi_val / 100,
            (price - ma20) / ma20 if ma20 != 0 else 0,  
            (price - ma50) / ma50 if ma50 != 0 else 0, 
            (ma20 - ma50) / ma50 if ma50 != 0 else 0,
        ]
        
        future_price = df['Close'].iloc[i + 5]
        target = (future_price - price) / price
        
        if not any(pd.isna(features)) and not pd.isna(target):
            ml_features.append(features)
            ml_targets.append(target)

if len(ml_features) > 100:
    X = np.array(ml_features)
    y = np.array(ml_targets)
    
    ml_model = LinearRegression()
    ml_model.fit(X, y)
    
    for i, stock in enumerate(stock_analysis):
        features = [
            stock['RSI'] / 100,
            (stock['Price'] - stock['MA_20']) / stock['MA_20'] if stock['MA_20'] != 0 else 0,
            (stock['Price'] - stock['MA_50']) / stock['MA_50'] if stock['MA_50'] != 0 else 0,
            (stock['MA_20'] - stock['MA_50']) / stock['MA_50'] if stock['MA_50'] != 0 else 0,
        ]
        
        if not any(pd.isna(features)):
            prediction = ml_model.predict([features])[0]
            stock_analysis[i]['ML_Prediction'] = prediction * 100
        else:
            stock_analysis[i]['ML_Prediction'] = 0

for i, stock in enumerate(stock_analysis):
    if stock['RSI'] < 30:
        rsi_score = 100
    elif stock['RSI'] < 40:
        rsi_score = 80
    elif stock['RSI'] < 60:
        rsi_score = 60
    elif stock['RSI'] < 70:
        rsi_score = 40
    else:
        rsi_score = 20
    
    if stock['MACD'] > 2:
        macd_score = 100
    elif stock['MACD'] > 0:
        macd_score = 70
    elif stock['MACD'] > -2:
        macd_score = 40
    else:
        macd_score = 20  
    
    ma_score = 50
    if stock['Price'] > stock['MA_20'] and stock['Price'] > stock['MA_50']:
        ma_score = 90
    elif stock['Price'] > stock['MA_20']:
        ma_score = 70
    elif stock['Price'] > stock['MA_50']:
        ma_score = 50
    else:
        ma_score = 30
    
    ml_pred = stock.get('ML_Prediction', 0)
    ml_score = min(100, max(0, 50 + (ml_pred * 10)))  
    
    vol_score = min(100, max(0, stock['Volume_Ratio'] * 50))
    
    momentum_score = 50 + min(25, max(-25, stock['Change_5D'] * 2))
    
    stoch_k = stock.get('Stoch_K', 50)
    if stoch_k < 20:
        stoch_score = 100
    elif stoch_k < 30:
        stoch_score = 80
    elif stoch_k < 70:
        stoch_score = 60
    elif stoch_k < 80:
        stoch_score = 40
    else:
        stoch_score = 20
    
    adx = stock.get('ADX', 25)
    if adx > 40:
        adx_score = 90
    elif adx > 25:
        adx_score = 70
    elif adx > 20:
        adx_score = 50
    else:
        adx_score = 30
    
    final_score = (
        0.20 * ml_score +
        0.15 * rsi_score +
        0.15 * macd_score +
        0.10 * ma_score +      
        0.10 * vol_score +
        0.10 * momentum_score +
        0.10 * stoch_score +
        0.10 * adx_score
    )
    
    stock_analysis[i]['Score'] = final_score
    stock_analysis[i]['RSI_Score'] = rsi_score
    stock_analysis[i]['MACD_Score'] = macd_score
    stock_analysis[i]['MA_Score'] = ma_score
    stock_analysis[i]['ML_Score'] = ml_score
    stock_analysis[i]['Stoch_Score'] = stoch_score
    stock_analysis[i]['ADX_Score'] = adx_score
    
    if final_score >= 75:
        signal = "🟢 STRONG BUY"
        signal_class = "signal-buy"
    elif final_score >= 60:
        signal = "🟡 BUY"
        signal_class = "signal-buy"
    elif final_score >= 40:
        signal = "⚪ HOLD"
        signal_class = "signal-hold"
    elif final_score >= 25:
        signal = "🟠 SELL"
        signal_class = "signal-sell"
    else:
        signal = "🔴 STRONG SELL"
        signal_class = "signal-sell"
    
    stock_analysis[i]['Signal'] = signal
    stock_analysis[i]['Signal_Class'] = signal_class

stock_analysis = sorted(stock_analysis, key=lambda x: x['Score'], reverse=True)

with tab1:
    st.markdown("### 📊 Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_stocks = len(stock_analysis)
    bullish = sum(1 for s in stock_analysis if s['Score'] >= 60)
    bearish = sum(1 for s in stock_analysis if s['Score'] < 40)
    neutral = total_stocks - bullish - bearish
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Stocks Analyzed</div>
            <div class="metric-value">{total_stocks}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🟢 Bullish</div>
            <div class="metric-value" style="color: #00b894;">{bullish}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🔴 Bearish</div>
            <div class="metric-value" style="color: #e74c3c;">{bearish}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚪ Neutral</div>
            <div class="metric-value" style="color: #f39c12;">{neutral}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        avg_score = sum(s['Score'] for s in stock_analysis) / len(stock_analysis)
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Market Sentiment", 'font': {'size': 20, 'color': 'white'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': 'white'},
                'bar': {'color': "#667eea"},
                'bgcolor': "#1a1a2e",
                'borderwidth': 2,
                'bordercolor': "white",
                'steps': [
                    {'range': [0, 25], 'color': '#e74c3c'},
                    {'range': [25, 50], 'color': '#f39c12'},
                    {'range': [50, 75], 'color': '#00cec9'},
                    {'range': [75, 100], 'color': '#00b894'}
                ],
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            height=300
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 Stock Score Distribution")
        scores = [s['Score'] for s in stock_analysis]
        fig_hist = px.histogram(x=scores, nbins=10, 
                                labels={'x': 'Score', 'y': 'Count'},
                                color_discrete_sequence=['#667eea'])
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            height=300,
            showlegend=False
        )
        fig_hist.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
        fig_hist.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### 📋 All Analyzed Stocks")
    
    display_df = pd.DataFrame([{
        'Symbol': s['Symbol'],
        'Price': f"₹{s['Price']:,.2f}",
        '1D %': f"{s['Change_1D']:+.2f}%",
        '5D %': f"{s['Change_5D']:+.2f}%",
        'RSI': f"{s['RSI']:.1f}",
        'Score': f"{s['Score']:.0f}",
        'Signal': s['Signal']
    } for s in stock_analysis])
    
    st.dataframe(display_df, use_container_width=True, height=400)

with tab2:
    st.markdown("### 🏆 AI-Recommended Top Stocks")
    st.markdown("---")
    
    top_5 = stock_analysis[:5]
    
    for idx, stock in enumerate(top_5):
        col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 2])
        
        with col1:
            st.markdown(f"### #{idx + 1}")
        
        with col2:
            st.markdown(f"### {stock['Symbol']}")
            st.markdown(f"₹{stock['Price']:,.2f}")
        
        with col3:
            change_color = "green" if stock['Change_1D'] >= 0 else "red"
            st.markdown(f"**1D Change**")
            st.markdown(f"<span style='color:{change_color}'>{stock['Change_1D']:+.2f}%</span>", unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"**Score**")
            st.markdown(f"<span class='score-badge'>{stock['Score']:.0f}/100</span>", unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"**Signal**")
            st.markdown(f"{stock['Signal']}")
        
        with st.expander(f"📊 Detailed Analysis for {stock['Symbol']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Technical Indicators")
                st.markdown(f"- **RSI (14):** {stock['RSI']:.1f} {'(Oversold)' if stock['RSI'] < 30 else '(Overbought)' if stock['RSI'] > 70 else ''}")
                st.markdown(f"- **MACD:** {stock['MACD']:.2f}")
                st.markdown(f"- **MA 20:** ₹{stock['MA_20']:,.2f}")
                st.markdown(f"- **MA 50:** ₹{stock['MA_50']:,.2f}")
                st.markdown(f"- **Volatility:** {stock['Volatility']:.1f}%")
                
                st.markdown("#### Score Breakdown")
                st.markdown(f"- ML Score: {stock['ML_Score']:.0f}/100")
                st.markdown(f"- RSI Score: {stock['RSI_Score']:.0f}/100")
                st.markdown(f"- MACD Score: {stock['MACD_Score']:.0f}/100")
                st.markdown(f"- MA Score: {stock['MA_Score']:.0f}/100")
            
            with col2:
                df = stock['Data']
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index[-30:],
                    open=df['Open'][-30:],
                    high=df['High'][-30:],
                    low=df['Low'][-30:],
                    close=df['Close'][-30:],
                    name='Price'
                ))
                fig.add_trace(go.Scatter(
                    x=df.index[-30:],
                    y=df['Close'].rolling(20).mean()[-30:],
                    name='MA 20',
                    line=dict(color='#667eea', width=1)
                ))
                fig.update_layout(
                    title=f"{stock['Symbol']} - Last 30 Days",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white'},
                    height=300,
                    showlegend=False,
                    xaxis_rangeslider_visible=False
                )
                fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
                fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
    
    st.markdown("### ⚠️ Stocks to Avoid (Lowest Scores)")
    bottom_5 = stock_analysis[-5:]
    
    avoid_df = pd.DataFrame([{
        'Symbol': s['Symbol'],
        'Price': f"₹{s['Price']:,.2f}",
        'Score': f"{s['Score']:.0f}",
        'RSI': f"{s['RSI']:.1f}",
        'Signal': s['Signal']
    } for s in reversed(bottom_5)])
    
    st.dataframe(avoid_df, use_container_width=True)

with tab3:
    st.markdown("### 💼 Paper Trading Portfolio")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    portfolio_value = st.session_state.cash
    holdings_value = 0
    
    for symbol, holding in st.session_state.portfolio.items():
        full_symbol = f"{symbol}.NS"
        if full_symbol in stock_data:
            current_price = stock_data[full_symbol]['Close'].iloc[-1]
            holdings_value += current_price * holding['quantity']
    
    total_value = st.session_state.cash + holdings_value
    initial_capital = 100000
    total_pnl = total_value - initial_capital
    total_pnl_pct = (total_pnl / initial_capital) * 100
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">💵 Cash Balance</div>
            <div class="metric-value">₹{st.session_state.cash:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📦 Holdings Value</div>
            <div class="metric-value">₹{holdings_value:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">💰 Total Value</div>
            <div class="metric-value">₹{total_value:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        pnl_color = "#00b894" if total_pnl >= 0 else "#e74c3c"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📈 Total P&L</div>
            <div class="metric-value" style="color: {pnl_color};">₹{total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📦 Current Holdings")
    
    if st.session_state.portfolio:
        holdings_data = []
        for symbol, holding in st.session_state.portfolio.items():
            full_symbol = f"{symbol}.NS"
            if full_symbol in stock_data:
                current_price = stock_data[full_symbol]['Close'].iloc[-1]
                buy_price = holding['avg_price']
                quantity = holding['quantity']
                current_value = current_price * quantity
                invested = buy_price * quantity
                pnl = current_value - invested
                pnl_pct = (pnl / invested) * 100
                
                holdings_data.append({
                    'Symbol': symbol,
                    'Quantity': quantity,
                    'Avg Buy Price': f"₹{buy_price:,.2f}",
                    'Current Price': f"₹{current_price:,.2f}",
                    'Current Value': f"₹{current_value:,.2f}",
                    'P&L': f"₹{pnl:+,.2f}",
                    'P&L %': f"{pnl_pct:+.2f}%"
                })
        
        if holdings_data:
            holdings_df = pd.DataFrame(holdings_data)
            st.dataframe(holdings_df, use_container_width=True)
            
            fig_pie = px.pie(
                values=[h['Quantity'] * float(h['Current Price'].replace('₹', '').replace(',', '')) for h in holdings_data],
                names=[h['Symbol'] for h in holdings_data],
                title='Portfolio Allocation',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("📭 No holdings yet. Start trading in the 'Trade Now' tab!")
    
    st.markdown("---")
    
    st.markdown("### 📜 Trade History")
    
    if st.session_state.trade_history:
        history_df = pd.DataFrame(st.session_state.trade_history)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("📭 No trades yet.")
    
    st.markdown("---")
    if st.button("🔄 Reset Portfolio (Start Fresh)", type="secondary"):
        st.session_state.portfolio = {}
        st.session_state.cash = 100000
        st.session_state.trade_history = []
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump({
                'holdings': {},
                'cash': 100000,
                'trade_history': []
            }, f)
        st.success("Portfolio reset! Starting fresh with ₹1,00,000")
        st.rerun()


with tab4:
    st.markdown("### 📈 Execute Paper Trades")
    
    acc_col1, acc_col2, acc_col3 = st.columns(3)
    with acc_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">💵 Available Cash</div>
            <div class="metric-value">₹{st.session_state.cash:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with acc_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📦 Holdings</div>
            <div class="metric-value">{len(st.session_state.portfolio)} Stocks</div>
        </div>
        """, unsafe_allow_html=True)
    with acc_col3:
        total_trades = len(st.session_state.trade_history)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📊 Total Trades</div>
            <div class="metric-value">{total_trades}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

    st.markdown("### 🔧 Manual Trading")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🟢 BUY Stocks")
        
        buy_stock = st.selectbox(
            "Select Stock to Buy",
            options=[s['Symbol'] for s in stock_analysis],
            key="buy_stock"
        )
        
        selected_stock = next((s for s in stock_analysis if s['Symbol'] == buy_stock), None)
        
        if selected_stock:
            st.markdown(f"**Current Price:** ₹{selected_stock['Price']:,.2f}")
            st.markdown(f"**Signal:** {selected_stock['Signal']}  |  **Score:** {selected_stock['Score']:.0f}/100")
            
            max_qty = int(st.session_state.cash / (selected_stock['Price'] * 1.015)) if selected_stock['Price'] > 0 else 0
            if max_qty > 0:
                # Ensure persisted value is within valid range
                default_qty = min(max(1, st.session_state.buy_qty_value), max_qty)
                buy_qty = st.number_input("Quantity", min_value=1, max_value=max_qty, value=default_qty, key="buy_qty")
                st.session_state.buy_qty_value = buy_qty
                
                base_value = buy_qty * selected_stock['Price']
                stt_charge = base_value * STT_BUY
                stamp_charge = base_value * STAMP_DUTY
                brokerage = min(20, base_value * BROKERAGE_PERCENT)
                exchange_fee = base_value * EXCHANGE_CHARGES
                sebi_fee = base_value * SEBI_CHARGES
                gst_charge = (brokerage + exchange_fee + sebi_fee) * GST_RATE
                total_charges = stt_charge + stamp_charge + brokerage + exchange_fee + sebi_fee + gst_charge
                total_cost = base_value + total_charges
                
                st.markdown("**💰 Cost Breakdown:**")
                st.write(f"Stock Value ({buy_qty} × ₹{selected_stock['Price']:,.2f}): **₹{base_value:,.2f}**")
                st.write(f"STT: ₹{stt_charge:,.2f} | Stamp: ₹{stamp_charge:,.2f} | Brokerage+GST: ₹{brokerage + gst_charge:,.2f}")
                st.success(f"**TOTAL: ₹{total_cost:,.2f}** (Charges: ₹{total_charges:,.2f})")
                
                if st.button("🟢 BUY NOW", key="buy_btn", type="primary"):
                    if total_cost <= st.session_state.cash:
                        st.session_state.cash -= total_cost
                        
                        if buy_stock in st.session_state.portfolio:
                            old_qty = st.session_state.portfolio[buy_stock]['quantity']
                            old_avg = st.session_state.portfolio[buy_stock]['avg_price']
                            new_qty = old_qty + buy_qty
                            new_avg = ((old_qty * old_avg) + total_cost) / new_qty
                            st.session_state.portfolio[buy_stock] = {'quantity': new_qty, 'avg_price': new_avg}
                        else:
                            st.session_state.portfolio[buy_stock] = {'quantity': buy_qty, 'avg_price': total_cost / buy_qty}
                        
                        st.session_state.trade_history.append({
                            'Date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                            'Type': 'BUY',
                            'Symbol': buy_stock,
                            'Quantity': buy_qty,
                            'Price': f"₹{selected_stock['Price']:,.2f}",
                            'Charges': f"₹{total_charges:,.2f}",
                            'Total': f"₹{total_cost:,.2f}"
                        })
                        
                        with open(PORTFOLIO_FILE, 'w') as f:
                            json.dump({'holdings': st.session_state.portfolio, 'cash': st.session_state.cash, 
                                      'trade_history': st.session_state.trade_history}, f)
                        
                        st.success(f"✅ Bought {buy_qty} shares of {buy_stock} @ ₹{selected_stock['Price']:,.2f} (Charges: ₹{total_charges:,.2f})")
                        st.rerun()
                    else:
                        st.error("❌ Insufficient funds!")
            else:
                st.warning("⚠️ Insufficient funds to buy any shares")
    
    with col2:
        st.markdown("#### 🔴 SELL Stocks")
        
        if st.session_state.portfolio:
            sell_stock = st.selectbox("Select Stock to Sell", options=list(st.session_state.portfolio.keys()), key="sell_stock")
            
            if sell_stock:
                holding = st.session_state.portfolio[sell_stock]
                full_symbol = f"{sell_stock}.NS"
                current_price = stock_data[full_symbol]['Close'].iloc[-1] if full_symbol in stock_data else 0
                
                st.markdown(f"**Holdings:** {holding['quantity']} shares @ ₹{holding['avg_price']:,.2f} avg")
                st.markdown(f"**Current Price:** ₹{current_price:,.2f}")
                
                pnl_per_share = current_price - holding['avg_price']
                pnl_pct = (pnl_per_share / holding['avg_price']) * 100 if holding['avg_price'] > 0 else 0
                pnl_emoji = "📈" if pnl_per_share >= 0 else "📉"
                st.metric("P&L per share", f"₹{pnl_per_share:+,.2f}", delta=f"{pnl_pct:+.2f}%")
                
                sell_qty = st.number_input("Quantity to Sell", min_value=1, max_value=holding['quantity'], value=holding['quantity'], key="sell_qty")
                
                base_value = sell_qty * current_price
                stt_charge = base_value * STT_SELL
                brokerage = min(20, base_value * BROKERAGE_PERCENT)
                exchange_fee = base_value * EXCHANGE_CHARGES
                sebi_fee = base_value * SEBI_CHARGES
                gst_charge = (brokerage + exchange_fee + sebi_fee) * GST_RATE
                dp_charge = DP_CHARGES if sell_qty > 0 else 0
                total_charges = stt_charge + brokerage + exchange_fee + sebi_fee + gst_charge + dp_charge
                net_proceeds = base_value - total_charges
                
                st.markdown("**💸 Sale Breakdown:**")
                st.write(f"Gross Value ({sell_qty} × ₹{current_price:,.2f}): **₹{base_value:,.2f}**")
                st.write(f"Charges - STT: ₹{stt_charge:,.2f} | Brokerage+GST: ₹{brokerage + gst_charge:,.2f} | DP: ₹{dp_charge:,.2f}")
                st.success(f"**NET PROCEEDS: ₹{net_proceeds:,.2f}** (Charges: ₹{total_charges:,.2f})")
                
                if st.button("🔴 SELL NOW", key="sell_btn", type="primary"):
                    st.session_state.cash += net_proceeds
                    
                    if sell_qty >= holding['quantity']:
                        del st.session_state.portfolio[sell_stock]
                    else:
                        st.session_state.portfolio[sell_stock]['quantity'] -= sell_qty
                    
                    st.session_state.trade_history.append({
                        'Date': datetime.now().strftime('%Y-%m-%d %H:%M'), 'Type': 'SELL', 'Symbol': sell_stock,
                        'Quantity': sell_qty, 'Price': f"₹{current_price:,.2f}", 'Charges': f"₹{total_charges:,.2f}",
                        'Total': f"₹{net_proceeds:,.2f}"
                    })
                    
                    with open(PORTFOLIO_FILE, 'w') as f:
                        json.dump({'holdings': st.session_state.portfolio, 'cash': st.session_state.cash, 
                                  'trade_history': st.session_state.trade_history}, f)
                    
                    st.success(f"✅ Sold {sell_qty} shares of {sell_stock} @ ₹{current_price:,.2f} (Charges: ₹{total_charges:,.2f})")
                    st.rerun()
        else:
            st.info("📭 No stocks to sell. Buy some stocks first!")
