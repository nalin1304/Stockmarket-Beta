# 🚀 Smart Stock Trading Bot

An AI-powered stock trading bot for the **Indian Stock Market (NSE)** with paper trading capabilities, technical analysis, and machine learning predictions.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

### 📊 Technical Analysis
- **RSI (Relative Strength Index)** - Identify overbought/oversold conditions
- **MACD (Moving Average Convergence Divergence)** - Trend momentum analysis
- **Bollinger Bands** - Volatility and price level detection
- **Stochastic Oscillator** - Momentum indicator
- **ADX (Average Directional Index)** - Trend strength measurement
- **ATR (Average True Range)** - Volatility measurement
- **Moving Averages** - 20-day and 50-day SMA

### 🤖 Machine Learning
- Linear Regression model for 5-day price prediction
- Feature engineering using technical indicators
- Scoring system combining ML predictions with technical signals

### 💼 Paper Trading
- Start with ₹1,00,000 virtual capital
- Realistic Indian market charges:
  - STT (Securities Transaction Tax)
  - Stamp Duty
  - Brokerage fees
  - GST
  - Exchange charges
  - SEBI charges
  - DP charges
- Portfolio tracking and P&L analysis
- Complete trade history

### 🎨 Modern UI
- Dark theme with gradient styling
- Interactive charts using Plotly
- Real-time market status indicator
- Responsive dashboard layout

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/smart-stock-trading-bot.git
   cd smart-stock-trading-bot
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

Run the Streamlit app:

```bash
streamlit run stock_trading_bot.py
```

The app will open in your browser at `http://localhost:8501`

## 📱 Dashboard Tabs

1. **📊 Market Dashboard** - Overview of analyzed stocks, market sentiment gauge, score distribution
2. **🏆 Top Picks** - AI-recommended stocks with detailed technical analysis and charts
3. **💼 Portfolio** - Track your paper trading portfolio, holdings, and P&L
4. **📈 Trade Now** - Execute buy/sell paper trades with realistic charge calculation

## 📈 Stock Universe

The bot analyzes **50 Nifty stocks** including:
- IT: TCS, INFY, WIPRO, HCLTECH, TECHM
- Banking: HDFCBANK, ICICIBANK, SBIN, KOTAKBANK, AXISBANK
- FMCG: HINDUNILVR, ITC, NESTLEIND, BRITANNIA
- Auto: MARUTI, TATAMOTORS, M&M, BAJAJ-AUTO
- Pharma: SUNPHARMA, DRREDDY, CIPLA, DIVISLAB
- And many more...

## 🔧 Configuration

Adjust the number of stocks to analyze using the sidebar slider (10-50 stocks).

## ⚠️ Disclaimer

This is a **paper trading simulation** for educational purposes only. The signals and predictions are not financial advice. Always do your own research before making real investment decisions.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

Made with ❤️ for Indian Stock Market enthusiasts
