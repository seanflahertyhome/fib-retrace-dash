# 📈 Fibonacci Trading Dashboard

A comprehensive, production-ready Streamlit dashboard for stock market analysis using Fibonacci retracement levels and options flow analysis to identify high-probability trading opportunities.

![Dashboard Preview](https://via.placeholder.com/800x400/0E1117/00D4AA?text=Fibonacci+Trading+Dashboard)

## 🌟 Features

### 📊 Single Stock Analysis
- **Interactive Fibonacci Charts**: Candlestick charts with automatically calculated Fibonacci retracement and extension levels
- **Technical Indicators**: RSI, MACD, Moving Averages, Bollinger Bands
- **Trading Signals**: AI-powered buy/sell signals with confidence scores (1-10)
- **Options Flow Analysis**: Real-time put/call ratios and volume analysis
- **Price Targets**: Automated price targets and stop-loss levels

### 🔍 Bull Run Scanner
- **Multi-Stock Scanning**: Scan up to 100+ stocks simultaneously
- **Composite Scoring**: 0-100 score based on:
  - Options flow (25 points)
  - Technical trend (25 points)
  - Volume accumulation (25 points)
  - Fibonacci positioning (25 points)
- **Ranked Results**: Sortable, filterable results table
- **Top Pick Analysis**: Detailed breakdown of highest-scoring stocks

### 📋 Watchlist Management
- **Custom Watchlists**: Save and track your favorite stocks
- **Batch Analysis**: Analyze entire watchlist at once
- **Quick Stats**: Instant overview of watchlist performance

### 📊 Backtesting Engine
- **Historical Performance**: Test Fibonacci strategy on historical data
- **Performance Metrics**: Win rate, profit factor, max drawdown
- **vs Buy & Hold**: Compare strategy returns against simple buy-and-hold
- **Trade Journal**: Complete trade history with entry/exit points

### 📚 Educational Resources
- Fibonacci retracement explanations
- Options trading fundamentals
- Best practice trading strategies
- Risk management guidelines

## 🚀 Quick Start

### Option 1: Deploy to Streamlit Cloud (Recommended)

1. **Fork this repository** to your GitHub account

2. **Go to [share.streamlit.io](https://share.streamlit.io)**

3. **Click "New app"**

4. **Connect your GitHub** repository:
   - Repository: `your-username/fibonacci-trading-dashboard`
   - Branch: `main`
   - Main file path: `app.py`

5. **Click "Deploy"**

6. Your app will be live at: `https://your-app-name.streamlit.app`

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/fibonacci-trading-dashboard.git
cd fibonacci-trading-dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
