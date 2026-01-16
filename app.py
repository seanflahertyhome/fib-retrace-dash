"""
Fibonacci Trading Dashboard
A comprehensive stock analysis tool using Fibonacci retracement and options flow analysis
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
import ta
from typing import Tuple, Dict, List, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fibonacci Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #1E2329;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #00D4AA;
    }
    .buy-signal {
        background-color: #0D4D3D;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #00D4AA;
    }
    .sell-signal {
        background-color: #4D0D0D;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #FF4B4B;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# DATA FETCHING FUNCTIONS
# =============================================================================

@st.cache_data(ttl=900)  # Cache for 15 minutes
def fetch_stock_data(ticker: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Fetch stock price data using yfinance
    
    Args:
        ticker: Stock ticker symbol
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            st.error(f"No data found for {ticker}")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=900)
def fetch_options_data(ticker: str) -> Dict:
    """
    Fetch options chain data for a given ticker
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary containing options data and metrics
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get available expiration dates
        expirations = stock.options
        
        if not expirations or len(expirations) == 0:
            return {"error": "No options data available"}
        
        # Get nearest expiration
        nearest_exp = expirations[0]
        opt_chain = stock.option_chain(nearest_exp)
        
        calls = opt_chain.calls
        puts = opt_chain.puts
        
        # Calculate metrics
        total_call_volume = calls['volume'].sum()
        total_put_volume = puts['volume'].sum()
        total_call_oi = calls['openInterest'].sum()
        total_put_oi = puts['openInterest'].sum()
        
        put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0
        oi_put_call_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        return {
            "calls": calls,
            "puts": puts,
            "expiration": nearest_exp,
            "total_call_volume": total_call_volume,
            "total_put_volume": total_put_volume,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "put_call_ratio": put_call_ratio,
            "oi_put_call_ratio": oi_put_call_ratio,
            "expirations": expirations
        }
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=900)
def fetch_stock_info(ticker: str) -> Dict:
    """Fetch basic stock information"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except Exception as e:
        return {"error": str(e)}

# =============================================================================
# FIBONACCI ANALYSIS FUNCTIONS
# =============================================================================

def find_swing_points(df: pd.DataFrame, order: int = 20) -> Tuple[pd.Series, pd.Series]:
    """
    Find swing highs and lows using local extrema detection
    
    Args:
        df: DataFrame with price data
        order: Number of points on each side to use for comparison
    
    Returns:
        Tuple of (swing_highs, swing_lows) as boolean Series
    """
    highs = df['High'].values
    lows = df['Low'].values
    
    # Find local maxima and minima
    high_indices = argrelextrema(highs, np.greater, order=order)[0]
    low_indices = argrelextrema(lows, np.less, order=order)[0]
    
    swing_highs = pd.Series(False, index=df.index)
    swing_lows = pd.Series(False, index=df.index)
    
    swing_highs.iloc[high_indices] = True
    swing_lows.iloc[low_indices] = True
    
    return swing_highs, swing_lows

def calculate_fibonacci_levels(high: float, low: float, trend: str = "uptrend") -> Dict[str, float]:
    """
    Calculate Fibonacci retracement and extension levels
    
    Args:
        high: Swing high price
        low: Swing low price
        trend: "uptrend" or "downtrend"
    
    Returns:
        Dictionary of Fibonacci levels
    """
    diff = high - low
    
    if trend == "uptrend":
        levels = {
            "0.0%": low,
            "23.6%": high - (diff * 0.236),
            "38.2%": high - (diff * 0.382),
            "50.0%": high - (diff * 0.50),
            "61.8%": high - (diff * 0.618),
            "78.6%": high - (diff * 0.786),
            "100.0%": high,
            # Extensions
            "127.2%": high + (diff * 0.272),
            "161.8%": high + (diff * 0.618),
            "200.0%": high + (diff * 1.0),
            "261.8%": high + (diff * 1.618)
        }
    else:  # downtrend
        levels = {
            "0.0%": high,
            "23.6%": low + (diff * 0.236),
            "38.2%": low + (diff * 0.382),
            "50.0%": low + (diff * 0.50),
            "61.8%": low + (diff * 0.618),
            "78.6%": low + (diff * 0.786),
            "100.0%": low,
            # Extensions
            "127.2%": low - (diff * 0.272),
            "161.8%": low - (diff * 0.618),
            "200.0%": low - (diff * 1.0),
            "261.8%": low - (diff * 1.618)
        }
    
    return levels

def get_latest_swing_points(df: pd.DataFrame, lookback: int = 100) -> Tuple[float, float, str]:
    """
    Get the most recent significant swing high and low
    
    Returns:
        Tuple of (swing_high, swing_low, trend)
    """
    recent_df = df.tail(lookback)
    
    swing_high = recent_df['High'].max()
    swing_low = recent_df['Low'].min()
    
    # Determine trend based on position of high and low
    high_idx = recent_df['High'].idxmax()
    low_idx = recent_df['Low'].idxmin()
    
    trend = "uptrend" if high_idx > low_idx else "downtrend"
    
    return swing_high, swing_low, trend

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for confirmation
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with added technical indicators
    """
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    
    # Moving Averages
    df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
    df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
    df['SMA_200'] = ta.trend.SMAIndicator(df['Close'], window=200).sma_indicator()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    df['BB_Mid'] = bollinger.bollinger_mavg()
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    
    return df

def generate_trading_signal(df: pd.DataFrame, fib_levels: Dict[str, float], 
                           current_price: float) -> Dict:
    """
    Generate trading signals based on Fibonacci levels and technical indicators
    
    Returns:
        Dictionary with signal information
    """
    latest = df.iloc[-1]
    
    # Get Fibonacci levels
    fib_618 = fib_levels.get("61.8%", 0)
    fib_786 = fib_levels.get("78.6%", 0)
    fib_50 = fib_levels.get("50.0%", 0)
    fib_382 = fib_levels.get("38.2%", 0)
    
    # Initialize signal
    signal = {
        "action": "HOLD",
        "strength": 0,
        "price_target": None,
        "stop_loss": None,
        "reasons": []
    }
    
    # Check RSI
    rsi_oversold = latest['RSI'] < 35
    rsi_overbought = latest['RSI'] > 70
    
    # Check MACD
    macd_bullish = latest['MACD'] > latest['MACD_Signal']
    macd_bearish = latest['MACD'] < latest['MACD_Signal']
    
    # Check volume
    volume_surge = latest['Volume'] > latest['Volume_SMA'] * 1.5
    
    # BUY SIGNAL LOGIC
    buy_score = 0
    
    # Price near key Fibonacci levels
    if fib_786 * 0.98 <= current_price <= fib_786 * 1.02:
        buy_score += 3
        signal["reasons"].append("Price at 78.6% retracement (strong support)")
    elif fib_618 * 0.98 <= current_price <= fib_618 * 1.02:
        buy_score += 2
        signal["reasons"].append("Price at 61.8% retracement (golden ratio)")
    elif fib_50 * 0.98 <= current_price <= fib_50 * 1.02:
        buy_score += 1
        signal["reasons"].append("Price at 50% retracement")
    
    # Technical confirmations
    if rsi_oversold:
        buy_score += 2
        signal["reasons"].append("RSI oversold (<35)")
    
    if macd_bullish:
        buy_score += 2
        signal["reasons"].append("MACD bullish crossover")
    
    if volume_surge and current_price < latest['Close']:
        buy_score += 1
        signal["reasons"].append("High volume on dip")
    
    # SELL SIGNAL LOGIC
    sell_score = 0
    
    # Price near extension levels
    if current_price >= fib_levels.get("161.8%", float('inf')) * 0.98:
        sell_score += 3
        signal["reasons"].append("Price at 161.8% extension (take profit)")
    elif current_price >= fib_levels.get("127.2%", float('inf')) * 0.98:
        sell_score += 2
        signal["reasons"].append("Price at 127.2% extension")
    
    if rsi_overbought:
        sell_score += 2
        signal["reasons"].append("RSI overbought (>70)")
    
    if macd_bearish and latest['RSI'] > 60:
        sell_score += 2
        signal["reasons"].append("MACD bearish crossover")
    
    if current_price < fib_786 * 0.95:
        sell_score += 2
        signal["reasons"].append("Price broke below 78.6% support")
    
    # Determine final signal
    if buy_score >= 4:
        signal["action"] = "BUY"
        signal["strength"] = min(buy_score, 10)
        signal["price_target"] = fib_levels.get("127.2%")
        signal["stop_loss"] = fib_786 * 0.97
    elif sell_score >= 4:
        signal["action"] = "SELL"
        signal["strength"] = min(sell_score, 10)
        signal["price_target"] = fib_50
        signal["stop_loss"] = fib_levels.get("161.8%", current_price * 1.05)
    else:
        signal["strength"] = abs(buy_score - sell_score)
        if buy_score > sell_score:
            signal["reasons"].append("Weak buy setup - wait for confirmation")
        elif sell_score > buy_score:
            signal["reasons"].append("Weak sell setup - consider taking profits")
        else:
            signal["reasons"].append("No clear signal - sideways action")
    
    return signal

# =============================================================================
# OPTIONS ANALYSIS FUNCTIONS
# =============================================================================

def calculate_bull_run_score(ticker: str, df: pd.DataFrame, options_data: Dict) -> Dict:
    """
    Calculate comprehensive bull run probability score
    
    Returns:
        Dictionary with score breakdown
    """
    scores = {
        "options_flow": 0,
        "technical_trend": 0,
        "volume_accumulation": 0,
        "fibonacci_position": 0,
        "total": 0,
        "details": {}
    }
    
    if "error" in options_data:
        scores["details"]["options_error"] = options_data["error"]
        return scores
    
    # 1. OPTIONS FLOW SCORE (0-25 points)
    options_score = 0
    
    # Put/Call Ratio (lower is more bullish)
    pc_ratio = options_data.get("put_call_ratio", 1.0)
    if pc_ratio < 0.5:
        options_score += 10
        scores["details"]["pc_ratio"] = "Very Bullish (<0.5)"
    elif pc_ratio < 0.7:
        options_score += 7
        scores["details"]["pc_ratio"] = "Bullish (<0.7)"
    elif pc_ratio < 1.0:
        options_score += 4
        scores["details"]["pc_ratio"] = "Neutral"
    else:
        scores["details"]["pc_ratio"] = "Bearish (>1.0)"
    
    # Call volume surge
    call_vol = options_data.get("total_call_volume", 0)
    put_vol = options_data.get("total_put_volume", 0)
    
    if call_vol > put_vol * 2:
        options_score += 10
        scores["details"]["volume_surge"] = "Strong call buying"
    elif call_vol > put_vol * 1.5:
        options_score += 5
        scores["details"]["volume_surge"] = "Moderate call buying"
    
    # Open Interest
    call_oi = options_data.get("total_call_oi", 0)
    put_oi = options_data.get("total_put_oi", 0)
    
    if call_oi > put_oi * 1.5:
        options_score += 5
        scores["details"]["open_interest"] = "Bullish OI positioning"
    
    scores["options_flow"] = min(options_score, 25)
    
    # 2. TECHNICAL TREND SCORE (0-25 points)
    technical_score = 0
    
    latest = df.iloc[-1]
    
    # Price above moving averages
    if latest['Close'] > latest['SMA_20']:
        technical_score += 5
    if latest['Close'] > latest['SMA_50']:
        technical_score += 7
    if latest['Close'] > latest['SMA_200']:
        technical_score += 8
        scores["details"]["trend"] = "Strong uptrend (above all MAs)"
    elif latest['Close'] > latest['SMA_50']:
        scores["details"]["trend"] = "Uptrend (above 50-day MA)"
    
    # MACD
    if latest['MACD'] > latest['MACD_Signal'] and latest['MACD_Diff'] > 0:
        technical_score += 5
    
    scores["technical_trend"] = min(technical_score, 25)
    
    # 3. VOLUME ACCUMULATION SCORE (0-25 points)
    volume_score = 0
    
    # Recent volume trend
    recent_vol = df['Volume'].tail(20).mean()
    older_vol = df['Volume'].tail(60).head(40).mean()
    
    if recent_vol > older_vol * 1.5:
        volume_score += 15
        scores["details"]["volume_trend"] = "Strong accumulation"
    elif recent_vol > older_vol * 1.2:
        volume_score += 10
        scores["details"]["volume_trend"] = "Moderate accumulation"
    
    # Volume on up days vs down days
    df_recent = df.tail(20).copy()
    df_recent['Price_Change'] = df_recent['Close'].pct_change()
    
    up_volume = df_recent[df_recent['Price_Change'] > 0]['Volume'].mean()
    down_volume = df_recent[df_recent['Price_Change'] < 0]['Volume'].mean()
    
    if up_volume > down_volume * 1.3:
        volume_score += 10
        scores["details"]["volume_bias"] = "Buying pressure"
    
    scores["volume_accumulation"] = min(volume_score, 25)
    
    # 4. FIBONACCI POSITION SCORE (0-25 points)
    fib_score = 0
    
    swing_high, swing_low, trend = get_latest_swing_points(df)
    fib_levels = calculate_fibonacci_levels(swing_high, swing_low, trend)
    current_price = latest['Close']
    
    # Reward positions near support levels
    fib_618 = fib_levels.get("61.8%", 0)
    fib_786 = fib_levels.get("78.6%", 0)
    fib_50 = fib_levels.get("50.0%", 0)
    
    if fib_786 * 0.95 <= current_price <= fib_786 * 1.05:
        fib_score += 20
        scores["details"]["fib_position"] = "Near 78.6% support (strong buy zone)"
    elif fib_618 * 0.95 <= current_price <= fib_618 * 1.05:
        fib_score += 15
        scores["details"]["fib_position"] = "Near 61.8% support (golden ratio)"
    elif fib_50 * 0.95 <= current_price <= fib_50 * 1.05:
        fib_score += 10
        scores["details"]["fib_position"] = "Near 50% retracement"
    elif current_price > swing_high:
        fib_score += 5
        scores["details"]["fib_position"] = "Breaking to new highs"
    
    # Bonus for bouncing off support
    if len(df) >= 5:
        recent_low = df['Low'].tail(5).min()
        if recent_low <= fib_618 * 1.02 and current_price > fib_618 * 1.05:
            fib_score += 5
            scores["details"]["fib_bounce"] = "Recent bounce from support"
    
    scores["fibonacci_position"] = min(fib_score, 25)
    
    # TOTAL SCORE
    scores["total"] = (scores["options_flow"] + scores["technical_trend"] + 
                       scores["volume_accumulation"] + scores["fibonacci_position"])
    
    return scores

def scan_multiple_stocks(tickers: List[str], progress_bar) -> pd.DataFrame:
    """
    Scan multiple stocks for bull run potential
    
    Returns:
        DataFrame with ranked results
    """
    results = []
    
    for i, ticker in enumerate(tickers):
        try:
            progress_bar.progress((i + 1) / len(tickers), text=f"Scanning {ticker}...")
            
            # Fetch data
            df = fetch_stock_data(ticker, period="6mo", interval="1d")
            if df is None or len(df) < 50:
                continue
            
            df = calculate_technical_indicators(df)
            options_data = fetch_options_data(ticker)
            
            # Calculate score
            score_data = calculate_bull_run_score(ticker, df, options_data)
            
            # Get current price info
            current_price = df['Close'].iloc[-1]
            price_change = ((current_price - df['Close'].iloc[-20]) / df['Close'].iloc[-20]) * 100
            
            results.append({
                "Ticker": ticker,
                "Score": score_data["total"],
                "Price": f"${current_price:.2f}",
                "20D Change": f"{price_change:+.2f}%",
                "Options Flow": score_data["options_flow"],
                "Technical": score_data["technical_trend"],
                "Volume": score_data["volume_accumulation"],
                "Fibonacci": score_data["fibonacci_position"],
                "P/C Ratio": options_data.get("put_call_ratio", "N/A"),
                "Signal": "🟢 Strong" if score_data["total"] >= 70 else 
                         "🟡 Moderate" if score_data["total"] >= 50 else "⚪ Weak"
            })
            
        except Exception as e:
            st.warning(f"Error scanning {ticker}: {str(e)}")
            continue
    
    if not results:
        return pd.DataFrame()
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("Score", ascending=False)
    
    return df_results

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_fibonacci_chart(df: pd.DataFrame, fib_levels: Dict[str, float], 
                          swing_high: float, swing_low: float, ticker: str) -> go.Figure:
    """
    Create interactive candlestick chart with Fibonacci levels
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{ticker} Price with Fibonacci Levels', 'Volume', 'RSI')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#00D4AA',
            decreasing_line_color='#FF4B4B'
        ),
        row=1, col=1
    )
    
    # Fibonacci levels
    colors = {
        "0.0%": "#FF4B4B",
        "23.6%": "#FFA500",
        "38.2%": "#FFFF00",
        "50.0%": "#00D4AA",
        "61.8%": "#00FF00",
        "78.6%": "#0000FF",
        "100.0%": "#FF4B4B",
        "127.2%": "#800080",
        "161.8%": "#FF00FF",
        "200.0%": "#FFC0CB",
        "261.8%": "#FFB6C1"
    }
    
    for level_name, level_price in fib_levels.items():
        if level_price < df['Low'].min() * 0.5 or level_price > df['High'].max() * 1.5:
            continue  # Skip levels too far from price range
            
        fig.add_hline(
            y=level_price,
            line_dash="dash",
            line_color=colors.get(level_name, "#FFFFFF"),
            line_width=1,
            annotation_text=f"{level_name}: ${level_price:.2f}",
            annotation_position="right",
            row=1, col=1
        )
    
    # Add shaded buy/sell zones
    fib_618 = fib_levels.get("61.8%", 0)
    fib_786 = fib_levels.get("78.6%", 0)
    
    # Buy zone (between 61.8% and 78.6%)
    fig.add_hrect(
        y0=fib_786, y1=fib_618,
        fillcolor="green", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="Buy Zone",
        annotation_position="left",
        row=1, col=1
    )
    
    # Moving averages
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['SMA_20'],
            name='SMA 20',
            line=dict(color='orange', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['SMA_50'],
            name='SMA 50',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    # Volume
    colors_volume = ['red' if row['Close'] < row['Open'] else 'green' 
                     for _, row in df.iterrows()]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color=colors_volume,
            showlegend=False
        ),
        row=2, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['RSI'],
            name='RSI',
            line=dict(color='purple', width=2)
        ),
        row=3, col=1
    )
    
    # RSI reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=3, col=1)
    
    # Update layout
    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        template='plotly_dark',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    
    return fig

def create_options_flow_chart(options_data: Dict) -> go.Figure:
    """
    Create options flow visualization
    """
    if "error" in options_data:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Options data not available: {options_data['error']}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Call vs Put Volume', 'Call vs Put Open Interest'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Volume comparison
    fig.add_trace(
        go.Bar(
            x=['Calls', 'Puts'],
            y=[options_data['total_call_volume'], options_data['total_put_volume']],
            marker_color=['#00D4AA', '#FF4B4B'],
            name='Volume',
            text=[f"{options_data['total_call_volume']:,.0f}", 
                  f"{options_data['total_put_volume']:,.0f}"],
            textposition='auto',
        ),
        row=1, col=1
    )
    
    # Open Interest comparison
    fig.add_trace(
        go.Bar(
            x=['Calls', 'Puts'],
            y=[options_data['total_call_oi'], options_data['total_put_oi']],
            marker_color=['#00D4AA', '#FF4B4B'],
            name='Open Interest',
            text=[f"{options_data['total_call_oi']:,.0f}", 
                  f"{options_data['total_put_oi']:,.0f}"],
            textposition='auto',
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        template='plotly_dark'
    )
    
    return fig

def create_bull_score_gauge(score: float) -> go.Figure:
    """
    Create bull run probability gauge chart
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Bull Run Probability Score", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "#00D4AA"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#00D4AA"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#FF4B4B'},
                {'range': [30, 50], 'color': '#FFA500'},
                {'range': [50, 70], 'color': '#FFFF00'},
                {'range': [70, 100], 'color': '#00D4AA'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        template='plotly_dark',
        font={'color': "white", 'family': "Arial"}
    )
    
    return fig

# =============================================================================
# BACKTESTING FUNCTIONS
# =============================================================================

def backtest_fibonacci_strategy(df: pd.DataFrame, initial_capital: float = 10000) -> Dict:
    """
    Backtest Fibonacci trading strategy
    
    Returns:
        Dictionary with backtest results
    """
    df = df.copy()
    df = calculate_technical_indicators(df)
    
    capital = initial_capital
    position = 0
    trades = []
    equity_curve = [initial_capital]
    
    for i in range(100, len(df)):
        current_data = df.iloc[:i+1]
        current_price = current_data['Close'].iloc[-1]
        
        # Get Fibonacci levels
        swing_high, swing_low, trend = get_latest_swing_points(current_data, lookback=50)
        fib_levels = calculate_fibonacci_levels(swing_high, swing_low, trend)
        
        # Generate signal
        signal = generate_trading_signal(current_data, fib_levels, current_price)
        
        # Execute trades
        if signal['action'] == 'BUY' and position == 0 and signal['strength'] >= 5:
            # Buy
            shares = capital / current_price
            position = shares
            capital = 0
            trades.append({
                'date': current_data.index[-1],
                'action': 'BUY',
                'price': current_price,
                'shares': shares
            })
        
        elif signal['action'] == 'SELL' and position > 0 and signal['strength'] >= 5:
            # Sell
            capital = position * current_price
            trades.append({
                'date': current_data.index[-1],
                'action': 'SELL',
                'price': current_price,
                'shares': position,
                'profit': capital - initial_capital
            })
            position = 0
        
        # Calculate equity
        if position > 0:
            equity = position * current_price
        else:
            equity = capital
        
        equity_curve.append(equity)
    
    # Close any open position
    if position > 0:
        final_price = df['Close'].iloc[-1]
        capital = position * final_price
        trades.append({
            'date': df.index[-1],
            'action': 'SELL',
            'price': final_price,
            'shares': position,
            'profit': capital - initial_capital
        })
    
    # Calculate metrics
    total_return = ((capital - initial_capital) / initial_capital) * 100
    
    # Buy and hold comparison
    buy_hold_shares = initial_capital / df['Close'].iloc[100]
    buy_hold_final = buy_hold_shares * df['Close'].iloc[-1]
    buy_hold_return = ((buy_hold_final - initial_capital) / initial_capital) * 100
    
    # Win rate
    winning_trades = [t for t in trades if t['action'] == 'SELL' and t.get('profit', 0) > 0]
    losing_trades = [t for t in trades if t['action'] == 'SELL' and t.get('profit', 0) <= 0]
    
    total_trades = len(winning_trades) + len(losing_trades)
    win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
    
    # Max drawdown
    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.expanding().max()
    drawdown = (equity_series - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()
    
    return {
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'total_trades': total_trades,
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'final_capital': capital,
        'trades': trades,
        'equity_curve': equity_curve
    }

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application function"""
    
    # Header
    st.markdown('<p class="big-font">📈 Fibonacci Trading Dashboard</p>', unsafe_allow_html=True)
    st.markdown("### Advanced Stock Analysis using Fibonacci Retracement & Options Flow")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/0E1117/00D4AA?text=FIB+TRADER", use_container_width=True)
        st.markdown("---")
        
        # Stock Selection
        st.subheader("📊 Stock Selection")
        ticker = st.text_input("Enter Ticker Symbol", value="AAPL").upper()
        
        # Date Range
        st.subheader("📅 Date Range")
        period_options = {
            "1 Month": "1mo",
            "3 Months": "3mo",
            "6 Months": "6mo",
            "1 Year": "1y",
            "2 Years": "2y",
            "5 Years": "5y"
        }
        period = st.selectbox("Select Period", list(period_options.keys()), index=3)
        
        # Interval
        interval_options = {
            "Daily": "1d",
            "Weekly": "1wk",
            "Monthly": "1mo"
        }
        interval = st.selectbox("Select Interval", list(interval_options.keys()), index=0)
        
        # Fibonacci Settings
        st.subheader("⚙️ Fibonacci Settings")
        lookback_period = st.slider("Swing Detection Lookback", 10, 50, 20)
        swing_window = st.slider("Swing Point Window", 5, 30, 20)
        
        # Watchlist
        st.subheader("⭐ Watchlist")
        if 'watchlist' not in st.session_state:
            st.session_state.watchlist = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        
        if st.button("➕ Add to Watchlist"):
            if ticker not in st.session_state.watchlist:
                st.session_state.watchlist.append(ticker)
                st.success(f"Added {ticker} to watchlist")
        
        for saved_ticker in st.session_state.watchlist:
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(saved_ticker, key=f"wl_{saved_ticker}", use_container_width=True):
                    ticker = saved_ticker
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{saved_ticker}"):
                    st.session_state.watchlist.remove(saved_ticker)
                    st.rerun()
        
        st.markdown("---")
        
        # Quick Stats (will be populated after data fetch)
        st.subheader("💹 Quick Stats")
        stats_placeholder = st.empty()
    
    # Main Content Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Single Stock Analysis",
        "🔍 Bull Run Scanner",
        "📋 Watchlist Analysis",
        "📊 Backtesting",
        "📚 Educational"
    ])
    
    # Fetch data for selected stock
    with st.spinner(f"Fetching data for {ticker}..."):
        df = fetch_stock_data(ticker, period=period_options[period], interval=interval_options[interval])
        
        if df is None or len(df) == 0:
            st.error(f"Could not fetch data for {ticker}. Please check the ticker symbol and try again.")
            return
        
        df = calculate_technical_indicators(df)
        stock_info = fetch_stock_info(ticker)
        options_data = fetch_options_data(ticker)
    
    # Update Quick Stats in sidebar
    with stats_placeholder.container():
        current_price = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
        price_change = current_price - prev_close
        price_change_pct = (price_change / prev_close) * 100
        
        st.metric("Price", f"${current_price:.2f}", 
                 f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
        
        st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}",
                 f"{((df['Volume'].iloc[-1] / df['Volume_SMA'].iloc[-1] - 1) * 100):+.1f}% vs avg")
        
        st.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}",
                 "Overbought" if df['RSI'].iloc[-1] > 70 else 
                 "Oversold" if df['RSI'].iloc[-1] < 30 else "Neutral")
    
    # =============================================================================
    # TAB 1: SINGLE STOCK ANALYSIS
    # =============================================================================
    
    with tab1:
        # Calculate Fibonacci levels
        swing_high, swing_low, trend = get_latest_swing_points(df, lookback=lookback_period)
        fib_levels = calculate_fibonacci_levels(swing_high, swing_low, trend)
        
        # Generate trading signal
        signal = generate_trading_signal(df, fib_levels, current_price)
        
        # Display Signal
        col1, col2, col3 = st.columns([2, 2, 3])
        
        with col1:
            if signal['action'] == 'BUY':
                st.markdown(f"""
                <div class="buy-signal">
                    <h2>🟢 BUY SIGNAL</h2>
                    <h3>Strength: {signal['strength']}/10</h3>
                    <p><b>Entry:</b> ${current_price:.2f}</p>
                    <p><b>Target:</b> ${signal['price_target']:.2f}</p>
                    <p><b>Stop Loss:</b> ${signal['stop_loss']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            elif signal['action'] == 'SELL':
                st.markdown(f"""
                <div class="sell-signal">
                    <h2>🔴 SELL SIGNAL</h2>
                    <h3>Strength: {signal['strength']}/10</h3>
                    <p><b>Exit:</b> ${current_price:.2f}</p>
                    <p><b>Target:</b> ${signal['price_target']:.2f}</p>
                    <p><b>Stop Loss:</b> ${signal['stop_loss']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info(f"**HOLD** - No strong signal (Strength: {signal['strength']}/10)")
        
        with col2:
            st.markdown("### Signal Reasons")
            for reason in signal['reasons']:
                st.markdown(f"✓ {reason}")
        
        with col3:
            st.markdown("### Key Fibonacci Levels")
            
            # Find closest level
            min_diff = float('inf')
            closest_level = None
            
            for level_name, level_price in fib_levels.items():
                diff = abs(current_price - level_price)
                if diff < min_diff:
                    min_diff = diff
                    closest_level = (level_name, level_price)
            
            st.metric("Current Price Position", closest_level[0],
                     f"${min_diff:.2f} away")
            
            # Display key levels
            for level in ["78.6%", "61.8%", "50.0%", "38.2%"]:
                if level in fib_levels:
                    distance = ((current_price - fib_levels[level]) / current_price) * 100
                    st.metric(f"Fib {level}", f"${fib_levels[level]:.2f}",
                             f"{distance:+.2f}%")
        
        st.markdown("---")
        
        # Fibonacci Chart
        st.subheader(f"📊 {ticker} Price Chart with Fibonacci Levels")
        fig = create_fibonacci_chart(df, fib_levels, swing_high, swing_low, ticker)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Technical Analysis Details
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Technical Indicators")
            
            latest = df.iloc[-1]
            
            # Trend Analysis
            if latest['Close'] > latest['SMA_200']:
                trend_text = "🟢 Strong Uptrend (Above 200 SMA)"
            elif latest['Close'] > latest['SMA_50']:
                trend_text = "🟡 Uptrend (Above 50 SMA)"
            elif latest['Close'] > latest['SMA_20']:
                trend_text = "🟠 Short-term Uptrend (Above 20 SMA)"
            else:
                trend_text = "🔴 Downtrend (Below Moving Averages)"
            
            st.info(trend_text)
            
            # RSI Analysis
            rsi = latest['RSI']
            if rsi > 70:
                rsi_text = f"🔴 Overbought (RSI: {rsi:.1f})"
            elif rsi < 30:
                rsi_text = f"🟢 Oversold (RSI: {rsi:.1f})"
            else:
                rsi_text = f"🟡 Neutral (RSI: {rsi:.1f})"
            
            st.info(rsi_text)
            
            # MACD Analysis
            if latest['MACD'] > latest['MACD_Signal']:
                macd_text = "🟢 MACD Bullish (Above Signal Line)"
            else:
                macd_text = "🔴 MACD Bearish (Below Signal Line)"
            
            st.info(macd_text)
            
            # Volume Analysis
            vol_ratio = latest['Volume'] / latest['Volume_SMA']
            if vol_ratio > 1.5:
                vol_text = f"🟢 High Volume ({vol_ratio:.1f}x average)"
            elif vol_ratio > 1.0:
                vol_text = f"🟡 Above Average Volume ({vol_ratio:.1f}x average)"
            else:
                vol_text = f"🔴 Below Average Volume ({vol_ratio:.1f}x average)"
            
            st.info(vol_text)
        
        with col2:
            st.subheader("📋 Fibonacci Level Table")
            
            # Create DataFrame of levels
            fib_df = pd.DataFrame([
                {"Level": k, "Price": f"${v:.2f}", 
                 "Distance": f"{((current_price - v) / current_price * 100):+.2f}%"}
                for k, v in fib_levels.items()
            ])
            
            st.dataframe(fib_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Options Flow Analysis
        st.subheader("📊 Options Flow Analysis")
        
        if "error" not in options_data:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Put/Call Ratio", f"{options_data['put_call_ratio']:.2f}",
                         "Bullish" if options_data['put_call_ratio'] < 0.7 else "Bearish")
            
            with col2:
                st.metric("Call Volume", f"{options_data['total_call_volume']:,.0f}")
            
            with col3:
                st.metric("Put Volume", f"{options_data['total_put_volume']:,.0f}")
            
            # Options chart
            fig_options = create_options_flow_chart(options_data)
            st.plotly_chart(fig_options, use_container_width=True)
        else:
            st.warning(f"Options data not available: {options_data['error']}")
        
        # Export options
        st.markdown("---")
        st.subheader("💾 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV Export
            csv = df.to_csv()
            st.download_button(
                label="📥 Download Price Data (CSV)",
                data=csv,
                file_name=f"{ticker}_data.csv",
                mime="text/csv"
            )
        
        with col2:
            # Fibonacci Levels Export
            fib_csv = pd.DataFrame([
                {"Level": k, "Price": v}
                for k, v in fib_levels.items()
            ]).to_csv(index=False)
            
            st.download_button(
                label="📥 Download Fib Levels (CSV)",
                data=fib_csv,
                file_name=f"{ticker}_fibonacci.csv",
                mime="text/csv"
            )
        
        with col3:
            # Signal Report
            report = f"""
            {ticker} Trading Signal Report
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Current Price: ${current_price:.2f}
            Signal: {signal['action']}
            Strength: {signal['strength']}/10
            
            Reasons:
            {chr(10).join(['- ' + r for r in signal['reasons']])}
            
            Fibonacci Levels:
            {chr(10).join([f"{k}: ${v:.2f}" for k, v in fib_levels.items()])}
            """
            
            st.download_button(
                label="📥 Download Signal Report",
                data=report,
                file_name=f"{ticker}_signal_report.txt",
                mime="text/plain"
            )
    
    # =============================================================================
    # TAB 2: BULL RUN SCANNER
    # =============================================================================
    
    with tab2:
        st.subheader("🔍 Bull Run Probability Scanner")
        st.markdown("Scan multiple stocks to identify those with highest potential for bull runs")
        
        # Stock list selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            scan_option = st.selectbox(
                "Select Stock List",
                ["Watchlist", "Popular Stocks", "Custom List"]
            )
        
        with col2:
            min_score = st.slider("Minimum Score", 0, 100, 50)
        
        # Define stock lists
        if scan_option == "Watchlist":
            stocks_to_scan = st.session_state.watchlist
        elif scan_option == "Popular Stocks":
            stocks_to_scan = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'AMD',
                'NFLX', 'COST', 'AVGO', 'CSCO', 'ADBE', 'PEP', 'INTC'
            ]
        else:
            custom_input = st.text_input("Enter tickers (comma separated)", "AAPL,MSFT,GOOGL")
            stocks_to_scan = [t.strip().upper() for t in custom_input.split(',')]
        
        # Scan button
        if st.button("🔍 Start Scan", type="primary", use_container_width=True):
            progress_bar = st.progress(0, text="Initializing scan...")
            
            with st.spinner("Scanning stocks..."):
                results_df = scan_multiple_stocks(stocks_to_scan, progress_bar)
            
            progress_bar.empty()
            
            if not results_df.empty:
                # Filter by minimum score
                filtered_df = results_df[results_df['Score'] >= min_score]
                
                st.success(f"✅ Scan complete! Found {len(filtered_df)} stocks above {min_score} score")
                
                # Display results
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Score": st.column_config.ProgressColumn(
                            "Score",
                            help="Bull run probability score (0-100)",
                            format="%d",
                            min_value=0,
                            max_value=100,
                        ),
                    }
                )
                
                # Top pick analysis
                if len(filtered_df) > 0:
                    st.markdown("---")
                    st.subheader("⭐ Top Pick Analysis")
                    
                    top_ticker = filtered_df.iloc[0]['Ticker']
                    top_score = filtered_df.iloc[0]['Score']
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f"""
                        ### {top_ticker}
                        **Score: {top_score}/100**
                        
                        {filtered_df.iloc[0]['Signal']}
                        """)
                        
                        # Detailed breakdown
                        st.metric("Options Flow", filtered_df.iloc[0]['Options Flow'], 
                                 help="Score based on options activity")
                        st.metric("Technical", filtered_df.iloc[0]['Technical'],
                                 help="Score based on technical indicators")
                        st.metric("Volume", filtered_df.iloc[0]['Volume'],
                                 help="Score based on volume analysis")
                        st.metric("Fibonacci", filtered_df.iloc[0]['Fibonacci'],
                                 help="Score based on Fibonacci positioning")
                    
                    with col2:
                        # Fetch detailed data for top pick
                        top_df = fetch_stock_data(top_ticker, period="6mo", interval="1d")
                        if top_df is not None:
                            top_df = calculate_technical_indicators(top_df)
                            top_options = fetch_options_data(top_ticker)
                            top_bull_score = calculate_bull_run_score(top_ticker, top_df, top_options)
                            
                            # Gauge chart
                            fig_gauge = create_bull_score_gauge(top_score)
                            st.plotly_chart(fig_gauge, use_container_width=True)
                            
                            # Detailed reasons
                            st.markdown("### Score Breakdown")
                            for key, value in top_bull_score['details'].items():
                                st.info(f"**{key.replace('_', ' ').title()}**: {value}")
                
                # Export results
                st.markdown("---")
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Scan Results (CSV)",
                    data=csv,
                    file_name=f"bull_run_scan_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No results found. Try adjusting your filters or stock list.")
    
    # =============================================================================
    # TAB 3: WATCHLIST ANALYSIS
    # =============================================================================
    
    with tab3:
        st.subheader("📋 Watchlist Analysis")
        st.markdown("Comprehensive analysis of all stocks in your watchlist")
        
        if len(st.session_state.watchlist) == 0:
            st.info("Your watchlist is empty. Add some stocks from the sidebar!")
        else:
            if st.button("🔄 Analyze Watchlist", type="primary"):
                progress_bar = st.progress(0, text="Analyzing watchlist...")
                
                watchlist_data = []
                
                for i, wl_ticker in enumerate(st.session_state.watchlist):
                    progress_bar.progress((i + 1) / len(st.session_state.watchlist), 
                                         text=f"Analyzing {wl_ticker}...")
                    
                    try:
                        wl_df = fetch_stock_data(wl_ticker, period="3mo", interval="1d")
                        if wl_df is None:
                            continue
                        
                        wl_df = calculate_technical_indicators(wl_df)
                        wl_options = fetch_options_data(wl_ticker)
                        
                        # Get Fibonacci signal
                        swing_high, swing_low, trend = get_latest_swing_points(wl_df)
                        fib_levels = calculate_fibonacci_levels(swing_high, swing_low, trend)
                        signal = generate_trading_signal(wl_df, fib_levels, wl_df['Close'].iloc[-1])
                        
                        # Get bull score
                        bull_score = calculate_bull_run_score(wl_ticker, wl_df, wl_options)
                        
                        watchlist_data.append({
                            "Ticker": wl_ticker,
                            "Price": f"${wl_df['Close'].iloc[-1]:.2f}",
                            "Signal": signal['action'],
                            "Strength": signal['strength'],
                            "Bull Score": bull_score['total'],
                            "RSI": f"{wl_df['RSI'].iloc[-1]:.1f}",
                            "Trend": trend.title(),
                            "P/C Ratio": f"{wl_options.get('put_call_ratio', 'N/A'):.2f}" 
                                        if isinstance(wl_options.get('put_call_ratio'), (int, float)) 
                                        else "N/A"
                        })
                    
                    except Exception as e:
                        st.warning(f"Error analyzing {wl_ticker}: {str(e)}")
                        continue
                
                progress_bar.empty()
                
                if watchlist_data:
                    wl_df = pd.DataFrame(watchlist_data)
                    
                    # Display summary
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        buy_signals = len(wl_df[wl_df['Signal'] == 'BUY'])
                        st.metric("Buy Signals", buy_signals)
                    
                    with col2:
                        sell_signals = len(wl_df[wl_df['Signal'] == 'SELL'])
                        st.metric("Sell Signals", sell_signals)
                    
                    with col3:
                        avg_bull_score = wl_df['Bull Score'].mean()
                        st.metric("Avg Bull Score", f"{avg_bull_score:.1f}")
                    
                    with col4:
                        strong_signals = len(wl_df[wl_df['Strength'] >= 7])
                        st.metric("Strong Signals", strong_signals)
                    
                    # Display table
                    st.dataframe(
                        wl_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Signal": st.column_config.TextColumn(
                                "Signal",
                                help="Trading signal",
                            ),
                            "Bull Score": st.column_config.ProgressColumn(
                                "Bull Score",
                                format="%d",
                                min_value=0,
                                max_value=100,
                            ),
                        }
                    )
                    
                    # Export
                    csv = wl_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Watchlist Analysis",
                        data=csv,
                        file_name=f"watchlist_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
    
    # =============================================================================
    # TAB 4: BACKTESTING
    # =============================================================================
    
    with tab4:
        st.subheader("📊 Strategy Backtesting")
        st.markdown(f"Test the Fibonacci strategy on historical {ticker} data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            initial_capital = st.number_input("Initial Capital ($)", 
                                            min_value=1000, 
                                            max_value=1000000, 
                                            value=10000, 
                                            step=1000)
        
        with col2:
            backtest_period = st.selectbox("Backtest Period", 
                                          ["1 Year", "2 Years", "5 Years"],
                                          index=1)
        
        if st.button("🚀 Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                # Fetch data for backtesting
                period_map = {"1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}
                backtest_df = fetch_stock_data(ticker, period=period_map[backtest_period], interval="1d")
                
                if backtest_df is not None and len(backtest_df) >= 100:
                    results = backtest_fibonacci_strategy(backtest_df, initial_capital)
                    
                    # Display results
                    st.success("✅ Backtest Complete!")
                    
                    # Performance metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Return", 
                                 f"{results['total_return']:.2f}%",
                                 f"vs {results['buy_hold_return']:.2f}% B&H")
                    
                    with col2:
                        st.metric("Total Trades", results['total_trades'])
                    
                    with col3:
                        st.metric("Win Rate", f"{results['win_rate']:.1f}%")
                    
                    with col4:
                        st.metric("Max Drawdown", f"{results['max_drawdown']:.2f}%")
                    
                    # Additional metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Winning Trades", results['winning_trades'])
                    
                    with col2:
                        st.metric("Losing Trades", results['losing_trades'])
                    
                    with col3:
                        st.metric("Final Capital", f"${results['final_capital']:.2f}")
                    
                    # Equity curve
                    st.markdown("---")
                    st.subheader("📈 Equity Curve")
                    
                    equity_df = pd.DataFrame({
                        'Equity': results['equity_curve']
                    })
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=equity_df['Equity'],
                        mode='lines',
                        name='Strategy Equity',
                        line=dict(color='#00D4AA', width=2)
                    ))
                    
                    # Add buy & hold comparison
                    buy_hold_equity = [initial_capital]
                    for i in range(100, len(backtest_df)):
                        shares = initial_capital / backtest_df['Close'].iloc[100]
                        equity = shares * backtest_df['Close'].iloc[i]
                        buy_hold_equity.append(equity)
                    
                    fig.add_trace(go.Scatter(
                        y=buy_hold_equity,
                        mode='lines',
                        name='Buy & Hold',
                        line=dict(color='orange', width=2, dash='dash')
                    ))
                    
                    fig.update_layout(
                        title="Strategy Performance vs Buy & Hold",
                        yaxis_title="Portfolio Value ($)",
                        xaxis_title="Time Period",
                        template='plotly_dark',
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Trade history
                    st.markdown("---")
                    st.subheader("📋 Trade History")
                    
                    if results['trades']:
                        trades_df = pd.DataFrame(results['trades'])
                        trades_df['date'] = pd.to_datetime(trades_df['date']).dt.strftime('%Y-%m-%d')
                        trades_df['price'] = trades_df['price'].apply(lambda x: f"${x:.2f}")
                        trades_df['shares'] = trades_df['shares'].apply(lambda x: f"{x:.2f}")
                        
                        if 'profit' in trades_df.columns:
                            trades_df['profit'] = trades_df['profit'].apply(
                                lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
                            )
                        
                        st.dataframe(trades_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No trades executed in backtest period")
                    
                    # Performance analysis
                    st.markdown("---")
                    st.subheader("📊 Performance Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Strategy Strengths")
                        if results['total_return'] > results['buy_hold_return']:
                            st.success("✅ Outperformed buy & hold")
                        if results['win_rate'] > 50:
                            st.success(f"✅ Positive win rate ({results['win_rate']:.1f}%)")
                        if results['max_drawdown'] > -20:
                            st.success(f"✅ Limited drawdown ({results['max_drawdown']:.1f}%)")
                    
                    with col2:
                        st.markdown("### Areas to Consider")
                        if results['total_return'] < results['buy_hold_return']:
                            st.warning("⚠️ Underperformed buy & hold")
                        if results['win_rate'] < 50:
                            st.warning(f"⚠️ Win rate below 50% ({results['win_rate']:.1f}%)")
                        if results['total_trades'] < 5:
                            st.warning("⚠️ Limited number of trades")
                
                else:
                    st.error("Insufficient data for backtesting. Try a shorter period.")
    
    # =============================================================================
    # TAB 5: EDUCATIONAL
    # =============================================================================
    
    with tab5:
        st.subheader("📚 Educational Resources")
        
        # Fibonacci explanation
        with st.expander("🔢 What is Fibonacci Retracement?", expanded=True):
            st.markdown("""
            ### Fibonacci Retracement
            
            Fibonacci retracement is a technical analysis tool that uses horizontal lines to indicate areas of 
            support or resistance at the key Fibonacci levels before the price continues in the original direction.
            
            **Key Levels:**
            - **23.6%**: Minor retracement level
            - **38.2%**: Moderate support/resistance
            - **50.0%**: Psychological level (not a Fibonacci ratio but widely watched)
            - **61.8%**: Golden ratio - most important level
            - **78.6%**: Strong support/resistance
            
            **Extension Levels:**
            - **127.2%**: First profit target
            - **161.8%**: Major extension level
            - **261.8%**: Extreme extension
            
            ### How to Use
            
            1. **Identify Trend**: Find a significant swing high and swing low
            2. **Draw Levels**: Calculate Fibonacci ratios between these points
            3. **Watch for Reactions**: Price often bounces at these levels
            4. **Confirm with Indicators**: Use RSI, MACD, volume for confirmation
            5. **Set Stop Loss**: Below key support levels
            """)
        
        with st.expander("📊 Understanding Options Flow"):
            st.markdown("""
            ### Options Flow Analysis
            
            Options flow refers to the real-time tracking of options contracts being traded.
            
            **Key Metrics:**
            
            - **Put/Call Ratio**: Ratio of put options to call options
              - < 0.7: Bullish sentiment
              - 0.7 - 1.0: Neutral
              - > 1.0: Bearish sentiment
            
            - **Open Interest**: Total number of outstanding options contracts
              - Increasing call OI: Bullish
              - Increasing put OI: Bearish
            
            - **Volume Surges**: Unusual high volume can indicate smart money activity
            
            - **Implied Volatility**: Expected volatility priced into options
            
            ### Bull Run Indicators
            
            1. **High call volume** relative to puts
            2. **Low put/call ratio** (< 0.7)
            3. **Increasing open interest** in calls
            4. **Large block trades** in call options
            5. **LEAP activity** (long-term bullish bets)
            """)
        
        with st.expander("💡 Trading Strategy Tips"):
            st.markdown("""
            ### Best Practices
            
            1. **Risk Management**
               - Never risk more than 1-2% per trade
               - Always use stop losses
               - Position sizing is crucial
            
            2. **Entry Timing**
               - Wait for confirmation at Fibonacci levels
               - Look for volume increase
               - Check multiple timeframes
            
            3. **Exit Strategy**
               - Take partial profits at targets
               - Trail stop loss as price moves favorably
               - Don't be greedy at extension levels
            
            4. **Confirmation Signals**
               - RSI divergence
               - MACD crossover
               - Volume spike
               - Candlestick patterns
            
            5. **Common Mistakes to Avoid**
               - Trading without stop loss
               - Ignoring the overall trend
               - Over-trading
               - Not waiting for confirmation
               - Risking too much per trade
            """)
        
        with st.expander("⚠️ Disclaimer"):
            st.markdown("""
            ### Important Legal Disclaimer
            
            **This dashboard is for educational and informational purposes only.**
            
            - ❌ **NOT FINANCIAL ADVICE**: Nothing on this platform constitutes financial advice
            - ❌ **NOT INVESTMENT RECOMMENDATION**: This is not a recommendation to buy or sell securities
            - ❌ **PAST PERFORMANCE**: Past performance does not guarantee future results
            - ❌ **RISK WARNING**: Trading stocks and options involves substantial risk of loss
            - ❌ **DO YOUR RESEARCH**: Always conduct your own research and due diligence
            
            **Risk Factors:**
            - You can lose your entire investment
            - Options trading is especially risky
            - Markets are unpredictable
            - Technical analysis is not foolproof
            - Always consult with a licensed financial advisor
            
            **Data Accuracy:**
            - Data is provided "as-is" from public sources
            - We make no guarantees about accuracy or completeness
            - Real-time data may have delays
            - Always verify information from multiple sources
            
            By using this dashboard, you acknowledge that:
            - You understand the risks involved in trading
            - You are solely responsible for your trading decisions
            - You will not hold the creators liable for any losses
            - You have read and understood this disclaimer
            """)
        
        # Video tutorials section
        st.markdown("---")
        st.subheader("🎥 Video Tutorials")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Recommended Learning Resources
            
            1. **Fibonacci Trading Basics**
               - Introduction to Fibonacci sequences
               - Drawing retracement levels
               - Common patterns
            
            2. **Options Trading 101**
               - What are options?
               - Calls vs Puts
               - Reading options chains
            
            3. **Technical Analysis**
               - Chart patterns
               - Indicators (RSI, MACD)
               - Volume analysis
            """)
        
        with col2:
            st.markdown("""
            ### Practice Recommendations
            
            1. **Paper Trading**
               - Practice with virtual money first
               - Test strategies without risk
               - Build confidence
            
            2. **Start Small**
               - Begin with small position sizes
               - Focus on learning, not profits
               - Gradually increase as you gain experience
            
            3. **Keep a Journal**
               - Document all trades
               - Note reasons for entry/exit
               - Review and learn from mistakes
            """)

# =============================================================================
# FOOTER
# =============================================================================

def show_footer():
    """Display footer with disclaimer and credits"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **⚠️ Risk Warning**
        
        Trading involves substantial risk. 
        Past performance does not guarantee future results.
        Always do your own research.
        """)
    
    with col2:
        st.markdown("""
        **📊 Data Sources**
        
        - Yahoo Finance (yfinance)
        - Real-time options data
        - Historical price data
        """)
    
    with col3:
        st.markdown("""
        **📝 Version Info**
        
        Dashboard v1.0.0
        
        Built with Streamlit
        """)
    
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>© 2024 Fibonacci Trading Dashboard | "
        "For Educational Purposes Only | Not Financial Advice</p>",
        unsafe_allow_html=True
    )

# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()
    show_footer()
