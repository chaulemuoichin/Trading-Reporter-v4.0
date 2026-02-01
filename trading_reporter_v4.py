#!/usr/bin/env python3
"""
================================================================================
TRADING REPORTER v4.0 - COMPLETE EDITION
================================================================================
Features:
    1. Better News (Yahoo + Google RSS + Finviz)
    2. Technical Indicators (RSI, MACD, Bollinger, MA50/200, ATR)
    3. Smarter Event Detection (gaps, volume spikes, earnings)
    4. Better Rule-Based Analysis (1-10 scoring system)
    5. Sector Comparison (vs ETF benchmarks)
    6. Trading Signals (Golden Cross, Death Cross, Oversold, etc.)
    7. Focused Gemini AI (ONLY analyzes price events + news - minimal tokens!)

Install:
    pip install yfinance pandas numpy requests google-genai

Usage:
    python trading_reporter_v4.py AAPL
    python trading_reporter_v4.py NVDA MSFT --no-ai
    python trading_reporter_v4.py GOOG --debug
================================================================================
"""

import os
import sys
import json
import time
import random
import hashlib
import logging
import argparse
import warnings
import re
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from xml.etree import ElementTree
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings('ignore')

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# IMPORTS
# =============================================================================

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    print("⚠ yfinance not installed - run: pip install yfinance")

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 2.0
    max_delay: float = 30.0
    jitter_min: float = 0.7
    jitter_max: float = 1.3

@dataclass 
class ValuationConfig:
    risk_free_rate: float = 0.045
    equity_risk_premium: float = 0.055
    default_tax_rate: float = 0.21
    cost_of_debt_spread: float = 0.02
    dcf_growth_short: float = 0.08
    dcf_growth_terminal: float = 0.025
    dcf_forecast_years: int = 5
    wacc_floor: float = 0.05
    wacc_cap: float = 0.20
    fair_pe_multiple: float = 20.0
    fair_pb_multiple: float = 3.5
    fair_ev_ebitda_multiple: float = 12.0
    epv_required_return: float = 0.10
    dcf_weight: float = 0.40
    multiples_weight: float = 0.35
    epv_weight: float = 0.25

@dataclass
class Config:
    gemini_api_key: Optional[str] = field(default_factory=lambda: os.getenv('GEMINI_API_KEY'))
    output_dir: str = 'reports'
    cache_dir: str = 'cache'
    performance_windows: List[int] = field(default_factory=lambda: [1, 7, 30, 90])
    price_history_days: int = 400
    news_lookback_days: int = 30
    valuation: ValuationConfig = field(default_factory=ValuationConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    z_threshold: float = 2.5
    min_abs_return: float = 0.025
    min_volume_ratio: float = 1.8
    rolling_window: int = 20
    news_window_minutes: List[int] = field(default_factory=lambda: [120, 480, 1440])
    max_news_per_event: int = 3
    throttle: float = 1.5
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    @property
    def ai_enabled(self) -> bool:
        return bool(self.gemini_api_key and GEMINI_AVAILABLE)

CONFIG = Config()

# Sector ETF mappings
SECTOR_ETFS = {
    'Technology': 'XLK', 'Healthcare': 'XLV', 'Financial Services': 'XLF',
    'Financials': 'XLF', 'Consumer Cyclical': 'XLY', 'Consumer Defensive': 'XLP',
    'Industrials': 'XLI', 'Energy': 'XLE', 'Utilities': 'XLU',
    'Real Estate': 'XLRE', 'Materials': 'XLB', 'Communication Services': 'XLC',
    'Basic Materials': 'XLB',
}

# =============================================================================
# UTILITIES
# =============================================================================

def safe_number(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        if pd.isna(value) or np.isinf(value):
            return default
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(',', '').replace('$', ''))
        except:
            return default
    return default

def fmt_large(n: float) -> str:
    if n is None:
        return "N/A"
    if n >= 1e12:
        return f"${n/1e12:.1f}T"
    if n >= 1e9:
        return f"${n/1e9:.1f}B"
    if n >= 1e6:
        return f"${n/1e6:.1f}M"
    return f"${n:,.0f}"

# =============================================================================
# 2. TECHNICAL INDICATORS
# =============================================================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_macd(prices: pd.Series) -> Dict:
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return {'macd': macd, 'signal': signal, 'histogram': histogram}

def calculate_bollinger(prices: pd.Series, period: int = 20) -> Dict:
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    return {'upper': sma + 2*std, 'middle': sma, 'lower': sma - 2*std}

def calculate_moving_averages(prices: pd.Series) -> Dict:
    return {
        'sma_10': prices.rolling(10).mean(),
        'sma_20': prices.rolling(20).mean(),
        'sma_50': prices.rolling(50).mean(),
        'sma_200': prices.rolling(200).mean(),
    }

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df['high'], df['low'], df['close'].shift(1)
    tr = pd.concat([h-l, abs(h-c), abs(l-c)], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calculate_support_resistance(df: pd.DataFrame) -> Dict:
    recent = df.tail(20)
    pivot = (recent['high'].iloc[-1] + recent['low'].iloc[-1] + recent['close'].iloc[-1]) / 3
    r1 = 2 * pivot - recent['low'].iloc[-1]
    s1 = 2 * pivot - recent['high'].iloc[-1]
    return {
        'pivot': round(pivot, 2),
        'resistance_1': round(r1, 2),
        'support_1': round(s1, 2),
    }

def get_technicals(df: pd.DataFrame) -> Dict:
    if df is None or len(df) < 50:
        return {}
    
    close = df['close']
    price = float(close.iloc[-1])
    
    # RSI
    rsi_series = calculate_rsi(close)
    rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else None
    
    # MACD
    macd_data = calculate_macd(close)
    macd = float(macd_data['macd'].iloc[-1]) if not pd.isna(macd_data['macd'].iloc[-1]) else None
    macd_signal = float(macd_data['signal'].iloc[-1]) if not pd.isna(macd_data['signal'].iloc[-1]) else None
    macd_hist = float(macd_data['histogram'].iloc[-1]) if not pd.isna(macd_data['histogram'].iloc[-1]) else None
    
    # Bollinger
    bb = calculate_bollinger(close)
    bb_upper = float(bb['upper'].iloc[-1]) if not pd.isna(bb['upper'].iloc[-1]) else None
    bb_lower = float(bb['lower'].iloc[-1]) if not pd.isna(bb['lower'].iloc[-1]) else None
    bb_mid = float(bb['middle'].iloc[-1]) if not pd.isna(bb['middle'].iloc[-1]) else None
    
    # Moving Averages
    mas = calculate_moving_averages(close)
    sma_50 = float(mas['sma_50'].iloc[-1]) if not pd.isna(mas['sma_50'].iloc[-1]) else None
    sma_200 = float(mas['sma_200'].iloc[-1]) if not pd.isna(mas['sma_200'].iloc[-1]) else None
    sma_20 = float(mas['sma_20'].iloc[-1]) if not pd.isna(mas['sma_20'].iloc[-1]) else None
    
    # ATR
    atr = calculate_atr(df)
    atr_val = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None
    
    # Support/Resistance
    sr = calculate_support_resistance(df)
    
    # BB position (0-100)
    bb_pos = None
    if bb_upper and bb_lower and bb_upper > bb_lower:
        bb_pos = (price - bb_lower) / (bb_upper - bb_lower) * 100
    
    return {
        'rsi': round(rsi, 1) if rsi else None,
        'rsi_signal': 'OVERSOLD' if rsi and rsi < 30 else ('OVERBOUGHT' if rsi and rsi > 70 else 'NEUTRAL'),
        'macd': round(macd, 3) if macd else None,
        'macd_signal': round(macd_signal, 3) if macd_signal else None,
        'macd_histogram': round(macd_hist, 3) if macd_hist else None,
        'macd_trend': 'BULLISH' if macd_hist and macd_hist > 0 else 'BEARISH',
        'bb_upper': round(bb_upper, 2) if bb_upper else None,
        'bb_middle': round(bb_mid, 2) if bb_mid else None,
        'bb_lower': round(bb_lower, 2) if bb_lower else None,
        'bb_position': round(bb_pos, 1) if bb_pos else None,
        'sma_20': round(sma_20, 2) if sma_20 else None,
        'sma_50': round(sma_50, 2) if sma_50 else None,
        'sma_200': round(sma_200, 2) if sma_200 else None,
        'above_sma_50': price > sma_50 if sma_50 else None,
        'above_sma_200': price > sma_200 if sma_200 else None,
        'atr': round(atr_val, 2) if atr_val else None,
        'atr_pct': round(atr_val/price*100, 2) if atr_val else None,
        **sr
    }

# =============================================================================
# 6. TRADING SIGNALS / ALERTS
# =============================================================================

def detect_signals(df: pd.DataFrame, technicals: Dict) -> List[Dict]:
    signals = []
    if df is None or len(df) < 200:
        return signals
    
    close = df['close']
    price = float(close.iloc[-1])
    mas = calculate_moving_averages(close)
    
    # Golden Cross / Death Cross
    sma_50, sma_200 = mas['sma_50'], mas['sma_200']
    if len(sma_50) > 2 and not pd.isna(sma_50.iloc[-1]) and not pd.isna(sma_200.iloc[-1]):
        if sma_50.iloc[-1] > sma_200.iloc[-1] and sma_50.iloc[-2] <= sma_200.iloc[-2]:
            signals.append({'signal': 'GOLDEN_CROSS', 'type': 'BULLISH', 'strength': 'STRONG',
                           'desc': '50-day MA crossed ABOVE 200-day MA'})
        elif sma_50.iloc[-1] < sma_200.iloc[-1] and sma_50.iloc[-2] >= sma_200.iloc[-2]:
            signals.append({'signal': 'DEATH_CROSS', 'type': 'BEARISH', 'strength': 'STRONG',
                           'desc': '50-day MA crossed BELOW 200-day MA'})
    
    # RSI
    rsi = technicals.get('rsi')
    if rsi:
        if rsi < 30:
            signals.append({'signal': 'RSI_OVERSOLD', 'type': 'BULLISH', 'strength': 'MODERATE',
                           'desc': f'RSI={rsi:.0f} - Oversold, potential bounce'})
        elif rsi > 70:
            signals.append({'signal': 'RSI_OVERBOUGHT', 'type': 'BEARISH', 'strength': 'MODERATE',
                           'desc': f'RSI={rsi:.0f} - Overbought, potential pullback'})
    
    # MACD Crossover
    macd_data = calculate_macd(close)
    if len(macd_data['macd']) > 2:
        m, s = macd_data['macd'], macd_data['signal']
        if m.iloc[-1] > s.iloc[-1] and m.iloc[-2] <= s.iloc[-2]:
            signals.append({'signal': 'MACD_BULLISH', 'type': 'BULLISH', 'strength': 'MODERATE',
                           'desc': 'MACD crossed above signal line'})
        elif m.iloc[-1] < s.iloc[-1] and m.iloc[-2] >= s.iloc[-2]:
            signals.append({'signal': 'MACD_BEARISH', 'type': 'BEARISH', 'strength': 'MODERATE',
                           'desc': 'MACD crossed below signal line'})
    
    # Bollinger Band touch
    bb_pos = technicals.get('bb_position')
    if bb_pos:
        if bb_pos < 5:
            signals.append({'signal': 'BB_OVERSOLD', 'type': 'BULLISH', 'strength': 'MODERATE',
                           'desc': 'Price at lower Bollinger Band'})
        elif bb_pos > 95:
            signals.append({'signal': 'BB_OVERBOUGHT', 'type': 'BEARISH', 'strength': 'MODERATE',
                           'desc': 'Price at upper Bollinger Band'})
    
    # 52-Week High/Low
    h52 = df['high'].tail(252).max()
    l52 = df['low'].tail(252).min()
    if price >= h52 * 0.98:
        signals.append({'signal': '52W_HIGH', 'type': 'BULLISH', 'strength': 'STRONG',
                       'desc': f'Near 52-week high (${h52:.2f})'})
    elif price <= l52 * 1.02:
        signals.append({'signal': '52W_LOW', 'type': 'BEARISH', 'strength': 'STRONG',
                       'desc': f'Near 52-week low (${l52:.2f})'})
    
    # Price vs 200 MA
    sma_200_val = technicals.get('sma_200')
    if sma_200_val:
        prev = float(close.iloc[-2])
        if price > sma_200_val and prev <= sma_200_val:
            signals.append({'signal': 'BREAK_ABOVE_200MA', 'type': 'BULLISH', 'strength': 'STRONG',
                           'desc': f'Broke above 200-day MA (${sma_200_val:.2f})'})
        elif price < sma_200_val and prev >= sma_200_val:
            signals.append({'signal': 'BREAK_BELOW_200MA', 'type': 'BEARISH', 'strength': 'STRONG',
                           'desc': f'Broke below 200-day MA (${sma_200_val:.2f})'})
    
    # Volume spike
    if 'volume' in df.columns:
        avg_vol = df['volume'].tail(20).mean()
        cur_vol = df['volume'].iloc[-1]
        if cur_vol > avg_vol * 2:
            signals.append({'signal': 'VOLUME_SPIKE', 'type': 'NEUTRAL', 'strength': 'MODERATE',
                           'desc': f'Volume {cur_vol/avg_vol:.1f}x above average'})
    
    return signals

# =============================================================================
# 1. BETTER NEWS FETCHING
# =============================================================================

def fetch_news_yahoo(ticker: str, days: int = 30) -> List[Dict]:
    if not YF_AVAILABLE:
        return []
    news_list = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    try:
        stock = yf.Ticker(ticker)
        for item in (stock.news or []):
            pub = item.get('providerPublishTime')
            if not pub:
                continue
            dt = pd.Timestamp(pub, unit='s', tz='UTC')
            if dt < cutoff:
                continue
            news_list.append({
                'title': item.get('title', ''),
                'publisher': item.get('publisher', 'Yahoo'),
                'timestamp': pub,
                'dt': dt,
                'datetime_str': dt.strftime('%Y-%m-%d %H:%M'),
                'source': 'yahoo'
            })
    except:
        pass
    return news_list

def fetch_news_google(ticker: str, company_name: str = "", days: int = 30) -> List[Dict]:
    news_list = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    
    terms = [f"{ticker} stock"]
    if company_name:
        clean = company_name.replace(' Inc.', '').replace(' Corp.', '').split(',')[0]
        terms.append(f"{clean} stock")
    
    for term in terms[:2]:
        try:
            url = f"https://news.google.com/rss/search?q={quote(term)}&hl=en-US&gl=US&ceid=US:en"
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
            if resp.status_code != 200:
                continue
            
            root = ElementTree.fromstring(resp.content)
            for item in root.findall('.//item')[:10]:
                title_el = item.find('title')
                pub_el = item.find('pubDate')
                if title_el is None:
                    continue
                
                title = title_el.text or ''
                try:
                    dt = pd.to_datetime(pub_el.text) if pub_el is not None else pd.Timestamp.now(tz='UTC')
                    if dt.tzinfo is None:
                        dt = dt.tz_localize('UTC')
                except:
                    dt = pd.Timestamp.now(tz='UTC')
                
                if dt < cutoff:
                    continue
                if any(title[:30].lower() == n['title'][:30].lower() for n in news_list):
                    continue
                
                news_list.append({
                    'title': title,
                    'publisher': 'Google News',
                    'timestamp': int(dt.timestamp()),
                    'dt': dt,
                    'datetime_str': dt.strftime('%Y-%m-%d %H:%M'),
                    'source': 'google'
                })
        except:
            continue
    return news_list

def fetch_news_finviz(ticker: str) -> List[Dict]:
    news_list = []
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
        if resp.status_code != 200:
            return []
        
        pattern = r'<a[^>]*class="tab-link-news"[^>]*>([^<]*)</a>'
        matches = re.findall(pattern, resp.text)
        
        dt = pd.Timestamp.now(tz='UTC')
        for title in matches[:8]:
            if not title.strip():
                continue
            news_list.append({
                'title': title.strip(),
                'publisher': 'Finviz',
                'timestamp': int(dt.timestamp()),
                'dt': dt,
                'datetime_str': dt.strftime('%Y-%m-%d %H:%M'),
                'source': 'finviz'
            })
    except:
        pass
    return news_list

def fetch_all_news(ticker: str, company_name: str = "") -> List[Dict]:
    all_news = []
    seen = set()
    
    for n in fetch_news_yahoo(ticker, CONFIG.news_lookback_days):
        key = n['title'][:30].lower()
        if key not in seen:
            all_news.append(n)
            seen.add(key)
    
    for n in fetch_news_google(ticker, company_name, CONFIG.news_lookback_days):
        key = n['title'][:30].lower()
        if key not in seen:
            all_news.append(n)
            seen.add(key)
    
    for n in fetch_news_finviz(ticker):
        key = n['title'][:30].lower()
        if key not in seen:
            all_news.append(n)
            seen.add(key)
    
    all_news.sort(key=lambda x: x['timestamp'], reverse=True)
    return all_news[:20]

# =============================================================================
# 3. SMARTER EVENT DETECTION
# =============================================================================

def detect_events(df: pd.DataFrame, earnings: List[Dict] = None) -> List[Dict]:
    if df is None or len(df) < 25:
        return []
    
    df = df.copy().sort_index()
    df['ret_std'] = df['return'].rolling(CONFIG.rolling_window).std().replace(0, np.nan)
    df['z'] = df['return'] / df['ret_std']
    df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    
    events = []
    for i in range(CONFIG.rolling_window, len(df)):
        row = df.iloc[i]
        ret = row.get('return', 0)
        z = row.get('z', 0)
        vol_ratio = row.get('volume_ratio', 1)
        gap = row.get('gap', 0)
        
        if pd.isna(ret) or pd.isna(z):
            continue
        
        reasons = []
        event_type = None
        
        if abs(z) >= CONFIG.z_threshold:
            event_type = 'surge' if ret > 0 else 'drop'
            reasons.append(f"Z={z:.1f}")
        
        if abs(ret) >= CONFIG.min_abs_return and vol_ratio >= CONFIG.min_volume_ratio:
            event_type = 'surge' if ret > 0 else 'drop'
            reasons.append(f"Vol={vol_ratio:.1f}x")
        
        if abs(gap) >= 2.0:
            event_type = 'gap_up' if gap > 0 else 'gap_down'
            reasons.append(f"Gap={gap:+.1f}%")
        
        if event_type:
            ts = row.name
            if not isinstance(ts, pd.Timestamp):
                ts = pd.Timestamp(ts)
            if ts.tzinfo is None:
                ts = ts.tz_localize('UTC')
            
            # Check if near earnings
            is_earnings = False
            if earnings:
                for e in earnings:
                    ed = e.get('date')
                    if ed and abs((ts - ed).days) <= 2:
                        is_earnings = True
                        reasons.append("Earnings")
                        break
            
            events.append({
                'timestamp': ts,
                'date_str': ts.strftime('%Y-%m-%d'),
                'type': event_type,
                'return_pct': round(ret * 100, 2),
                'gap_pct': round(gap, 2) if abs(gap) >= 1 else None,
                'volume_ratio': round(vol_ratio, 1) if not pd.isna(vol_ratio) else None,
                'is_earnings': is_earnings,
                'reasons': reasons,
                'matched_news': [],
            })
    
    return sorted(events, key=lambda x: abs(x['return_pct']), reverse=True)[:10]

def match_news_to_events(events: List[Dict], news: List[Dict]) -> List[Dict]:
    if not events or not news:
        return events
    
    for event in events:
        event_ts = event.get('timestamp')
        if not isinstance(event_ts, pd.Timestamp):
            event_ts = pd.Timestamp(event_ts, tz='UTC')
        
        matched = []
        for window in CONFIG.news_window_minutes:
            for n in news:
                news_dt = n.get('dt')
                if news_dt.tzinfo is None:
                    news_dt = news_dt.tz_localize('UTC')
                
                diff = abs((news_dt - event_ts).total_seconds() / 60)
                if diff <= window:
                    matched.append({
                        'title': n['title'],
                        'publisher': n['publisher'],
                        'time_delta_min': round(diff, 0),
                    })
            if matched:
                break
        
        event['matched_news'] = matched[:CONFIG.max_news_per_event]
    return events

# =============================================================================
# 5. SECTOR COMPARISON
# =============================================================================

def fetch_sector_data(sector: str, days: int = 30) -> Optional[Dict]:
    etf = SECTOR_ETFS.get(sector)
    if not etf or not YF_AVAILABLE:
        return None
    
    try:
        stock = yf.Ticker(etf)
        df = stock.history(period=f"{days+10}d")
        if df.empty:
            return None
        
        current = float(df['Close'].iloc[-1])
        returns = {}
        for w in [1, 7, 30]:
            if len(df) >= w + 1:
                start = float(df['Close'].iloc[-1-w])
                returns[w] = round((current/start - 1) * 100, 2)
        
        return {'etf': etf, 'sector': sector, 'returns': returns}
    except:
        return None

def compare_to_sector(stock_perf: Dict, sector_data: Optional[Dict]) -> Dict:
    if not sector_data:
        return {'comparison': 'N/A', 'details': {}}
    
    details = {}
    outperform = 0
    
    for w in [1, 7, 30]:
        stock_ret = stock_perf.get('windows', {}).get(w, {}).get('return_pct', 0)
        sector_ret = sector_data.get('returns', {}).get(w, 0)
        diff = stock_ret - sector_ret
        details[f'{w}d'] = {'stock': stock_ret, 'sector': sector_ret, 'diff': round(diff, 2)}
        if diff > 0:
            outperform += 1
    
    verdict = 'OUTPERFORMING' if outperform >= 2 else ('UNDERPERFORMING' if outperform == 0 else 'INLINE')
    return {'comparison': verdict, 'etf': sector_data.get('etf'), 'details': details}

# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_price_data(ticker: str, days: int = 400) -> Optional[pd.DataFrame]:
    if not YF_AVAILABLE:
        return None
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=f"{days}d")
        if df.empty:
            return None
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize('UTC')
        df = df.sort_index()
        df['return'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        return df
    except Exception as e:
        return None

def fetch_fundamentals(ticker: str) -> Dict:
    if not YF_AVAILABLE:
        return {}
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        data = {
            'ticker': ticker,
            'name': info.get('longName') or info.get('shortName') or ticker,
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'current_price': safe_number(info.get('currentPrice')) or safe_number(info.get('regularMarketPrice')),
            'fifty_two_week_high': safe_number(info.get('fiftyTwoWeekHigh')),
            'fifty_two_week_low': safe_number(info.get('fiftyTwoWeekLow')),
            'beta': safe_number(info.get('beta'), 1.0),
            'market_cap': safe_number(info.get('marketCap')),
            'shares_outstanding': safe_number(info.get('sharesOutstanding')),
            'pe_trailing': safe_number(info.get('trailingPE')),
            'pe_forward': safe_number(info.get('forwardPE')),
            'pb_ratio': safe_number(info.get('priceToBook')),
            'profit_margin': safe_number(info.get('profitMargins')),
            'roe': safe_number(info.get('returnOnEquity')),
            'roa': safe_number(info.get('returnOnAssets')),
            'revenue_growth': safe_number(info.get('revenueGrowth')),
            'debt_to_equity': safe_number(info.get('debtToEquity')),
            'total_debt': safe_number(info.get('totalDebt')),
            'total_cash': safe_number(info.get('totalCash')),
            'free_cash_flow': safe_number(info.get('freeCashflow')),
            'eps_trailing': safe_number(info.get('trailingEps')),
            'book_value': safe_number(info.get('bookValue')),
            'ebitda': safe_number(info.get('ebitda')),
            'interest_expense': safe_number(info.get('interestExpense')),
            'target_mean': safe_number(info.get('targetMeanPrice')),
            'recommendation': info.get('recommendationKey'),
        }
        if not data['shares_outstanding'] and data['current_price'] and data['market_cap']:
            data['shares_outstanding'] = data['market_cap'] / data['current_price']
        return data
    except:
        return {}

def fetch_earnings(ticker: str) -> List[Dict]:
    if not YF_AVAILABLE:
        return []
    try:
        stock = yf.Ticker(ticker)
        ed = stock.earnings_dates
        if ed is None or ed.empty:
            return []
        
        earnings = []
        for idx, row in ed.head(8).iterrows():
            dt = pd.to_datetime(idx)
            if dt.tzinfo is None:
                dt = dt.tz_localize('UTC')
            earnings.append({
                'date': dt,
                'date_str': dt.strftime('%Y-%m-%d'),
                'eps_estimate': safe_number(row.get('EPS Estimate')),
                'eps_actual': safe_number(row.get('Reported EPS')),
                'surprise_pct': safe_number(row.get('Surprise(%)')),
            })
        return earnings
    except:
        return []

# =============================================================================
# PERFORMANCE ANALYSIS
# =============================================================================

def analyze_performance(df: pd.DataFrame) -> Dict:
    if df is None or df.empty:
        return {}
    
    df = df.sort_index()
    current = float(df['close'].iloc[-1])
    results = {'current_price': round(current, 2), 'windows': {}}
    
    for w in CONFIG.performance_windows:
        if len(df) < w + 1:
            continue
        start = float(df['close'].iloc[-1-w])
        ret = (current/start - 1) if start > 0 else 0
        wdf = df.iloc[-w:]
        log_ret = wdf['log_return'].dropna()
        vol = float(log_ret.std()) * np.sqrt(252) if len(log_ret) >= 2 else None
        
        results['windows'][w] = {
            'return_pct': round(ret * 100, 2),
            'vol_ann_pct': round(vol * 100, 2) if vol else None,
            'high': round(float(wdf['high'].max()), 2),
            'low': round(float(wdf['low'].min()), 2),
        }
    return results

# =============================================================================
# VALUATION
# =============================================================================

def calculate_wacc(info: Dict) -> float:
    cfg = CONFIG.valuation
    beta = max(0.3, min(safe_number(info.get('beta'), 1.0), 3.0))
    coe = cfg.risk_free_rate + beta * cfg.equity_risk_premium
    mcap = safe_number(info.get('market_cap'))
    debt = safe_number(info.get('total_debt'), 0)
    e_wt = mcap / (mcap + debt) if mcap else 0.8
    d_wt = min(1 - e_wt, 0.5)
    e_wt = 1 - d_wt
    
    interest = safe_number(info.get('interest_expense'))
    cod = abs(interest)/debt if interest and debt else cfg.risk_free_rate + cfg.cost_of_debt_spread
    cod_post = cod * (1 - cfg.default_tax_rate)
    
    wacc = e_wt * coe + d_wt * cod_post
    return max(cfg.wacc_floor, min(cfg.wacc_cap, wacc))

def dcf_valuation(info: Dict) -> Dict:
    cfg = CONFIG.valuation
    fcf = safe_number(info.get('free_cash_flow'))
    shares = safe_number(info.get('shares_outstanding'))
    if not fcf or fcf <= 0 or not shares:
        return {'intrinsic_value': None}
    
    debt = safe_number(info.get('total_debt'), 0)
    cash = safe_number(info.get('total_cash'), 0)
    wacc = calculate_wacc(info)
    g_s, g_t = cfg.dcf_growth_short, cfg.dcf_growth_terminal
    if wacc <= g_t:
        wacc = g_t + 0.03
    
    pv, cf = 0, fcf
    for y in range(1, cfg.dcf_forecast_years + 1):
        cf *= (1 + g_s)
        pv += cf / ((1 + wacc) ** y)
    
    tv = cf * (1 + g_t) / (wacc - g_t)
    pv_tv = tv / ((1 + wacc) ** cfg.dcf_forecast_years)
    ev = pv + pv_tv
    iv = (ev - (debt - cash)) / shares
    
    return {'method': 'DCF', 'intrinsic_value': round(iv, 2) if iv > 0 else None}

def multiples_valuation(info: Dict) -> Dict:
    cfg = CONFIG.valuation
    implied = []
    
    eps = safe_number(info.get('eps_trailing'))
    if eps and eps > 0:
        implied.append(eps * cfg.fair_pe_multiple)
    
    bv = safe_number(info.get('book_value'))
    if bv and bv > 0:
        implied.append(bv * cfg.fair_pb_multiple)
    
    ebitda = safe_number(info.get('ebitda'))
    shares = safe_number(info.get('shares_outstanding'))
    debt = safe_number(info.get('total_debt'), 0)
    cash = safe_number(info.get('total_cash'), 0)
    if ebitda and ebitda > 0 and shares:
        ev = ebitda * cfg.fair_ev_ebitda_multiple
        p = (ev - (debt - cash)) / shares
        if p > 0:
            implied.append(p)
    
    target = safe_number(info.get('target_mean'))
    if target:
        implied.append(target)
    
    if not implied:
        return {'intrinsic_value': None}
    
    return {
        'method': 'Multiples',
        'intrinsic_value': round(float(np.median(implied)), 2),
        'range_low': round(min(implied), 2),
        'range_high': round(max(implied), 2),
    }

def epv_valuation(info: Dict) -> Dict:
    eps = safe_number(info.get('eps_trailing'))
    if not eps or eps <= 0:
        return {'intrinsic_value': None}
    return {'method': 'EPV', 'intrinsic_value': round(eps / CONFIG.valuation.epv_required_return, 2)}

def comprehensive_valuation(info: Dict) -> Dict:
    cfg = CONFIG.valuation
    cp = safe_number(info.get('current_price'), 0)
    
    dcf = dcf_valuation(info)
    mult = multiples_valuation(info)
    epv = epv_valuation(info)
    
    values = []
    if dcf.get('intrinsic_value'):
        values.append(('DCF', dcf['intrinsic_value'], cfg.dcf_weight))
    if mult.get('intrinsic_value'):
        values.append(('Multiples', mult['intrinsic_value'], cfg.multiples_weight))
    if epv.get('intrinsic_value'):
        values.append(('EPV', epv['intrinsic_value'], cfg.epv_weight))
    
    if not values:
        return {'current_price': cp, 'intrinsic_value': None, 'verdict': 'INSUFFICIENT DATA'}
    
    tot_wt = sum(w for _, _, w in values)
    weighted = sum(v * w for _, v, w in values) / tot_wt
    all_v = [v for _, v, _ in values]
    r_lo, r_hi = min(all_v), max(all_v)
    up = ((weighted / cp) - 1) * 100 if cp > 0 else 0
    
    if up > 25: verdict = 'SIGNIFICANTLY UNDERVALUED'
    elif up > 10: verdict = 'UNDERVALUED'
    elif up > -10: verdict = 'FAIRLY VALUED'
    elif up > -25: verdict = 'OVERVALUED'
    else: verdict = 'SIGNIFICANTLY OVERVALUED'
    
    return {
        'current_price': round(cp, 2),
        'intrinsic_value': round(weighted, 2),
        'range_low': round(r_lo, 2),
        'range_high': round(r_hi, 2),
        'upside_pct': round(up, 1),
        'verdict': verdict,
        'components': [{'method': m, 'value': round(v, 2), 'weight': w} for m, v, w in values],
    }

# =============================================================================
# 4. BETTER RULE-BASED ANALYSIS (SCORING)
# =============================================================================

def calculate_score(info: Dict, perf: Dict, val: Dict, technicals: Dict, signals: List, sector_comp: Dict) -> Dict:
    scores = {'valuation': 5, 'technicals': 5, 'momentum': 5, 'fundamentals': 5, 'sector': 5}
    factors = []
    
    # Valuation (upside)
    up = val.get('upside_pct', 0) or 0
    if up > 30:
        scores['valuation'] = 9
        factors.append(f"Strong value: {up:+.0f}% upside")
    elif up > 15:
        scores['valuation'] = 7
    elif up > 0:
        scores['valuation'] = 6
    elif up > -15:
        scores['valuation'] = 4
    elif up > -30:
        scores['valuation'] = 3
        factors.append(f"Overvalued: {up:.0f}%")
    else:
        scores['valuation'] = 1
        factors.append(f"Severely overvalued: {up:.0f}%")
    
    # Technicals
    t_score = 5
    rsi = technicals.get('rsi')
    if rsi:
        if rsi < 30:
            t_score += 2
            factors.append(f"RSI oversold ({rsi:.0f})")
        elif rsi > 70:
            t_score -= 2
            factors.append(f"RSI overbought ({rsi:.0f})")
    if technicals.get('above_sma_200'):
        t_score += 1
    else:
        t_score -= 1
    if technicals.get('macd_trend') == 'BULLISH':
        t_score += 1
    else:
        t_score -= 1
    scores['technicals'] = max(1, min(10, t_score))
    
    # Momentum
    m_score = 5
    ret_30d = perf.get('windows', {}).get(30, {}).get('return_pct', 0)
    if ret_30d > 10:
        m_score += 2
        factors.append(f"Strong momentum: {ret_30d:+.1f}%")
    elif ret_30d > 0:
        m_score += 1
    elif ret_30d < -10:
        m_score -= 2
        factors.append(f"Weak momentum: {ret_30d:.1f}%")
    else:
        m_score -= 1
    scores['momentum'] = max(1, min(10, m_score))
    
    # Fundamentals
    f_score = 5
    roe = info.get('roe')
    if roe:
        if roe > 0.20:
            f_score += 2
            factors.append(f"Strong ROE: {roe*100:.0f}%")
        elif roe > 0.10:
            f_score += 1
        elif roe < 0:
            f_score -= 2
    
    rev_g = info.get('revenue_growth')
    if rev_g and rev_g > 0.15:
        f_score += 1
        factors.append(f"Revenue growth: {rev_g*100:.0f}%")
    
    de = info.get('debt_to_equity')
    if de and de > 200:
        f_score -= 1
        factors.append(f"High debt: D/E {de:.0f}")
    scores['fundamentals'] = max(1, min(10, f_score))
    
    # Sector
    if sector_comp.get('comparison') == 'OUTPERFORMING':
        scores['sector'] = 7
        factors.append("Outperforming sector")
    elif sector_comp.get('comparison') == 'UNDERPERFORMING':
        scores['sector'] = 3
        factors.append("Underperforming sector")
    
    # Signals adjustment
    for sig in signals:
        if sig['type'] == 'BULLISH' and sig['strength'] == 'STRONG':
            scores['technicals'] = min(10, scores['technicals'] + 1)
        elif sig['type'] == 'BEARISH' and sig['strength'] == 'STRONG':
            scores['technicals'] = max(1, scores['technicals'] - 1)
    
    # Overall
    weights = {'valuation': 0.30, 'technicals': 0.20, 'momentum': 0.15, 'fundamentals': 0.25, 'sector': 0.10}
    overall = sum(scores[k] * weights[k] for k in scores)
    
    if overall >= 7.5:
        rec, conf = 'STRONG BUY', 'HIGH'
    elif overall >= 6.5:
        rec, conf = 'BUY', 'MEDIUM'
    elif overall >= 5.5:
        rec, conf = 'HOLD', 'MEDIUM'
    elif overall >= 4.0:
        rec, conf = 'HOLD', 'LOW'
    elif overall >= 3.0:
        rec, conf = 'SELL', 'MEDIUM'
    else:
        rec, conf = 'STRONG SELL', 'HIGH'
    
    return {
        'overall': round(overall, 1),
        'scores': scores,
        'recommendation': rec,
        'confidence': conf,
        'factors': factors[:5],
    }

def generate_rule_analysis(ticker: str, info: Dict, perf: Dict, events: List, news: List,
                           val: Dict, technicals: Dict, signals: List, sector_comp: Dict, score: Dict) -> Dict:
    name = info.get('name', ticker)
    up = val.get('upside_pct', 0) or 0
    verdict = val.get('verdict', 'N/A')
    ret_30d = perf.get('windows', {}).get(30, {}).get('return_pct', 0)
    
    summary = f"{name} scores {score['overall']}/10 ({score['recommendation']}). "
    summary += f"Stock is {verdict.lower()} with {up:+.1f}% upside. "
    summary += f"30-day return: {ret_30d:+.1f}%."
    
    # Event analysis
    event_analysis = "No significant price events detected."
    if events:
        e = events[0]
        event_analysis = f"Biggest move: {e['return_pct']:+.1f}% on {e['date_str']}"
        if e.get('matched_news'):
            event_analysis += f" - possibly linked to \"{e['matched_news'][0]['title'][:40]}...\""
        else:
            event_analysis += " - no clear news catalyst found."
    
    # Bull case
    bull = []
    if up > 10:
        bull.append(f"Trading {up:.0f}% below fair value")
    if info.get('roe') and info['roe'] > 0.15:
        bull.append(f"Strong ROE ({info['roe']*100:.0f}%)")
    if info.get('revenue_growth') and info['revenue_growth'] > 0.10:
        bull.append(f"Growing revenue ({info['revenue_growth']*100:.0f}%)")
    if technicals.get('rsi') and technicals['rsi'] < 35:
        bull.append(f"Oversold (RSI {technicals['rsi']:.0f})")
    if technicals.get('above_sma_200'):
        bull.append("Above 200-day MA (bullish trend)")
    for s in signals:
        if s['type'] == 'BULLISH':
            bull.append(s['desc'])
    if not bull:
        bull = ['Established market position', 'Industry tailwinds']
    
    # Bear case
    bear = []
    if up < -10:
        bear.append(f"Trading {abs(up):.0f}% above fair value")
    if info.get('pe_trailing') and info['pe_trailing'] > 35:
        bear.append(f"High P/E ({info['pe_trailing']:.0f}x)")
    if info.get('debt_to_equity') and info['debt_to_equity'] > 150:
        bear.append(f"High leverage (D/E {info['debt_to_equity']:.0f})")
    if technicals.get('rsi') and technicals['rsi'] > 65:
        bear.append(f"Overbought (RSI {technicals['rsi']:.0f})")
    if not technicals.get('above_sma_200'):
        bear.append("Below 200-day MA (bearish trend)")
    for s in signals:
        if s['type'] == 'BEARISH':
            bear.append(s['desc'])
    if not bear:
        bear = ['Market/macro risk', 'Competition']
    
    return {
        'source': 'RULE_BASED',
        'recommendation': score['recommendation'],
        'confidence': score['confidence'],
        'score': score['overall'],
        'scores': score['scores'],
        'summary': summary,
        'event_analysis': event_analysis,
        'bull_case': bull[:5],
        'bear_case': bear[:5],
        'factors': score['factors'],
        'fair_value': f"${val.get('range_low', 0):.2f} - ${val.get('range_high', 0):.2f}",
    }

# =============================================================================
# 7. FOCUSED GEMINI AI (MINIMAL TOKENS - PRICE EVENT ANALYSIS ONLY)
# =============================================================================

class GeminiClient:
    RETRYABLE = ['429', 'RESOURCE_EXHAUSTED', 'Too Many Requests', 'quota']
    
    def __init__(self, api_key: str, retry_cfg: RetryConfig = None):
        self.api_key = api_key
        self.retry_cfg = retry_cfg or RetryConfig()
        self.client = None
        self.available = False
        
        if GEMINI_AVAILABLE and api_key:
            try:
                self.client = genai.Client(api_key=api_key)
                self.available = True
            except:
                pass
    
    def generate(self, prompt: str, model: str = "gemini-2.0-flash-lite") -> Optional[str]:
        if not self.available:
            return None
        
        for attempt in range(self.retry_cfg.max_retries):
            try:
                response = self.client.models.generate_content(model=model, contents=prompt)
                if response and response.text:
                    return response.text
                return None
            except Exception as e:
                err = str(e).lower()
                if any(x.lower() in err for x in self.RETRYABLE):
                    if attempt < self.retry_cfg.max_retries - 1:
                        delay = min(self.retry_cfg.max_delay, 
                                   self.retry_cfg.base_delay * (2 ** attempt)) * random.uniform(0.7, 1.3)
                        print(f"    ⚠ Gemini 429. Retry {attempt+1}/{self.retry_cfg.max_retries} in {delay:.1f}s...")
                        time.sleep(delay)
                    continue
                else:
                    print(f"    ⚠ Gemini error: {str(e)[:50]}")
                    break
        return None


def get_ai_event_analysis(ticker: str, info: Dict, events: List, news: List, 
                          val: Dict, score: Dict) -> Optional[Dict]:
    """
    FOCUSED Gemini call - ONLY analyzes price events + news
    Uses minimal tokens to avoid rate limits
    """
    if not CONFIG.ai_enabled:
        return None
    
    client = GeminiClient(CONFIG.gemini_api_key, CONFIG.retry)
    if not client.available:
        return None
    
    # Build minimal event text
    event_text = ""
    for e in events[:3]:
        event_text += f"• {e['date_str']}: {e['return_pct']:+.1f}% ({e['type']})"
        if e.get('reasons'):
            event_text += f" [{', '.join(e['reasons'])}]"
        if e.get('matched_news'):
            event_text += f"\n  News: \"{e['matched_news'][0]['title'][:50]}...\""
        event_text += "\n"
    
    if not event_text:
        event_text = "No significant price events detected.\n"
    
    # Build minimal news text
    news_text = ""
    for n in news[:5]:
        news_text += f"• {n['title'][:60]}\n"
    
    if not news_text:
        news_text = "No recent news.\n"
    
    # MINIMAL PROMPT - focused only on price events
    prompt = f"""Analyze {ticker}'s price moves:

PRICE EVENTS:
{event_text}
RECENT NEWS:
{news_text}
CONTEXT: Stock is {val.get('verdict', 'N/A')}. Score: {score['overall']}/10.

In 3-4 sentences:
1. What likely caused the biggest price move? (Link to specific news if relevant)
2. Is this BULLISH or BEARISH going forward?
3. One action recommendation.

If no clear news matches the price move, say "No clear catalyst identified."
Be specific. Don't hallucinate news."""

    print(f"    → Calling Gemini (minimal prompt: {len(prompt)} chars)...")
    response = client.generate(prompt)
    
    if not response:
        return None
    
    # Parse response
    lines = response.strip().split('\n')
    clean_response = ' '.join(l.strip() for l in lines if l.strip())
    
    # Determine sentiment from response
    resp_lower = clean_response.lower()
    if 'bullish' in resp_lower or 'buy' in resp_lower or 'positive' in resp_lower:
        sentiment = 'BULLISH'
    elif 'bearish' in resp_lower or 'sell' in resp_lower or 'negative' in resp_lower:
        sentiment = 'BEARISH'
    else:
        sentiment = 'NEUTRAL'
    
    return {
        'ai_analysis': clean_response,
        'ai_sentiment': sentiment,
        'source': 'GEMINI_AI'
    }

# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(ticker: str, info: Dict, perf: Dict, events: List, news: List,
                    val: Dict, analysis: Dict, technicals: Dict, signals: List,
                    sector_comp: Dict, ai_result: Optional[Dict]) -> str:
    lines = []
    
    # Header
    lines.append("=" * 80)
    lines.append(f"{'TRADING REPORTER v4.0':^80}")
    lines.append(f"{ticker} - {info.get('name', ticker)[:50]:^80}")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}")
    lines.append("")
    
    # Summary Line
    cp = val.get('current_price', 0) or 0
    iv = val.get('intrinsic_value', 0) or 0
    up = val.get('upside_pct', 0) or 0
    
    lines.append("-" * 80)
    lines.append(f"SCORE: {analysis.get('score', 'N/A')}/10 | {analysis.get('recommendation')} ({analysis.get('confidence')})")
    lines.append(f"Price: ${cp:.2f} → Fair Value: ${iv:.2f} | {val.get('verdict', 'N/A')} | {up:+.1f}%")
    lines.append("-" * 80)
    lines.append("")
    
    # Component Scores
    scores = analysis.get('scores', {})
    if scores:
        lines.append("SCORES: Valuation={}/10 | Technicals={}/10 | Momentum={}/10 | Fundamentals={}/10 | Sector={}/10".format(
            scores.get('valuation', '-'), scores.get('technicals', '-'), scores.get('momentum', '-'),
            scores.get('fundamentals', '-'), scores.get('sector', '-')
        ))
        lines.append("")
    
    # Summary
    lines.append("SUMMARY:")
    lines.append(f"  {analysis.get('summary', 'N/A')}")
    lines.append("")
    
    # AI Analysis (if available)
    if ai_result and ai_result.get('ai_analysis'):
        lines.append("=" * 40)
        lines.append(f"AI ANALYSIS ({ai_result.get('ai_sentiment', 'N/A')})")
        lines.append("=" * 40)
        # Word wrap the AI response
        ai_text = ai_result['ai_analysis']
        words = ai_text.split()
        line = "  "
        for word in words:
            if len(line) + len(word) > 75:
                lines.append(line)
                line = "  " + word
            else:
                line += " " + word if line != "  " else word
        if line.strip():
            lines.append(line)
        lines.append("")
    
    # Price Event Analysis
    lines.append("PRICE EVENT ANALYSIS:")
    lines.append(f"  {analysis.get('event_analysis', 'N/A')}")
    lines.append("")
    
    # Key Factors
    factors = analysis.get('factors', [])
    if factors:
        lines.append("KEY FACTORS:")
        for f in factors:
            lines.append(f"  • {f}")
        lines.append("")
    
    # Bull/Bear Case
    lines.append("BULL CASE:")
    for b in analysis.get('bull_case', [])[:4]:
        lines.append(f"  + {b}")
    lines.append("")
    lines.append("BEAR CASE:")
    for b in analysis.get('bear_case', [])[:4]:
        lines.append(f"  - {b}")
    lines.append("")
    
    # Trading Signals
    if signals:
        lines.append("=" * 40)
        lines.append("TRADING SIGNALS")
        lines.append("=" * 40)
        for s in signals[:5]:
            icon = "🟢" if s['type'] == 'BULLISH' else ("🔴" if s['type'] == 'BEARISH' else "⚪")
            lines.append(f"  {icon} {s['signal']}: {s['desc']}")
        lines.append("")
    
    # Technical Indicators
    lines.append("=" * 40)
    lines.append("TECHNICAL INDICATORS")
    lines.append("=" * 40)
    lines.append(f"  RSI(14): {technicals.get('rsi', 'N/A')} ({technicals.get('rsi_signal', '')})")
    lines.append(f"  MACD: {technicals.get('macd', 'N/A')} | Signal: {technicals.get('macd_signal', 'N/A')} ({technicals.get('macd_trend', '')})")
    lines.append(f"  Bollinger: ${technicals.get('bb_lower', 0):.2f} - ${technicals.get('bb_upper', 0):.2f}")
    lines.append(f"  50-day MA: ${technicals.get('sma_50', 0):.2f} | 200-day MA: ${technicals.get('sma_200', 0):.2f}")
    lines.append(f"  Support: ${technicals.get('support_1', 0):.2f} | Resistance: ${technicals.get('resistance_1', 0):.2f}")
    lines.append("")
    
    # Performance
    lines.append("=" * 40)
    lines.append("PERFORMANCE")
    lines.append("=" * 40)
    for w, d in perf.get('windows', {}).items():
        vol = d.get('vol_ann_pct')
        v_str = f" | Vol: {vol:.1f}%" if vol else ""
        lines.append(f"  {w}D: {d.get('return_pct', 0):+.1f}%{v_str} | H: ${d.get('high', 0):.2f} L: ${d.get('low', 0):.2f}")
    lines.append("")
    
    # Sector Comparison
    if sector_comp.get('comparison') != 'N/A':
        lines.append("=" * 40)
        lines.append("SECTOR COMPARISON")
        lines.append("=" * 40)
        lines.append(f"  vs {sector_comp.get('etf', 'Sector')}: {sector_comp.get('comparison')}")
        for period, data in sector_comp.get('details', {}).items():
            lines.append(f"    {period}: Stock {data.get('stock', 0):+.1f}% vs Sector {data.get('sector', 0):+.1f}% ({data.get('diff', 0):+.1f}%)")
        lines.append("")
    
    # Price Events
    if events:
        lines.append("=" * 40)
        lines.append("PRICE EVENTS")
        lines.append("=" * 40)
        for e in events[:5]:
            icon = "📈" if 'surge' in e['type'] or 'up' in e['type'] else "📉"
            reasons = ", ".join(e.get('reasons', []))
            lines.append(f"  {icon} {e['date_str']}: {e['type'].upper()} {e['return_pct']:+.1f}% ({reasons})")
            for n in e.get('matched_news', [])[:1]:
                lines.append(f"      → \"{n['title'][:55]}...\"")
        lines.append("")
    
    # News
    if news:
        lines.append("=" * 40)
        lines.append("RECENT NEWS")
        lines.append("=" * 40)
        for n in news[:8]:
            lines.append(f"  [{n['datetime_str']}] {n['title'][:55]}")
        lines.append("")
    
    # Valuation
    lines.append("=" * 40)
    lines.append("VALUATION")
    lines.append("=" * 40)
    lines.append(f"  Fair Value: {analysis.get('fair_value', 'N/A')}")
    for c in val.get('components', []):
        lines.append(f"    {c['method']}: ${c['value']:.2f} ({c['weight']*100:.0f}%)")
    lines.append("")
    
    # Fundamentals
    lines.append("=" * 40)
    lines.append("FUNDAMENTALS")
    lines.append("=" * 40)
    if info.get('pe_trailing'):
        lines.append(f"  P/E: {info['pe_trailing']:.1f}x")
    if info.get('pb_ratio'):
        lines.append(f"  P/B: {info['pb_ratio']:.2f}x")
    if info.get('roe'):
        lines.append(f"  ROE: {info['roe']*100:.1f}%")
    if info.get('profit_margin'):
        lines.append(f"  Profit Margin: {info['profit_margin']*100:.1f}%")
    if info.get('revenue_growth'):
        lines.append(f"  Revenue Growth: {info['revenue_growth']*100:.1f}%")
    if info.get('debt_to_equity'):
        lines.append(f"  D/E: {info['debt_to_equity']:.1f}")
    lines.append(f"  Market Cap: {fmt_large(info.get('market_cap'))}")
    lines.append("")
    
    # Disclaimer
    lines.append("=" * 80)
    lines.append("DISCLAIMER: Educational only. NOT financial advice.")
    lines.append("=" * 80)
    
    return '\n'.join(lines)

# =============================================================================
# MAIN
# =============================================================================

def analyze_stock(ticker: str, no_ai: bool = False, quiet: bool = False, debug: bool = False) -> Dict:
    ticker = ticker.upper()
    
    if not quiet:
        print(f"\n{'='*60}")
        print(f"Analyzing {ticker}...")
        print('='*60)
    
    # 1. Price data
    if not quiet: print("[1/9] Fetching price data...")
    df = fetch_price_data(ticker)
    if df is None:
        print(f"  ✗ No price data for {ticker}")
        return {'ticker': ticker, 'error': 'No price data'}
    if not quiet: print(f"  ✓ {len(df)} days")
    
    # 2. Fundamentals
    if not quiet: print("[2/9] Fetching fundamentals...")
    info = fetch_fundamentals(ticker)
    if not quiet: print(f"  ✓ {info.get('name', ticker)}")
    
    # 3. News (ENHANCED - 3 sources)
    if not quiet: print("[3/9] Fetching news (Yahoo + Google + Finviz)...")
    news = fetch_all_news(ticker, info.get('name', ''))
    if not quiet: print(f"  ✓ {len(news)} articles")
    
    # 4. Earnings
    if not quiet: print("[4/9] Fetching earnings...")
    earnings = fetch_earnings(ticker)
    if not quiet: print(f"  ✓ {len(earnings)} earnings records")
    
    # 5. Performance & Events (ENHANCED)
    if not quiet: print("[5/9] Detecting price events...")
    perf = analyze_performance(df)
    events = detect_events(df, earnings)
    events = match_news_to_events(events, news)
    if not quiet: print(f"  ✓ {len(events)} events detected")
    
    # 6. Technical Indicators (NEW)
    if not quiet: print("[6/9] Calculating technicals (RSI, MACD, BB, MA)...")
    technicals = get_technicals(df)
    signals = detect_signals(df, technicals)
    if not quiet: print(f"  ✓ {len(signals)} signals")
    
    # 7. Sector Comparison (NEW)
    if not quiet: print("[7/9] Comparing to sector...")
    sector_data = fetch_sector_data(info.get('sector', ''))
    sector_comp = compare_to_sector(perf, sector_data)
    if not quiet: print(f"  ✓ {sector_comp.get('comparison', 'N/A')} vs {sector_comp.get('etf', 'N/A')}")
    
    # 8. Valuation
    if not quiet: print("[8/9] Running valuation...")
    val = comprehensive_valuation(info)
    if not quiet:
        iv = val.get('intrinsic_value')
        if iv:
            print(f"  ✓ ${iv:.2f} ({val.get('verdict')})")
        else:
            print(f"  ⚠ Insufficient data")
    
    # 9. Analysis (Rule-based + AI)
    if not quiet: print("[9/9] Generating analysis...")
    
    # Calculate score
    score = calculate_score(info, perf, val, technicals, signals, sector_comp)
    
    # Generate rule-based analysis
    analysis = generate_rule_analysis(ticker, info, perf, events, news, val, technicals, signals, sector_comp, score)
    
    # Get AI analysis (FOCUSED - only price events)
    ai_result = None
    if not no_ai and CONFIG.ai_enabled:
        ai_result = get_ai_event_analysis(ticker, info, events, news, val, score)
        if ai_result:
            analysis['source'] = 'HYBRID (Rules + AI)'
            if not quiet: print(f"  ✓ AI analysis received")
        else:
            if not quiet: print(f"  ⚠ AI unavailable, using rules only")
    else:
        if not quiet: print(f"  ✓ Rule-based analysis")
    
    if not quiet: print(f"  ✓ {analysis['recommendation']} (Score: {analysis['score']}/10)")
    
    # Generate report
    report = generate_report(ticker, info, perf, events, news, val, analysis, technicals, signals, sector_comp, ai_result)
    
    # Print summary
    if not quiet:
        print("\n" + "=" * 80)
        cp = val.get('current_price', 0) or 0
        iv = val.get('intrinsic_value', 0) or 0
        print(f"{ticker}: ${cp:.2f} → Fair ${iv:.2f} | Score: {analysis['score']}/10 | {analysis['recommendation']}")
        if signals:
            sig_names = [s['signal'] for s in signals[:3]]
            print(f"Signals: {', '.join(sig_names)}")
        if ai_result:
            print(f"AI Sentiment: {ai_result.get('ai_sentiment', 'N/A')}")
        print("=" * 80)
    
    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = Path(CONFIG.output_dir) / f"{ticker}_REPORT_{ts}.txt"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)
    
    if not quiet:
        print(f"\n✓ Report saved: {filepath}")
    
    return {'ticker': ticker, 'file': str(filepath), 'analysis': analysis, 'ai': ai_result}


def main():
    parser = argparse.ArgumentParser(
        description='Trading Reporter v4.0 - Complete Edition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FEATURES (All included):
  1. Better News (Yahoo + Google RSS + Finviz)
  2. Technical Indicators (RSI, MACD, Bollinger, MA50/200)
  3. Smarter Event Detection (gaps, volume, earnings)
  4. 1-10 Scoring System
  5. Sector Comparison
  6. Trading Signals (Golden Cross, Death Cross, etc.)
  7. Focused AI Analysis (price events only - minimal tokens!)

EXAMPLES:
    python trading_reporter_v4.py AAPL
    python trading_reporter_v4.py NVDA MSFT GOOG
    python trading_reporter_v4.py AAPL --no-ai

ENVIRONMENT:
    GEMINI_API_KEY - Optional, for AI price event analysis
        """
    )
    
    parser.add_argument('tickers', nargs='*', help='Stock ticker(s)')
    parser.add_argument('--no-ai', action='store_true', help='Skip AI (rule-based only)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--throttle', type=float, default=CONFIG.throttle, help='Delay between tickers')
    
    args = parser.parse_args()
    
    if not args.tickers:
        print("\n" + "=" * 60)
        print("TRADING REPORTER v4.0 - Complete Edition")
        print("=" * 60)
        print("\nFEATURES:")
        print("  ✓ Technical Indicators (RSI, MACD, Bollinger, MA)")
        print("  ✓ News from 3 sources (Yahoo + Google + Finviz)")
        print("  ✓ Trading Signals (Golden Cross, Death Cross, etc.)")
        print("  ✓ Sector Comparison")
        print("  ✓ 1-10 Scoring System")
        print("  ✓ Focused AI Analysis (optional)")
        print("\nUsage: python trading_reporter_v4.py AAPL")
        
        try:
            inp = input("\nEnter ticker(s): ").strip()
            args.tickers = inp.upper().split() if inp else []
        except:
            return 0
    
    if not args.tickers:
        print("No tickers")
        return 1
    
    for i, ticker in enumerate(args.tickers):
        try:
            analyze_stock(ticker.upper(), no_ai=args.no_ai, quiet=args.quiet, debug=args.debug)
            
            if i < len(args.tickers) - 1 and args.throttle > 0:
                if not args.quiet:
                    print(f"\n⏳ Next ticker in {args.throttle}s...")
                time.sleep(args.throttle)
                
        except KeyboardInterrupt:
            print("\n\nInterrupted")
            return 130
        except Exception as e:
            print(f"Error: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
    
    print(f"\n✓ Done! Reports in {CONFIG.output_dir}/")
    return 0

if __name__ == '__main__':
    sys.exit(main())
