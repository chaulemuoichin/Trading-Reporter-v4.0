#!/usr/bin/env python3
"""
================================================================================
TRADING REPORTER v1.3 - NEWS-AWARE STOCK ANALYSIS
================================================================================
Fixed: Upcoming events, AI connection, better error handling

Usage:
    python trading_reporter.py AAPL
    python trading_reporter.py GOOG MSFT
================================================================================
"""

# =============================================================================
# AUTO-INSTALL DEPENDENCIES
# =============================================================================
import subprocess
import sys

def install_package(package):
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q', '--disable-pip-version-check'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        return False

def check_and_install():
    packages = {'yfinance': 'yfinance', 'pandas': 'pandas', 'numpy': 'numpy', 
                'requests': 'requests', 'google.generativeai': 'google-generativeai'}
    
    print("=" * 60)
    print("TRADING REPORTER v1.3")
    print("=" * 60)
    print("\nChecking dependencies...")
    
    for module, package in packages.items():
        try:
            __import__(module.split('.')[0])
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  Installing {package}...", end=' ', flush=True)
            if install_package(package):
                print("✓")
            else:
                print("✗")
    print("")

check_and_install()

# =============================================================================
# IMPORTS
# =============================================================================
import warnings
warnings.filterwarnings('ignore')

import os
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

GEMINI_OK = False
try:
    import google.generativeai as genai
    GEMINI_OK = True
except ImportError:
    print("⚠ google-generativeai not installed")

# =============================================================================
# CONFIGURATION - UPDATE YOUR API KEY HERE IF NEEDED
# =============================================================================

# Get FREE API key at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY = "AIzaSyCeBCgvKI8J9vFm8dZgabhF8a5Q081nPBs"

@dataclass
class Config:
    windows: List[int] = field(default_factory=lambda: [1, 7, 30])
    output_dir: str = 'reports'

CONFIG = Config()
Path(CONFIG.output_dir).mkdir(exist_ok=True)

# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_price_data(ticker: str, days: int = 90) -> Optional[pd.DataFrame]:
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=f"{days}d", interval='1d')
        if df.empty:
            return None
        df = df.reset_index()
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        df['return'] = df['close'].pct_change()
        df['vol_avg_20'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_avg_20']
        return df
    except Exception as e:
        print(f"  ✗ Price error: {e}")
        return None

def fetch_stock_info(ticker: str) -> Dict:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'name': info.get('longName') or info.get('shortName') or ticker,
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'current_price': info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose'),
            '52w_high': info.get('fiftyTwoWeekHigh'),
            '52w_low': info.get('fiftyTwoWeekLow'),
            'beta': info.get('beta'),
            'market_cap': info.get('marketCap'),
            'shares_outstanding': info.get('sharesOutstanding'),
            'pe_trailing': info.get('trailingPE'),
            'pe_forward': info.get('forwardPE'),
            'peg_ratio': info.get('pegRatio'),
            'pb_ratio': info.get('priceToBook'),
            'ev_ebitda': info.get('enterpriseToEbitda'),
            'profit_margin': info.get('profitMargins'),
            'operating_margin': info.get('operatingMargins'),
            'roe': info.get('returnOnEquity'),
            'roa': info.get('returnOnAssets'),
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
            'current_ratio': info.get('currentRatio'),
            'debt_to_equity': info.get('debtToEquity'),
            'total_debt': info.get('totalDebt'),
            'total_cash': info.get('totalCash'),
            'free_cash_flow': info.get('freeCashflow'),
            'eps_trailing': info.get('trailingEps'),
            'book_value': info.get('bookValue'),
            'ebitda': info.get('ebitda'),
            'dividend_yield': info.get('dividendYield'),
            'dividend_rate': info.get('dividendRate'),
            'ex_dividend_date': info.get('exDividendDate'),
            'target_high': info.get('targetHighPrice'),
            'target_low': info.get('targetLowPrice'),
            'target_mean': info.get('targetMeanPrice'),
            'recommendation': info.get('recommendationKey'),
            'num_analysts': info.get('numberOfAnalystOpinions'),
            'employees': info.get('fullTimeEmployees'),
        }
    except Exception as e:
        print(f"  ✗ Info error: {e}")
        return {}

def fetch_upcoming_events(ticker: str) -> Dict:
    """Fetch upcoming events - earnings, dividends, etc."""
    events = {}
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Method 1: Try calendar
        try:
            cal = stock.calendar
            if cal is not None:
                if isinstance(cal, pd.DataFrame) and not cal.empty:
                    for col in cal.columns:
                        for idx in cal.index:
                            val = cal.loc[idx, col]
                            if pd.notna(val):
                                key = f"{idx}"
                                if isinstance(val, (datetime, pd.Timestamp)):
                                    events[key] = val.strftime('%Y-%m-%d')
                                else:
                                    events[key] = str(val)[:10]
                elif isinstance(cal, dict):
                    for k, v in cal.items():
                        if v is not None:
                            if isinstance(v, list) and len(v) > 0:
                                events[k] = str(v[0])[:10]
                            else:
                                events[k] = str(v)[:10]
        except Exception as e:
            pass
        
        # Method 2: Try earnings_dates
        try:
            ed = stock.earnings_dates
            if ed is not None and not ed.empty:
                future = ed[ed.index >= datetime.now()]
                if not future.empty:
                    next_date = future.index[0]
                    events['Next Earnings'] = next_date.strftime('%Y-%m-%d')
                    if 'EPS Estimate' in future.columns:
                        est = future['EPS Estimate'].iloc[0]
                        if pd.notna(est):
                            events['EPS Estimate'] = f"${est:.2f}"
        except:
            pass
        
        # Method 3: Check info for dates
        try:
            # Ex-dividend date
            ex_div = info.get('exDividendDate')
            if ex_div:
                ex_div_date = datetime.fromtimestamp(ex_div)
                if ex_div_date > datetime.now():
                    events['Ex-Dividend Date'] = ex_div_date.strftime('%Y-%m-%d')
                    if info.get('dividendRate'):
                        events['Dividend Rate'] = f"${info['dividendRate']:.2f}/share"
        except:
            pass
        
        # Method 4: Try to get next earnings from quarterly data
        try:
            if 'Next Earnings' not in events:
                # Estimate next earnings based on last quarter
                quarterly = stock.quarterly_earnings
                if quarterly is not None and not quarterly.empty:
                    last_date = quarterly.index[0]
                    if isinstance(last_date, (datetime, pd.Timestamp)):
                        next_est = last_date + timedelta(days=90)
                        if next_est > datetime.now():
                            events['Est. Next Earnings'] = next_est.strftime('%Y-%m-%d') + " (estimated)"
        except:
            pass
            
    except Exception as e:
        print(f"  Events fetch error: {e}")
    
    return events

def fetch_earnings_history(ticker: str) -> List[Dict]:
    """Fetch earnings history with surprises"""
    earnings = []
    
    try:
        stock = yf.Ticker(ticker)
        
        # Try earnings_dates first
        try:
            ed = stock.earnings_dates
            if ed is not None and not ed.empty:
                for idx, row in ed.head(8).iterrows():
                    date_str = idx.strftime('%Y-%m-%d') if isinstance(idx, (datetime, pd.Timestamp)) else str(idx)[:10]
                    
                    entry = {'date': date_str}
                    
                    if 'EPS Estimate' in row.index and pd.notna(row['EPS Estimate']):
                        entry['eps_estimate'] = row['EPS Estimate']
                    if 'Reported EPS' in row.index and pd.notna(row['Reported EPS']):
                        entry['eps_actual'] = row['Reported EPS']
                    if 'Surprise(%)' in row.index and pd.notna(row['Surprise(%)']):
                        entry['surprise'] = row['Surprise(%)']
                    
                    earnings.append(entry)
        except:
            pass
        
        # Try quarterly earnings as backup
        if not earnings:
            try:
                qe = stock.quarterly_earnings
                if qe is not None and not qe.empty:
                    for idx, row in qe.head(6).iterrows():
                        date_str = str(idx)
                        entry = {'date': date_str}
                        if 'Earnings' in row.index:
                            entry['eps_actual'] = row['Earnings']
                        if 'Revenue' in row.index:
                            entry['revenue'] = row['Revenue']
                        earnings.append(entry)
            except:
                pass
                
    except Exception as e:
        pass
    
    return earnings

def fetch_news_yfinance(ticker: str) -> List[Dict]:
    news_list = []
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if news:
            for item in news[:10]:
                title = item.get('title', '')
                if not title:
                    continue
                pub_time = item.get('providerPublishTime', 0)
                news_list.append({
                    'title': title,
                    'publisher': item.get('publisher', 'Unknown'),
                    'link': item.get('link', ''),
                    'timestamp': pub_time,
                    'datetime': datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d %H:%M') if pub_time else 'Unknown',
                    'date': datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d') if pub_time else 'Unknown',
                })
    except:
        pass
    return news_list

def fetch_news_google(ticker: str, company_name: str = "") -> List[Dict]:
    news_list = []
    try:
        search_terms = []
        if company_name:
            clean = company_name.replace(' Inc.', '').replace(' Corp.', '').replace(' Corporation', '')
            search_terms.append(clean.split()[0])
        search_terms.append(ticker)
        
        for term in search_terms:
            if not term or len(news_list) >= 5:
                continue
            url = f"https://news.google.com/rss/search?q={term}+stock&hl=en-US&gl=US&ceid=US:en"
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            try:
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    import re
                    items = re.findall(r'<item>(.*?)</item>', resp.text, re.DOTALL)
                    for item in items[:6]:
                        title_m = re.search(r'<title>(.*?)</title>', item)
                        pub_m = re.search(r'<pubDate>(.*?)</pubDate>', item)
                        src_m = re.search(r'<source.*?>(.*?)</source>', item)
                        
                        if title_m:
                            title = title_m.group(1).replace('<![CDATA[', '').replace(']]>', '').strip()
                            if any(title[:40] == n['title'][:40] for n in news_list):
                                continue
                            
                            pub_date = pub_m.group(1) if pub_m else ''
                            source = src_m.group(1) if src_m else 'News'
                            
                            try:
                                dt = datetime.strptime(pub_date[:25], '%a, %d %b %Y %H:%M:%S')
                                timestamp = dt.timestamp()
                                date_str = dt.strftime('%Y-%m-%d')
                                datetime_str = dt.strftime('%Y-%m-%d %H:%M')
                            except:
                                timestamp = time.time()
                                date_str = datetime.now().strftime('%Y-%m-%d')
                                datetime_str = datetime.now().strftime('%Y-%m-%d %H:%M')
                            
                            news_list.append({
                                'title': title, 'publisher': source, 'link': '',
                                'timestamp': timestamp, 'datetime': datetime_str, 'date': date_str,
                            })
            except:
                continue
    except:
        pass
    return sorted(news_list, key=lambda x: x.get('timestamp', 0), reverse=True)

def fetch_all_news(ticker: str, company_name: str = "") -> List[Dict]:
    all_news = []
    yahoo_news = fetch_news_yfinance(ticker)
    all_news.extend(yahoo_news)
    
    google_news = fetch_news_google(ticker, company_name)
    existing = {n['title'][:40].lower() for n in all_news}
    for n in google_news:
        if n['title'][:40].lower() not in existing:
            all_news.append(n)
            existing.add(n['title'][:40].lower())
    
    return sorted(all_news, key=lambda x: x.get('timestamp', 0), reverse=True)[:12]

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_performance(df: pd.DataFrame, windows: List[int] = [1, 7, 30]) -> Dict:
    if df is None or df.empty:
        return {}
    current_price = df['close'].iloc[-1]
    results = {'current_price': current_price, 'windows': {}}
    
    for window in windows:
        if len(df) < window + 1:
            continue
        start_price = df['close'].iloc[-window - 1]
        window_df = df.iloc[-window - 1:]
        pct_return = (current_price - start_price) / start_price
        daily_returns = window_df['return'].dropna()
        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
        
        results['windows'][window] = {
            'days': window,
            'start_price': round(start_price, 2),
            'end_price': round(current_price, 2),
            'return_pct': round(pct_return * 100, 2),
            'volatility_ann': round(volatility * 100, 2) if volatility else 0,
            'high': round(window_df['high'].max(), 2),
            'low': round(window_df['low'].min(), 2),
        }
    return results

def detect_events(df: pd.DataFrame) -> List[Dict]:
    if df is None or len(df) < 20:
        return []
    events = []
    df = df.copy()
    df['return_std'] = df['return'].rolling(20).std()
    df['z_score'] = df['return'] / df['return_std']
    
    for i in range(20, len(df)):
        row = df.iloc[i]
        z = row.get('z_score', 0)
        ret = row.get('return', 0)
        vol_ratio = row.get('vol_ratio', 1)
        if pd.isna(z):
            continue
        if abs(z) >= 1.5 or (abs(ret) > 0.02 and vol_ratio > 1.3):
            events.append({
                'date': str(row['date'])[:10] if 'date' in row.index else 'Unknown',
                'type': 'surge' if ret > 0 else 'drop',
                'return_pct': round(ret * 100, 2),
                'z_score': round(z, 2) if not pd.isna(z) else 0,
                'volume_ratio': round(vol_ratio, 2) if not pd.isna(vol_ratio) else 1,
                'close': round(row['close'], 2),
            })
    return sorted(events, key=lambda x: abs(x.get('return_pct', 0)), reverse=True)[:8]

def match_news_to_events(events: List[Dict], news: List[Dict]) -> List[Dict]:
    if not events or not news:
        return events
    news_by_date = {}
    for n in news:
        date = n.get('date', '')
        if date and date != 'Unknown':
            if date not in news_by_date:
                news_by_date[date] = []
            news_by_date[date].append(n)
    
    for event in events:
        event_date = event.get('date', '')
        matched = []
        if event_date in news_by_date:
            matched.extend(news_by_date[event_date])
        try:
            prev = (datetime.strptime(event_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
            if prev in news_by_date:
                matched.extend(news_by_date[prev])
        except:
            pass
        event['matched_news'] = matched[:3]
    return events

# =============================================================================
# VALUATION
# =============================================================================

def calculate_wacc(info: Dict) -> float:
    beta = info.get('beta', 1.0) or 1.0
    beta = max(0.5, min(beta, 2.5))
    risk_free = 0.045
    erp = 0.055
    cost_of_equity = risk_free + beta * erp
    de = (info.get('debt_to_equity', 50) or 50) / 100
    debt_weight = min(de / (1 + de), 0.4)
    equity_weight = 1 - debt_weight
    cost_of_debt = (risk_free + 0.02) * 0.75
    wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt
    return max(wacc, 0.07)

def dcf_valuation(info: Dict) -> Dict:
    fcf = info.get('free_cash_flow')
    shares = info.get('shares_outstanding')
    if not fcf or not shares or fcf <= 0:
        return {'error': 'Need positive FCF'}
    wacc = calculate_wacc(info)
    g_short, g_term = 0.08, 0.025
    if wacc <= g_term:
        wacc = g_term + 0.04
    
    pv_fcf = 0
    current_fcf = fcf
    for year in range(1, 6):
        current_fcf *= (1 + g_short)
        pv_fcf += current_fcf / ((1 + wacc) ** year)
    
    terminal_fcf = current_fcf * (1 + g_term)
    terminal_value = terminal_fcf / (wacc - g_term)
    pv_terminal = terminal_value / ((1 + wacc) ** 5)
    
    ev = pv_fcf + pv_terminal
    debt = info.get('total_debt', 0) or 0
    cash = info.get('total_cash', 0) or 0
    equity = ev - debt + cash
    intrinsic = equity / shares
    current = info.get('current_price', 0) or 1
    
    return {
        'method': 'DCF', 'intrinsic': round(intrinsic, 2),
        'upside': round((intrinsic - current) / current * 100, 1),
        'wacc': round(wacc * 100, 1), 'growth': 8.0, 'terminal': 2.5,
    }

def multiples_valuation(info: Dict) -> Dict:
    current = info.get('current_price', 0) or 1
    implied = []
    
    eps = info.get('eps_trailing')
    if eps and eps > 0:
        price = eps * 20
        implied.append({'metric': 'P/E (20x)', 'price': round(price, 2), 'upside': round((price - current) / current * 100, 1)})
    
    bv = info.get('book_value')
    if bv and bv > 0:
        price = bv * 3.5
        implied.append({'metric': 'P/B (3.5x)', 'price': round(price, 2), 'upside': round((price - current) / current * 100, 1)})
    
    ebitda = info.get('ebitda')
    shares = info.get('shares_outstanding')
    if ebitda and shares and ebitda > 0:
        ev = ebitda * 12
        debt = info.get('total_debt', 0) or 0
        cash = info.get('total_cash', 0) or 0
        price = (ev - debt + cash) / shares
        implied.append({'metric': 'EV/EBITDA (12x)', 'price': round(price, 2), 'upside': round((price - current) / current * 100, 1)})
    
    target = info.get('target_mean')
    if target:
        implied.append({'metric': 'Analyst Target', 'price': round(target, 2), 'upside': round((target - current) / current * 100, 1)})
    
    avg = np.mean([x['price'] for x in implied]) if implied else current
    return {'method': 'Multiples', 'implied': implied, 'average': round(avg, 2), 'upside': round((avg - current) / current * 100, 1)}

def epv_valuation(info: Dict) -> Dict:
    eps = info.get('eps_trailing')
    if not eps or eps <= 0:
        return {'error': 'Need positive EPS'}
    epv = eps / 0.10
    current = info.get('current_price', 0) or 1
    return {'method': 'EPV', 'intrinsic': round(epv, 2), 'upside': round((epv - current) / current * 100, 1)}

def comprehensive_valuation(info: Dict) -> Dict:
    current = info.get('current_price', 0)
    dcf = dcf_valuation(info)
    mult = multiples_valuation(info)
    epv = epv_valuation(info)
    
    values = []
    if 'intrinsic' in dcf:
        values.append(('DCF', dcf['intrinsic'], 0.40))
    if mult.get('average'):
        values.append(('Multiples', mult['average'], 0.35))
    if 'intrinsic' in epv:
        values.append(('EPV', epv['intrinsic'], 0.25))
    
    if not values:
        return {'current': current, 'verdict': 'INSUFFICIENT DATA'}
    
    total_w = sum(w for _, _, w in values)
    weighted = sum(v * w for _, v, w in values) / total_w
    upside = (weighted - current) / current * 100 if current else 0
    
    if upside > 25: verdict = 'SIGNIFICANTLY UNDERVALUED'
    elif upside > 10: verdict = 'UNDERVALUED'
    elif upside > -10: verdict = 'FAIRLY VALUED'
    elif upside > -25: verdict = 'OVERVALUED'
    else: verdict = 'SIGNIFICANTLY OVERVALUED'
    
    return {
        'current': current, 'intrinsic': round(weighted, 2), 'upside': round(upside, 1),
        'range_low': round(min(v for _, v, _ in values), 2),
        'range_high': round(max(v for _, v, _ in values), 2),
        'verdict': verdict,
        'components': [{'method': m, 'value': v} for m, v, _ in values],
        'dcf': dcf, 'multiples': mult, 'epv': epv,
    }

# =============================================================================
# GEMINI AI ANALYSIS
# =============================================================================

class GeminiAnalyzer:
    def __init__(self):
        self.model = None
        self.available = False
        self.error_msg = ""
        
        if not GEMINI_OK:
            self.error_msg = "Package not installed"
            return
        
        if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_API_KEY_HERE":
            self.error_msg = "No API key - get free key at https://aistudio.google.com/app/apikey"
            return
        
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Test the connection with a simple prompt
            test_response = self.model.generate_content(
                "Reply with just: OK",
                generation_config={'max_output_tokens': 10}
            )
            
            if test_response and test_response.text:
                self.available = True
                print("  ✓ Gemini AI connected!")
            else:
                self.error_msg = "Empty response from API"
                
        except Exception as e:
            err = str(e)
            if 'API_KEY_INVALID' in err or 'invalid' in err.lower():
                self.error_msg = "Invalid API key - get new key at https://aistudio.google.com/app/apikey"
            elif 'quota' in err.lower():
                self.error_msg = "API quota exceeded - try again later or get new key"
            elif 'blocked' in err.lower():
                self.error_msg = "API blocked - check your Google Cloud console"
            else:
                self.error_msg = f"Connection error: {err[:80]}"
    
    def analyze(self, ticker: str, info: Dict, performance: Dict, events: List[Dict], 
                news: List[Dict], valuation: Dict, upcoming_events: Dict) -> Dict:
        
        if not self.available:
            print(f"  ⚠ AI unavailable: {self.error_msg}")
            return self._rule_based(ticker, info, performance, events, news, valuation)
        
        # Format data
        news_text = "\n".join([f"- [{n.get('date','')}] {n.get('title','')} ({n.get('publisher','')})" 
                              for n in news[:8]]) if news else "No news"
        
        events_text = "\n".join([f"- {e.get('date')}: {'UP' if e.get('type')=='surge' else 'DOWN'} {e.get('return_pct',0):+.1f}%" 
                                for e in events[:5]]) if events else "No significant events"
        
        perf_text = "\n".join([f"- {w}D: {d.get('return_pct',0):+.1f}%" for w, d in performance.get('windows', {}).items()])
        
        upcoming_text = "\n".join([f"- {k}: {v}" for k, v in upcoming_events.items()]) if upcoming_events else "None found"
        
        prompt = f"""Analyze {ticker} ({info.get('name', ticker)}) stock. Be specific and data-driven.

COMPANY:
- Sector: {info.get('sector', 'N/A')}
- Market Cap: ${(info.get('market_cap',0) or 0)/1e9:.1f}B
- P/E: {info.get('pe_trailing', 'N/A')}, P/B: {info.get('pb_ratio', 'N/A')}
- Profit Margin: {(info.get('profit_margin',0) or 0)*100:.1f}%
- Revenue Growth: {(info.get('revenue_growth',0) or 0)*100:.1f}%
- ROE: {(info.get('roe',0) or 0)*100:.1f}%

VALUATION:
- Current: ${info.get('current_price', 0):.2f}
- Fair Value: ${valuation.get('intrinsic', 0):.2f}
- Verdict: {valuation.get('verdict', 'N/A')}
- Upside: {valuation.get('upside', 0):+.1f}%
- Analyst Target: ${info.get('target_mean', 0) or 0:.2f}

PERFORMANCE:
{perf_text}

PRICE EVENTS:
{events_text}

UPCOMING EVENTS:
{upcoming_text}

NEWS:
{news_text}

Respond in this EXACT format:

RECOMMENDATION: [STRONG BUY/BUY/HOLD/SELL/STRONG SELL]
CONFIDENCE: [HIGH/MEDIUM/LOW]

SUMMARY:
[3-4 sentence investment thesis based on the data above]

NEWS_ANALYSIS:
[2-3 sentences analyzing how the news affected stock price. Be specific about which news caused which move.]

PRICE_EXPLANATION:
[Explain what drove the recent price movements based on news and events]

BULL_CASE:
- [Specific data-backed bull point]
- [Specific data-backed bull point]
- [Specific data-backed bull point]

BEAR_CASE:
- [Specific data-backed bear point]
- [Specific data-backed bear point]
- [Specific data-backed bear point]

FAIR_VALUE: $[low] - $[high]

CATALYSTS:
- [Upcoming catalyst with date if known]
- [Another catalyst]

ACTION:
[Specific actionable advice with price levels]"""

        try:
            response = self.model.generate_content(prompt)
            
            if not response or not response.text:
                print("  ⚠ Empty AI response")
                return self._rule_based(ticker, info, performance, events, news, valuation)
            
            result = self._parse_response(response.text)
            result['source'] = 'GEMINI_AI'
            return result
            
        except Exception as e:
            print(f"  ⚠ AI error: {str(e)[:60]}")
            return self._rule_based(ticker, info, performance, events, news, valuation)
    
    def _parse_response(self, text: str) -> Dict:
        result = {
            'recommendation': 'HOLD', 'confidence': 'MEDIUM', 'summary': '',
            'news_analysis': '', 'price_explanation': '', 'bull_case': [],
            'bear_case': [], 'fair_value': '', 'catalysts': [], 'action': ''
        }
        
        section = None
        for line in text.strip().split('\n'):
            line = line.strip()
            if line.startswith('RECOMMENDATION:'): result['recommendation'] = line.split(':', 1)[1].strip()
            elif line.startswith('CONFIDENCE:'): result['confidence'] = line.split(':', 1)[1].strip()
            elif line.startswith('SUMMARY:'): section = 'summary'
            elif line.startswith('NEWS_ANALYSIS:'): section = 'news'
            elif line.startswith('PRICE_EXPLANATION:'): section = 'price'
            elif line.startswith('BULL_CASE:'): section = 'bull'
            elif line.startswith('BEAR_CASE:'): section = 'bear'
            elif line.startswith('FAIR_VALUE:'): result['fair_value'] = line.split(':', 1)[1].strip(); section = None
            elif line.startswith('CATALYSTS:'): section = 'catalysts'
            elif line.startswith('ACTION:'): section = 'action'
            elif line.startswith('- '):
                item = line[2:]
                if section == 'bull': result['bull_case'].append(item)
                elif section == 'bear': result['bear_case'].append(item)
                elif section == 'catalysts': result['catalysts'].append(item)
            elif line:
                if section == 'summary': result['summary'] += line + ' '
                elif section == 'news': result['news_analysis'] += line + ' '
                elif section == 'price': result['price_explanation'] += line + ' '
                elif section == 'action': result['action'] += line + ' '
        
        for k in ['summary', 'news_analysis', 'price_explanation', 'action']:
            result[k] = result[k].strip()
        
        return result
    
    def _rule_based(self, ticker: str, info: Dict, performance: Dict, events: List[Dict], 
                    news: List[Dict], valuation: Dict) -> Dict:
        upside = valuation.get('upside', 0)
        
        if upside > 20: rec, conf = 'BUY', 'MEDIUM'
        elif upside > 10: rec, conf = 'BUY', 'LOW'
        elif upside < -20: rec, conf = 'SELL', 'LOW'
        elif upside < -10: rec, conf = 'HOLD', 'LOW'
        else: rec, conf = 'HOLD', 'MEDIUM'
        
        perf_30d = performance.get('windows', {}).get(30, {}).get('return_pct', 0)
        
        summary = f"{info.get('name', ticker)} appears {valuation.get('verdict', 'mixed').lower()}. "
        summary += f"Stock moved {perf_30d:+.1f}% over 30 days. "
        if info.get('target_mean'):
            target_up = (info['target_mean'] - (info.get('current_price') or 1)) / (info.get('current_price') or 1) * 100
            summary += f"Analysts target ${info['target_mean']:.2f} ({target_up:+.1f}%)."
        
        news_text = f"Found {len(news)} news items. " + ("; ".join([n['title'][:35] for n in news[:3]]) if news else "Check financial news sites.")
        
        if events:
            biggest = max(events, key=lambda x: abs(x.get('return_pct', 0)))
            price_text = f"Biggest move: {biggest.get('return_pct', 0):+.1f}% on {biggest.get('date', 'N/A')}. Overall {perf_30d:+.1f}% over 30 days."
        else:
            price_text = f"Stock moved {perf_30d:+.1f}% over 30 days with moderate volatility."
        
        bull, bear = [], []
        if info.get('revenue_growth') and info['revenue_growth'] > 0.10: bull.append(f"Strong revenue growth ({info['revenue_growth']*100:.1f}%)")
        if info.get('roe') and info['roe'] > 0.15: bull.append(f"High ROE ({info['roe']*100:.1f}%)")
        if info.get('target_mean') and info['target_mean'] > (info.get('current_price') or 0): bull.append(f"Analyst target above current (${info['target_mean']:.2f})")
        if len(bull) < 3: bull.extend(['Strong market position', 'Industry tailwinds'][:3-len(bull)])
        
        if info.get('pe_trailing') and info['pe_trailing'] > 30: bear.append(f"High P/E ({info['pe_trailing']:.1f}x)")
        if info.get('debt_to_equity') and info['debt_to_equity'] > 100: bear.append(f"Elevated debt ({info['debt_to_equity']:.0f}% D/E)")
        if upside < -10: bear.append('Valuation stretched')
        if len(bear) < 3: bear.extend(['Competition risk', 'Macro uncertainty'][:3-len(bear)])
        
        return {
            'source': 'RULE_BASED (AI unavailable)',
            'recommendation': rec, 'confidence': conf, 'summary': summary,
            'news_analysis': news_text, 'price_explanation': price_text,
            'bull_case': bull[:4], 'bear_case': bear[:4],
            'fair_value': f"${valuation.get('range_low', 0):.2f} - ${valuation.get('range_high', 0):.2f}",
            'catalysts': ['Earnings release', 'Product announcements', 'Macro data'],
            'action': f"Fair value ${valuation.get('intrinsic', 0):.2f} vs current ${info.get('current_price', 0):.2f}. {valuation.get('verdict', '')}."
        }

# =============================================================================
# REPORT GENERATOR
# =============================================================================

def generate_report(ticker: str, info: Dict, performance: Dict, events: List[Dict],
                   news: List[Dict], valuation: Dict, analysis: Dict,
                   upcoming_events: Dict, earnings: List[Dict]) -> str:
    lines = []
    
    # Header
    lines.append("=" * 80)
    lines.append(f"{'TRADING REPORTER - COMPREHENSIVE ANALYSIS':^80}")
    lines.append(f"{ticker} - {info.get('name', ticker)[:50]:^80}")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}")
    if info.get('market_cap'):
        emp = f" | Employees: {info['employees']:,}" if info.get('employees') else ""
        lines.append(f"Market Cap: ${info['market_cap']/1e9:.1f}B{emp}")
    lines.append("")
    
    # Investment Thesis
    lines.append("=" * 80)
    lines.append(f"{'INVESTMENT THESIS':^80}")
    lines.append("=" * 80)
    rec = analysis.get('recommendation', 'HOLD')
    icon = "🟢" if 'BUY' in rec else ("🔴" if 'SELL' in rec else "🟡")
    lines.append(f"\n  {icon} RECOMMENDATION: {rec}")
    lines.append(f"  CONFIDENCE: {analysis.get('confidence', 'N/A')}")
    lines.append(f"  Analysis: {analysis.get('source', 'N/A')}")
    lines.append(f"\n  SUMMARY:")
    _wrap(lines, analysis.get('summary', 'N/A'))
    lines.append(f"\n  FAIR VALUE: {analysis.get('fair_value', 'N/A')}")
    lines.append("\n  BULL CASE:")
    for p in analysis.get('bull_case', [])[:4]: lines.append(f"    + {p}")
    lines.append("\n  BEAR CASE:")
    for p in analysis.get('bear_case', [])[:4]: lines.append(f"    - {p}")
    lines.append("\n  CATALYSTS:")
    for c in analysis.get('catalysts', [])[:3]: lines.append(f"    * {c}")
    lines.append(f"\n  ACTION:")
    _wrap(lines, analysis.get('action', 'N/A'))
    lines.append("")
    
    # Performance
    lines.append("=" * 80)
    lines.append(f"{'SHORT-TERM PERFORMANCE':^80}")
    lines.append("=" * 80)
    lines.append(f"\n  Current Price: ${performance.get('current_price', 0):.2f}\n")
    lines.append("  ┌─────────┬────────────┬────────────┬────────────┬────────────┐")
    lines.append("  │ Window  │   Return   │ Volatility │    High    │    Low     │")
    lines.append("  ├─────────┼────────────┼────────────┼────────────┼────────────┤")
    for w in [1, 7, 30]:
        d = performance.get('windows', {}).get(w, {})
        if d:
            lines.append(f"  │ {w:2}D     │ {d.get('return_pct',0):+8.1f}%  │ {d.get('volatility_ann',0):8.1f}%  │ ${d.get('high',0):>8.2f} │ ${d.get('low',0):>8.2f} │")
    lines.append("  └─────────┴────────────┴────────────┴────────────┴────────────┘")
    lines.append("")
    
    # Price & News
    lines.append("=" * 80)
    lines.append(f"{'PRICE MOVEMENTS & NEWS':^80}")
    lines.append("=" * 80)
    if events:
        lines.append("\n  SIGNIFICANT PRICE EVENTS:")
        for e in events[:5]:
            icon = "📈" if e.get('type') == 'surge' else "📉"
            lines.append(f"\n  {icon} {e.get('date')}: {e.get('type','').upper()} {e.get('return_pct',0):+.1f}%")
            lines.append(f"     Volume: {e.get('volume_ratio',1):.1f}x normal")
            matched = e.get('matched_news', [])
            if matched:
                lines.append("     Related news:")
                for n in matched[:2]: lines.append(f"       → {n.get('title', '')[:55]}...")
    else:
        lines.append("\n  No major price events detected.")
    lines.append("\n" + "-" * 80)
    lines.append("  PRICE EXPLANATION:")
    _wrap(lines, analysis.get('price_explanation', 'N/A'))
    lines.append("\n" + "-" * 80)
    lines.append("  NEWS IMPACT:")
    _wrap(lines, analysis.get('news_analysis', 'N/A'))
    lines.append("")
    
    # News
    lines.append("=" * 80)
    lines.append(f"{'RECENT NEWS':^80}")
    lines.append("=" * 80)
    if news:
        for i, n in enumerate(news[:7], 1):
            lines.append(f"\n  {i}. {n.get('title', '')[:65]}")
            lines.append(f"     {n.get('publisher', '')} | {n.get('datetime', '')}")
    else:
        lines.append("\n  No recent news found.")
    lines.append("")
    
    # Valuation
    lines.append("=" * 80)
    lines.append(f"{'VALUATION ANALYSIS':^80}")
    lines.append("=" * 80)
    verdict = valuation.get('verdict', 'N/A')
    v_icon = "🟢" if 'UNDER' in verdict else ("🔴" if 'OVER' in verdict else "🟡")
    lines.append(f"\n  {v_icon} VERDICT: {verdict}")
    lines.append(f"  Current Price: ${valuation.get('current', 0):.2f}")
    lines.append(f"  Fair Value: ${valuation.get('intrinsic', 0):.2f}")
    lines.append(f"  Upside/Downside: {valuation.get('upside', 0):+.1f}%")
    lines.append(f"  Range: ${valuation.get('range_low', 0):.2f} - ${valuation.get('range_high', 0):.2f}")
    lines.append("\n  Components:")
    for c in valuation.get('components', []): lines.append(f"    - {c['method']}: ${c['value']:.2f}")
    dcf = valuation.get('dcf', {})
    if 'intrinsic' in dcf: lines.append(f"\n  DCF: WACC {dcf.get('wacc',0)}%, Growth {dcf.get('growth',0)}%, Terminal {dcf.get('terminal',0)}%")
    mult = valuation.get('multiples', {})
    if mult.get('implied'):
        lines.append("\n  Multiples:")
        for m in mult['implied']: lines.append(f"    - {m['metric']}: ${m['price']:.2f} ({m['upside']:+.1f}%)")
    lines.append("")
    
    # Fundamentals
    lines.append("=" * 80)
    lines.append(f"{'KEY FUNDAMENTALS':^80}")
    lines.append("=" * 80)
    lines.append("\n  VALUATION:")
    for k, l in [('pe_trailing', 'P/E'), ('pe_forward', 'Fwd P/E'), ('pb_ratio', 'P/B'), ('ev_ebitda', 'EV/EBITDA')]:
        if info.get(k): lines.append(f"    {l}: {info[k]:.2f}")
    lines.append("\n  PROFITABILITY:")
    for k, l in [('profit_margin', 'Margin'), ('roe', 'ROE'), ('roa', 'ROA')]:
        if info.get(k): lines.append(f"    {l}: {info[k]*100:.1f}%")
    lines.append("\n  GROWTH:")
    for k, l in [('revenue_growth', 'Revenue'), ('earnings_growth', 'Earnings')]:
        if info.get(k): lines.append(f"    {l}: {info[k]*100:.1f}%")
    if info.get('52w_low') and info.get('52w_high'):
        current = info.get('current_price', 0)
        pos = (current - info['52w_low']) / (info['52w_high'] - info['52w_low']) * 100 if info['52w_high'] > info['52w_low'] else 50
        lines.append(f"\n  52-WEEK: ${info['52w_low']:.2f} - ${info['52w_high']:.2f} (now at {pos:.0f}%)")
    if info.get('recommendation'):
        rec_line = f"\n  ANALYSTS: {info['recommendation'].upper()}"
        if info.get('target_mean'): rec_line += f" | Target: ${info['target_mean']:.2f}"
        lines.append(rec_line)
    lines.append("")
    
    # Upcoming Events
    lines.append("=" * 80)
    lines.append(f"{'UPCOMING EVENTS':^80}")
    lines.append("=" * 80)
    
    has_events = False
    if upcoming_events:
        for k, v in upcoming_events.items():
            lines.append(f"  📅 {k}: {v}")
            has_events = True
    
    if earnings:
        lines.append("\n  EARNINGS HISTORY:")
        for e in earnings[:4]:
            date_str = e.get('date', 'N/A')
            est = f"Est ${e['eps_estimate']:.2f}" if e.get('eps_estimate') else "Est N/A"
            act = f"Act ${e['eps_actual']:.2f}" if e.get('eps_actual') else "Act N/A"
            surp = f" ({e['surprise']:+.1f}%)" if e.get('surprise') else ""
            lines.append(f"    {date_str}: {est} / {act}{surp}")
        has_events = True
    
    if not has_events:
        lines.append("  No upcoming events data available.")
        lines.append("  Check investor relations website for earnings calendar.")
    lines.append("")
    
    # Disclaimer
    lines.append("=" * 80)
    lines.append("DISCLAIMER: Educational purposes only. NOT financial advice.")
    lines.append("Past performance ≠ future results. Do your own research.")
    lines.append("=" * 80)
    
    return '\n'.join(lines)

def _wrap(lines: List[str], text: str, width: int = 75):
    words = (text or 'N/A').split()
    line = "  "
    for word in words:
        if len(line) + len(word) > width:
            lines.append(line)
            line = "  " + word
        else:
            line += " " + word if line.strip() else "  " + word
    if line.strip():
        lines.append(line)

# =============================================================================
# MAIN
# =============================================================================

def analyze_stock(ticker: str) -> Dict:
    print(f"\n{'='*60}")
    print(f"Analyzing {ticker}...")
    print('='*60)
    
    print("\n[1/7] Fetching price data...")
    df = fetch_price_data(ticker)
    if df is None:
        print("  ✗ Failed")
        return {'error': 'No data'}
    print(f"  ✓ {len(df)} days")
    
    print("[2/7] Fetching fundamentals...")
    info = fetch_stock_info(ticker)
    print(f"  ✓ {info.get('name', ticker)}")
    
    print("[3/7] Fetching news...")
    news = fetch_all_news(ticker, info.get('name', ''))
    print(f"  ✓ {len(news)} articles")
    
    print("[4/7] Fetching upcoming events...")
    upcoming_events = fetch_upcoming_events(ticker)
    earnings = fetch_earnings_history(ticker)
    print(f"  ✓ {len(upcoming_events)} events, {len(earnings)} earnings records")
    
    print("[5/7] Analyzing performance...")
    performance = analyze_performance(df)
    events = detect_events(df)
    events = match_news_to_events(events, news)
    print(f"  ✓ {len(events)} price events")
    
    print("[6/7] Running valuation...")
    valuation = comprehensive_valuation(info)
    print(f"  ✓ Fair value: ${valuation.get('intrinsic', 0):.2f} ({valuation.get('verdict', 'N/A')})")
    
    print("[7/7] AI analysis...")
    analyzer = GeminiAnalyzer()
    analysis = analyzer.analyze(ticker, info, performance, events, news, valuation, upcoming_events)
    
    print("\nGenerating report...")
    report = generate_report(ticker, info, performance, events, news, valuation, analysis, upcoming_events, earnings)
    
    print("\n" + report)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = Path(CONFIG.output_dir) / f"{ticker}_REPORT_{ts}.txt"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✓ Report saved: {filepath}")
    return {'ticker': ticker, 'file': str(filepath)}

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Trading Reporter v1.3')
    parser.add_argument('tickers', nargs='*')
    args = parser.parse_args()
    
    if not args.tickers:
        print("\nFeatures:")
        print("  ✓ 1D/7D/30D performance")
        print("  ✓ News + Google backup")
        print("  ✓ Price surge detection")
        print("  ✓ DCF/Multiples/EPV valuation")
        print("  ✓ Upcoming events & earnings")
        print("  ✓ Gemini AI analysis")
        print("")
        print("═" * 50)
        print("GET FREE GEMINI API KEY:")
        print("  1. Go to: https://aistudio.google.com/app/apikey")
        print("  2. Click 'Create API Key'")
        print("  3. Copy the key and replace GEMINI_API_KEY in script")
        print("═" * 50)
        inp = input("\nEnter ticker(s): ").strip()
        args.tickers = inp.upper().split() if inp else []
    
    if not args.tickers:
        print("No ticker")
        return
    
    for ticker in args.tickers:
        try:
            analyze_stock(ticker.upper())
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✓ Done! Reports in {CONFIG.output_dir}/")

if __name__ == "__main__":
    main()
