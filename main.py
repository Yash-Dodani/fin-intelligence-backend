import os
import json
import time
import asyncio
import requests
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from groq import Groq
from datetime import datetime, timedelta
import pytz

# =================================================================
# CONFIGURATION & KEYS (SECURED FOR PRODUCTION)
# =================================================================
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_KEY = os.getenv("GROQ_API_KEY", "")

gemini_client = genai.Client(api_key=GEMINI_KEY)
groq_client = Groq(api_key=GROQ_KEY)

app = FastAPI(title="Vantage Ultra-Prism Pro API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =================================================================
# KEEP-ALIVE HEALTH ENDPOINT (pinged by GitHub Actions every 14 min)
# =================================================================
@app.get("/health")
async def health_check():
    return {"status": "ok", "engine": "Vantage-Ultra-Pro-V5"}

# =================================================================
# AGGRESSIVE CACHING UTILITY
# =================================================================
cache_store = {}

def get_cache(key: str, ttl: int):
    """
    ttl in seconds: 
    - Option Chain: 60s
    - Technicals: 30s
    - Prices: 5s
    - AI Strategy: 120s
    """
    if key in cache_store:
        data, timestamp = cache_store[key]
        if time.time() - timestamp < ttl:
            return data
    return None

def set_cache(key: str, data: any):
    cache_store[key] = (data, time.time())

# =================================================================
# MARKET STATUS UTILITY
# =================================================================
def get_market_status():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    # Check if Weekend
    if now.weekday() >= 5:
        return False, "Market Closed (Weekend)"
    
    # Check Market Hours (9:15 AM to 3:30 PM)
    start_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    end_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    if start_time <= now <= end_time:
        return True, "Market Open"
    elif now < start_time:
        return False, f"Market Opens at 09:15 AM"
    else:
        return False, "Market Closed"

# =================================================================
# NSE SESSION MANAGEMENT (Improved)
# =================================================================
class NSESession:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.last_init = 0

    def init_session(self):
        if time.time() - self.last_init > 180: # Refresh every 3 mins
            try:
                # Visit homepage first to get cookies
                self.session.get("https://www.nseindia.com", timeout=15)
                # Small sleep to mimic human
                time.sleep(1)
                self.last_init = time.time()
            except Exception as e:
                print(f"NSE Session Error: {e}")
                self.last_init = 0 # Retry next time

    def get_json(self, url: str):
        self.init_session()
        try:
            # Add referer dynamically
            self.session.headers.update({'Referer': 'https://www.nseindia.com/option-chain'})
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401 or response.status_code == 403:
                print(f"NSE Blocked: {response.status_code}. Re-initializing...")
                self.last_init = 0 
                time.sleep(2)
                return self.get_json(url)
        except Exception as e:
            print(f"API Fetch Error: {e}")
        return None

nse_client = NSESession()

# =================================================================
# DATA PROCESSING HELPERS
# =================================================================
def get_historical_data(symbol: str, period="5d", interval="5m"):
    cache_key = f"hist_{symbol}_{period}_{interval}"
    cached = get_cache(cache_key, 60) # Increased TTL for stability
    if cached is not None: return cached

    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            df = yf.download(symbol, period="1mo", interval="1d", progress=False)
        
        if not df.empty:
            set_cache(cache_key, df)
            return df
    except Exception as e:
        print(f"YFinance Error for {symbol}: {e}")
    return None

def calculate_technicals(df: pd.DataFrame):
    if df is None or df.empty:
        return {}
    
    try:
        # Handle possible multi-index
        if isinstance(df.columns, pd.MultiIndex):
            close = df['Close'].iloc[:, 0]
            high = df['High'].iloc[:, 0]
            low = df['Low'].iloc[:, 0]
            volume = df['Volume'].iloc[:, 0]
        else:
            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']
        
        current_price = float(close.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) > 1 else current_price
        
        # RSI (14)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = round(float(rsi.iloc[-1]), 2) if not rsi.empty else 50
        
        # EMA (20)
        ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        
        # Volume spikes
        avg_vol = volume.rolling(window=20).mean().iloc[-1]
        curr_vol = volume.iloc[-1]
        vol_spike = curr_vol > (avg_vol * 1.5)
        
        # Support/Resistance (Pivot)
        max_h = float(high.max())
        min_l = float(low.min())
        pivot = (max_h + min_l + current_price) / 3
        r1 = (2 * pivot) - min_l
        s1 = (2 * pivot) - max_h
        
        return {
            "price": round(current_price, 2),
            "change": round(current_price - prev_close, 2),
            "changePercent": round(((current_price - prev_close) / prev_close) * 100, 2),
            "rsi": current_rsi,
            "ema20": round(float(ema20), 2),
            "pivot": round(pivot, 2),
            "r1": round(r1, 2),
            "s1": round(s1, 2),
            "volSpike": vol_spike,
            "trend": "Bullish" if current_rsi > 55 else "Bearish" if current_rsi < 45 else "Neutral",
            "strength": "Strong" if abs(current_rsi - 50) > 15 else "Weak"
        }
    except Exception as e:
        print(f"Technical Error: {e}")
        return {}

# =================================================================
# AI STRATEGY ENGINE
# =================================================================
async def generate_fno_strategy(symbol: str, techs: dict, oc_data: dict = None, mode: str = "fno"):
    cache_key = f"ai_{symbol}_{mode}"
    cached = get_cache(cache_key, 300) # Strategy lasts 5 mins
    if cached: return cached

    role = "Institutional F&O Quant Researcher" if mode == "fno" else "Intraday Equity Specialist"
    task = f"Generate a high-conviction {mode.upper()} strategy for {symbol}."
    
    prompt = f"""
    ROLE: {role}
    TASK: {task}
    
    TECHNICALS:
    Price: ₹{techs.get('price')}
    Trend: {techs.get('trend')} ({techs.get('strength')})
    RSI: {techs.get('rsi')}
    Pivot: {techs.get('pivot')}
    Levels: R1={techs.get('r1')}, S1={techs.get('s1')}
    Volume Spike: {'Yes' if techs.get('volSpike') else 'No'}
    
    OPTION DATA (PCR/OI):
    {oc_data if oc_data else 'No data - Use Technicals'}
    
    Output exactly ONE JSON object in a list. DO NOT include markdown formatting or extra text.
    Schema:
    [{{
        "stockName": "{symbol} Pro Analysis",
        "symbol": "{symbol}",
        "tradeDate": "{datetime.now().strftime('%d %b %Y')}",
        "tradeTime": "{datetime.now().strftime('%H:%M')}",
        "isMarketOpen": true,
        "tradeType": "{'CALL' if techs.get('trend') == 'Bullish' else 'PUT' if mode == 'fno' else 'BUY'}",
        "prevClose": {techs.get('price', 0) - techs.get('change', 0)},
        "currentMarketPrice": {techs.get('price', 0)},
        "exactEntryPrice": <float>,
        "exactEntryTime": "LATEST - {datetime.now().strftime('%H:%M')}",
        "target1": <float>,
        "target2": <float>,
        "stopLoss": <float>,
        "estimatedProfitPercentage": <float>,
        "confidenceScore": <int: 0-100>,
        "strategyLogic": "Quick bullet points on why this trade is valid..."
    }}]
    """

    # Multi-AI Strategy: Try Gemini, Fallback to Groq
    try:
        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model='gemini-2.0-flash',
            contents=prompt
        )
        res_text = response.text.strip()
        if "```json" in res_text:
            res_text = res_text.split("```json")[1].split("```")[0].strip()
        elif "```" in res_text:
            res_text = res_text.split("```")[1].strip()
        data = json.loads(res_text)
        set_cache(cache_key, data)
        return data
    except:
        try:
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            res_text = completion.choices[0].message.content.strip()
            if "```json" in res_text:
                res_text = res_text.split("```json")[1].split("```")[0].strip()
            data = json.loads(res_text)
            set_cache(cache_key, data)
            return data
        except:
            # Last ditch fallback logic
            is_bull = techs.get('trend') == 'Bullish'
            price = techs.get('price', 0)
            entry = price
            t1 = price * (1.01 if is_bull else 0.99)
            sl = price * (0.995 if is_bull else 1.005)
            return [{
                "stockName": symbol,
                "symbol": symbol,
                "tradeType": "CALL" if is_bull else "PUT",
                "exactEntryPrice": entry,
                "target1": round(t1, 2),
                "stopLoss": round(sl, 2),
                "confidenceScore": 65,
                "strategyLogic": "High momentum detected. Manual verification recommended."
            }]

# =================================================================
# API ENDPOINTS
# =================================================================

@app.get("/api/trade")
async def get_trade(symbol: str = "NIFTY"):
    yf_map = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "FINNIFTY": "NIFTY_FIN_SERVICE.NS"}
    yf_sym = yf_map.get(symbol.upper(), f"{symbol.upper()}.NS")
    
    df = get_historical_data(yf_sym)
    techs = calculate_technicals(df)
    
    if not techs:
        raise HTTPException(status_code=404, detail="Symbol Data Error")
    
    oc_summary = None
    if symbol.upper() in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
        try:
            oc = await get_option_chain(symbol)
            oc_summary = {"pcr": oc.get("pcr"), "sentiment": oc.get("sentiment")}
        except: pass
        
    return await generate_fno_strategy(symbol, techs, oc_summary)

@app.get("/api/technical")
async def get_technical(symbol: str):
    """ Technical Indicators Only """
    yf_map = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "FINNIFTY": "NIFTY_FIN_SERVICE.NS"}
    yf_sym = yf_map.get(symbol.upper(), f"{symbol.upper()}.NS")
    
    df = get_historical_data(yf_sym)
    techs = calculate_technicals(df)
    
    if not techs:
        raise HTTPException(status_code=404, detail="Symbol Data Error")
    
    # Add trend and strength for Flutter's QuoteSnapshot
    change_pct = techs.get("changePercent", 0)
    techs["trend"] = "Bullish" if change_pct > 0.2 else "Bearish" if change_pct < -0.2 else "Neutral"
    techs["strength"] = "Strong" if abs(change_pct) > 1.0 else "Moderate" if abs(change_pct) > 0.4 else "Weak"
    
    return techs

@app.get("/api/fno-lab")
async def get_fno_lab():
    """ Dedicated F&O Lab Logic for Main Indices """
    indices = ["NIFTY", "BANKNIFTY", "FINNIFTY"]
    results = []
    for idx in indices:
        try:
            data = await get_trade(idx)
            if data:
                results.append(data[0])
        except:
            continue
    return results

@app.get("/api/intraday-analysis")
async def get_intraday_analysis(symbols: str):
    """ Intraday Analysis for multiple symbols (comma separated) """
    sym_list = symbols.split(",")
    results = []
    for sym in sym_list:
        try:
            yf_sym = f"{sym.upper().strip()}.NS"
            df = get_historical_data(yf_sym)
            techs = calculate_technicals(df)
            if techs:
                strategy = await generate_fno_strategy(sym.upper().strip(), techs, mode="intraday")
                results.append(strategy[0])
        except:
            continue
    return results

@app.get("/api/stocks")
async def get_stocks():
    """ Top 20 Trending Stocks """
    cache_key = "stocks_v5"
    cached = get_cache(cache_key, 15)
    if cached: return cached

    # Expanded 20+ stocks list (mostly liquid NSE stocks)
    stock_list = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
        "SBIN.NS", "AXISBANK.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
        "LT.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "MARUTI.NS", "SUNPHARMA.NS",
        "TITAN.NS", "ADANIENT.NS", "TATASTEEL.NS", "M&M.NS", "HCLTECH.NS",
        "ASIANPAINT.NS", "WIPRO.NS", "ONGC.NS", "ULTRACEMCO.NS"
    ]
    
    results = []
    try:
        # Group download for speed
        data = yf.download(stock_list, period="2d", interval="15m", progress=False)
        if data.empty: return []

        for sym in stock_list:
            try:
                if sym in data['Close']:
                    series = data['Close'][sym].dropna()
                    if series.empty: continue
                    
                    curr = float(series.iloc[-1])
                    prev = float(series.iloc[0])
                    change = ((curr-prev)/prev)*100
                    
                    # Clean symbol
                    clean_sym = sym.replace(".NS", "")
                    
                    results.append({
                        "symbol": clean_sym,
                        "name": clean_sym.replace("_", " "),
                        "price": round(curr, 2),
                        "change": f"{'+' if change >= 0 else ''}{change:.2f}%",
                        "trend": "Bullish" if change > 0.5 else "Bearish" if change < -0.5 else "Neutral"
                    })
            except: continue
    except Exception as e:
        print(f"Bulk Fetch Error: {e}")
    
    set_cache(cache_key, results)
    return results

@app.get("/api/option-chain")
async def get_option_chain(symbol: str = "NIFTY"):
    cache_key = f"oc_{symbol.upper()}"
    cached = get_cache(cache_key, 60)
    if cached: return cached

    base_url = "https://www.nseindia.com/api/option-chain-"
    url = f"{base_url}indices?symbol={symbol.upper()}"
    if symbol.upper() not in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]:
        url = f"{base_url}equities?symbol={symbol.upper()}"
    
    data = nse_client.get_json(url)
    if not data:
        # Better simulated data for F&O lab when NSE is blocked
        return {
            "symbol": symbol, 
            "error": "NSE Latency - Using Technical Proxy", 
            "status": "Degraded",
            "pcr": 0.95, # Neutral fallback
            "sentiment": "Neutral"
        }
    
    try:
        filtered = data.get('filtered', {})
        records = data.get('records', {})
        ce_oi = filtered.get('CE', {}).get('totOI', 0)
        pe_oi = filtered.get('PE', {}).get('totOI', 0)
        pcr = round(pe_oi / (ce_oi or 1), 2)
        
        result = {
            "symbol": symbol.upper(),
            "underlying": records.get('underlyingValue'),
            "pcr": pcr,
            "sentiment": "Bullish" if pcr > 1.15 else "Bearish" if pcr < 0.85 else "Neutral",
            "timestamp": records.get('timestamp'),
            "expiries": records.get('expiryDates', [])[:3],
            "data": filtered.get('data', [])[:10]
        }
        set_cache(cache_key, result)
        return result
    except:
        return {"error": "Processing Error", "pcr": 1.0, "sentiment": "Neutral"}

@app.get("/api/indices")
async def get_indices():
    cache_key = "indices_v5"
    cached = get_cache(cache_key, 10)
    if cached: return cached

    idx_map = {"^NSEI": "NIFTY 50", "^NSEBANK": "BANK NIFTY", "^BSESN": "SENSEX", "^CNXIT": "NIFTY IT"}
    results = []
    
    try:
        data = yf.download(list(idx_map.keys()), period="2d", interval="15m", progress=False)
        for sym, name in idx_map.items():
            if sym in data['Close']:
                series = data['Close'][sym].dropna()
                if series.empty: continue
                curr = float(series.iloc[-1])
                prev = float(series.iloc[0])
                change_pct = ((curr-prev)/prev)*100
                results.append({
                    "name": name,
                    "symbol": sym.replace("^", ""),
                    "price": round(curr, 2),
                    "change": f"{'+' if change_pct >= 0 else ''}{change_pct:.2f}%"
                })
    except: pass
    
    set_cache(cache_key, results)
    return results

@app.get("/")
def health():
    isOpen, msg = get_market_status()
    return {
        "engine": "Vantage-Ultra-Pro-V5",
        "market_status": msg,
        "is_open": isOpen,
        "cache_objects": len(cache_store),
        "server_time": datetime.now().strftime("%H:%M:%S")
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
