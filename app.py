"""
AlphaEdge API — Flask Backend
Serves real NSE signals to the web app
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time

app = Flask(__name__)
CORS(app)  # Allow frontend to call this API

# ── Universe ──────────────────────────────────────────────────────────────────
STOCKS = [
    ("HDFCBANK",   "HDFC Bank",        "Banking"),
    ("ICICIBANK",  "ICICI Bank",       "Banking"),
    ("SBIN",       "SBI",              "Banking"),
    ("RELIANCE",   "Reliance",         "Energy"),
    ("INFY",       "Infosys",          "IT"),
    ("TCS",        "TCS",              "IT"),
    ("LT",         "L&T",              "Infra"),
    ("AXISBANK",   "Axis Bank",        "Banking"),
    ("KOTAKBANK",  "Kotak Bank",       "Banking"),
    ("BHARTIARTL", "Bharti Airtel",    "Telecom"),
    ("BAJFINANCE", "Bajaj Finance",    "NBFC"),
    ("MARUTI",     "Maruti",           "Auto"),
    ("NTPC",       "NTPC",             "Power"),
    ("ONGC",       "ONGC",             "Energy"),
    ("COALINDIA",  "Coal India",       "Commodities"),
    ("BEL",        "BEL",              "Defence"),
    ("HAL",        "HAL",              "Defence"),
    ("TRENT",      "Trent",            "Retail"),
    ("ULTRACEMCO", "UltraTech",        "Cement"),
    ("HINDUNILVR", "HUL",              "FMCG"),
    ("ITC",        "ITC",              "FMCG"),
    ("SUNPHARMA",  "Sun Pharma",       "Pharma"),
    ("WIPRO",      "Wipro",            "IT"),
    ("HCLTECH",    "HCL Tech",         "IT"),
    ("POWERGRID",  "Power Grid",       "Power"),
    ("JSWSTEEL",   "JSW Steel",        "Metals"),
    ("TATASTEEL",  "Tata Steel",       "Metals"),
    ("CIPLA",      "Cipla",            "Pharma"),
    ("EICHERMOT",  "Eicher Motors",    "Auto"),
    ("M&M",        "M&M",              "Auto"),
    ("INDUSINDBK", "IndusInd Bank",    "Banking"),
    ("RECLTD",     "REC",              "NBFC"),
]

# ── Indicators ────────────────────────────────────────────────────────────────
def calc_rsi(closes, period=14):
    d = closes.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = (-d.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return float((100 - 100 / (1 + rs)).iloc[-1])

def calc_ema(closes, period):
    return float(closes.ewm(span=period, adjust=False).mean().iloc[-1])

def calc_atr(hist, period=14):
    h = hist["High"]; l = hist["Low"]; pc = hist["Close"].shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])

def higher_highs_lows(closes, n=5):
    r = closes.iloc[-n:].values
    return bool(all(r[i] >= r[i-1] for i in range(1, len(r))))

def fetch(sym):
    try:
        t = yf.Ticker(sym + ".NS")
        hist = t.history(period="60d", interval="1d")
        return None if hist.empty or len(hist) < 20 else hist
    except:
        return None

# ── Market Context ────────────────────────────────────────────────────────────
def get_market():
    try:
        hist   = yf.Ticker("^NSEI").history(period="60d", interval="1d")
        closes = hist["Close"]
        last   = float(closes.iloc[-1])
        prev   = float(closes.iloc[-2])
        chg    = ((last - prev) / prev) * 100
        rsi    = calc_rsi(closes)
        ema20  = calc_ema(closes, 20)
        typ    = (float(hist["High"].iloc[-1]) + float(hist["Low"].iloc[-1]) + last) / 3
        vwap   = float(((hist["High"] + hist["Low"] + hist["Close"]) / 3).iloc[-5:].mean())
        av20   = last > ema20
        av_vwap= typ > vwap
        b3     = closes.iloc[-3:].values
        bread  = bool(b3[-1] > b3[0])
        mkt_pts= int(sum([av_vwap*5, av20*5, bread*5, (rsi>50)*5]))
        bear   = sum([chg<-0.5, not av20, rsi<45, not av_vwap])
        bull   = sum([chg>0.5, av20, rsi>55, av_vwap])
        bias   = ("STRONGLY BEARISH" if bear>=3 else "BEARISH" if bear==2
                  else "STRONGLY BULLISH" if bull>=3 else "BULLISH" if bull==2
                  else "NEUTRAL")
        return dict(price=round(last,1), chg=round(chg,2), rsi=round(rsi,1),
                    av20=av20, av_vwap=av_vwap, bread=bread,
                    mkt_pts=mkt_pts, bias=bias, bear=bear, bull=bull,
                    block_long=(bear>=2), block_short=(bull>=3))
    except Exception as e:
        return dict(price=0, chg=0, rsi=50, av20=True, av_vwap=True,
                    bread=True, mkt_pts=10, bias="NEUTRAL",
                    bear=0, bull=0, block_long=False, block_short=False,
                    error=str(e))

# ── Score Engine ──────────────────────────────────────────────────────────────
def score_stock(sym, name, sector, hist, mkt):
    try:
        closes = hist["Close"]; vols = hist["Volume"]
        price  = float(closes.iloc[-1])
        vol    = float(vols.iloc[-1])
        avg10  = float(vols.iloc[-10:].mean())
        rvol   = round(vol / avg10 if avg10 > 0 else 1.0, 1)
        atr    = calc_atr(hist)
        rsi    = calc_rsi(closes)
        e20    = calc_ema(closes, 20)
        e50    = calc_ema(closes, 50)
        ae20   = price > e20
        ae50   = price > e50
        hhhl   = higher_highs_lows(closes)
        typ    = (float(hist["High"].iloc[-1]) + float(hist["Low"].iloc[-1]) + price) / 3
        vwap   = float(((hist["High"] + hist["Low"] + hist["Close"]) / 3).iloc[-5:].mean())
        av     = typ > vwap
        h52    = float(closes.max())
        pfh    = round(((price - h52) / h52) * 100, 1)

        bull_sigs = sum([av, ae20, ae50, hhhl, rsi>55])
        direction = "LONG" if bull_sigs >= 3 else "SHORT"
        is_long   = direction == "LONG"

        s_mkt  = mkt["mkt_pts"]
        s_sec  = (int(ae50)*10 + int(rsi>60)*5) if is_long else (int(not ae50)*10 + int(rsi<45)*5)
        s_mom  = (int(ae20)*5 + int(ae50)*5 + int(hhhl)*5) if is_long else (int(not ae20)*5 + int(not ae50)*5 + int(not hhhl)*5)
        s_vwap = (10 if av else 0) if is_long else (10 if not av else 0)
        s_vol  = (10 if rvol>2 else 7 if rvol>1.5 else 5 if rvol>1.0 else 3 if rvol>0.7 else 0)
        if is_long:
            s_rsi = (10 if 60<=rsi<=70 else 5 if 55<=rsi<60 else 3 if 50<=rsi<55 else 2 if rsi>75 else 0)
        else:
            s_rsi = (10 if rsi<35 else 8 if rsi<40 else 5 if rsi<45 else 3 if rsi<50 else 0)
        s_opt  = (int(rvol>1.5 and av)*5 + int(ae20 and ae50)*5) if is_long else (int(rvol>1.5 and not av)*5 + int(not ae20 and not ae50)*5)
        s_risk = 10
        if rvol < 0.5:       s_risk -= 3
        if rsi > 80:         s_risk -= 3
        if abs(pfh) < 1:     s_risk -= 2
        if atr/price > 0.03: s_risk -= 2
        s_risk = max(0, s_risk)

        total = s_mkt + s_sec + s_mom + s_vwap + s_vol + s_rsi + s_opt + s_risk

        if is_long:
            sl = round(price - atr*1.5, 1)
            t1 = round(price + atr*2.0, 1)
            t2 = round(price + atr*3.5, 1)
        else:
            sl = round(price + atr*1.5, 1)
            t1 = round(price - atr*2.0, 1)
            t2 = round(price - atr*3.5, 1)

        rr     = round(abs(t1-price)/abs(sl-price), 1) if abs(sl-price) > 0 else 1.5
        strike = int(round(price/50)*50)
        atm    = f"{sym} {strike} {'CE' if is_long else 'PE'}"
        conf   = "High" if total>=85 else "Medium" if total>=70 else "Low"

        stars  = 5 if total>=90 else 4 if total>=80 else 3 if total>=70 else 2

        return dict(
            sym=sym, name=name, sector=sector,
            price=round(price,1), direction=direction,
            score=total, confidence=conf, stars=stars,
            rsi=round(rsi,1), rvol=rvol, atr=round(atr,1),
            above_vwap=av, above_ema20=ae20, above_ema50=ae50,
            hhhl=hhhl, pct_from_high=pfh,
            components=dict(
                market_trend=s_mkt, sector_strength=s_sec,
                stock_momentum=s_mom, vwap=s_vwap,
                volume=s_vol, rsi_score=s_rsi,
                option_chain=s_opt, risk_filters=s_risk
            ),
            entry=round(price,1), sl=sl, t1=t1, t2=t2,
            rr=rr, atm=atm,
        )
    except Exception as e:
        return None

# ── Cache (avoid hammering Yahoo Finance) ────────────────────────────────────
cache = {"data": None, "time": None, "market": None}
CACHE_MINUTES = 15

def is_cache_valid():
    if not cache["time"]: return False
    diff = (datetime.now() - cache["time"]).seconds / 60
    return diff < CACHE_MINUTES

# ── API Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return jsonify({"status": "AlphaEdge API running", "version": "v3"})

@app.route("/api/market")
def market():
    """Get Nifty market context"""
    try:
        mkt = get_market()
        return jsonify({"success": True, "data": mkt})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/scan")
def scan():
    """Scan top N stocks and return ranked signals"""
    n = min(int(request.args.get("n", 10)), 32)
    min_score = int(request.args.get("min_score", 0))
    direction = request.args.get("direction", "all")  # all, long, short

    # Use cache if valid
    if is_cache_valid() and cache["data"]:
        results = cache["data"]
        mkt = cache["market"]
    else:
        mkt = get_market()
        batch = STOCKS[:n]
        results = []
        for sym, name, sector in batch:
            hist = fetch(sym)
            if hist is not None:
                r = score_stock(sym, name, sector, hist, mkt)
                if r:
                    results.append(r)
            time.sleep(0.2)
        results.sort(key=lambda x: x["score"], reverse=True)
        cache["data"] = results
        cache["market"] = mkt
        cache["time"] = datetime.now()

    # Filter
    filtered = [r for r in results if r["score"] >= min_score]
    if direction == "long":
        filtered = [r for r in filtered if r["direction"] == "LONG"]
    elif direction == "short":
        filtered = [r for r in filtered if r["direction"] == "SHORT"]

    # Market gate
    gated = []
    for r in filtered:
        if mkt["block_long"] and r["direction"] == "LONG": continue
        if mkt["block_short"] and r["direction"] == "SHORT": continue
        gated.append(r)

    return jsonify({
        "success": True,
        "date": datetime.now().strftime("%d %B %Y %H:%M"),
        "market": mkt,
        "count": len(gated),
        "signals": gated[:10],
        "cached": is_cache_valid()
    })

@app.route("/api/stock/<sym>")
def single_stock(sym):
    """Get signal for a single stock"""
    match = [(s,n,sec) for s,n,sec in STOCKS if s.upper() == sym.upper()]
    if not match:
        return jsonify({"success": False, "error": f"{sym} not in universe"}), 404
    s, n, sec = match[0]
    mkt  = get_market()
    hist = fetch(s)
    if hist is None:
        return jsonify({"success": False, "error": "No data"}), 500
    r = score_stock(s, n, sec, hist, mkt)
    return jsonify({"success": True, "market": mkt, "signal": r})

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "time": datetime.now().isoformat(),
        "cache_valid": is_cache_valid()
    })

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
