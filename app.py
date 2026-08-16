"""
AlphaEdge API — Flask Backend
Serves NSE signals and Zerodha Kite Connect authentication
"""

from flask import Flask, jsonify, request, redirect
from flask_cors import CORS

import os
import time
from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as np

from kiteconnect import KiteConnect


app = Flask(__name__)
CORS(app)


# =============================================================================
# ZERODHA KITE CONNECT
# =============================================================================

KITE_API_KEY = os.environ.get("KITE_API_KEY")
KITE_API_SECRET = os.environ.get("KITE_API_SECRET")

kite = KiteConnect(api_key=KITE_API_KEY) if KITE_API_KEY else None

# Temporary in-memory token.
# IMPORTANT:
# This will disappear when the Render instance restarts.
# We will add secure token persistence in a later step.
kite_access_token = None


# =============================================================================
# UNIVERSE
# =============================================================================

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


# =============================================================================
# INDICATORS
# =============================================================================

def calc_rsi(closes, period=14):
    d = closes.diff()

    gain = d.clip(lower=0).rolling(period).mean()
    loss = (-d.clip(upper=0)).rolling(period).mean()

    rs = gain / loss

    return float((100 - 100 / (1 + rs)).iloc[-1])


def calc_ema(closes, period):
    return float(
        closes.ewm(span=period, adjust=False).mean().iloc[-1]
    )


def calc_atr(hist, period=14):
    h = hist["High"]
    l = hist["Low"]
    pc = hist["Close"].shift(1)

    tr = pd.concat(
        [
            h - l,
            (h - pc).abs(),
            (l - pc).abs()
        ],
        axis=1
    ).max(axis=1)

    return float(
        tr.rolling(period).mean().iloc[-1]
    )


def higher_highs_lows(closes, n=5):
    r = closes.iloc[-n:].values

    return bool(
        all(r[i] >= r[i - 1] for i in range(1, len(r)))
    )


# =============================================================================
# YAHOO DATA — CURRENT RESEARCH FALLBACK
# =============================================================================

def fetch(sym):
    """
    Current research data source.

    NOTE:
    This is daily data and is NOT yet our final live intraday engine.
    We will replace this with Zerodha historical/intraday data next.
    """

    try:
        ticker = yf.Ticker(sym + ".NS")

        hist = ticker.history(
            period="60d",
            interval="1d"
        )

        if hist.empty or len(hist) < 20:
            return None

        return hist

    except Exception:
        return None


# =============================================================================
# MARKET CONTEXT
# =============================================================================

def get_market():

    try:

        hist = None

        for ticker in [
            "^NSEI",
            "NIFTY50.NS",
            "^CNX500",
            "NIFTYBEES.NS"
        ]:

            try:

                t = yf.Ticker(ticker)

                h = t.history(
                    period="30d",
                    interval="1d"
                )

                if not h.empty and len(h) >= 2:
                    hist = h
                    break

            except Exception:
                continue


        # IMPORTANT:
        # Do not generate fake market values.
        if hist is None or len(hist) < 2:

            return {
                "available": False,
                "price": None,
                "chg": None,
                "rsi": None,
                "av20": None,
                "av_vwap": None,
                "bread": None,
                "mkt_pts": None,
                "bias": "NO_DATA",
                "bear": None,
                "bull": None,
                "block_long": True,
                "block_short": True,
                "note": "Nifty data unavailable — NO TRADE"
            }


        closes = hist["Close"].dropna()

        last = float(closes.iloc[-1])
        prev = float(closes.iloc[-2])

        chg = ((last - prev) / prev) * 100

        rsi = calc_rsi(closes)

        ema20 = calc_ema(closes, 20)

        typ = (
            float(hist["High"].iloc[-1])
            + float(hist["Low"].iloc[-1])
            + last
        ) / 3

        # Existing VWAP proxy.
        # Will be replaced by TRUE intraday VWAP.
        vwap = float(
            (
                (hist["High"] + hist["Low"] + hist["Close"]) / 3
            ).iloc[-5:].mean()
        )

        av20 = last > ema20

        av_vwap = typ > vwap

        b3 = closes.iloc[-3:].values

        bread = bool(
            b3[-1] > b3[0]
        )

        mkt_pts = int(
            sum(
                [
                    av_vwap * 5,
                    av20 * 5,
                    bread * 5,
                    (rsi > 50) * 5
                ]
            )
        )

        bear = sum(
            [
                chg < -0.5,
                not av20,
                rsi < 45,
                not av_vwap
            ]
        )

        bull = sum(
            [
                chg > 0.5,
                av20,
                rsi > 55,
                av_vwap
            ]
        )

        bias = (
            "STRONGLY BEARISH"
            if bear >= 3
            else "BEARISH"
            if bear == 2
            else "STRONGLY BULLISH"
            if bull >= 3
            else "BULLISH"
            if bull == 2
            else "NEUTRAL"
        )

        return {
            "available": True,
            "price": round(last, 1),
            "chg": round(chg, 2),
            "rsi": round(rsi, 1),
            "av20": av20,
            "av_vwap": av_vwap,
            "bread": bread,
            "mkt_pts": mkt_pts,
            "bias": bias,
            "bear": bear,
            "bull": bull,
            "block_long": bear >= 2,
            "block_short": bull >= 3
        }


    except Exception as e:

        return {
            "available": False,
            "price": None,
            "chg": None,
            "rsi": None,
            "av20": None,
            "av_vwap": None,
            "bread": None,
            "mkt_pts": None,
            "bias": "NO_DATA",
            "bear": None,
            "bull": None,
            "block_long": True,
            "block_short": True,
            "error": str(e),
            "note": "Market engine error — NO TRADE"
        }


# =============================================================================
# SCORE ENGINE
# =============================================================================

def score_stock(sym, name, sector, hist, mkt):

    try:

        # Do not score stocks when market data is unavailable.
        if not mkt.get("available", False):
            return None


        closes = hist["Close"]
        vols = hist["Volume"]

        price = float(closes.iloc[-1])

        vol = float(vols.iloc[-1])

        avg10 = float(
            vols.iloc[-10:].mean()
        )

        rvol = round(
            vol / avg10 if avg10 > 0 else 1.0,
            1
        )

        atr = calc_atr(hist)

        rsi = calc_rsi(closes)

        e20 = calc_ema(closes, 20)

        e50 = calc_ema(closes, 50)

        ae20 = price > e20

        ae50 = price > e50

        hhhl = higher_highs_lows(closes)

        typ = (
            float(hist["High"].iloc[-1])
            + float(hist["Low"].iloc[-1])
            + price
        ) / 3

        # Existing VWAP proxy.
        vwap = float(
            (
                (hist["High"] + hist["Low"] + hist["Close"]) / 3
            ).iloc[-5:].mean()
        )

        av = typ > vwap

        h52 = float(closes.max())

        pfh = round(
            ((price - h52) / h52) * 100,
            1
        )


        bull_sigs = sum(
            [
                av,
                ae20,
                ae50,
                hhhl,
                rsi > 55
            ]
        )

        direction = (
            "LONG"
            if bull_sigs >= 3
            else "SHORT"
        )

        is_long = direction == "LONG"


        # -------------------------------------------------------------
        # SCORING
        # -------------------------------------------------------------

        s_mkt = mkt["mkt_pts"]

        s_sec = (
            int(ae50) * 10
            + int(rsi > 60) * 5
        ) if is_long else (
            int(not ae50) * 10
            + int(rsi < 45) * 5
        )

        s_mom = (
            int(ae20) * 5
            + int(ae50) * 5
            + int(hhhl) * 5
        ) if is_long else (
            int(not ae20) * 5
            + int(not ae50) * 5
            + int(not hhhl) * 5
        )

        s_vwap = (
            10 if av else 0
        ) if is_long else (
            10 if not av else 0
        )

        s_vol = (
            10 if rvol > 2
            else 7 if rvol > 1.5
            else 5 if rvol > 1.0
            else 3 if rvol > 0.7
            else 0
        )

        if is_long:

            s_rsi = (
                10 if 60 <= rsi <= 70
                else 5 if 55 <= rsi < 60
                else 3 if 50 <= rsi < 55
                else 2 if rsi > 75
                else 0
            )

        else:

            s_rsi = (
                10 if rsi < 35
                else 8 if rsi < 40
                else 5 if rsi < 45
                else 3 if rsi < 50
                else 0
            )


        # Existing option proxy.
        # Will be replaced with real option-chain data.
        s_opt = (
            int(rvol > 1.5 and av) * 5
            + int(ae20 and ae50) * 5
        ) if is_long else (
            int(rvol > 1.5 and not av) * 5
            + int(not ae20 and not ae50) * 5
        )


        s_risk = 10

        if rvol < 0.5:
            s_risk -= 3

        if rsi > 80:
            s_risk -= 3

        if abs(pfh) < 1:
            s_risk -= 2

        if atr / price > 0.03:
            s_risk -= 2

        s_risk = max(0, s_risk)


        total = (
            s_mkt
            + s_sec
            + s_mom
            + s_vwap
            + s_vol
            + s_rsi
            + s_opt
            + s_risk
        )


        # -------------------------------------------------------------
        # TRADE LEVELS
        # -------------------------------------------------------------

        if is_long:

            sl = round(
                price - atr * 1.5,
                1
            )

            t1 = round(
                price + atr * 2.0,
                1
            )

            t2 = round(
                price + atr * 3.5,
                1
            )

        else:

            sl = round(
                price + atr * 1.5,
                1
            )

            t1 = round(
                price - atr * 2.0,
                1
            )

            t2 = round(
                price - atr * 3.5,
                1
            )


        rr = round(
            abs(t1 - price)
            / abs(sl - price),
            1
        ) if abs(sl - price) > 0 else 1.5


        strike = int(
            round(price / 50) * 50
        )

        atm = (
            f"{sym} {strike} "
            f"{'CE' if is_long else 'PE'}"
        )


        conf = (
            "High"
            if total >= 85
            else "Medium"
            if total >= 70
            else "Low"
        )

        stars = (
            5 if total >= 90
            else 4 if total >= 80
            else 3 if total >= 70
            else 2
        )


        return {

            "sym": sym,
            "name": name,
            "sector": sector,

            "price": round(price, 1),

            "direction": direction,

            "score": total,

            "confidence": conf,

            "stars": stars,

            "rsi": round(rsi, 1),

            "rvol": rvol,

            "atr": round(atr, 1),

            "above_vwap": av,

            "above_ema20": ae20,

            "above_ema50": ae50,

            "hhhl": hhhl,

            "pct_from_high": pfh,

            "components": {

                "market_trend": s_mkt,

                "sector_strength": s_sec,

                "stock_momentum": s_mom,

                "vwap": s_vwap,

                "volume": s_vol,

                "rsi_score": s_rsi,

                "option_chain": s_opt,

                "risk_filters": s_risk
            },

            "entry": round(price, 1),

            "sl": sl,

            "t1": t1,

            "t2": t2,

            "rr": rr,

            "atm": atm
        }


    except Exception:
        return None


# =============================================================================
# CACHE
# =============================================================================

cache = {
    "data": None,
    "time": None,
    "market": None
}

CACHE_MINUTES = 15


def is_cache_valid():

    if not cache["time"]:
        return False

    diff = (
        datetime.now() - cache["time"]
    ).seconds / 60

    return diff < CACHE_MINUTES


# =============================================================================
# API — BASIC
# =============================================================================

@app.route("/")
def home():

    return jsonify({
        "status": "AlphaEdge API running",
        "version": "v4",
        "kite_configured": bool(KITE_API_KEY)
    })


# =============================================================================
# API — ZERODHA AUTHENTICATION
# =============================================================================

@app.route("/api/kite/login")
def kite_login():
    """
    Generate the Zerodha Kite Connect login URL.
    """

    if not kite:

        return jsonify({
            "success": False,
            "error": "KITE_API_KEY is not configured on the server"
        }), 500


    try:

        login_url = kite.login_url()

        return jsonify({
            "success": True,
            "login_url": login_url
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/kite/callback")
def kite_callback():
    """
    Handle Zerodha OAuth callback.

    Zerodha redirects here after successful login.
    """

    global kite_access_token

    request_token = request.args.get(
        "request_token"
    )

    if not request_token:

        return jsonify({
            "success": False,
            "error": "Missing request_token"
        }), 400


    if not kite:

        return jsonify({
            "success": False,
            "error": "KITE_API_KEY is not configured"
        }), 500


    if not KITE_API_SECRET:

        return jsonify({
            "success": False,
            "error": "KITE_API_SECRET is not configured"
        }), 500


    try:

        session = kite.generate_session(
            request_token,
            api_secret=KITE_API_SECRET
        )

        kite_access_token = session["access_token"]

        kite.set_access_token(
            kite_access_token
        )

        return jsonify({

            "success": True,

            "message":
                "Zerodha authentication successful",

            "user_id":
                session.get("user_id"),

            "user_name":
                session.get("user_name"),

            "login_time":
                session.get("login_time")
        })


    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)
        }), 500


@app.route("/api/kite/status")
def kite_status():

    return jsonify({

        "success": True,

        "configured":
            bool(KITE_API_KEY),

        "connected":
            kite_access_token is not None
    })


# =============================================================================
# API — TEST ZERODHA QUOTE
# =============================================================================

@app.route("/api/kite/quote")

@app.route("/api/kite/candles")
def kite_candles():
    """
    Fetch 5-minute historical candles from Zerodha.

    Example:
    /api/kite/candles?instrument_token=256265&days=5
    """

    if not kite_access_token:
        return jsonify({
            "success": False,
            "error": "Zerodha is not authenticated"
        }), 401

    try:
        kite.set_access_token(kite_access_token)

        instrument_token = int(
            request.args.get("instrument_token", 256265)
        )

        days = min(
            int(request.args.get("days", 5)),
            60
        )

        to_date = datetime.now()
        from_date = to_date - pd.Timedelta(days=days)

        candles = kite.historical_data(
            instrument_token,
            from_date,
            to_date,
            "5minute",
            continuous=False,
            oi=True
        )

        return jsonify({
            "success": True,
            "instrument_token": instrument_token,
            "interval": "5minute",
            "count": len(candles),
            "candles": candles
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def kite_quote():
    """
    Test authenticated Zerodha market data.

    Default instrument: NSE:NIFTY 50
    """

    if not kite_access_token:

        return jsonify({

            "success": False,

            "error":
                "Zerodha is not authenticated. "
                "Open /api/kite/login first."
        }), 401


    try:

        kite.set_access_token(
            kite_access_token
        )

        instrument = request.args.get(
            "instrument",
            "NSE:NIFTY 50"
        )

        quote = kite.quote(
            [instrument]
        )

        return jsonify({

            "success": True,

            "instrument": instrument,

            "data": quote.get(instrument)
        })


    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)
        }), 500


# =============================================================================
# API — MARKET
# =============================================================================

@app.route("/api/market")
def market():

    try:

        mkt = get_market()

        return jsonify({

            "success": True,

            "data": mkt
        })


    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# =============================================================================
# API — SCANNER
# =============================================================================

@app.route("/api/scan")
def scan():

    """
    Scan stocks and return ranked signals.
    """

    n = min(
        int(request.args.get("n", 10)),
        len(STOCKS)
    )

    min_score = int(
        request.args.get(
            "min_score",
            0
        )
    )

    direction = request.args.get(
        "direction",
        "all"
    )


    # -------------------------------------------------------------
    # Cache
    # -------------------------------------------------------------

    if (
        is_cache_valid()
        and cache["data"]
        and len(cache["data"]) > 0
    ):

        results = cache["data"]

        mkt = cache["market"]


    else:

        cache["data"] = None

        cache["time"] = None

        mkt = get_market()


        # No market data = no scan.
        if not mkt.get("available", False):

            return jsonify({

                "success": True,

                "date":
                    datetime.now().strftime(
                        "%d %B %Y %H:%M"
                    ),

                "market": mkt,

                "count": 0,

                "signals": [],

                "cached": False,

                "note":
                    "Market data unavailable — NO TRADE"
            })


        batch = STOCKS[:n]

        results = []


        for sym, name, sector in batch:

            hist = fetch(sym)

            if hist is not None:

                r = score_stock(
                    sym,
                    name,
                    sector,
                    hist,
                    mkt
                )

                if r:
                    results.append(r)

            time.sleep(0.2)


        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )


        cache["data"] = results

        cache["market"] = mkt

        cache["time"] = datetime.now()


    # -------------------------------------------------------------
    # Filters
    # -------------------------------------------------------------

    filtered = [

        r for r in results

        if r["score"] >= min_score
    ]


    if direction == "long":

        filtered = [

            r for r in filtered

            if r["direction"] == "LONG"
        ]


    elif direction == "short":

        filtered = [

            r for r in filtered

            if r["direction"] == "SHORT"
        ]


    # -------------------------------------------------------------
    # Market gate
    # -------------------------------------------------------------

    gated = []


    for r in filtered:

        if (
            mkt["block_long"]
            and r["direction"] == "LONG"
        ):
            continue


        if (
            mkt["block_short"]
            and r["direction"] == "SHORT"
        ):
            continue


        gated.append(r)


    return jsonify({

        "success": True,

        "date":
            datetime.now().strftime(
                "%d %B %Y %H:%M"
            ),

        "market": mkt,

        "count": len(gated),

        "signals": gated[:10],

        "cached": is_cache_valid()
    })


# =============================================================================
# API — SINGLE STOCK
# =============================================================================

@app.route("/api/stock/<sym>")
def single_stock(sym):

    match = [

        (s, n, sec)

        for s, n, sec in STOCKS

        if s.upper() == sym.upper()
    ]


    if not match:

        return jsonify({

            "success": False,

            "error":
                f"{sym} not in universe"

        }), 404


    s, n, sec = match[0]

    mkt = get_market()

    hist = fetch(s)


    if hist is None:

        return jsonify({

            "success": False,

            "error": "No data"

        }), 500


    r = score_stock(
        s,
        n,
        sec,
        hist,
        mkt
    )


    return jsonify({

        "success": True,

        "market": mkt,

        "signal": r
    })


# =============================================================================
# HEALTH
# =============================================================================

@app.route("/api/health")
def health():

    return jsonify({

        "status": "ok",

        "time":
            datetime.now().isoformat(),

        "cache_valid":
            is_cache_valid(),

        "kite_configured":
            bool(KITE_API_KEY),

        "kite_connected":
            kite_access_token is not None
    })


# =============================================================================
# LOCAL DEVELOPMENT
# =============================================================================

if __name__ == "__main__":

    app.run(
        debug=False,
        host="0.0.0.0",
        port=5000
    )