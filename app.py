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

@app.route("/api/kite/instrument")
def kite_instrument():
    """
    Find an NSE instrument in Zerodha's instrument master.

    Example:
    /api/kite/instrument?symbol=RELIANCE
    """

    if not kite_access_token:
        return jsonify({
            "success": False,
            "error": "Zerodha is not authenticated"
        }), 401

    try:
        kite.set_access_token(kite_access_token)

        symbol = request.args.get(
            "symbol",
            "RELIANCE"
        ).upper().strip()

        instruments = kite.instruments("NSE")

        matches = [
            x for x in instruments
            if x.get("tradingsymbol", "").upper() == symbol
        ]

        if not matches:
            return jsonify({
                "success": False,
                "error": f"{symbol} not found in NSE instruments"
            }), 404

        instrument = matches[0]

        return jsonify({
            "success": True,
            "instrument": {
                "instrument_token":
                    instrument.get("instrument_token"),

                "exchange_token":
                    instrument.get("exchange_token"),

                "tradingsymbol":
                    instrument.get("tradingsymbol"),

                "name":
                    instrument.get("name"),

                "exchange":
                    instrument.get("exchange"),

                "segment":
                    instrument.get("segment"),

                "instrument_type":
                    instrument.get("instrument_type")
            }
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/kite/candles")

@app.route("/api/kite/stock-candles")
@app.route("/api/kite/stock-analysis")

@app.route("/api/kite/backtest")
def backtest():

    if not kite_access_token:
        return jsonify({
            "success": False,
            "error": "Zerodha is not authenticated"
        }), 401

    try:
        kite.set_access_token(kite_access_token)

        instrument_token = int(
            request.args.get("instrument_token", 738561)
        )

        symbol = request.args.get("symbol", "RELIANCE")

        days = int(request.args.get("days", 20))

        min_score = int(request.args.get("score", 70))

        to_date = datetime.now()
        from_date = to_date - pd.Timedelta(days=days)

        candles = kite.historical_data(
            instrument_token,
            from_date,
            to_date,
            "5minute",
            continuous=False,
            oi=False
        )

        df = pd.DataFrame(candles)

        df["date"] = pd.to_datetime(df["date"])

        # ---------- Indicators ----------
        close = df["close"]

        df["ema20"] = close.ewm(span=20).mean()
        df["ema50"] = close.ewm(span=50).mean()

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        rs = (
            gain.ewm(alpha=1/14).mean() /
            loss.ewm(alpha=1/14).mean()
        )

        df["rsi"] = 100 - (100/(1+rs))

        tr = pd.concat([
            df["high"]-df["low"],
            (df["high"]-close.shift()).abs(),
            (df["low"]-close.shift()).abs()
        ], axis=1).max(axis=1)

        df["atr"] = tr.ewm(alpha=1/14).mean()

        # VWAP
        df["session"] = df["date"].dt.date

        tp = (df["high"]+df["low"]+df["close"])/3

        pv = tp*df["volume"]

        df["vwap"] = (
            pv.groupby(df["session"]).cumsum() /
            df["volume"].groupby(df["session"]).cumsum()
        )

        trades = []

        i = 60

        while i < len(df)-20:

            row = df.iloc[i]

            score = 0

            if row["close"] > row["vwap"]:
                score += 20

            if row["close"] > row["ema20"]:
                score += 15

            if row["close"] > row["ema50"]:
                score += 15

            if row["ema20"] > row["ema50"]:
                score += 15

            if row["rsi"] > 50:
                score += 10

            if 55 <= row["rsi"] <= 70:
                score += 10

            if score < min_score:
                i += 1
                continue

            entry = row["close"]
            atr = row["atr"]

            sl = entry - atr
            target = entry + atr*2

            outcome = "TIME_EXIT"
            exit_price = df.iloc[i+20]["close"]

            for j in range(i+1, min(i+21, len(df))):

                c = df.iloc[j]

                if c["low"] <= sl:
                    outcome = "SL"
                    exit_price = sl
                    break

                if c["high"] >= target:
                    outcome = "TARGET"
                    exit_price = target
                    break

            pnl = round(exit_price-entry, 2)

            trades.append({
                "entry_time": str(row["date"]),
                "entry": round(entry,2),
                "sl": round(sl,2),
                "target": round(target,2),
                "exit": round(exit_price,2),
                "score": score,
                "result": outcome,
                "pnl": pnl
            })

            i += 20

        wins = len([t for t in trades if t["pnl"] > 0])
        losses = len([t for t in trades if t["pnl"] <= 0])

        total = len(trades)

        win_rate = round((wins/total)*100,1) if total else 0

        gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))

        profit_factor = round(
            gross_profit/gross_loss,2
        ) if gross_loss else 99

        return jsonify({
            "success": True,
            "symbol": symbol,
            "days": days,
            "min_score": min_score,
            "summary": {
                "trades": total,
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "gross_profit": round(gross_profit,2),
                "gross_loss": round(gross_loss,2)
            },
            "recent_trades": trades[-10:]
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def kite_stock_analysis():
    """
    Analyse an NSE stock using Zerodha 5-minute candles.

    Example:
    /api/kite/stock-analysis?symbol=RELIANCE&instrument_token=738561
    """

    if not kite_access_token:
        return jsonify({
            "success": False,
            "error": "Zerodha is not authenticated"
        }), 401

    try:
        kite.set_access_token(kite_access_token)

        symbol = request.args.get(
            "symbol",
            "RELIANCE"
        ).upper().strip()

        instrument_token = int(
            request.args.get(
                "instrument_token",
                738561
            )
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
            oi=False
        )

        if not candles or len(candles) < 60:
            return jsonify({
                "success": False,
                "error": "Insufficient 5-minute candle data"
            }), 500

        df = pd.DataFrame(candles)

        # =============================================================
        # CLEAN DATA
        # =============================================================

        df["date"] = pd.to_datetime(df["date"])

        for col in [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna(
            subset=[
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        ).copy()

        # Remove impossible records
        df = df[
            (df["high"] >= df["low"]) &
            (df["volume"] >= 0)
        ].copy()

        close = df["close"]

        # =============================================================
        # EMA 20 / EMA 50
        # =============================================================

        df["ema20"] = close.ewm(
            span=20,
            adjust=False
        ).mean()

        df["ema50"] = close.ewm(
            span=50,
            adjust=False
        ).mean()

        # =============================================================
        # RSI 14
        # =============================================================

        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(
            alpha=1 / 14,
            adjust=False,
            min_periods=14
        ).mean()

        avg_loss = loss.ewm(
            alpha=1 / 14,
            adjust=False,
            min_periods=14
        ).mean()

        rs = avg_gain / avg_loss.replace(
            0,
            np.nan
        )

        df["rsi14"] = (
            100 - (100 / (1 + rs))
        )

        # =============================================================
        # ATR 14
        # =============================================================

        previous_close = close.shift(1)

        true_range = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - previous_close).abs(),
                (df["low"] - previous_close).abs()
            ],
            axis=1
        ).max(axis=1)

        df["atr14"] = true_range.ewm(
            alpha=1 / 14,
            adjust=False,
            min_periods=14
        ).mean()

        # =============================================================
        # SESSION IDENTIFIER
        # =============================================================

        df["session_date"] = (
            df["date"].dt.date
        )

        # =============================================================
        # TRUE SESSION VWAP
        # =============================================================

        df["typical_price"] = (
            df["high"]
            + df["low"]
            + df["close"]
        ) / 3

        df["price_volume"] = (
            df["typical_price"]
            * df["volume"]
        )

        df["cumulative_pv"] = (
            df.groupby("session_date")[
                "price_volume"
            ].cumsum()
        )

        df["cumulative_volume"] = (
            df.groupby("session_date")[
                "volume"
            ].cumsum()
        )

        df["vwap"] = (
            df["cumulative_pv"]
            / df["cumulative_volume"].replace(
                0,
                np.nan
            )
        )

        # =============================================================
        # RELATIVE VOLUME
        # =============================================================
        #
        # First version:
        # Compare current 5-minute candle volume with
        # the previous 20 five-minute candles.
        #
        # We will later replace this with time-of-day
        # adjusted relative volume.
        # =============================================================

        df["volume_avg20"] = (
            df["volume"]
            .rolling(
                window=20,
                min_periods=10
            )
            .mean()
        )

        df["relative_volume"] = (
            df["volume"]
            / df["volume_avg20"].replace(
                0,
                np.nan
            )
        )

        # =============================================================
        # MOMENTUM
        # =============================================================

        df["momentum_3"] = (
            close
            - close.shift(3)
        )

        # =============================================================
        # LATEST CANDLE
        # =============================================================

        last = df.iloc[-1]

        price = float(last["close"])
        ema20 = float(last["ema20"])
        ema50 = float(last["ema50"])
        rsi = float(last["rsi14"])
        atr = float(last["atr14"])
        vwap = float(last["vwap"])
        relative_volume = float(
            last["relative_volume"]
        )

        momentum = float(
            last["momentum_3"]
        )

        # =============================================================
        # CONDITIONS
        # =============================================================

        above_vwap = price > vwap
        above_ema20 = price > ema20
        above_ema50 = price > ema50

        ema_bullish = ema20 > ema50

        rsi_bullish = rsi > 50
        rsi_strong = 55 <= rsi <= 70

        volume_confirmed = (
            relative_volume >= 1.20
        )

        momentum_positive = (
            momentum > 0
        )

        # =============================================================
        # LONG SCORE
        # =============================================================

        long_score = 0

        if above_vwap:
            long_score += 20

        if above_ema20:
            long_score += 15

        if above_ema50:
            long_score += 15

        if ema_bullish:
            long_score += 15

        if rsi_bullish:
            long_score += 10

        if rsi_strong:
            long_score += 10

        if volume_confirmed:
            long_score += 10

        if momentum_positive:
            long_score += 5

        # =============================================================
        # SHORT SCORE
        # =============================================================

        below_vwap = price < vwap
        below_ema20 = price < ema20
        below_ema50 = price < ema50

        ema_bearish = ema20 < ema50

        rsi_bearish = rsi < 50
        rsi_weak = 30 <= rsi <= 45

        momentum_negative = (
            momentum < 0
        )

        short_score = 0

        if below_vwap:
            short_score += 20

        if below_ema20:
            short_score += 15

        if below_ema50:
            short_score += 15

        if ema_bearish:
            short_score += 15

        if rsi_bearish:
            short_score += 10

        if rsi_weak:
            short_score += 10

        if volume_confirmed:
            short_score += 10

        if momentum_negative:
            short_score += 5

        # =============================================================
        # SETUP CLASSIFICATION
        # =============================================================

        if (
            long_score >= 75
            and long_score > short_score
        ):
            bias = "BULLISH"
            setup = "LONG_SETUP"
            score = long_score

        elif (
            short_score >= 75
            and short_score > long_score
        ):
            bias = "BEARISH"
            setup = "SHORT_SETUP"
            score = short_score

        else:
            bias = "NEUTRAL"
            setup = "NO_TRADE"
            score = max(
                long_score,
                short_score
            )

        # =============================================================
        # SESSION STATUS
        # =============================================================

        latest_session = last["session_date"]

        today = datetime.now().date()

        data_status = (
            "CURRENT_SESSION"
            if latest_session == today
            else "HISTORICAL_LAST_SESSION"
        )

        # =============================================================
        # RESPONSE
        # =============================================================

        return jsonify({

            "success": True,

            "instrument": symbol,

            "instrument_token":
                instrument_token,

            "data_status":
                data_status,

            "session_date":
                str(latest_session),

            "latest_candle":
                str(last["date"]),

            "price":
                round(price, 2),

            "indicators": {

                "ema20":
                    round(ema20, 2),

                "ema50":
                    round(ema50, 2),

                "rsi14":
                    round(rsi, 2),

                "atr14":
                    round(atr, 2),

                "vwap":
                    round(vwap, 2),

                "relative_volume":
                    round(
                        relative_volume,
                        2
                    ),

                "momentum_3_bars":
                    round(momentum, 2)
            },

            "conditions": {

                "above_vwap":
                    above_vwap,

                "above_ema20":
                    above_ema20,

                "above_ema50":
                    above_ema50,

                "ema20_above_ema50":
                    ema_bullish,

                "rsi_above_50":
                    rsi_bullish,

                "rsi_in_strong_zone":
                    rsi_strong,

                "volume_confirmed":
                    volume_confirmed,

                "momentum_positive":
                    momentum_positive
            },

            "scores": {

                "long":
                    long_score,

                "short":
                    short_score,

                "final":
                    score
            },

            "signal": {

                "bias":
                    bias,

                "setup":
                    setup,

                "trade_gate":
                    "OPEN"
                    if setup != "NO_TRADE"
                    else "NO_TRADE"
            },

            "note":
                "Relative volume currently uses a "
                "20-candle rolling baseline. "
                "Time-of-day-adjusted relative volume "
                "will be added after baseline validation."
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
def kite_stock_candles():
    """
    Fetch 5-minute candles for an NSE stock.

    Example:
    /api/kite/stock-candles?instrument_token=738561&days=5
    """

    if not kite_access_token:
        return jsonify({
            "success": False,
            "error": "Zerodha is not authenticated"
        }), 401

    try:
        kite.set_access_token(kite_access_token)

        instrument_token = int(
            request.args.get("instrument_token", 738561)
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
            oi=False
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

@app.route("/api/kite/nifty-analysis")
@app.route("/api/kite/nifty-validation")
def nifty_validation():
    """
    Independent validation of NIFTY 50 5-minute indicators.

    This endpoint is for calculation validation only.
    It does NOT generate a trading signal.
    """

    if not kite_access_token:
        return jsonify({
            "success": False,
            "error": "Zerodha is not authenticated"
        }), 401

    try:
        kite.set_access_token(kite_access_token)

        instrument_token = 256265

        to_date = datetime.now()
        from_date = to_date - pd.Timedelta(days=5)

        candles = kite.historical_data(
            instrument_token,
            from_date,
            to_date,
            "5minute",
            continuous=False,
            oi=False
        )

        if not candles or len(candles) < 60:
            return jsonify({
                "success": False,
                "error": "Insufficient candle data"
            }), 500

        df = pd.DataFrame(candles)

        df["date"] = pd.to_datetime(df["date"])

        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna(
            subset=["open", "high", "low", "close"]
        ).copy()

        close = df["close"]

        # =============================================================
        # EMA 20 / EMA 50
        # =============================================================

        df["ema20"] = close.ewm(
            span=20,
            adjust=False
        ).mean()

        df["ema50"] = close.ewm(
            span=50,
            adjust=False
        ).mean()

        # =============================================================
        # RSI 14 — Wilder smoothing
        # =============================================================

        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(
            alpha=1 / 14,
            adjust=False,
            min_periods=14
        ).mean()

        avg_loss = loss.ewm(
            alpha=1 / 14,
            adjust=False,
            min_periods=14
        ).mean()

        rs = avg_gain / avg_loss.replace(
            0,
            np.nan
        )

        df["rsi14"] = (
            100 - (100 / (1 + rs))
        )

        # =============================================================
        # ATR 14 — Wilder smoothing
        # =============================================================

        previous_close = close.shift(1)

        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - previous_close).abs(),
                (df["low"] - previous_close).abs()
            ],
            axis=1
        ).max(axis=1)

        df["atr14"] = tr.ewm(
            alpha=1 / 14,
            adjust=False,
            min_periods=14
        ).mean()

        # =============================================================
        # Session reference
        # =============================================================

        df["session_date"] = df["date"].dt.date

        df["typical_price"] = (
            df["high"]
            + df["low"]
            + df["close"]
        ) / 3

        df["session_reference"] = (
            df.groupby("session_date")[
                "typical_price"
            ]
            .transform("mean")
        )

        # =============================================================
        # Latest candle
        # =============================================================

        last = df.iloc[-1]

        # =============================================================
        # Recent candles for sanity checking
        # =============================================================

        recent = df.tail(10)

        recent_candles = []

        for _, row in recent.iterrows():

            recent_candles.append({
                "time": str(row["date"]),
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2)
            })

        # =============================================================
        # Validation flags
        # =============================================================

        checks = {

            "ema20_available":
                pd.notna(last["ema20"]),

            "ema50_available":
                pd.notna(last["ema50"]),

            "rsi14_available":
                pd.notna(last["rsi14"]),

            "atr14_available":
                pd.notna(last["atr14"]),

            "ohlc_valid":
                (
                    float(last["high"])
                    >= float(last["low"])
                ),

            "close_inside_range":
                (
                    float(last["low"])
                    <= float(last["close"])
                    <= float(last["high"])
                ),

            "index_volume_available":
                bool(
                    "volume" in df.columns
                    and df["volume"].fillna(0).sum() > 0
                )
        }

        return jsonify({

            "success": True,

            "validation": True,

            "instrument": "NIFTY 50",

            "data_status":
                "HISTORICAL_LAST_SESSION",

            "session_date":
                str(last["session_date"]),

            "latest_candle":
                str(last["date"]),

            "candle_count":
                len(df),

            "latest": {

                "price":
                    round(
                        float(last["close"]),
                        2
                    ),

                "ema20":
                    round(
                        float(last["ema20"]),
                        2
                    ),

                "ema50":
                    round(
                        float(last["ema50"]),
                        2
                    ),

                "rsi14":
                    round(
                        float(last["rsi14"]),
                        2
                    ),

                "atr14":
                    round(
                        float(last["atr14"]),
                        2
                    ),

                "session_reference":
                    round(
                        float(
                            last[
                                "session_reference"
                            ]
                        ),
                        2
                    )
            },

            "checks": checks,

            "vwap_status": {

                "true_vwap":
                    False,

                "reason":
                    "NIFTY 50 index candles returned by "
                    "Zerodha have zero volume. True "
                    "volume-weighted VWAP cannot be "
                    "calculated from this dataset."
            },

            "relative_volume_status": {

                "available":
                    checks["index_volume_available"],

                "reason":
                    "NIFTY 50 index candle volume is "
                    "not available for relative-volume "
                    "calculation."
            },

            "recent_candles":
                recent_candles
        })

    except Exception as e:

        return jsonify({

            "success": False,

            "validation": False,

            "error": str(e)

        }), 500

def nifty_analysis():
    """
    Analyse the latest available NIFTY 50 5-minute candles.

    On a non-trading day, this will analyse the latest completed
    trading session and explicitly mark the result as HISTORICAL.
    """

    if not kite_access_token:
        return jsonify({
            "success": False,
            "error": "Zerodha is not authenticated"
        }), 401

    try:
        kite.set_access_token(kite_access_token)

        instrument_token = 256265  # NIFTY 50

        days = 5

        to_date = datetime.now()
        from_date = to_date - pd.Timedelta(days=days)

        candles = kite.historical_data(
            instrument_token,
            from_date,
            to_date,
            "5minute",
            continuous=False,
            oi=False
        )

        if not candles or len(candles) < 50:
            return jsonify({
                "success": False,
                "error": "Insufficient 5-minute candle data"
            }), 500

        df = pd.DataFrame(candles)

        # -------------------------------------------------------------
        # Basic cleanup
        # -------------------------------------------------------------

        df["date"] = pd.to_datetime(df["date"])

        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna(
            subset=["open", "high", "low", "close"]
        ).copy()

        closes = df["close"]

        # -------------------------------------------------------------
        # EMA 20 / EMA 50
        # -------------------------------------------------------------

        df["ema20"] = closes.ewm(
            span=20,
            adjust=False
        ).mean()

        df["ema50"] = closes.ewm(
            span=50,
            adjust=False
        ).mean()

        # -------------------------------------------------------------
        # RSI 14 — Wilder-style calculation
        # -------------------------------------------------------------

        delta = closes.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(
            alpha=1 / 14,
            adjust=False,
            min_periods=14
        ).mean()

        avg_loss = loss.ewm(
            alpha=1 / 14,
            adjust=False,
            min_periods=14
        ).mean()

        rs = avg_gain / avg_loss.replace(
            0,
            np.nan
        )

        df["rsi14"] = (
            100 - (100 / (1 + rs))
        )

        # -------------------------------------------------------------
        # ATR 14
        # -------------------------------------------------------------

        previous_close = closes.shift(1)

        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - previous_close).abs(),
                (df["low"] - previous_close).abs()
            ],
            axis=1
        ).max(axis=1)

        df["atr14"] = tr.ewm(
            alpha=1 / 14,
            adjust=False,
            min_periods=14
        ).mean()

        # -------------------------------------------------------------
        # TRUE SESSION VWAP
        # -------------------------------------------------------------
        #
        # Important:
        # VWAP resets at the beginning of each trading session.
        #

        df["session_date"] = df["date"].dt.date

        df["typical_price"] = (
            df["high"]
            + df["low"]
            + df["close"]
        ) / 3

        # Zerodha index candles have volume = 0.
        # Therefore we cannot calculate a genuine volume-weighted
        # VWAP for NIFTY 50 from these index candles.
        #
        # We calculate price-only session VWAP here as a temporary
        # reference and explicitly label it accordingly.

        df["tp_cumulative"] = (
            df.groupby("session_date")["typical_price"]
            .cumsum()
        )

        df["bars_in_session"] = (
            df.groupby("session_date")
            .cumcount() + 1
        )

        df["session_price_average"] = (
            df["tp_cumulative"]
            / df["bars_in_session"]
        )

        # -------------------------------------------------------------
        # Latest candle
        # -------------------------------------------------------------

        last = df.iloc[-1]

        price = float(last["close"])
        ema20 = float(last["ema20"])
        ema50 = float(last["ema50"])
        rsi = float(last["rsi14"])
        atr = float(last["atr14"])

        session_reference = float(
            last["session_price_average"]
        )

        # -------------------------------------------------------------
        # Conditions
        # -------------------------------------------------------------

        above_ema20 = price > ema20
        above_ema50 = price > ema50
        above_session_reference = (
            price > session_reference
        )

        rsi_bullish = rsi > 50
        rsi_strong = 55 <= rsi <= 70

        ema_bullish = ema20 > ema50

        # -------------------------------------------------------------
        # Market score
        # -------------------------------------------------------------

        score = 0

        if above_session_reference:
            score += 20

        if above_ema20:
            score += 20

        if above_ema50:
            score += 20

        if ema_bullish:
            score += 15

        if rsi_bullish:
            score += 15

        if rsi_strong:
            score += 10

        # -------------------------------------------------------------
        # Bias
        # -------------------------------------------------------------

        if score >= 75:
            bias = "BULLISH"

        elif score >= 55:
            bias = "MILDLY BULLISH"

        elif score <= 25:
            bias = "BEARISH"

        elif score <= 45:
            bias = "MILDLY BEARISH"

        else:
            bias = "NEUTRAL"

        # -------------------------------------------------------------
        # Trade gate
        # -------------------------------------------------------------

        if score >= 75:
            trade_gate = "OPEN_LONG"

        elif score <= 25:
            trade_gate = "OPEN_SHORT"

        else:
            trade_gate = "NO_TRADE"

        # -------------------------------------------------------------
        # Determine latest completed session
        # -------------------------------------------------------------

        latest_session = last["session_date"]

        today = datetime.now().date()

        is_today = latest_session == today

        market_data_status = (
            "CURRENT_SESSION"
            if is_today
            else "HISTORICAL_LAST_SESSION"
        )

        return jsonify({

            "success": True,

            "instrument": "NIFTY 50",

            "instrument_token": instrument_token,

            "data_status": market_data_status,

            "session_date": str(
                latest_session
            ),

            "latest_candle": str(
                last["date"]
            ),

            "price": round(price, 2),

            "indicators": {

                "rsi14": round(rsi, 2),

                "ema20": round(ema20, 2),

                "ema50": round(ema50, 2),

                "atr14": round(atr, 2),

                "session_price_reference":
                    round(
                        session_reference,
                        2
                    )
            },

            "conditions": {

                "above_ema20":
                    above_ema20,

                "above_ema50":
                    above_ema50,

                "above_session_reference":
                    above_session_reference,

                "ema20_above_ema50":
                    ema_bullish,

                "rsi_above_50":
                    rsi_bullish,

                "rsi_in_strong_zone":
                    rsi_strong
            },

            "market": {

                "score": score,

                "bias": bias,

                "trade_gate": trade_gate
            },

            "note":
                "Session price reference is a temporary "
                "price-average proxy because NIFTY 50 index "
                "candles have zero volume. True VWAP will be "
                "implemented using appropriate volume data."
        })

    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500

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