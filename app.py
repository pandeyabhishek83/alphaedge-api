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

        # =============================================================
        # PARAMETERS
        # =============================================================

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

        days = max(
            5,
            min(
                int(request.args.get("days", 60)),
                365
            )
        )

        min_score = max(
            0,
            min(
                int(request.args.get("score", 70)),
                100
            )
        )

        direction_filter = request.args.get(
            "direction",
            "BOTH"
        ).upper().strip()

        if direction_filter not in [
            "LONG",
            "SHORT",
            "BOTH"
        ]:
            direction_filter = "BOTH"

        atr_stop = float(
            request.args.get("atr_stop", 1.0)
        )

        risk_reward = float(
            request.args.get("rr", 2.0)
        )

        # Exit experiment modes:
        # BASELINE = existing fixed SL + fixed target
        # TRAIL_1R = trailing protection after +1R
        # TRAIL_1_5R = trailing protection after +1.5R
        # TRAIL_2R = trailing protection after +2R
        exit_mode = request.args.get(
            "exit_mode", "BASELINE"
        ).upper().strip()

        valid_exit_modes = {
            "BASELINE",
            "TRAIL_1R",
            "TRAIL_1_5R",
            "TRAIL_2R"
        }

        if exit_mode not in valid_exit_modes:
            exit_mode = "BASELINE"

        max_hold = max(
            1,
            min(
                int(request.args.get("hold", 20)),
                100
            )
        )

        # Approximate costs for research purposes.
        # These are deliberately configurable.
        slippage_pct = float(
            request.args.get(
                "slippage_pct",
                0.0005
            )
        )

        brokerage_per_trade = float(
            request.args.get(
                "brokerage",
                0.0
            )
        )

        # =============================================================
        # DOWNLOAD HISTORICAL DATA
        # =============================================================

        to_date = datetime.now()

        from_date = (
            to_date
            - pd.Timedelta(days=days)
        )

        candles = kite.historical_data(
            instrument_token,
            from_date,
            to_date,
            "5minute",
            continuous=False,
            oi=False
        )

        if not candles or len(candles) < 100:
            return jsonify({
                "success": False,
                "error": (
                    "Insufficient historical data "
                    "for backtesting"
                )
            }), 400

        df = pd.DataFrame(candles)

        # =============================================================
        # CLEAN DATA
        # =============================================================

        df["date"] = pd.to_datetime(
            df["date"]
        )

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        for col in numeric_columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna(
            subset=numeric_columns
        ).copy()

        df = df.sort_values(
            "date"
        ).reset_index(drop=True)

        # =============================================================
        # EMA 20 / EMA 50
        # =============================================================

        df["ema20"] = df["close"].ewm(
            span=20,
            adjust=False
        ).mean()

        df["ema50"] = df["close"].ewm(
            span=50,
            adjust=False
        ).mean()

        # =============================================================
        # RSI 14 — Wilder style
        # =============================================================

        delta = df["close"].diff()

        gain = delta.clip(
            lower=0
        )

        loss = -delta.clip(
            upper=0
        )

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

        rs = (
            avg_gain
            / avg_loss.replace(
                0,
                np.nan
            )
        )

        df["rsi"] = (
            100
            - (
                100
                / (1 + rs)
            )
        )

        # =============================================================
        # ATR 14 — Wilder style
        # =============================================================

        previous_close = (
            df["close"].shift(1)
        )

        true_range = pd.concat(
            [
                df["high"] - df["low"],

                (
                    df["high"]
                    - previous_close
                ).abs(),

                (
                    df["low"]
                    - previous_close
                ).abs()
            ],
            axis=1
        ).max(axis=1)

        df["atr"] = true_range.ewm(
            alpha=1 / 14,
            adjust=False,
            min_periods=14
        ).mean()

        # =============================================================
        # SESSION VWAP
        # =============================================================

        df["session"] = (
            df["date"].dt.date
        )

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
            df.groupby("session")[
                "price_volume"
            ].cumsum()
        )

        df["cumulative_volume"] = (
            df.groupby("session")[
                "volume"
            ].cumsum()
        )

        df["vwap"] = (
            df["cumulative_pv"]
            /
            df["cumulative_volume"].replace(
                0,
                np.nan
            )
        )

        # =============================================================
        # RELATIVE VOLUME
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
            /
            df["volume_avg20"].replace(
                0,
                np.nan
            )
        )

        # =============================================================
        # MOMENTUM
        # =============================================================

        df["momentum_3"] = (
            df["close"]
            - df["close"].shift(3)
        )

        # =============================================================
        # REMOVE INITIAL INDICATOR WARM-UP
        # =============================================================

        df = df.dropna(
            subset=[
                "ema20",
                "ema50",
                "rsi",
                "atr",
                "vwap",
                "relative_volume",
                "momentum_3"
            ]
        ).reset_index(
            drop=True
        )

        # =============================================================
        # TRADE STORAGE
        # =============================================================

        trades = []

        i = 1

        # =============================================================
        # MAIN BACKTEST LOOP
        # =============================================================

        while i < len(df) - max_hold:

            row = df.iloc[i]

            price = float(
                row["close"]
            )

            ema20 = float(
                row["ema20"]
            )

            ema50 = float(
                row["ema50"]
            )

            rsi = float(
                row["rsi"]
            )

            atr = float(
                row["atr"]
            )

            vwap = float(
                row["vwap"]
            )

            relative_volume = float(
                row["relative_volume"]
            )

            momentum = float(
                row["momentum_3"]
            )

            if atr <= 0:
                i += 1
                continue

            # =========================================================
            # LONG SCORE
            # =========================================================

            long_score = 0

            if price > vwap:
                long_score += 20

            if price > ema20:
                long_score += 15

            if price > ema50:
                long_score += 15

            if ema20 > ema50:
                long_score += 15

            if rsi > 50:
                long_score += 10

            if 55 <= rsi <= 70:
                long_score += 10

            if relative_volume >= 1.20:
                long_score += 10

            if momentum > 0:
                long_score += 5

            # =========================================================
            # SHORT SCORE
            # =========================================================

            short_score = 0

            if price < vwap:
                short_score += 20

            if price < ema20:
                short_score += 15

            if price < ema50:
                short_score += 15

            if ema20 < ema50:
                short_score += 15

            if rsi < 50:
                short_score += 10

            if 30 <= rsi <= 45:
                short_score += 10

            if relative_volume >= 1.20:
                short_score += 10

            if momentum < 0:
                short_score += 5

            # =========================================================
            # DETERMINE SIGNAL
            # =========================================================

            # LONG ONLY
            if direction_filter == "LONG":

                if long_score >= min_score:

                    direction = "LONG"
                    score = long_score

                else:

                    i += 1
                    continue


            # SHORT ONLY
            elif direction_filter == "SHORT":

                if short_score >= min_score:

                    direction = "SHORT"
                    score = short_score

                else:

                    i += 1
                    continue


            # BOTH DIRECTIONS
            else:

                if (
                    long_score >= min_score
                    and long_score > short_score
                ):

                    direction = "LONG"
                    score = long_score

                elif (
                    short_score >= min_score
                    and short_score > long_score
                ):

                    direction = "SHORT"
                    score = short_score

                else:

                    i += 1
                    continue
            # =========================================================
            # ENTRY WITH SLIPPAGE
            # =========================================================

            raw_entry = price

            if direction == "LONG":

                entry = (
                    raw_entry
                    * (1 + slippage_pct)
                )

            else:

                entry = (
                    raw_entry
                    * (1 - slippage_pct)
                )

            risk = atr * atr_stop

            if risk <= 0:
                i += 1
                continue

            # =========================================================
            # STOP + TARGET
            # =========================================================

            if direction == "LONG":

                sl = entry - risk

                target = (
                    entry
                    + risk * risk_reward
                )

            else:

                sl = entry + risk

                target = (
                    entry
                    - risk * risk_reward
                )

            # =========================================================
            # SEARCH EXIT
            # =========================================================
            #
            # Entry rules remain unchanged.
            # BASELINE uses the original fixed SL/target.
            # TRAIL modes activate a 1-ATR trailing stop after
            # the selected favorable R threshold.
            # =========================================================

            exit_price = None
            exit_time = None
            result = "TIME_EXIT"

            exit_index = min(
                i + max_hold,
                len(df) - 1
            )

            trail_trigger_r = {
                "TRAIL_1R": 1.0,
                "TRAIL_1_5R": 1.5,
                "TRAIL_2R": 2.0
            }.get(exit_mode)

            best_price = entry
            trailing_active = False
            trailing_stop = sl

            for j in range(
                i + 1,
                exit_index + 1
            ):
                candle = df.iloc[j]

                candle_high = float(
                    candle["high"]
                )
                candle_low = float(
                    candle["low"]
                )

                if direction == "LONG":
                    best_price = max(
                        best_price,
                        candle_high
                    )
                else:
                    best_price = min(
                        best_price,
                        candle_low
                    )

                # Activate/update trailing protection.
                if trail_trigger_r is not None:

                    if direction == "LONG":

                        favorable_r = (
                            best_price - entry
                        ) / risk

                        if (
                            not trailing_active
                            and favorable_r >= trail_trigger_r
                        ):
                            trailing_active = True
                            trailing_stop = (
                                best_price - risk
                            )

                        elif trailing_active:
                            trailing_stop = max(
                                trailing_stop,
                                best_price - risk
                            )

                    else:

                        favorable_r = (
                            entry - best_price
                        ) / risk

                        if (
                            not trailing_active
                            and favorable_r >= trail_trigger_r
                        ):
                            trailing_active = True
                            trailing_stop = (
                                best_price + risk
                            )

                        elif trailing_active:
                            trailing_stop = min(
                                trailing_stop,
                                best_price + risk
                            )

                active_sl = (
                    trailing_stop
                    if trailing_active
                    else sl
                )

                if direction == "LONG":

                    hit_sl = (
                        candle_low <= active_sl
                    )

                    hit_target = (
                        exit_mode == "BASELINE"
                        and candle_high >= target
                    )

                    if hit_sl and hit_target:
                        result = "SL_AMBIGUOUS"
                        exit_price = active_sl
                        exit_time = candle["date"]
                        break

                    if hit_sl:
                        result = (
                            "TRAIL_SL"
                            if trailing_active
                            else "SL"
                        )
                        exit_price = active_sl
                        exit_time = candle["date"]
                        break

                    if hit_target:
                        result = "TARGET"
                        exit_price = target
                        exit_time = candle["date"]
                        break

                else:

                    hit_sl = (
                        candle_high >= active_sl
                    )

                    hit_target = (
                        exit_mode == "BASELINE"
                        and candle_low <= target
                    )

                    if hit_sl and hit_target:
                        result = "SL_AMBIGUOUS"
                        exit_price = active_sl
                        exit_time = candle["date"]
                        break

                    if hit_sl:
                        result = (
                            "TRAIL_SL"
                            if trailing_active
                            else "SL"
                        )
                        exit_price = active_sl
                        exit_time = candle["date"]
                        break

                    if hit_target:
                        result = "TARGET"
                        exit_price = target
                        exit_time = candle["date"]
                        break

            # =========================================================
            # EXIT SLIPPAGE
            # =========================================================

            if direction == "LONG":

                exit_price_after_slippage = (
                    exit_price
                    * (1 - slippage_pct)
                )

            else:

                exit_price_after_slippage = (
                    exit_price
                    * (1 + slippage_pct)
                )

            # =========================================================
            # GROSS P&L
            # =========================================================

            if direction == "LONG":

                gross_pnl = (
                    exit_price_after_slippage
                    - entry
                )

            else:

                gross_pnl = (
                    entry
                    - exit_price_after_slippage
                )

            # =========================================================
            # COSTS
            # =========================================================

            total_cost = (
                brokerage_per_trade
            )

            net_pnl = (
                gross_pnl
                - total_cost
            )

            # =========================================================
            # R MULTIPLE
            # =========================================================

            r_multiple = (
                net_pnl / risk
            )

            # =========================================================
            # MAE / MFE DIAGNOSTICS
            # =========================================================

            mae_price = 0.0
            mfe_price = 0.0
            mae_bar = 0
            mfe_bar = 0

            for k in range(i + 1, exit_index + 1):
                excursion_candle = df.iloc[k]
                candle_high = float(excursion_candle["high"])
                candle_low = float(excursion_candle["low"])
                bars_from_entry = k - i

                if direction == "LONG":
                    favorable = candle_high - entry
                    adverse = entry - candle_low
                else:
                    favorable = entry - candle_low
                    adverse = candle_high - entry

                if favorable > mfe_price:
                    mfe_price = favorable
                    mfe_bar = bars_from_entry

                if adverse > mae_price:
                    mae_price = adverse
                    mae_bar = bars_from_entry

            mae_r = (-mae_price / risk) if risk > 0 else 0
            mfe_r = (mfe_price / risk) if risk > 0 else 0

            # =========================================================
            # TRADE RECORD
            # =========================================================

            trades.append({

                "direction":
                    direction,

                "entry_time":
                    str(row["date"]),

                "exit_time":
                    str(exit_time),

                "entry":
                    round(entry, 2),

                "sl":
                    round(sl, 2),

                "target":
                    round(target, 2),

                "exit":
                    round(
                        exit_price_after_slippage,
                        2
                    ),

                "score":
                    int(score),

                "long_score":
                    int(long_score),

                "short_score":
                    int(short_score),

                "rsi":
                    round(rsi, 2),

                "atr":
                    round(atr, 2),

                "vwap":
                    round(vwap, 2),

                "relative_volume":
                    round(
                        relative_volume,
                        2
                    ),

                "result":
                    result,

                "gross_pnl":
                    round(
                        gross_pnl,
                        2
                    ),

                "cost":
                    round(
                        total_cost,
                        2
                    ),

                "net_pnl":
                    round(
                        net_pnl,
                        2
                    ),

                "r_multiple":
                    round(
                        r_multiple,
                        3
                    ),

                "mae_price": round(mae_price, 2),
                "mfe_price": round(mfe_price, 2),
                "mae_r": round(mae_r, 3),
                "mfe_r": round(mfe_r, 3),
                "mae_bars": int(mae_bar),
                "mfe_bars": int(mfe_bar)
            })

            # =========================================================
            # MOVE PAST TRADE
            # =========================================================

            i = exit_index + 1

        # =============================================================
        # STATISTICS
        # =============================================================

        total_trades = len(trades)

        long_trades = [
            t for t in trades
            if t["direction"] == "LONG"
        ]

        short_trades = [
            t for t in trades
            if t["direction"] == "SHORT"
        ]

        winning_trades = [
            t for t in trades
            if t["net_pnl"] > 0
        ]

        losing_trades = [
            t for t in trades
            if t["net_pnl"] <= 0
        ]

        wins = len(
            winning_trades
        )

        losses = len(
            losing_trades
        )

        win_rate = (
            (wins / total_trades) * 100
            if total_trades
            else 0
        )

        gross_profit = sum(
            t["net_pnl"]
            for t in winning_trades
        )

        gross_loss = abs(
            sum(
                t["net_pnl"]
                for t in losing_trades
            )
        )

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else 999
        )

        average_win = (
            gross_profit / wins
            if wins
            else 0
        )

        average_loss = (
            gross_loss / losses
            if losses
            else 0
        )

        expectancy = (
            sum(
                t["net_pnl"]
                for t in trades
            ) / total_trades
            if total_trades
            else 0
        )

        # =============================================================
        # EQUITY CURVE + MAX DRAWDOWN
        # =============================================================

        equity = 0.0
        peak = 0.0
        max_drawdown = 0.0

        equity_curve = []

        for t in trades:

            equity += t["net_pnl"]

            peak = max(
                peak,
                equity
            )

            drawdown = (
                peak - equity
            )

            max_drawdown = max(
                max_drawdown,
                drawdown
            )

            equity_curve.append({
                "time":
                    t["exit_time"],

                "equity":
                    round(
                        equity,
                        2
                    ),

                "drawdown":
                    round(
                        drawdown,
                        2
                    )
            })

        # =============================================================
        # CONSECUTIVE WINS / LOSSES
        # =============================================================

        max_consecutive_wins = 0
        max_consecutive_losses = 0

        current_wins = 0
        current_losses = 0

        for t in trades:

            if t["net_pnl"] > 0:

                current_wins += 1
                current_losses = 0

                max_consecutive_wins = max(
                    max_consecutive_wins,
                    current_wins
                )

            else:

                current_losses += 1
                current_wins = 0

                max_consecutive_losses = max(
                    max_consecutive_losses,
                    current_losses
                )

        # =============================================================
        # R-MULTIPLE STATISTICS
        # =============================================================

        average_r = (
            sum(
                t["r_multiple"]
                for t in trades
            ) / total_trades
            if total_trades
            else 0
        )

        # =============================================================
        # SCORE / DIRECTION DIAGNOSTICS
        # =============================================================

        # Bucket executed trades into 5-point score bands. This is a
        # diagnostic view only; it does not alter the trading rules.
        score_buckets = {}

        def ensure_score_bucket(bucket):
            if bucket not in score_buckets:
                score_buckets[bucket] = {
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "gross_profit": 0.0,
                    "gross_loss": 0.0,
                    "net_pnl": 0.0,
                    "r_sum": 0.0
                }

        for t in trades:

            score = int(t["score"])
            bucket = (score // 5) * 5
            ensure_score_bucket(bucket)

            item = score_buckets[bucket]
            pnl = float(t["net_pnl"])

            item["trades"] += 1
            item["net_pnl"] += pnl
            item["r_sum"] += float(t["r_multiple"])

            if pnl > 0:
                item["wins"] += 1
                item["gross_profit"] += pnl
            else:
                item["losses"] += 1
                item["gross_loss"] += abs(pnl)

        for bucket in sorted(score_buckets):

            item = score_buckets[bucket]

            item["win_rate"] = round(
                (item["wins"] / item["trades"]) * 100
                if item["trades"] else 0,
                1
            )

            item["profit_factor"] = round(
                item["gross_profit"] / item["gross_loss"]
                if item["gross_loss"] > 0 else 999,
                2
            )

            item["average_pnl"] = round(
                item["net_pnl"] / item["trades"]
                if item["trades"] else 0,
                2
            )

            item["average_r"] = round(
                item["r_sum"] / item["trades"]
                if item["trades"] else 0,
                3
            )

            item["gross_profit"] = round(
                item["gross_profit"], 2
            )
            item["gross_loss"] = round(
                item["gross_loss"], 2
            )
            item["net_pnl"] = round(
                item["net_pnl"], 2
            )
            item["r_sum"] = round(
                item["r_sum"], 3
            )

            # Keep the public response compact.
            del item["r_sum"]

        # Direction-level diagnostics are useful when BOTH is selected.
        direction_diagnostics = {}

        for direction_name in ["LONG", "SHORT"]:

            direction_trades = [
                t for t in trades
                if t["direction"] == direction_name
            ]

            d_wins = [
                t for t in direction_trades
                if t["net_pnl"] > 0
            ]

            d_losses = [
                t for t in direction_trades
                if t["net_pnl"] <= 0
            ]

            d_gp = sum(
                t["net_pnl"] for t in d_wins
            )

            d_gl = abs(sum(
                t["net_pnl"] for t in d_losses
            ))

            d_n = len(direction_trades)

            direction_diagnostics[direction_name] = {
                "trades": d_n,
                "wins": len(d_wins),
                "losses": len(d_losses),
                "win_rate": round(
                    len(d_wins) / d_n * 100
                    if d_n else 0,
                    1
                ),
                "profit_factor": round(
                    d_gp / d_gl
                    if d_gl > 0 else 999,
                    2
                ),
                "net_pnl": round(
                    sum(t["net_pnl"] for t in direction_trades),
                    2
                ),
                "average_r": round(
                    sum(t["r_multiple"] for t in direction_trades) / d_n
                    if d_n else 0,
                    3
                )
            }

        # Outcome diagnostics help identify whether the issue is the
        # entry signal, stop/target construction, or time exit.
        outcome_diagnostics = {}

        for outcome in [
            "TARGET",
            "SL",
            "TRAIL_SL",
            "SL_AMBIGUOUS",
            "TIME_EXIT"
        ]:

            outcome_trades = [
                t for t in trades
                if t["result"] == outcome
            ]

            outcome_diagnostics[outcome] = {
                "trades": len(outcome_trades),
                "net_pnl": round(
                    sum(t["net_pnl"] for t in outcome_trades),
                    2
                )
            }

        # Highest-performing score band among bands that have at least
        # five observations. This is descriptive, not an optimization.
        eligible_bands = [
            (bucket, data)
            for bucket, data in score_buckets.items()
            if data["trades"] >= 5
        ]

        best_score_band = None

        if eligible_bands:
            best_score_band, best_data = max(
                eligible_bands,
                key=lambda x: x[1]["average_pnl"]
            )

            best_score_band = {
                "score_band": f"{best_score_band}-{best_score_band + 4}",
                "trades": best_data["trades"],
                "win_rate": best_data["win_rate"],
                "profit_factor": best_data["profit_factor"],
                "average_pnl": best_data["average_pnl"],
                "average_r": best_data["average_r"]
            }

        # =============================================================
        # MAE / MFE DIAGNOSTICS
        # =============================================================

        def excursion_stats(trade_list):
            if not trade_list:
                return {
                    "trades": 0,
                    "average_mae_price": 0,
                    "average_mfe_price": 0,
                    "average_mae_r": 0,
                    "average_mfe_r": 0,
                    "average_mae_bars": 0,
                    "average_mfe_bars": 0
                }

            return {
                "trades": len(trade_list),
                "average_mae_price": round(sum(t["mae_price"] for t in trade_list) / len(trade_list), 2),
                "average_mfe_price": round(sum(t["mfe_price"] for t in trade_list) / len(trade_list), 2),
                "average_mae_r": round(sum(t["mae_r"] for t in trade_list) / len(trade_list), 3),
                "average_mfe_r": round(sum(t["mfe_r"] for t in trade_list) / len(trade_list), 3),
                "average_mae_bars": round(sum(t["mae_bars"] for t in trade_list) / len(trade_list), 1),
                "average_mfe_bars": round(sum(t["mfe_bars"] for t in trade_list) / len(trade_list), 1)
            }

        mae_mfe_diagnostics = {
            "purpose": (
                "Descriptive diagnostics only. MAE is maximum adverse excursion "
                "and MFE is maximum favorable excursion after entry through the "
                "simulated exit candle. These statistics do not change trading rules."
            ),
            "overall": excursion_stats(trades),
            "TARGET": excursion_stats([t for t in trades if t["result"] == "TARGET"]),
            "SL": excursion_stats([t for t in trades if t["result"] == "SL"]),
            "SL_AMBIGUOUS": excursion_stats([t for t in trades if t["result"] == "SL_AMBIGUOUS"]),
            "TIME_EXIT": excursion_stats([t for t in trades if t["result"] == "TIME_EXIT"])
        }

        diagnostic_report = {
            "purpose": (
                "Descriptive diagnostics only. These statistics do not "
                "change the trading rules or optimize parameters."
            ),
            "score_bands": score_buckets,
            "direction": direction_diagnostics,
            "outcomes": outcome_diagnostics,
            "best_score_band_min_5_trades": best_score_band,
            "mae_mfe": mae_mfe_diagnostics
        }

        # =============================================================
        # FINAL RESULT
        # =============================================================

        net_pnl = sum(
            t["net_pnl"]
            for t in trades
        )

        return jsonify({

            "success": True,

            "backtest_version":
                "2.3-exit-experiment",

            "symbol":
                symbol,

            "instrument_token":
                instrument_token,

            "days":
                days,

            "parameters": {
	    	"direction":
        	direction_filter,

                "min_score":
                    min_score,

                "atr_stop":
                    atr_stop,

                "risk_reward":
                    risk_reward,

                "exit_mode":
                    exit_mode,

                "max_hold_candles":
                    max_hold,

                "slippage_pct":
                    slippage_pct,

                "brokerage_per_trade":
                    brokerage_per_trade
            },

            "summary": {

                "trades":
                    total_trades,

                "long_trades":
                    len(long_trades),

                "short_trades":
                    len(short_trades),

                "wins":
                    wins,

                "losses":
                    losses,

                "win_rate":
                    round(
                        win_rate,
                        2
                    ),

                "gross_profit":
                    round(
                        gross_profit,
                        2
                    ),

                "gross_loss":
                    round(
                        gross_loss,
                        2
                    ),

                "net_pnl":
                    round(
                        net_pnl,
                        2
                    ),

                "profit_factor":
                    round(
                        profit_factor,
                        2
                    ),

                "average_win":
                    round(
                        average_win,
                        2
                    ),

                "average_loss":
                    round(
                        average_loss,
                        2
                    ),

                "expectancy":
                    round(
                        expectancy,
                        2
                    ),

                "average_r":
                    round(
                        average_r,
                        3
                    ),

                "max_drawdown":
                    round(
                        max_drawdown,
                        2
                    ),

                "max_consecutive_wins":
                    max_consecutive_wins,

                "max_consecutive_losses":
                    max_consecutive_losses
            },

            "score_breakdown":
                score_buckets,

            "diagnostic_report":
                diagnostic_report,

            "equity_curve":
                equity_curve,

            "recent_trades":
                trades[-20:],

            "note":
                (
                    "Research backtest only. "
                    "OHLC data cannot determine the "
                    "intrabar order when both stop "
                    "and target are touched in the "
                    "same candle; such cases are "
                    "handled conservatively as SL."
                )
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