"""
AlphaEdge API — Flask Backend
Serves NSE signals and Zerodha Kite Connect authentication
"""

from flask import Flask, jsonify, request
from flask_cors import CORS

import os
import time
import io
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
import requests

from kiteconnect import KiteConnect


app = Flask(__name__)
CORS(app)


# =============================================================================
# ZERODHA KITE CONNECT
# =============================================================================

KITE_API_KEY = os.environ.get("KITE_API_KEY")
KITE_API_SECRET = os.environ.get("KITE_API_SECRET")

kite = KiteConnect(api_key=KITE_API_KEY) if KITE_API_KEY else None

# ── Token persistence ────────────────────────────────────────
# Saves token to a file so it survives Render restarts.
# On free tier, /tmp is writable and persists within a session.
# For production, use a database or environment variable instead.

TOKEN_FILE = "/tmp/kite_token.txt"

def load_token():
    """Load token from file if it exists."""
    try:
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, "r") as f:
                token = f.read().strip()
                if token:
                    return token
    except Exception:
        pass
    return None

def save_token(token):
    """Save token to file and update os.environ for current process."""
    try:
        with open(TOKEN_FILE, "w") as f:
            f.write(token)
        # Also set in current process environment
        os.environ["KITE_ACCESS_TOKEN"] = token
    except Exception:
        pass

def clear_token():
    """Clear saved token."""
    try:
        if os.path.exists(TOKEN_FILE):
            os.remove(TOKEN_FILE)
    except Exception:
        pass

# Load token on startup
# Priority: 1) Env var, 2) File, 3) None
def get_stored_token():
    # Check environment variable first
    env_token = os.environ.get("KITE_ACCESS_TOKEN", "").strip()
    if env_token:
        return env_token
    # Fall back to file
    return load_token()

kite_access_token = get_stored_token()

if kite_access_token and kite:
    try:
        kite.set_access_token(kite_access_token)
    except Exception:
        kite_access_token = None
        clear_token()


# =============================================================================
# UNIVERSE
# =============================================================================

NIFTY_INDEX_URLS = {
    "NIFTY 50": [
        "https://nsearchives.nseindia.com/content/indices/"
        "ind_nifty50list.csv",
        "https://www.niftyindices.com/IndexConstituent/"
        "ind_nifty50list.csv"
    ],
    "NIFTY NEXT 50": [
        "https://nsearchives.nseindia.com/content/indices/"
        "ind_niftynext50list.csv",
        "https://www.niftyindices.com/IndexConstituent/"
        "ind_niftynext50list.csv"
    ]
}
NSE_HOME_URL = "https://www.nseindia.com/"
NSE_INDEX_API_URL = "https://www.nseindia.com/api/equity-stockIndices"
NSE_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache"
}
UNIVERSE_CACHE_SECONDS = 6 * 60 * 60
universe_cache = {
    "loaded_at_monotonic": 0.0,
    "loaded_at": None,
    "stocks": [],
    "counts": {},
    "source_urls": {},
    "cache_status": "EMPTY",
    "refresh_error": None,
    "data_source": "NSE_INDICES_OFFICIAL"
}


def _parse_index_constituents(index_name, source_url, content):
    try:
        frame = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise LiveDataError(
            f"The official {index_name} constituent file could not be parsed",
            status_code=503,
            code="INVALID_UNIVERSE_DATA",
            details={"index": index_name, "source_url": source_url}
        ) from exc

    normalized_columns = {
        str(column).strip().lower(): column for column in frame.columns
    }
    symbol_column = normalized_columns.get("symbol")
    name_column = normalized_columns.get("company name")
    industry_column = normalized_columns.get("industry")
    if not symbol_column or not name_column:
        raise LiveDataError(
            f"The official {index_name} file is missing Symbol or Company Name",
            status_code=503,
            code="INVALID_UNIVERSE_DATA",
            details={
                "index": index_name,
                "source_url": source_url,
                "columns": [str(column) for column in frame.columns]
            }
        )

    stocks = []
    for _, row in frame.iterrows():
        symbol = str(row.get(symbol_column, "")).upper().strip()
        name = str(row.get(name_column, "")).strip()
        industry = (
            str(row.get(industry_column, "")).strip()
            if industry_column else ""
        )
        if not symbol or symbol == "NAN" or not name or name.lower() == "nan":
            continue
        stocks.append((
            symbol,
            name,
            industry if industry and industry.lower() != "nan" else "Unknown",
            index_name
        ))

    if not 45 <= len(stocks) <= 55:
        raise LiveDataError(
            f"The official {index_name} file returned an unexpected constituent count",
            status_code=503,
            code="INVALID_UNIVERSE_COUNT",
            details={
                "index": index_name,
                "count": len(stocks),
                "source_url": source_url
            }
        )
    return stocks


def _parse_nse_api_constituents(index_name, source_url, payload):
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise LiveDataError(
            f"The official {index_name} API returned an invalid response",
            status_code=503,
            code="INVALID_UNIVERSE_DATA",
            details={"index": index_name, "source_url": source_url}
        )

    stocks = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            continue

        symbol = str(row.get("symbol") or "").upper().strip()
        # The API's data array starts with an index-summary row.
        if not symbol or symbol == index_name.upper() or symbol in seen:
            continue

        meta = row.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        name = str(
            meta.get("companyName")
            or row.get("companyName")
            or row.get("name")
            or symbol
        ).strip()
        industry = str(
            meta.get("industry")
            or row.get("industry")
            or "Unknown"
        ).strip()
        seen.add(symbol)
        stocks.append((symbol, name, industry or "Unknown", index_name))

    if not 45 <= len(stocks) <= 55:
        raise LiveDataError(
            f"The official {index_name} API returned an unexpected constituent count",
            status_code=503,
            code="INVALID_UNIVERSE_COUNT",
            details={
                "index": index_name,
                "count": len(stocks),
                "source_url": source_url
            }
        )
    return stocks


def _download_nse_api_constituents(index_name):
    session = requests.Session()
    try:
        session.headers.update(NSE_BROWSER_HEADERS)

        # NSE rejects a direct API request without cookies set by its website.
        landing_response = session.get(NSE_HOME_URL, timeout=(4, 10))
        landing_response.raise_for_status()

        response = session.get(
            NSE_INDEX_API_URL,
            params={"index": index_name},
            headers={
                "Accept": "application/json, text/plain, */*",
                "Referer": NSE_HOME_URL
            },
            timeout=(4, 10)
        )
        response.raise_for_status()
        source_url = getattr(response, "url", NSE_INDEX_API_URL)
        try:
            payload = response.json()
        except ValueError as exc:
            raise LiveDataError(
                f"The official {index_name} API response was not valid JSON",
                status_code=503,
                code="INVALID_UNIVERSE_DATA",
                details={"index": index_name, "source_url": source_url}
            ) from exc

        stocks = _parse_nse_api_constituents(
            index_name,
            source_url,
            payload
        )
        return stocks, source_url
    finally:
        session.close()


def _download_index_constituents(index_name, source_urls):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; AlphaEdge/5.0; "
            "+https://www.nseindia.com/)"
        ),
        "Accept": "text/csv,application/octet-stream;q=0.9,*/*;q=0.8"
    }
    errors = []

    for source_url in source_urls:
        try:
            response = requests.get(
                source_url,
                headers=headers,
                timeout=(4, 10)
            )
            response.raise_for_status()
            stocks = _parse_index_constituents(
                index_name,
                source_url,
                response.content
            )
            return stocks, source_url
        except (requests.RequestException, LiveDataError) as exc:
            errors.append({
                "source_url": source_url,
                "error": str(exc)
            })

    try:
        return _download_nse_api_constituents(index_name)
    except (requests.RequestException, LiveDataError) as exc:
        errors.append({
            "source_url": NSE_INDEX_API_URL,
            "error": str(exc)
        })

    raise LiveDataError(
        f"Could not load the current {index_name} constituents from official sources",
        status_code=503,
        code="UNIVERSE_SOURCE_UNAVAILABLE",
        details={
            "universe_data_source": "NSE_INDICES_OFFICIAL",
            "index": index_name,
            "source_urls": list(source_urls) + [NSE_INDEX_API_URL],
            "source_errors": errors
        }
    )


def get_stock_universe(force=False):
    now_monotonic = time.monotonic()
    cache_valid = (
        universe_cache["stocks"]
        and not force
        and now_monotonic - universe_cache["loaded_at_monotonic"]
        < UNIVERSE_CACHE_SECONDS
    )
    if cache_valid:
        return list(universe_cache["stocks"])

    combined = []
    counts = {}
    source_urls = {}
    seen = set()

    try:
        for index_name, urls in NIFTY_INDEX_URLS.items():
            constituents, source_url = _download_index_constituents(
                index_name,
                urls
            )
            counts[index_name] = len(constituents)
            source_urls[index_name] = source_url
            for stock in constituents:
                if stock[0] not in seen:
                    seen.add(stock[0])
                    combined.append(stock)

        if len(combined) < 90:
            raise LiveDataError(
                "The combined NIFTY 50 and NIFTY Next 50 universe is incomplete",
                status_code=503,
                code="INVALID_UNIVERSE_COUNT",
                details={"count": len(combined), "counts": counts}
            )
    except LiveDataError as exc:
        if universe_cache["stocks"]:
            universe_cache["cache_status"] = "LAST_SUCCESSFUL"
            universe_cache["refresh_error"] = {
                "error": str(exc),
                "error_code": exc.code
            }
            return list(universe_cache["stocks"])
        raise

    universe_cache["loaded_at_monotonic"] = now_monotonic
    universe_cache["loaded_at"] = datetime.now(IST).isoformat()
    universe_cache["stocks"] = combined
    universe_cache["counts"] = counts
    universe_cache["source_urls"] = source_urls
    universe_cache["cache_status"] = "FRESH"
    universe_cache["refresh_error"] = None
    return list(combined)


# =============================================================================
# LIVE ZERODHA DATA HELPERS
# =============================================================================

IST = ZoneInfo("Asia/Kolkata")
LIVE_DATA_MAX_AGE_MINUTES = max(
    5,
    int(os.environ.get("LIVE_DATA_MAX_AGE_MINUTES", "20"))
)
INSTRUMENT_CACHE_SECONDS = 6 * 60 * 60

instrument_cache = {
    "loaded_at": 0.0,
    "by_symbol": {},
    "by_token": {}
}


class LiveDataError(Exception):
    """A client-safe failure from the authenticated live-data pipeline."""

    def __init__(self, message, status_code=502, code="KITE_DATA_ERROR", details=None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.details = details or {}


def _kite_error(action, exc):
    message = str(exc)
    lowered = message.lower()
    auth_markers = (
        "token", "permission", "api key", "api_key", "session",
        "login", "unauthor", "forbidden"
    )
    if any(marker in lowered for marker in auth_markers):
        return LiveDataError(
            f"Zerodha authentication failed while {action}: {message}",
            status_code=401,
            code="KITE_AUTH_FAILED"
        )
    return LiveDataError(
        f"Zerodha failed while {action}: {message}",
        status_code=502,
        code="KITE_DATA_ERROR"
    )


def _require_kite():
    if not kite or not KITE_API_KEY:
        raise LiveDataError(
            "KITE_API_KEY is not configured on the server",
            status_code=503,
            code="KITE_NOT_CONFIGURED"
        )
    if not kite_access_token:
        raise LiveDataError(
            "Zerodha is not authenticated. Open /api/kite/login first.",
            status_code=401,
            code="KITE_NOT_AUTHENTICATED"
        )
    try:
        kite.set_access_token(kite_access_token)
    except Exception as exc:
        raise _kite_error("setting the access token", exc) from exc
    return kite


def _to_ist_datetime(value):
    if value is None or value == "":
        return None
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            return None
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize(IST)
        else:
            stamp = stamp.tz_convert(IST)
        return stamp.to_pydatetime()
    except Exception:
        return None


def _load_nse_instruments(force=False):
    now_monotonic = time.monotonic()
    cache_is_valid = (
        instrument_cache["by_symbol"]
        and not force
        and now_monotonic - instrument_cache["loaded_at"] < INSTRUMENT_CACHE_SECONDS
    )
    if cache_is_valid:
        return instrument_cache

    client = _require_kite()
    try:
        instruments = client.instruments("NSE")
    except Exception as exc:
        raise _kite_error("loading the NSE instrument master", exc) from exc

    by_symbol = {}
    by_token = {}
    for instrument in instruments or []:
        symbol = str(instrument.get("tradingsymbol", "")).upper().strip()
        token = instrument.get("instrument_token")
        if not symbol or token is None or token == "":
            continue

        # If the master contains duplicate symbols, prefer the cash-equity row.
        existing = by_symbol.get(symbol)
        is_equity = str(instrument.get("instrument_type", "")).upper() == "EQ"
        existing_is_equity = (
            str(existing.get("instrument_type", "")).upper() == "EQ"
            if existing else False
        )
        if existing is None or (is_equity and not existing_is_equity):
            by_symbol[symbol] = instrument
        by_token[int(token)] = instrument

    if not by_symbol:
        raise LiveDataError(
            "Zerodha returned an empty NSE instrument master",
            status_code=502,
            code="EMPTY_INSTRUMENT_MASTER"
        )

    instrument_cache["loaded_at"] = now_monotonic
    instrument_cache["by_symbol"] = by_symbol
    instrument_cache["by_token"] = by_token
    return instrument_cache


def _resolve_nse_instrument(symbol=None, instrument_token=None):
    master = _load_nse_instruments()
    normalized_symbol = str(symbol or "").upper().strip()
    if normalized_symbol.startswith("NSE:"):
        normalized_symbol = normalized_symbol.split(":", 1)[1]

    resolved = None
    if normalized_symbol:
        resolved = master["by_symbol"].get(normalized_symbol)
        if not resolved:
            raise LiveDataError(
                f"{normalized_symbol} was not found in the current Zerodha NSE instrument master",
                status_code=404,
                code="INSTRUMENT_NOT_FOUND"
            )

    parsed_token = None
    if instrument_token is not None and instrument_token != "":
        try:
            parsed_token = int(instrument_token)
        except (TypeError, ValueError) as exc:
            raise LiveDataError(
                "instrument_token must be an integer",
                status_code=400,
                code="INVALID_INSTRUMENT_TOKEN"
            ) from exc

        token_instrument = master["by_token"].get(parsed_token)
        if not token_instrument:
            raise LiveDataError(
                f"Instrument token {parsed_token} was not found in the current Zerodha NSE instrument master",
                status_code=404,
                code="INSTRUMENT_NOT_FOUND"
            )
        if resolved and int(resolved["instrument_token"]) != parsed_token:
            raise LiveDataError(
                "The supplied instrument_token does not match the requested symbol",
                status_code=400,
                code="SYMBOL_TOKEN_MISMATCH",
                details={
                    "symbol": normalized_symbol,
                    "resolved_instrument_token": int(resolved["instrument_token"]),
                    "supplied_instrument_token": parsed_token
                }
            )
        resolved = token_instrument

    if not resolved:
        raise LiveDataError(
            "Provide symbol or instrument_token",
            status_code=400,
            code="MISSING_INSTRUMENT"
        )
    return resolved


def _quote_snapshot(exchange_symbol):
    client = _require_kite()
    try:
        quotes = client.quote([exchange_symbol])
    except Exception as exc:
        raise _kite_error(f"fetching the live quote for {exchange_symbol}", exc) from exc
    quote = (quotes or {}).get(exchange_symbol)
    if not quote or quote.get("last_price") is None or quote.get("last_price") == "":
        raise LiveDataError(
            f"Zerodha returned no live quote for {exchange_symbol}",
            status_code=502,
            code="QUOTE_UNAVAILABLE"
        )
    return quote


def _resolve_index_token(exchange_symbol="NSE:NIFTY 50"):
    """Resolve an index token from a current Zerodha quote, not a constant."""
    quote = _quote_snapshot(exchange_symbol)
    instrument_token = quote.get("instrument_token")
    if instrument_token is None or instrument_token == "":
        symbol = exchange_symbol.split(":", 1)[-1]
        instrument = _resolve_nse_instrument(symbol=symbol)
        instrument_token = instrument["instrument_token"]
    return int(instrument_token), quote


def _quote_time(quote):
    return _to_ist_datetime(
        quote.get("timestamp") or quote.get("last_trade_time")
    )


def _fetch_five_minute_dataframe(instrument_token, days=5):
    client = _require_kite()
    days = max(2, min(int(days), 60))
    to_date = datetime.now(IST)
    from_date = to_date - pd.Timedelta(days=days)
    try:
        candles = client.historical_data(
            int(instrument_token),
            from_date,
            to_date,
            "5minute",
            continuous=False,
            oi=False
        )
    except Exception as exc:
        raise _kite_error(
            f"fetching 5-minute candles for token {instrument_token}",
            exc
        ) from exc

    if not candles or len(candles) < 60:
        raise LiveDataError(
            "Zerodha returned insufficient 5-minute candle data",
            status_code=503,
            code="INSUFFICIENT_CANDLES",
            details={"candle_count": len(candles or [])}
        )

    frame = pd.DataFrame(candles)
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise LiveDataError(
            f"Zerodha candle data is missing: {', '.join(missing)}",
            status_code=502,
            code="INVALID_CANDLE_DATA"
        )

    frame["date"] = pd.to_datetime(frame["date"])
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=required).copy()
    frame = frame[
        (frame["high"] >= frame["low"])
        & (frame["volume"] >= 0)
        & (frame["close"] > 0)
    ].sort_values("date").drop_duplicates("date", keep="last")

    if len(frame) < 60:
        raise LiveDataError(
            "Too few valid 5-minute candles remain after validation",
            status_code=503,
            code="INSUFFICIENT_VALID_CANDLES",
            details={"candle_count": len(frame)}
        )
    return frame


def _freshness_details(quote_time, candle_time):
    now_ist = datetime.now(IST)
    quote_age = (
        max(0.0, (now_ist - quote_time).total_seconds() / 60)
        if quote_time else None
    )
    candle_age = (
        max(0.0, (now_ist - candle_time).total_seconds() / 60)
        if candle_time else None
    )
    available_times = [value for value in (quote_time, candle_time) if value]
    last_data_time = min(available_times) if len(available_times) == 2 else None
    return {
        "checked_at": now_ist.isoformat(),
        "last_quote_time": quote_time.isoformat() if quote_time else None,
        "last_candle_time": candle_time.isoformat() if candle_time else None,
        # Conservative: this is the older of the two inputs required by the result.
        "last_data_time": last_data_time.isoformat() if last_data_time else None,
        "quote_age_minutes": round(quote_age, 2) if quote_age is not None else None,
        "candle_age_minutes": round(candle_age, 2) if candle_age is not None else None,
        "max_age_minutes": LIVE_DATA_MAX_AGE_MINUTES
    }


def _require_fresh_data(symbol, quote_time, candle_time):
    freshness = _freshness_details(quote_time, candle_time)
    ages = (freshness["quote_age_minutes"], freshness["candle_age_minutes"])
    if any(age is None for age in ages) or any(
        age > LIVE_DATA_MAX_AGE_MINUTES for age in ages if age is not None
    ):
        raise LiveDataError(
            f"Live Zerodha data for {symbol} is stale or missing",
            status_code=503,
            code="STALE_LIVE_DATA",
            details={
                "symbol": symbol,
                "data_source": "ZERODHA_KITE",
                **freshness
            }
        )
    return freshness


def _confidence(score):
    if score >= 85:
        return "High"
    if score >= 75:
        return "Medium"
    return "Low"


def _analyze_live_stock(symbol=None, instrument_token=None, days=5,
                        instrument=None, quote=None):
    instrument = instrument or _resolve_nse_instrument(
        symbol=symbol,
        instrument_token=instrument_token
    )
    resolved_symbol = str(instrument["tradingsymbol"]).upper()
    resolved_token = int(instrument["instrument_token"])
    exchange = str(instrument.get("exchange") or "NSE").upper()
    quote_key = f"{exchange}:{resolved_symbol}"
    quote = quote or _quote_snapshot(quote_key)

    quote_token = quote.get("instrument_token")
    if quote_token is not None and quote_token != "" and int(quote_token) != resolved_token:
        raise LiveDataError(
            "Zerodha quote token does not match the resolved instrument",
            status_code=502,
            code="QUOTE_TOKEN_MISMATCH"
        )

    frame = _fetch_five_minute_dataframe(resolved_token, days=days)
    quote_timestamp = _quote_time(quote)
    candle_timestamp = _to_ist_datetime(frame.iloc[-1]["date"])
    freshness = _require_fresh_data(
        resolved_symbol,
        quote_timestamp,
        candle_timestamp
    )

    close = frame["close"]
    frame["ema20"] = close.ewm(span=20, adjust=False).mean()
    frame["ema50"] = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    rsi14 = 100 - (100 / (1 + relative_strength))
    no_losses = (avg_loss == 0) & (avg_gain > 0)
    no_gains = (avg_gain == 0) & (avg_loss > 0)
    unchanged = (avg_gain == 0) & (avg_loss == 0)
    frame["rsi14"] = (
        rsi14
        .mask(no_losses, 100.0)
        .mask(no_gains, 0.0)
        .mask(unchanged, 50.0)
    )

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)
    frame["atr14"] = true_range.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    frame["session_date"] = frame["date"].dt.date
    frame["typical_price"] = (
        frame["high"] + frame["low"] + frame["close"]
    ) / 3
    frame["price_volume"] = frame["typical_price"] * frame["volume"]
    frame["cumulative_pv"] = frame.groupby("session_date")["price_volume"].cumsum()
    frame["cumulative_volume"] = frame.groupby("session_date")["volume"].cumsum()
    frame["vwap"] = frame["cumulative_pv"] / frame["cumulative_volume"].replace(0, np.nan)
    frame["volume_avg20"] = frame["volume"].rolling(20, min_periods=10).mean()
    frame["relative_volume"] = frame["volume"] / frame["volume_avg20"].replace(0, np.nan)
    frame["momentum_3"] = close - close.shift(3)

    last = frame.iloc[-1]
    live_price = float(quote["last_price"])
    candle_close = float(last["close"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    rsi = float(last["rsi14"])
    atr = float(last["atr14"])
    vwap = float(last["vwap"])
    relative_volume = float(last["relative_volume"])
    momentum = float(last["momentum_3"])

    numeric_values = [live_price, ema20, ema50, rsi, atr, vwap, relative_volume, momentum]
    if not all(np.isfinite(value) for value in numeric_values):
        raise LiveDataError(
            f"Indicators could not be calculated for {resolved_symbol}",
            status_code=503,
            code="INVALID_INDICATORS"
        )

    above_vwap = live_price > vwap
    above_ema20 = live_price > ema20
    above_ema50 = live_price > ema50
    ema_bullish = ema20 > ema50
    rsi_bullish = rsi > 50
    rsi_strong = 55 <= rsi <= 70
    volume_confirmed = relative_volume >= 1.20
    momentum_positive = momentum > 0

    long_score = sum([
        20 if above_vwap else 0,
        15 if above_ema20 else 0,
        15 if above_ema50 else 0,
        15 if ema_bullish else 0,
        10 if rsi_bullish else 0,
        10 if rsi_strong else 0,
        10 if volume_confirmed else 0,
        5 if momentum_positive else 0
    ])

    below_vwap = live_price < vwap
    below_ema20 = live_price < ema20
    below_ema50 = live_price < ema50
    ema_bearish = ema20 < ema50
    rsi_bearish = rsi < 50
    rsi_weak = 30 <= rsi <= 45
    momentum_negative = momentum < 0
    short_score = sum([
        20 if below_vwap else 0,
        15 if below_ema20 else 0,
        15 if below_ema50 else 0,
        15 if ema_bearish else 0,
        10 if rsi_bearish else 0,
        10 if rsi_weak else 0,
        10 if volume_confirmed else 0,
        5 if momentum_negative else 0
    ])

    if long_score >= 75 and long_score > short_score:
        bias, setup, score = "BULLISH", "LONG_SETUP", long_score
    elif short_score >= 75 and short_score > long_score:
        bias, setup, score = "BEARISH", "SHORT_SETUP", short_score
    else:
        bias, setup, score = "NEUTRAL", "NO_TRADE", max(long_score, short_score)

    return {
        "instrument": resolved_symbol,
        "instrument_token": resolved_token,
        "exchange": exchange,
        "data_source": "ZERODHA_KITE",
        "data_status": "LIVE_FRESH",
        **freshness,
        "session_date": str(last["session_date"]),
        "latest_candle": freshness["last_candle_time"],
        "price": round(live_price, 2),
        "live_price": round(live_price, 2),
        "candle_close": round(candle_close, 2),
        "indicators": {
            "ema20": round(ema20, 2),
            "ema50": round(ema50, 2),
            "rsi14": round(rsi, 2),
            "atr14": round(atr, 2),
            "vwap": round(vwap, 2),
            "relative_volume": round(relative_volume, 2),
            "momentum_3_bars": round(momentum, 2)
        },
        "conditions": {
            "above_vwap": above_vwap,
            "above_ema20": above_ema20,
            "above_ema50": above_ema50,
            "ema20_above_ema50": ema_bullish,
            "rsi_above_50": rsi_bullish,
            "rsi_in_strong_zone": rsi_strong,
            "volume_confirmed": volume_confirmed,
            "momentum_positive": momentum_positive
        },
        "scores": {
            "long": long_score,
            "short": short_score,
            "final": score
        },
        "signal": {
            "bias": bias,
            "setup": setup,
            "trade_gate": "OPEN" if setup != "NO_TRADE" else "NO_TRADE"
        },
        "note": (
            "Live price comes from Zerodha quote; indicators come from "
            "the latest Zerodha 5-minute candles."
        )
    }


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
    Legacy dataframe adapter backed only by authenticated Zerodha 5-minute data.

    Live endpoints must never silently downgrade to delayed daily Yahoo data.
    """
    try:
        instrument = _resolve_nse_instrument(symbol=sym)
        resolved_symbol = str(instrument["tradingsymbol"]).upper()
        quote = _quote_snapshot(f"NSE:{resolved_symbol}")
        frame = _fetch_five_minute_dataframe(
            instrument["instrument_token"],
            days=5
        )
        _require_fresh_data(
            resolved_symbol,
            _quote_time(quote),
            _to_ist_datetime(frame.iloc[-1]["date"])
        )
        frame = frame.rename(columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        }).set_index("Date")
        return frame[["Open", "High", "Low", "Close", "Volume"]]
    except LiveDataError:
        return None


# =============================================================================
# MARKET CONTEXT
# =============================================================================

def get_market():
    """
    Get a fresh NIFTY 50 snapshot from Zerodha only.

    The quote supplies the live price and its instrument token. The token then
    drives the 5-minute candle request used by the market indicators.
    """
    try:
        quote = _quote_snapshot("NSE:NIFTY 50")
        instrument_token = quote.get("instrument_token")
        if instrument_token is None or instrument_token == "":
            instrument = _resolve_nse_instrument(symbol="NIFTY 50")
            instrument_token = instrument["instrument_token"]

        frame = _fetch_five_minute_dataframe(instrument_token, days=5)
        quote_timestamp = _quote_time(quote)
        candle_timestamp = _to_ist_datetime(frame.iloc[-1]["date"])
        freshness = _require_fresh_data(
            "NIFTY 50",
            quote_timestamp,
            candle_timestamp
        )

        closes = frame["close"]
        price = float(quote["last_price"])
        previous_close = quote.get("ohlc", {}).get("close")
        if previous_close is None or float(previous_close) <= 0:
            previous_close = float(closes.iloc[-2])
        previous_close = float(previous_close)
        chg = ((price - previous_close) / previous_close) * 100
        rsi = calc_rsi(closes)
        ema20 = calc_ema(closes, 20)
        ema50 = calc_ema(closes, 50)

        recent = frame.iloc[-20:]
        vwap_proxy = float(
            ((recent["high"] + recent["low"] + recent["close"]) / 3).mean()
        )
        av20 = price > ema20
        av_vwap = price > vwap_proxy
        bread = bool(closes.iloc[-1] > closes.iloc[-3])
        mkt_pts = int(sum([
            av_vwap * 5,
            av20 * 5,
            bread * 5,
            (rsi > 50) * 5
        ]))
        bear = sum([chg < -0.5, not av20, rsi < 45, not av_vwap])
        bull = sum([chg > 0.5, av20, rsi > 55, av_vwap])
        bias = (
            "STRONGLY BEARISH" if bear >= 3 else
            "BEARISH" if bear == 2 else
            "STRONGLY BULLISH" if bull >= 3 else
            "BULLISH" if bull == 2 else
            "NEUTRAL"
        )

        return {
            "available": True,
            "source": "kite",
            "data_source": "ZERODHA_KITE",
            "data_status": "LIVE_FRESH",
            "instrument_token": int(instrument_token),
            **freshness,
            "last_candle": freshness["last_candle_time"],
            "price": round(price, 1),
            "chg": round(chg, 2),
            "rsi": round(rsi, 1),
            "ema20": round(ema20, 1),
            "ema50": round(ema50, 1),
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

    except LiveDataError as exc:
        return {
            "available": False,
            "source": "none",
            "data_source": "ZERODHA_KITE",
            "data_status": exc.code,
            "price": None,
            "chg": None,
            "rsi": None,
            "ema20": None,
            "ema50": None,
            "av20": None,
            "av_vwap": None,
            "bread": None,
            "mkt_pts": None,
            "bias": "NO_DATA",
            "bear": None,
            "bull": None,
            "block_long": True,
            "block_short": True,
            "error": str(exc),
            "error_code": exc.code,
            "http_status": exc.status_code,
            **exc.details,
            "note": "Fresh authenticated Zerodha market data is required - NO TRADE"
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
    "market": None,
    "scan_key": None,
    "errors": []
}

CACHE_MINUTES = max(
    0.25,
    float(os.environ.get("LIVE_SCAN_CACHE_MINUTES", "1"))
)


def is_cache_valid():

    if not cache["time"]:
        return False

    diff = (
        datetime.now(IST) - cache["time"]
    ).total_seconds() / 60

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

    except LiveDataError as exc:

        return jsonify({

            "success": False,

            "error": str(exc),

            "error_code": exc.code,

            **exc.details

        }), exc.status_code

    except Exception as e:

        import traceback

        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
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

        kite.set_access_token(kite_access_token)

        # Persist token so it survives restarts
        save_token(kite_access_token)

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

@app.route("/api/kite/instrument")
def kite_instrument():
    """Resolve a requested NSE symbol or token from Zerodha's current master."""

    symbol = request.args.get("symbol")
    instrument_token = request.args.get("instrument_token")
    if not symbol and not instrument_token:
        return jsonify({
            "success": False,
            "error": "Provide symbol or instrument_token",
            "error_code": "MISSING_INSTRUMENT"
        }), 400

    try:
        instrument = _resolve_nse_instrument(
            symbol=symbol,
            instrument_token=instrument_token
        )
        return jsonify({
            "success": True,
            "data_source": "ZERODHA_INSTRUMENT_MASTER",
            "instrument": {
                "instrument_token": instrument.get("instrument_token"),
                "exchange_token": instrument.get("exchange_token"),
                "tradingsymbol": instrument.get("tradingsymbol"),
                "name": instrument.get("name"),
                "exchange": instrument.get("exchange"),
                "segment": instrument.get("segment"),
                "instrument_type": instrument.get("instrument_type")
            }
        })
    except LiveDataError as exc:
        return jsonify({
            "success": False,
            "error": str(exc),
            "error_code": exc.code,
            **exc.details
        }), exc.status_code


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

        symbol_arg = request.args.get("symbol")
        token_arg = request.args.get("instrument_token")
        if not symbol_arg and not token_arg:
            return jsonify({
                "success": False,
                "error": "Provide symbol or instrument_token",
                "error_code": "MISSING_INSTRUMENT"
            }), 400

        instrument = _resolve_nse_instrument(
            symbol=symbol_arg,
            instrument_token=token_arg
        )
        symbol = str(instrument["tradingsymbol"]).upper()
        instrument_token = int(instrument["instrument_token"])

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

        # Defensive parameter parsing: some clients send an empty
        # query value (e.g. atr_stop=), which request.args.get()
        # returns as an empty string rather than the default.
        atr_stop_raw = request.args.get(
            "atr_stop"
        )
        risk_reward_raw = request.args.get(
            "rr"
        )

        try:
            atr_stop = (
                float(atr_stop_raw)
                if atr_stop_raw not in (None, "")
                else 1.0
            )
        except (TypeError, ValueError):
            atr_stop = 1.0

        try:
            risk_reward = (
                float(risk_reward_raw)
                if risk_reward_raw not in (None, "")
                else 2.0
            )
        except (TypeError, ValueError):
            risk_reward = 2.0

        if atr_stop <= 0:
            atr_stop = 1.0

        if risk_reward <= 0:
            risk_reward = 2.0

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

        def safe_int_arg(name, default, minimum, maximum):
            raw = request.args.get(name)
            if raw is None or str(raw).strip() == "":
                return int(default)
            try:
                value = int(raw)
                return max(minimum, min(value, maximum))
            except (TypeError, ValueError):
                return int(default)

        max_hold = safe_int_arg("hold", 20, 1, 100)

        # Approximate costs for research purposes.
        # These are deliberately configurable.
        # Defensive parsing for numeric backtest parameters.
        def safe_float_arg(name, default):
            raw = request.args.get(name)
            if raw is None or str(raw).strip() == "":
                return float(default)
            try:
                value = float(raw)
                return float(default) if not np.isfinite(value) else value
            except (TypeError, ValueError):
                return float(default)

        slippage_pct = safe_float_arg("slippage_pct", 0.0005)
        brokerage_per_trade = safe_float_arg("brokerage", 0.0)

        if slippage_pct < 0:
            slippage_pct = 0.0005

        if brokerage_per_trade < 0:
            brokerage_per_trade = 0.0

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

            atr_value = row["atr"]

            if pd.isna(atr_value):
                i += 1
                continue

            atr = float(atr_value)

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

            # Guard every value used in entry/risk arithmetic.
            if not all(
                np.isfinite(float(x))
                for x in (
                    raw_entry,
                    slippage_pct,
                    atr,
                    atr_stop,
                    risk_reward
                )
            ):
                i += 1
                continue

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

            if not np.isfinite(atr):
                i += 1
                continue

            risk = atr * atr_stop

            if not np.isfinite(risk) or risk <= 0:
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
                "2.3.2-runtime-diagnostics",

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

    except LiveDataError as exc:

        return jsonify({

            "success": False,

            "error": str(exc),

            "error_code": exc.code,

            **exc.details

        }), exc.status_code

    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500

@app.route("/api/kite/stock-analysis")
def kite_stock_analysis():
    """
    Analyse the requested NSE symbol with a live Zerodha quote and fresh
    Zerodha 5-minute candles. The instrument master is the token authority.

    Example:
    /api/kite/stock-analysis?symbol=HDFCBANK
    """
    symbol = request.args.get("symbol")
    instrument_token = request.args.get("instrument_token")
    if not symbol and not instrument_token:
        return jsonify({
            "success": False,
            "error": "Provide symbol or instrument_token",
            "error_code": "MISSING_INSTRUMENT"
        }), 400

    try:
        days = max(2, min(int(request.args.get("days", 5)), 60))
    except (TypeError, ValueError):
        return jsonify({
            "success": False,
            "error": "days must be an integer",
            "error_code": "INVALID_DAYS"
        }), 400

    try:
        analysis = _analyze_live_stock(
            symbol=symbol,
            instrument_token=instrument_token,
            days=days
        )
        return jsonify({
            "success": True,
            **analysis
        })
    except LiveDataError as exc:
        return jsonify({
            "success": False,
            "error": str(exc),
            "error_code": exc.code,
            "data_source": "ZERODHA_KITE",
            **exc.details
        }), exc.status_code


@app.route("/api/kite/stock-candles")
def kite_stock_candles():
    """Fetch Zerodha 5-minute candles for an explicitly requested NSE stock."""

    symbol = request.args.get("symbol")
    instrument_token = request.args.get("instrument_token")
    if not symbol and not instrument_token:
        return jsonify({
            "success": False,
            "error": "Provide symbol or instrument_token",
            "error_code": "MISSING_INSTRUMENT"
        }), 400

    try:
        days = max(2, min(int(request.args.get("days", 5)), 60))
        instrument = _resolve_nse_instrument(
            symbol=symbol,
            instrument_token=instrument_token
        )
        frame = _fetch_five_minute_dataframe(
            instrument["instrument_token"],
            days=days
        )
        last_data_time = _to_ist_datetime(frame.iloc[-1]["date"])
        candles = frame[
            ["date", "open", "high", "low", "close", "volume"]
        ].to_dict(orient="records")

        return jsonify({
            "success": True,
            "symbol": instrument["tradingsymbol"],
            "instrument_token": int(instrument["instrument_token"]),
            "data_source": "ZERODHA_KITE",
            "last_data_time": (
                last_data_time.isoformat() if last_data_time else None
            ),
            "interval": "5minute",
            "count": len(candles),
            "candles": candles
        })
    except LiveDataError as exc:
        return jsonify({
            "success": False,
            "error": str(exc),
            "error_code": exc.code,
            "data_source": "ZERODHA_KITE",
            **exc.details
        }), exc.status_code
    except (TypeError, ValueError):
        return jsonify({
            "success": False,
            "error": "days must be an integer",
            "error_code": "INVALID_DAYS"
        }), 400

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

        instrument_token, _ = _resolve_index_token("NSE:NIFTY 50")

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

@app.route("/api/kite/nifty-analysis")
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

        instrument_token, _ = _resolve_index_token("NSE:NIFTY 50")

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

@app.route("/api/kite/candles")
def kite_candles():
    """Fetch Zerodha 5-minute candles for a requested token, stock, or index."""

    try:
        _require_kite()
        token_arg = request.args.get("instrument_token")
        symbol = request.args.get("symbol")
        index_instrument = request.args.get("instrument", "NSE:NIFTY 50")

        if token_arg:
            instrument_token = int(token_arg)
            resolved_symbol = symbol
        elif symbol:
            resolved = _resolve_nse_instrument(symbol=symbol)
            instrument_token = int(resolved["instrument_token"])
            resolved_symbol = resolved["tradingsymbol"]
        else:
            instrument_token, _ = _resolve_index_token(index_instrument)
            resolved_symbol = index_instrument.split(":", 1)[-1]

        days = max(2, min(int(request.args.get("days", 5)), 60))
        frame = _fetch_five_minute_dataframe(instrument_token, days=days)
        last_data_time = _to_ist_datetime(frame.iloc[-1]["date"])
        candles = frame[
            ["date", "open", "high", "low", "close", "volume"]
        ].to_dict(orient="records")

        return jsonify({
            "success": True,
            "symbol": resolved_symbol,
            "instrument_token": instrument_token,
            "data_source": "ZERODHA_KITE",
            "last_data_time": (
                last_data_time.isoformat() if last_data_time else None
            ),
            "interval": "5minute",
            "count": len(candles),
            "candles": candles
        })
    except LiveDataError as exc:
        return jsonify({
            "success": False,
            "error": str(exc),
            "error_code": exc.code,
            "data_source": "ZERODHA_KITE",
            **exc.details
        }), exc.status_code
    except (TypeError, ValueError):
        return jsonify({
            "success": False,
            "error": "instrument_token and days must be integers",
            "error_code": "INVALID_PARAMETERS"
        }), 400

@app.route("/api/kite/quote")
def kite_quote():
    """Return a fresh authenticated Zerodha quote for a requested instrument."""

    instrument = request.args.get("instrument", "NSE:NIFTY 50")
    try:
        quote = _quote_snapshot(instrument)
        quote_timestamp = _quote_time(quote)
        freshness = _require_fresh_data(
            instrument,
            quote_timestamp,
            quote_timestamp
        )
        return jsonify({
            "success": True,
            "instrument": instrument,
            "data_source": "ZERODHA_KITE",
            "data_status": "LIVE_FRESH",
            **freshness,
            "data": quote
        })
    except LiveDataError as exc:
        return jsonify({
            "success": False,
            "error": str(exc),
            "error_code": exc.code,
            "data_source": "ZERODHA_KITE",
            **exc.details
        }), exc.status_code


# =============================================================================
# API — MARKET
# =============================================================================

@app.route("/api/market")
def market():
    mkt = get_market()
    if not mkt.get("available", False):
        return jsonify({
            "success": False,
            "error": mkt.get("error", "Fresh Zerodha market data is unavailable"),
            "error_code": mkt.get("error_code", "MARKET_DATA_UNAVAILABLE"),
            "data_source": "ZERODHA_KITE",
            "last_data_time": mkt.get("last_data_time"),
            "data": mkt
        }), int(mkt.get("http_status", 503))

    return jsonify({
        "success": True,
        "data_source": "ZERODHA_KITE",
        "last_data_time": mkt.get("last_data_time"),
        "data": mkt
    })


# =============================================================================
# API — SCANNER
# =============================================================================

@app.route("/api/scan")
def scan():
    """Scan the current official NIFTY 50 and NIFTY Next 50 constituents."""

    try:
        _require_kite()
        refresh_universe = (
            request.args.get("refresh_universe", "false").lower() == "true"
        )
        universe = get_stock_universe(force=refresh_universe)
    except LiveDataError as exc:
        return jsonify({
            "success": False,
            "error": str(exc),
            "error_code": exc.code,
            "data_source": "ZERODHA_KITE",
            **exc.details
        }), exc.status_code

    try:
        requested_n = int(request.args.get("n", len(universe)))
        n = max(1, min(requested_n, len(universe)))
        min_score = max(0, min(int(request.args.get("min_score", 0)), 100))
    except (TypeError, ValueError):
        return jsonify({
            "success": False,
            "error": "n and min_score must be integers",
            "error_code": "INVALID_SCAN_PARAMETERS"
        }), 400

    direction = request.args.get("direction", "all").lower().strip()
    if direction not in ("all", "long", "short"):
        return jsonify({
            "success": False,
            "error": "direction must be all, long, or short",
            "error_code": "INVALID_DIRECTION"
        }), 400

    force = request.args.get("force", "false").lower() == "true"
    batch = universe[:n]
    scan_key = tuple(stock[0] for stock in batch)

    if (
        not force
        and is_cache_valid()
        and cache["data"] is not None
        and cache["scan_key"] == scan_key
    ):
        results = list(cache["data"])
        mkt = cache["market"]
        scan_errors = list(cache["errors"])
        cached = True
    else:
        mkt = get_market()
        if not mkt.get("available", False):
            return jsonify({
                "success": False,
                "error": mkt.get("error", "Fresh Zerodha market data is unavailable"),
                "error_code": mkt.get("error_code", "MARKET_DATA_UNAVAILABLE"),
                "data_source": "ZERODHA_KITE",
                "data_status": mkt.get("data_status"),
                "last_data_time": mkt.get("last_data_time"),
                "market": mkt,
                "signals": [],
                "count": 0
            }), int(mkt.get("http_status", 503))

        resolved = []
        scan_errors = []
        for symbol, name, sector, index_name in batch:
            try:
                instrument = _resolve_nse_instrument(symbol=symbol)
                quote_key = (
                    f"{str(instrument.get('exchange') or 'NSE').upper()}:"
                    f"{str(instrument['tradingsymbol']).upper()}"
                )
                resolved.append((
                    symbol, name, sector, index_name, instrument, quote_key
                ))
            except LiveDataError as exc:
                scan_errors.append({
                    "symbol": symbol,
                    "error": str(exc),
                    "error_code": exc.code
                })

        quote_keys = [entry[5] for entry in resolved]
        try:
            client = _require_kite()
            quotes = client.quote(quote_keys) if quote_keys else {}
        except Exception as exc:
            kite_exc = _kite_error("fetching scanner quotes", exc)
            return jsonify({
                "success": False,
                "error": str(kite_exc),
                "error_code": kite_exc.code,
                "data_source": "ZERODHA_KITE",
                "signals": [],
                "count": 0
            }), kite_exc.status_code

        results = []
        for symbol, name, sector, index_name, instrument, quote_key in resolved:
            quote = (quotes or {}).get(quote_key)
            try:
                if not quote:
                    raise LiveDataError(
                        f"Zerodha returned no live quote for {quote_key}",
                        status_code=502,
                        code="QUOTE_UNAVAILABLE"
                    )
                analysis = _analyze_live_stock(
                    symbol=symbol,
                    days=5,
                    instrument=instrument,
                    quote=quote
                )

                if analysis["signal"]["trade_gate"] != "OPEN":
                    continue

                is_long = analysis["signal"]["setup"] == "LONG_SETUP"
                direction_value = "LONG" if is_long else "SHORT"
                price = float(analysis["live_price"])
                atr = float(analysis["indicators"]["atr14"])
                stop = price - 1.5 * atr if is_long else price + 1.5 * atr
                target_1 = price + 2.0 * atr if is_long else price - 2.0 * atr
                target_2 = price + 3.5 * atr if is_long else price - 3.5 * atr
                score = int(analysis["scores"]["final"])
                strike = int(round(price / 50) * 50)

                results.append({
                    "sym": analysis["instrument"],
                    "name": name,
                    "sector": sector,
                    "index": index_name,
                    "instrument_token": analysis["instrument_token"],
                    "data_source": analysis["data_source"],
                    "data_status": analysis["data_status"],
                    "last_data_time": analysis["last_data_time"],
                    "last_quote_time": analysis["last_quote_time"],
                    "last_candle_time": analysis["last_candle_time"],
                    "price": round(price, 2),
                    "direction": direction_value,
                    "score": score,
                    "confidence": _confidence(score),
                    "stars": 5 if score >= 90 else 4 if score >= 80 else 3,
                    "rsi": analysis["indicators"]["rsi14"],
                    "rvol": analysis["indicators"]["relative_volume"],
                    "atr": analysis["indicators"]["atr14"],
                    "above_vwap": analysis["conditions"]["above_vwap"],
                    "above_ema20": analysis["conditions"]["above_ema20"],
                    "above_ema50": analysis["conditions"]["above_ema50"],
                    "entry": round(price, 2),
                    "sl": round(stop, 2),
                    "t1": round(target_1, 2),
                    "t2": round(target_2, 2),
                    "rr": round(
                        abs(target_1 - price) / abs(stop - price),
                        2
                    ) if stop != price else None,
                    "atm": (
                        f"{analysis['instrument']} {strike} "
                        f"{'CE' if is_long else 'PE'}"
                    ),
                    "indicators": analysis["indicators"],
                    "conditions": analysis["conditions"],
                    "scores": analysis["scores"],
                    "signal": analysis["signal"]
                })
            except LiveDataError as exc:
                scan_errors.append({
                    "symbol": symbol,
                    "error": str(exc),
                    "error_code": exc.code,
                    **exc.details
                })

            # Kite historical-data rate limit is lower than its batch quote limit.
            time.sleep(0.35)

        results.sort(key=lambda item: item["score"], reverse=True)
        cache["data"] = list(results)
        cache["market"] = mkt
        cache["time"] = datetime.now(IST)
        cache["scan_key"] = scan_key
        cache["errors"] = list(scan_errors)
        cached = False

    filtered = [
        item for item in results
        if item["score"] >= min_score
        and (
            direction == "all"
            or item["direction"].lower() == direction
        )
    ]

    gated = []
    for item in filtered:
        if mkt["block_long"] and item["direction"] == "LONG":
            continue
        if mkt["block_short"] and item["direction"] == "SHORT":
            continue
        gated.append(item)

    if not results and scan_errors:
        stale_errors = [
            error for error in scan_errors
            if error.get("error_code") == "STALE_LIVE_DATA"
        ]
        status_code = 503 if stale_errors else 502
        error_code = "STALE_LIVE_DATA" if stale_errors else "SCAN_DATA_UNAVAILABLE"
        return jsonify({
            "success": False,
            "error": (
                "The scanner rejected stale Zerodha data"
                if stale_errors else
                "No constituent could be analysed from Zerodha"
            ),
            "error_code": error_code,
            "data_source": "ZERODHA_KITE",
            "market": mkt,
            "signals": [],
            "count": 0,
            "scan_errors": scan_errors[:20]
        }), status_code

    note = None
    if not gated:
        note = (
            "No fresh NIFTY 50 or NIFTY Next 50 constituent met "
            "the AlphaEdge score and market-gate criteria."
        )

    result_times = [
        item["last_data_time"] for item in gated if item.get("last_data_time")
    ]
    if mkt.get("last_data_time"):
        result_times.append(mkt["last_data_time"])

    return jsonify({
        "success": True,
        "date": datetime.now(IST).isoformat(),
        "data_source": "ZERODHA_KITE",
        "universe_data_source": universe_cache["data_source"],
        "universe_loaded_at": universe_cache["loaded_at"],
        "universe_counts": universe_cache["counts"],
        "last_data_time": min(result_times) if result_times else None,
        "market": mkt,
        "count": len(gated),
        "signals": gated[:15],
        "cached": cached,
        "note": note,
        "scanned": n,
        "analyzed": len(results),
        "skipped": len(scan_errors),
        "scan_errors": scan_errors[:20],
        "universe_size": len(universe)
    })


# =============================================================================
# API — SINGLE STOCK
# =============================================================================

@app.route("/api/stock/<sym>")
def single_stock(sym):
    """Return the same fresh Zerodha analysis used by the scanner."""

    try:
        analysis = _analyze_live_stock(symbol=sym, days=5)
        mkt = get_market()
        if not mkt.get("available", False):
            raise LiveDataError(
                mkt.get("error", "Fresh Zerodha market data is unavailable"),
                status_code=int(mkt.get("http_status", 503)),
                code=mkt.get("error_code", "MARKET_DATA_UNAVAILABLE"),
                details={
                    "data_source": "ZERODHA_KITE",
                    "last_data_time": mkt.get("last_data_time")
                }
            )

        return jsonify({
            "success": True,
            "data_source": "ZERODHA_KITE",
            "last_data_time": analysis["last_data_time"],
            "market": mkt,
            "signal": analysis
        })
    except LiveDataError as exc:
        return jsonify({
            "success": False,
            "error": str(exc),
            "error_code": exc.code,
            "data_source": "ZERODHA_KITE",
            **exc.details
        }), exc.status_code


# =============================================================================
# HEALTH
# =============================================================================

@app.route("/api/clear-cache")
def clear_cache_endpoint():
    cache["data"] = None
    cache["time"] = None
    cache["market"] = None
    cache["scan_key"] = None
    cache["errors"] = []
    return jsonify({"success": True, "message": "Cache cleared"})


@app.route("/api/kite/logout")
def kite_logout():
    """Clear Kite token — forces re-authentication."""
    global kite_access_token
    kite_access_token = None
    clear_token()
    if "KITE_ACCESS_TOKEN" in os.environ:
        del os.environ["KITE_ACCESS_TOKEN"]
    return jsonify({"success": True, "message": "Logged out — re-authenticate via /api/kite/login"})


@app.route("/api/kite/token-info")
def token_info():
    """
    Show current token status and instructions for permanent storage.
    Visit this after logging in to get your token for Render env vars.
    """
    token = kite_access_token
    if not token:
        return jsonify({
            "success": False,
            "connected": False,
            "message": "Not authenticated. Visit /api/kite/login first."
        })
    # Show partial token for security
    masked = token[:8] + "..." + token[-4:] if len(token) > 12 else "***"
    return jsonify({
        "success": True,
        "connected": True,
        "token_masked": masked,
        "token_length": len(token),
        "full_token": token,  # Needed to copy into Render env var
        "instructions": {
            "step1": "Copy the full_token value above",
            "step2": "Go to Render dashboard → your service → Environment",
            "step3": "Add/update variable: KITE_ACCESS_TOKEN = <paste token>",
            "step4": "Click Save — no redeploy needed",
            "note": "Kite tokens expire daily at midnight. Update this every morning."
        }
    })


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

@app.route("/api/debug/market")
def debug_market():
    """Diagnose the same Zerodha and official-index sources used in production."""

    results = {
        "kite_token_present": bool(kite_access_token),
        "kite_configured": bool(KITE_API_KEY),
        "data_source": "ZERODHA_KITE",
        "universe_data_source": "NIFTY_INDICES_OFFICIAL"
    }

    try:
        universe = get_stock_universe()
        results["universe_count"] = len(universe)
        results["universe_counts"] = universe_cache["counts"]
        results["universe_loaded_at"] = universe_cache["loaded_at"]
        results["universe_error"] = None
    except LiveDataError as exc:
        results["universe_count"] = 0
        results["universe_error"] = {
            "message": str(exc),
            "error_code": exc.code,
            **exc.details
        }

    results["market"] = get_market()
    return jsonify({
        "success": bool(results["market"].get("available")),
        "debug": results
    })


if __name__ == "__main__":

    app.run(
        debug=False,
        host="0.0.0.0",
        port=5000
    )
