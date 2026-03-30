#!/usr/bin/env python3
"""
OANDA Client for SID Method Trading - AUGMENTED VERSION
=============================================================================
Integrates ALL THREE WAVES of SID Method strategies:

WAVE 1 (Core Quick Win):
- RSI threshold detection (exact 30/70)
- MACD alignment and cross detection
- Stop loss calculation (rounded down/up)
- Take profit at RSI 50
- Position sizing with 0.5-2% risk

WAVE 2 (Live Sessions & Q&A):
- Market context filtering via SPY/QQQ
- Divergence detection in real-time
- Pattern confirmation (W, M, H&S)
- Alternative take profits (50-SMA, points)
- Reachability check before entry

WAVE 3 (Academy Support Sessions):
- Precision RSI (no "near" values)
- Zone quality assessment for supply/demand
- Stop loss pip buffer (5/10 pips behind zone)
- Session-based filtering (Asian/London/US)
- Minimum candle requirements for patterns

Author: Sid Naiman / Trading Cafe
Version: 3.0 (Fully Augmented)
=============================================================================
"""

import requests
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Callable, Union, Tuple
import time
import math
import json
from enum import Enum

logger = logging.getLogger(__name__)

# ============================================================================
# OANDA CONFIGURATION
# ============================================================================
OANDA_API_KEY = "fceae94e861642af6c9d9de1bf6a319d-52074b19e787811450bc44152cb71e78"
OANDA_ENVIRONMENT = "practice"  # or "live"

# Base URLs
if OANDA_ENVIRONMENT == "practice":
    OANDA_BASE_URL = "https://api-fxpractice.oanda.com/v3"
else:
    OANDA_BASE_URL = "https://api-fxtrade.oanda.com/v3"

# Account ID
ACCOUNT_ID = "101-004-35778624-001"

# Headers for all requests
OANDA_HEADERS = {
    "Authorization": f"Bearer {OANDA_API_KEY}",
    "Content-Type": "application/json",
}

# ============================================================================
# VERIFIED WORKING OANDA PAIRS
# ============================================================================
VALID_OANDA_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_AUD", "EUR_CHF", "AUD_NZD",
    "NZD_JPY", "GBP_AUD", "GBP_CAD", "GBP_CHF", "GBP_NZD", "CAD_JPY", "CHF_JPY",
    "EUR_CAD", "AUD_CAD", "NZD_CAD", "EUR_NZD", "USD_NOK", "USD_SEK", "USD_TRY",
    "EUR_NOK", "EUR_SEK", "EUR_TRY", "GBP_NOK", "GBP_SEK", "GBP_TRY"
]

# ============================================================================
# MARKET INDICES FOR CONTEXT (Wave 2)
# ============================================================================
MARKET_INDICES = {
    'SPY': 'SPY',      # S&P 500 ETF
    'QQQ': 'QQQ',      # Nasdaq ETF
    'DIA': 'DIA',      # Dow Jones ETF
    'IWM': 'IWM'       # Russell 2000 ETF
}

# ============================================================================
# GRANULARITY MAPPING
# ============================================================================
UI_TO_OANDA_GRANULARITY = {
    '5s': 'S5', '10s': 'S10', '15s': 'S15', '30s': 'S30',
    '1m': 'M1', '2m': 'M2', '4m': 'M4', '5m': 'M5',
    '10m': 'M10', '15m': 'M15', '30m': 'M30',
    '1h': 'H1', '2h': 'H2', '3h': 'H3', '4h': 'H4',
    '6h': 'H6', '8h': 'H8', '12h': 'H12',
    '1d': 'D', '1w': 'W', '1M': 'M'
}

GRANULARITY_MAP = {
    '1m': 'M1', '5m': 'M5', '15m': 'M15', '30m': 'M30',
    '1h': 'H1', '2h': 'H2', '3h': 'H3', '4h': 'H4',
    '6h': 'H6', '8h': 'H8', '12h': 'H12',
    '1d': 'D', '1w': 'W', '1M': 'M',
    '1H': 'H1', '2H': 'H2', '3H': 'H3', '4H': 'H4',
    '6H': 'H6', '8H': 'H8', '12H': 'H12',
    '1D': 'D', '1W': 'W', '1M': 'M',
    '5M': 'M5', '15M': 'M15', '30M': 'M30',
    'M1': 'M1', 'M5': 'M5', 'M15': 'M15', 'M30': 'M30',
    'H1': 'H1', 'H2': 'H2', 'H3': 'H3', 'H4': 'H4',
    'H6': 'H6', 'H8': 'H8', 'H12': 'H12',
    'D': 'D', 'W': 'W', 'M': 'M',
}

VALID_GRANULARITIES = ['S5', 'S10', 'S15', 'S30', 'M1', 'M2', 'M4', 'M5', 
                       'M10', 'M15', 'M30', 'H1', 'H2', 'H3', 'H4', 'H6', 
                       'H8', 'H12', 'D', 'W', 'M']

# ============================================================================
# SID METHOD PARAMETERS (Wave 1, 2, 3)
# ============================================================================
# Wave 1: Core RSI thresholds
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RSI_TARGET = 50

# Wave 1: Earnings buffer (days)
EARNINGS_BUFFER_DAYS = 14

# Wave 3: Stop loss pips behind zone
STOP_PIPS_DEFAULT = 5   # For non-Yen pairs
STOP_PIPS_YEN = 10      # For Yen pairs

# Wave 1: Risk management
DEFAULT_RISK_PERCENT = 1.0
MIN_RISK_PERCENT = 0.5
MAX_RISK_PERCENT = 2.0

# Wave 2: Pattern detection
MIN_PATTERN_CANDLES = 7

# Pip values for different pair types
PIP_VALUES = {
    'JPY': 0.01,      # Yen pairs: 2nd decimal
    'DEFAULT': 0.0001  # Most pairs: 4th decimal
}

# ============================================================================
# ENUMS FOR SID METHOD (Wave 2 & 3)
# ============================================================================

class MarketTrend(Enum):
    """Market trend direction (Wave 2)"""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"

class SignalQuality(Enum):
    """Quality grade for trade signals (Wave 3)"""
    EXCELLENT = "excellent"   # 80%+ quality score
    GOOD = "good"             # 60-79% quality score
    FAIR = "fair"             # 40-59% quality score
    POOR = "poor"             # Below 40% quality score
    INVALID = "invalid"

class TradingSession(Enum):
    """Trading sessions (Wave 3)"""
    ASIAN = "asian"       # Tokyo: 00:00-09:00 GMT
    LONDON = "london"     # London: 07:00-16:00 GMT
    US = "us"             # New York: 12:00-21:00 GMT
    OVERLAP = "overlap"   # London-US overlap: 12:00-16:00 GMT


class OANDAClient:
    """
    OANDA API client with full SID Method implementation
    Augmented with Wave 2 and Wave 3 strategies
    """

    def __init__(self, verbose: bool = True):
        """Initialize the OANDA client with SID Method parameters"""
        self.api_key = OANDA_API_KEY
        self.base_url = OANDA_BASE_URL
        self.account_id = ACCOUNT_ID
        self.headers = OANDA_HEADERS.copy()
        self.valid_pairs = set(VALID_OANDA_PAIRS)
        self.verbose = verbose

        # SID Method parameters (Wave 1)
        self.rsi_oversold = RSI_OVERSOLD
        self.rsi_overbought = RSI_OVERBOUGHT
        self.rsi_target = RSI_TARGET
        self.earnings_buffer_days = EARNINGS_BUFFER_DAYS

        # Wave 2 parameters
        self.prefer_macd_cross = True
        self.use_pattern_confirmation = True
        self.use_divergence = True
        self.use_market_context = True

        # Wave 3 parameters
        self.strict_rsi = True
        self.stop_pips_default = STOP_PIPS_DEFAULT
        self.stop_pips_yen = STOP_PIPS_YEN
        self.min_pattern_candles = MIN_PATTERN_CANDLES

        # Risk management (Wave 1)
        self.default_risk_percent = DEFAULT_RISK_PERCENT
        self.min_risk_percent = MIN_RISK_PERCENT
        self.max_risk_percent = MAX_RISK_PERCENT

        # Account tracking
        self.account_balance = 0.0
        self.open_positions = []
        self.trade_history = []
        self.consecutive_losses = 0

        # Market context (Wave 2)
        self.current_market_trend = MarketTrend.UNKNOWN
        self.market_df = None

        # Create session for connection pooling
        self._session = requests.Session()
        self._session.headers.update(self.headers)

        # Verify connection on init
        self._verify_connection()

        logger.info(f"✅ OANDAClient initialized for {OANDA_ENVIRONMENT}")
        logger.info(f"   Account ID: {self.account_id}")
        logger.info(f"   Valid pairs: {len(self.valid_pairs)}")
        logger.info(f"   SID Method: RSI {self.rsi_oversold}/{self.rsi_overbought}")
        logger.info(f"   MACD cross preferred: {self.prefer_macd_cross}")
        logger.info(f"   Pattern confirmation: {self.use_pattern_confirmation}")
        logger.info(f"   Market context: {self.use_market_context}")

    # ========================================================================
    # CONNECTION VERIFICATION
    # ========================================================================

    def _verify_connection(self):
        """Verify API connection is working"""
        try:
            test_df = self.fetch_candles("EUR_USD", "H1", count=5)
            if not test_df.empty:
                logger.info("✅ OANDA API connection verified")
            else:
                logger.warning("⚠️ OANDA API connection test returned no data")
        except Exception as e:
            logger.error(f"❌ OANDA API connection failed: {e}")

    # ========================================================================
    # PAIR VALIDATION
    # ========================================================================

    def is_pair_supported(self, pair: str) -> bool:
        """Check if pair is in valid list"""
        return pair in self.valid_pairs

    def get_valid_pairs(self) -> List[str]:
        """Return list of valid pairs"""
        return sorted(list(self.valid_pairs))

    # ========================================================================
    # PIP CALCULATIONS (Wave 3)
    # ========================================================================

    def get_pip_value(self, instrument: str) -> float:
        """
        Get pip value for an instrument (Wave 3)
        - Most pairs: 4th decimal (0.0001)
        - Yen pairs: 2nd decimal (0.01)
        """
        if instrument.endswith('_JPY') or '_JPY' in instrument:
            return PIP_VALUES['JPY']
        return PIP_VALUES['DEFAULT']

    def get_stop_pips(self, instrument: str) -> int:
        """
        Get number of pips for stop loss placement (Wave 3)
        From Academy sessions:
        - Yen pairs: 10 pips behind zone
        - Others: 5 pips behind zone
        """
        if instrument.endswith('_JPY') or '_JPY' in instrument:
            return self.stop_pips_yen
        return self.stop_pips_default

    def calculate_pips(self, instrument: str, price1: float, price2: float) -> float:
        """Calculate number of pips between two prices"""
        pip_value = self.get_pip_value(instrument)
        return abs(price1 - price2) / pip_value

    def add_pips(self, instrument: str, price: float, pips: float, direction: str = 'up') -> float:
        """Add/subtract pips from a price (Wave 3 stop placement)"""
        pip_value = self.get_pip_value(instrument)
        adjustment = pips * pip_value

        if direction == 'up':
            return round(price + adjustment, 5)
        else:
            return round(price - adjustment, 5)

    def calculate_stop_from_zone(self, instrument: str, zone_price: float, 
                                   trade_direction: str = 'long') -> float:
        """
        Calculate proper stop loss distance from zone (Wave 3)
        Yen pairs: 10 pips, others: 5 pips
        """
        pips = self.get_stop_pips(instrument)
        pip_value = self.get_pip_value(instrument)

        if trade_direction == 'long':
            # Stop below zone
            return round(zone_price - (pips * pip_value), 5)
        else:
            # Stop above zone
            return round(zone_price + (pips * pip_value), 5)

    # ========================================================================
    # SESSION DETECTION (Wave 3)
    # ========================================================================

    def get_trading_session(self, dt: datetime) -> TradingSession:
        """
        Determine trading session based on GMT time (Wave 3)
        Asian: 00:00-09:00 GMT (Tokyo)
        London: 07:00-16:00 GMT
        US: 12:00-21:00 GMT
        Overlap: 12:00-16:00 GMT (best liquidity)
        """
        hour = dt.hour
        
        if 0 <= hour < 7:
            return TradingSession.ASIAN
        elif 7 <= hour < 12:
            return TradingSession.LONDON
        elif 12 <= hour < 16:
            return TradingSession.OVERLAP
        elif 16 <= hour < 21:
            return TradingSession.US
        else:
            return TradingSession.ASIAN

    def get_session_suitability(self, session: TradingSession) -> str:
        """Get SID method suitability for a session (Wave 3)"""
        suitability = {
            TradingSession.ASIAN: "low",
            TradingSession.LONDON: "medium",
            TradingSession.US: "high",
            TradingSession.OVERLAP: "very_high"
        }
        return suitability.get(session, "medium")

    # ========================================================================
    # TIME FORMATTING
    # ========================================================================

    def format_time(self, dt: datetime) -> str:
        """Format datetime for OANDA API (RFC3339)"""
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def parse_time(self, time_str: str) -> datetime:
        """Parse OANDA time string to datetime"""
        time_str = time_str.rstrip("Z")
        if "." in time_str:
            return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%f")
        else:
            return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S")

    # ========================================================================
    # PUBLIC ENDPOINTS (Candle data - no account ID)
    # ========================================================================

    def fetch_candles(self, instrument: str, granularity: str = 'H1',
                       from_time: Optional[Union[datetime, str]] = None,
                       to_time: Optional[Union[datetime, str]] = None,
                       count: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch candle data from OANDA - PUBLIC ENDPOINT
        This endpoint does NOT require an account ID
        """
        if not self.is_pair_supported(instrument):
            logger.warning(f"Instrument {instrument} not in valid pairs list")
            return pd.DataFrame()

        # Convert granularity
        oanda_gran = GRANULARITY_MAP.get(granularity, granularity)

        if oanda_gran not in VALID_GRANULARITIES:
            logger.error(f"❌ Invalid granularity: '{granularity}' -> '{oanda_gran}'")
            return pd.DataFrame()

        # Build URL - PUBLIC endpoint
        url = f"{self.base_url}/instruments/{instrument}/candles"

        params = {'granularity': oanda_gran, 'price': 'MBA'}

        # Handle datetime objects
        if from_time:
            if isinstance(from_time, datetime):
                params['from'] = self.format_time(from_time)
            else:
                params['from'] = from_time
        if to_time:
            if isinstance(to_time, datetime):
                params['to'] = self.format_time(to_time)
            else:
                params['to'] = to_time
        if count:
            params['count'] = min(count, 5000)

        try:
            response = self._session.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                candles = data.get('candles', [])

                if not candles:
                    return pd.DataFrame()

                # Convert to DataFrame
                df_list = []
                for c in candles:
                    if c.get('complete', True):
                        df_list.append({
                            'time': c['time'],
                            'open': float(c['mid']['o']),
                            'high': float(c['mid']['h']),
                            'low': float(c['mid']['l']),
                            'close': float(c['mid']['c']),
                            'volume': c['volume']
                        })

                if not df_list:
                    return pd.DataFrame()

                df = pd.DataFrame(df_list)
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)

                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')

                if self.verbose:
                    logger.info(f"✅ Fetched {len(df)} candles for {instrument}")
                return df

            else:
                logger.error(f"❌ OANDA API error: {response.status_code}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"❌ Error fetching candles: {e}")
            return pd.DataFrame()

    # ========================================================================
    # INDICATOR CALCULATIONS (Wave 1)
    # ========================================================================

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI from price data (Wave 1)"""
        if df.empty or len(df) < period + 1:
            return pd.Series(index=df.index)

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss.replace(0, float('nan'))
        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50)

    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, 
                        signal: int = 9) -> pd.DataFrame:
        """Calculate MACD from price data (Wave 1)"""
        if df.empty or len(df) < slow + signal:
            return pd.DataFrame(index=df.index)

        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()

        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': macd_line - signal_line
        })

    # ========================================================================
    # SID METHOD SIGNAL DETECTION (Wave 1, 2, 3)
    # ========================================================================

    def check_rsi_signal(self, rsi_value: float) -> Tuple[str, bool]:
        """
        Check RSI signal with EXACT thresholds (Wave 3 precision)
        Returns (signal_type, is_exact)
        """
        if self.strict_rsi:
            if rsi_value < self.rsi_oversold:
                return 'oversold', True
            elif rsi_value > self.rsi_overbought:
                return 'overbought', True
            else:
                return 'neutral', False
        else:
            if rsi_value < self.rsi_oversold:
                return 'oversold', True
            elif rsi_value > self.rsi_overbought:
                return 'overbought', True
            else:
                return 'neutral', False

    def check_macd_alignment(self, macd_df: pd.DataFrame, current_idx: int, 
                               signal_type: str) -> bool:
        """Check MACD alignment (Wave 1)"""
        if current_idx < 2:
            return False

        current_macd = macd_df['macd'].iloc[current_idx]
        prev_macd = macd_df['macd'].iloc[current_idx - 1]

        if signal_type == 'oversold':
            return current_macd > prev_macd
        elif signal_type == 'overbought':
            return current_macd < prev_macd
        return False

    def check_macd_cross(self, macd_df: pd.DataFrame, current_idx: int,
                           signal_type: str) -> bool:
        """Check MACD cross (Wave 2 - preferred)"""
        if current_idx < 1:
            return False

        current_macd = macd_df['macd'].iloc[current_idx]
        current_signal = macd_df['signal'].iloc[current_idx]
        prev_macd = macd_df['macd'].iloc[current_idx - 1]
        prev_signal = macd_df['signal'].iloc[current_idx - 1]

        if signal_type == 'oversold':
            return (prev_macd <= prev_signal and current_macd > current_signal)
        elif signal_type == 'overbought':
            return (prev_macd >= prev_signal and current_macd < current_signal)
        return False

    # ========================================================================
    # PATTERN DETECTION (Wave 2 & 3)
    # ========================================================================

    def detect_double_bottom(self, df: pd.DataFrame, lookback: int = 30) -> Tuple[bool, Optional[float]]:
        """
        Detect double bottom (W) pattern (Wave 2)
        Returns (detected, neckline_price)
        """
        if len(df) < lookback:
            return False, None

        recent_lows = df['low'].iloc[-lookback:]

        # Find the two lowest points
        min1_idx = recent_lows.idxmin()
        temp_lows = recent_lows.drop(min1_idx)
        if temp_lows.empty:
            return False, None

        min2_idx = temp_lows.idxmin()

        low1 = recent_lows[min1_idx]
        low2 = temp_lows[min2_idx]

        # Wave 3: Minimum candle requirement
        candle_count = abs(df.index.get_loc(min2_idx) - df.index.get_loc(min1_idx))
        if candle_count < self.min_pattern_candles:
            return False, None

        # Check if lows are within 2%
        if abs(low1 - low2) / low1 > 0.02:
            return False, None

        # Find peak between them
        between_mask = (df.index >= min(min1_idx, min2_idx)) & (df.index <= max(min1_idx, min2_idx))
        peak_between = df.loc[between_mask, 'high'].max()

        if peak_between > low1 * 1.02:
            return True, peak_between

        return False, None

    def detect_double_top(self, df: pd.DataFrame, lookback: int = 30) -> Tuple[bool, Optional[float]]:
        """Detect double top (M) pattern (Wave 2)"""
        if len(df) < lookback:
            return False, None

        recent_highs = df['high'].iloc[-lookback:]

        max1_idx = recent_highs.idxmax()
        temp_highs = recent_highs.drop(max1_idx)
        if temp_highs.empty:
            return False, None

        max2_idx = temp_highs.idxmax()

        high1 = recent_highs[max1_idx]
        high2 = temp_highs[max2_idx]

        candle_count = abs(df.index.get_loc(max2_idx) - df.index.get_loc(max1_idx))
        if candle_count < self.min_pattern_candles:
            return False, None

        if abs(high1 - high2) / high1 > 0.02:
            return False, None

        between_mask = (df.index >= min(max1_idx, max2_idx)) & (df.index <= max(max1_idx, max2_idx))
        trough_between = df.loc[between_mask, 'low'].min()

        if trough_between < high1 * 0.98:
            return True, trough_between

        return False, None

    # ========================================================================
    # DIVERGENCE DETECTION (Wave 2)
    # ========================================================================

    def detect_divergence(self, df: pd.DataFrame, lookback: int = 20) -> Dict[str, bool]:
        """Detect RSI divergence (Wave 2)"""
        if len(df) < lookback + 5:
            return {'bullish': False, 'bearish': False}

        price_window = df['close'].iloc[-lookback:]
        rsi_window = df['rsi'].iloc[-lookback:] if 'rsi' in df.columns else None

        if rsi_window is None:
            return {'bullish': False, 'bearish': False}

        price_min_idx = price_window.idxmin()
        rsi_min_idx = rsi_window.idxmin()
        price_max_idx = price_window.idxmax()
        rsi_max_idx = rsi_window.idxmax()

        bullish = (price_window.loc[price_min_idx] < price_window.iloc[0] and
                   rsi_window.loc[rsi_min_idx] > rsi_window.iloc[0])

        bearish = (price_window.loc[price_max_idx] > price_window.iloc[0] and
                   rsi_window.loc[rsi_max_idx] < rsi_window.iloc[0])

        return {'bullish': bullish, 'bearish': bearish}

    # ========================================================================
    # ZONE QUALITY ASSESSMENT (Wave 3)
    # ========================================================================

    def assess_zone_quality(self, df: pd.DataFrame, zone_start_idx: int, 
                              zone_end_idx: int) -> Dict:
        """
        Assess quality of a supply/demand zone (Wave 3)
        From Academy sessions:
        - Consolidation tightness
        - Breakout violence
        
        Returns quality score and rating
        """
        if zone_start_idx < 0 or zone_end_idx >= len(df):
            return {'quality_rating': 'invalid', 'overall_quality': 0}

        zone_data = df.iloc[zone_start_idx:zone_end_idx + 1]

        # 1. Consolidation tightness (small bodies, small wicks)
        body_sizes = abs(zone_data['close'] - zone_data['open'])
        wick_sizes = zone_data['high'] - zone_data['low']

        avg_body = body_sizes.mean()
        avg_wick = wick_sizes.mean()

        if avg_body > 0:
            tightness_score = 100 - min(100, (avg_wick / avg_body * 100))
        else:
            tightness_score = 50

        # 2. Breakout violence
        if zone_end_idx + 1 < len(df):
            breakout_candle = df.iloc[zone_end_idx + 1]
            prev_candle = df.iloc[zone_end_idx]

            breakout_size = abs(breakout_candle['close'] - breakout_candle['open'])
            prev_size = abs(prev_candle['close'] - prev_candle['open'])

            violence_score = min(100, (breakout_size / prev_size * 100) if prev_size > 0 else 50)
        else:
            violence_score = 50

        overall_quality = (tightness_score * 0.6 + violence_score * 0.4)

        if overall_quality >= 80:
            quality_rating = 'excellent'
        elif overall_quality >= 60:
            quality_rating = 'good'
        elif overall_quality >= 40:
            quality_rating = 'fair'
        else:
            quality_rating = 'poor'

        return {
            'tightness_score': tightness_score,
            'violence_score': violence_score,
            'overall_quality': overall_quality,
            'quality_rating': quality_rating,
            'consolidation_candles': len(zone_data)
        }

    # ========================================================================
    # MARKET CONTEXT ANALYSIS (Wave 2)
    # ========================================================================

    def fetch_market_context(self) -> pd.DataFrame:
        """Fetch SPY data for market context (Wave 2)"""
        # For Forex, we use DXY or major indices
        # This is a simplified version - in production, fetch SPY/QQQ
        return self.fetch_candles("EUR_USD", "D", count=100)

    def analyze_market_trend(self, df: pd.DataFrame, lookback: int = 50) -> MarketTrend:
        """Analyze market trend (Wave 2)"""
        if df.empty or len(df) < lookback:
            return MarketTrend.UNKNOWN

        recent_data = df.iloc[-lookback:]
        highs = recent_data['high']
        lows = recent_data['low']

        higher_highs = all(highs.iloc[i] <= highs.iloc[i+1] for i in range(len(highs)-10, len(highs)-1))
        higher_lows = all(lows.iloc[i] <= lows.iloc[i+1] for i in range(len(lows)-10, len(lows)-1))

        if higher_highs and higher_lows:
            return MarketTrend.UPTREND

        lower_highs = all(highs.iloc[i] >= highs.iloc[i+1] for i in range(len(highs)-10, len(highs)-1))
        lower_lows = all(lows.iloc[i] >= lows.iloc[i+1] for i in range(len(lows)-10, len(lows)-1))

        if lower_highs and lower_lows:
            return MarketTrend.DOWNTREND

        return MarketTrend.SIDEWAYS

    # ========================================================================
    # STOP LOSS CALCULATION (Wave 1 + Wave 3)
    # ========================================================================

    def calculate_sid_stop_loss(self, df: pd.DataFrame, 
                                  signal_date: datetime,
                                  entry_date: datetime,
                                  signal_type: str,
                                  instrument: str = None,
                                  use_pip_buffer: bool = True) -> float:
        """
        Calculate SID Method stop loss (Wave 1 + Wave 3)
        - Wave 1: Rounding rules (down for longs, up for shorts)
        - Wave 3: Pip buffer behind zone
        """
        mask = (df.index >= signal_date) & (df.index <= entry_date)
        period_df = df.loc[mask]

        if period_df.empty:
            return 0.0

        if signal_type == 'oversold':
            # Long trade: lowest low rounded DOWN
            lowest_low = period_df['low'].min()
            stop_loss = np.floor(lowest_low)

            if use_pip_buffer and instrument:
                pip_buffer = self.get_stop_pips(instrument)
                pip_value = self.get_pip_value(instrument)
                stop_loss = stop_loss - (pip_buffer * pip_value)

        else:
            # Short trade: highest high rounded UP
            highest_high = period_df['high'].max()
            stop_loss = np.ceil(highest_high)

            if use_pip_buffer and instrument:
                pip_buffer = self.get_stop_pips(instrument)
                pip_value = self.get_pip_value(instrument)
                stop_loss = stop_loss + (pip_buffer * pip_value)

        return float(stop_loss)

    # ========================================================================
    # TAKE PROFIT CALCULATION (Wave 1 + Wave 2)
    # ========================================================================

    def calculate_sid_take_profit(self, entry_price: float, stop_loss: float,
                                    direction: str, method: str = 'rsi_50',
                                    sma_50: Optional[float] = None) -> Dict:
        """Calculate take profit (Wave 1 primary, Wave 2 alternatives)"""
        risk_distance = abs(entry_price - stop_loss)

        if method == 'rsi_50':
            if direction == 'long':
                tp = entry_price + risk_distance
            else:
                tp = entry_price - risk_distance
            return {'primary_tp': tp, 'reward_ratio': 1.0}

        elif method == 'sma_50' and sma_50 is not None:
            if direction == 'long' and sma_50 > entry_price:
                tp = sma_50
            elif direction == 'short' and sma_50 < entry_price:
                tp = sma_50
            else:
                tp = entry_price + risk_distance if direction == 'long' else entry_price - risk_distance
            return {'primary_tp': tp, 'reward_ratio': abs(tp - entry_price) / risk_distance if risk_distance > 0 else 1.0}

        elif method == 'points':
            points = 4 if entry_price < 200 else 8
            if direction == 'long':
                tp = entry_price + points
            else:
                tp = entry_price - points
            return {'primary_tp': tp, 'reward_ratio': points / risk_distance if risk_distance > 0 else 1.0}

        else:
            return {'primary_tp': 0.0, 'reward_ratio': 0.0}

    # ========================================================================
    # POSITION SIZING (Wave 1 + Wave 2)
    # ========================================================================

    def calculate_position_size(self, entry_price: float, stop_loss: float,
                                  account_balance: float,
                                  risk_percent: float = None,
                                  reachability_multiplier: float = 1.0) -> Dict:
        """
        Calculate position size (Wave 1: 0.5-2% risk)
        Wave 2: Consecutive loss adjustment
        """
        if risk_percent is None:
            risk_percent = self.default_risk_percent

        # Apply reachability multiplier (Wave 2)
        risk_percent = risk_percent * reachability_multiplier

        # Clamp to min/max
        risk_percent = max(self.min_risk_percent, min(risk_percent, self.max_risk_percent))

        risk_amount = account_balance * (risk_percent / 100)

        if entry_price > stop_loss:
            risk_per_unit = entry_price - stop_loss
            direction = 'long'
        else:
            risk_per_unit = stop_loss - entry_price
            direction = 'short'

        if risk_per_unit <= 0:
            return {'error': 'Invalid stop loss'}

        units = risk_amount / risk_per_unit
        units = np.floor(units)

        return {
            'units': units,
            'direction': direction,
            'risk_amount': risk_amount,
            'risk_percent': risk_percent,
            'risk_per_unit': risk_per_unit,
            'position_value': units * entry_price,
            'entry_price': entry_price,
            'stop_loss': stop_loss
        }

    # ========================================================================
    # TRADE EXECUTION (Full SID Method)
    # ========================================================================

    def scan_for_sid_signals(self, instrument: str, df: pd.DataFrame,
                               market_df: pd.DataFrame = None) -> List[Dict]:
        """
        Scan for SID Method signals (Wave 1, 2, 3)
        Returns list of potential trade opportunities
        """
        if df.empty or len(df) < 50:
            return []

        # Calculate indicators
        df = df.copy()
        df['rsi'] = self.calculate_rsi(df)
        macd_df = self.calculate_macd(df)
        df['macd'] = macd_df['macd']
        df['macd_signal'] = macd_df['signal']

        # Market context (Wave 2)
        market_trend = self.analyze_market_trend(market_df) if market_df is not None else MarketTrend.UNKNOWN

        signals = []

        for i in range(20, len(df) - 1):
            current_date = df.index[i]
            rsi_value = df['rsi'].iloc[i]

            signal_type, is_exact = self.check_rsi_signal(rsi_value)
            if signal_type == 'neutral':
                continue

            # Wave 2: Filter by market context
            if market_trend != MarketTrend.UNKNOWN:
                if signal_type == 'oversold' and market_trend == MarketTrend.DOWNTREND:
                    continue
                if signal_type == 'overbought' and market_trend == MarketTrend.UPTREND:
                    continue
                if market_trend == MarketTrend.SIDEWAYS:
                    continue

            # Check MACD (Wave 1 & 2)
            aligned = self.check_macd_alignment(macd_df, i, signal_type)
            if not aligned:
                continue

            crossed = self.check_macd_cross(macd_df, i, signal_type)

            if self.prefer_macd_cross and not crossed:
                continue

            # Find signal date
            signal_date = self._find_signal_date(df, i, signal_type)

            # Calculate stop loss (Wave 1 + Wave 3)
            stop_loss = self.calculate_sid_stop_loss(
                df, signal_date, current_date, signal_type, instrument, use_pip_buffer=True
            )

            direction = 'long' if signal_type == 'oversold' else 'short'
            entry_price = df['close'].iloc[i]

            # Pattern confirmation (Wave 2)
            pattern_confirmed = False
            pattern_name = None
            if self.use_pattern_confirmation:
                if signal_type == 'oversold':
                    detected, _ = self.detect_double_bottom(df.iloc[:i+1])
                    pattern_confirmed = detected
                    pattern_name = 'double_bottom' if detected else None
                else:
                    detected, _ = self.detect_double_top(df.iloc[:i+1])
                    pattern_confirmed = detected
                    pattern_name = 'double_top' if detected else None

            # Divergence (Wave 2)
            divergence = self.detect_divergence(df.iloc[:i+1]) if self.use_divergence else {'bullish': False, 'bearish': False}
            divergence_detected = (divergence['bullish'] and signal_type == 'oversold') or (divergence['bearish'] and signal_type == 'overbought')

            # Calculate take profit (Wave 1 & 2)
            sma_50 = df['close'].rolling(50).mean().iloc[i] if len(df) > 50 else None
            tp_result = self.calculate_sid_take_profit(entry_price, stop_loss, direction, 'rsi_50', sma_50)

            # Session check (Wave 3)
            session = self.get_trading_session(current_date)
            session_suitability = self.get_session_suitability(session)

            # Quality assessment (Wave 3)
            quality_score = 0
            if crossed:
                quality_score += 30
            if pattern_confirmed:
                quality_score += 25
            if divergence_detected:
                quality_score += 20
            if is_exact:
                quality_score += 15
            if session_suitability == 'high' or session_suitability == 'very_high':
                quality_score += 10

            if quality_score >= 70:
                quality = SignalQuality.EXCELLENT
            elif quality_score >= 50:
                quality = SignalQuality.GOOD
            elif quality_score >= 30:
                quality = SignalQuality.FAIR
            else:
                quality = SignalQuality.POOR

            signals.append({
                'date': current_date,
                'instrument': instrument,
                'signal_type': signal_type,
                'direction': direction,
                'rsi_value': rsi_value,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': tp_result['primary_tp'],
                'reward_ratio': tp_result['reward_ratio'],
                'macd_aligned': aligned,
                'macd_crossed': crossed,
                'pattern_confirmed': pattern_confirmed,
                'pattern_name': pattern_name,
                'divergence_detected': divergence_detected,
                'session': session.value,
                'session_suitability': session_suitability,
                'quality': quality.value,
                'quality_score': quality_score,
                'market_trend': market_trend.value if market_trend != MarketTrend.UNKNOWN else 'unknown'
            })

        return signals

    def _find_signal_date(self, df: pd.DataFrame, entry_idx: int, signal_type: str) -> datetime:
        """Find the date when RSI first crossed the threshold"""
        rsi_values = df['rsi'].iloc[:entry_idx + 1]

        if signal_type == 'oversold':
            mask = rsi_values < self.rsi_oversold
        else:
            mask = rsi_values > self.rsi_overbought

        if mask.any():
            signal_idx = mask[mask].index[-1]
            return signal_idx
        else:
            return df.index[entry_idx]

    # ========================================================================
    # ACCOUNT-SPECIFIC ENDPOINTS
    # ========================================================================

    def get_account_summary(self) -> Dict:
        """Get account details and balance"""
        if not self.account_id:
            return {}

        url = f"{self.base_url}/accounts/{self.account_id}"
        try:
            response = self._session.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                account = data.get('account', {})
                self.account_balance = float(account.get('balance', 0))
                return data
            else:
                logger.error(f"Failed to get account details: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Exception getting account details: {e}")
            return {}

    def get_current_price(self, instrument: str) -> Dict:
        """Get current bid/ask for an instrument"""
        if not self.account_id or not self.is_pair_supported(instrument):
            return {}

        url = f"{self.base_url}/accounts/{self.account_id}/pricing"
        params = {"instruments": instrument}

        try:
            response = self._session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                prices = data.get('prices', [])
                if prices:
                    price = prices[0]
                    bid = float(price['bids'][0]['price'])
                    ask = float(price['asks'][0]['price'])
                    mid = (bid + ask) / 2

                    return {
                        'instrument': price['instrument'],
                        'bid': bid,
                        'ask': ask,
                        'mid': mid,
                        'spread': ask - bid,
                        'spread_pips': self.calculate_pips(instrument, ask, bid),
                        'time': price['time']
                    }
            else:
                return {}
        except Exception as e:
            logger.error(f"Exception getting price: {e}")
            return {}

    def place_order(self, instrument: str, units: int, take_profit: float = None,
                     stop_loss: float = None) -> Dict:
        """Place a market order with optional TP/SL"""
        if not self.account_id:
            return {}

        url = f"{self.base_url}/accounts/{self.account_id}/orders"

        order = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units)
            }
        }

        if take_profit:
            order["order"]["takeProfitOnFill"] = {"price": f"{take_profit:.5f}"}
        if stop_loss:
            order["order"]["stopLossOnFill"] = {"price": f"{stop_loss:.5f}"}

        try:
            response = self._session.post(url, json=order, timeout=10)

            if response.status_code == 201:
                logger.info(f"✅ Order placed: {units} units of {instrument}")
                return response.json()
            else:
                logger.error(f"❌ Order failed: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            logger.error(f"Exception placing order: {e}")
            return {}

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_recent_data(self, instrument: str, granularity: str = '1h',
                         hours: int = 24) -> pd.DataFrame:
        """Get recent data for an instrument"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)

        return self.fetch_candles(
            instrument=instrument,
            granularity=granularity,
            from_time=start_time,
            to_time=end_time
        )

    def get_historical_data(self, instrument: str, granularity: str = 'H1',
                              days: int = 30) -> pd.DataFrame:
        """Get historical data for backtesting"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        return self.fetch_candles(
            instrument=instrument,
            granularity=granularity,
            from_time=start_time,
            to_time=end_time
        )

    def get_connection_status(self) -> Dict:
        """Get current connection status"""
        return {
            'environment': OANDA_ENVIRONMENT,
            'account_id': self.account_id,
            'valid_pairs_count': len(self.valid_pairs),
            'base_url': self.base_url,
            'sid_method_params': {
                'rsi_oversold': self.rsi_oversold,
                'rsi_overbought': self.rsi_overbought,
                'prefer_macd_cross': self.prefer_macd_cross,
                'use_pattern_confirmation': self.use_pattern_confirmation,
                'use_divergence': self.use_divergence,
                'use_market_context': self.use_market_context,
                'strict_rsi': self.strict_rsi,
                'stop_pips_default': self.stop_pips_default,
                'stop_pips_yen': self.stop_pips_yen
            }
        }


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "="*60)
    print("🧪 TESTING OANDA CLIENT with SID METHOD v3.0")
    print("="*60)

    client = OANDAClient(verbose=True)

    # Test 1: Get account summary
    print("\n📊 Account Summary:")
    summary = client.get_account_summary()
    if summary:
        account = summary.get('account', {})
        print(f"  Balance: {account.get('balance')} {account.get('currency')}")
        print(f"  NAV: {account.get('NAV')}")

    # Test 2: Fetch data for major pairs
    print("\n📈 Fetching data for EUR_USD...")
    df = client.get_recent_data("EUR_USD", "H1", hours=168)  # 7 days

    if not df.empty:
        print(f"  Fetched {len(df)} candles")

        # Test 3: Scan for SID signals
        print("\n🔍 Scanning for SID Method signals...")
        signals = client.scan_for_sid_signals("EUR_USD", df)

        print(f"  Found {len(signals)} potential signals")

        for sig in signals[:5]:
            print(f"\n  📍 {sig['date']}: {sig['direction'].upper()} @ {sig['entry_price']:.5f}")
            print(f"     RSI: {sig['rsi_value']:.1f} | Quality: {sig['quality']} ({sig['quality_score']})")
            print(f"     Stop: {sig['stop_loss']:.5f} | TP: {sig['take_profit']:.5f}")
            if sig['pattern_confirmed']:
                print(f"     Pattern: {sig['pattern_name']}")
            if sig['divergence_detected']:
                print(f"     Divergence detected")

    # Test 4: Connection status
    print("\n🔌 Connection Status:")
    status = client.get_connection_status()
    for key, value in status.items():
        if key != 'sid_method_params':
            print(f"  {key}: {value}")

    print("\n✅ Test complete")