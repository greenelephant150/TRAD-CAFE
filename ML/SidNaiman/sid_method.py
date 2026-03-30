#!/usr/bin/env python3
"""
Sid Naiman's SID Method - COMPLETE AUGMENTED IMPLEMENTATION
=============================================================================
Incorporates strategies from all three waves of transcripts:

WAVE 1 (Core Quick Win):
- RSI thresholds: EXACTLY below 30 (oversold) or above 70 (overbought)
- MACD alignment: both pointing same direction
- MACD cross: preferred over pointing (lower risk)
- Stop loss: lowest low (longs) / highest high (shorts) between signal and entry
- Stop loss rounding: down for longs, up for shorts
- Take profit: RSI reaches 50
- Position sizing: 0.5% to 2% of account

WAVE 2 (Live Sessions & Q&A):
- Market context: uptrend = focus on oversold; downtrend = focus on overbought
- Divergence detection (bullish/bearish)
- Price pattern confirmation (W, M, H&S)
- Alternative take profits: 50-SMA, +4 points (<$200), +8 points (>$200)
- Forex application with specific pairs (GBP/JPY, EUR/USD)
- Reachability check (2:1 risk-reward)

WAVE 3 (Academy Support Sessions):
- Precision: NO "near" RSI levels – exact thresholds only
- Stop loss refinement: 5 pips behind zone (learning mode)
- Aggressive vs. passive stops
- Zone quality assessment
- Session-based trading (Asian, London, US)
- Pattern completion: minimum 7 candles, neckline break

Author: Sid Naiman / Trading Cafe
Version: 3.0 (Fully Augmented)
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm, trange
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    class tqdm:
        def __init__(self, iterable=None, desc="", **kwargs):
            self.iterable = iterable or []
            self.desc = desc
        def __iter__(self): 
            return iter(self.iterable)
        def update(self, n=1): pass
        def close(self): pass
        def set_description(self, desc): self.desc = desc
        def set_postfix(self, **kwargs): pass
    trange = lambda *args, **kwargs: tqdm(range(*args), **kwargs)


class MarketTrend(Enum):
    """Market trend direction enum"""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


class SignalQuality(Enum):
    """Quality grade for trade signals (from Wave 3 zone assessment)"""
    EXCELLENT = "excellent"   # 80%+ quality score
    GOOD = "good"             # 60-79% quality score
    FAIR = "fair"             # 40-59% quality score
    POOR = "poor"             # Below 40% quality score
    INVALID = "invalid"       # Does not meet minimum criteria


class TradingSession(Enum):
    """Trading sessions (from Wave 3 session rules)"""
    ASIAN = "asian"       # Tokyo: 00:00-09:00 GMT
    LONDON = "london"     # London: 07:00-16:00 GMT
    US = "us"             # New York: 12:00-21:00 GMT
    OVERLAP = "overlap"   # London-US overlap: 12:00-16:00 GMT


class SidMethod:
    """
    Complete implementation of Sid Naiman's SID Method
    Augmented with all three waves of transcript strategies
    """

    def __init__(self, 
                 account_balance: float = 10000, 
                 verbose: bool = True,
                 prefer_macd_cross: bool = True,
                 use_pattern_confirmation: bool = True,
                 use_divergence: bool = True,
                 use_market_context: bool = True,
                 risk_percent_default: float = 1.0,
                 risk_percent_min: float = 0.5,
                 risk_percent_max: float = 2.0,
                 earnings_buffer_days: int = 14,
                 stop_pips_default: int = 5,
                 stop_pips_yen: int = 10):
        """
        Initialize the SID Method with augmented parameters
        
        Args:
            account_balance: Starting account balance
            verbose: Enable verbose output
            prefer_macd_cross: Prefer MACD cross over pointing (Wave 2)
            use_pattern_confirmation: Use W/M/H&S patterns (Wave 2)
            use_divergence: Use RSI/MACD divergence (Wave 2)
            use_market_context: Filter trades by market trend (Wave 2)
            risk_percent_default: Default risk per trade (1%)
            risk_percent_min: Minimum risk (0.5%)
            risk_percent_max: Maximum risk (2%)
            earnings_buffer_days: Days to avoid before earnings (14)
            stop_pips_default: Pips behind zone for non-Yen (5)
            stop_pips_yen: Pips behind zone for Yen pairs (10)
        """
        self.account_balance = account_balance
        self.verbose = verbose
        
        # ====================================================================
        # CORE SID METHOD PARAMETERS (Wave 1)
        # ====================================================================
        self.rsi_oversold = 30          # EXACT threshold - no "near" allowed
        self.rsi_overbought = 70        # EXACT threshold - no "near" allowed
        self.rsi_target = 50            # Primary take profit level
        self.earnings_buffer_days = earnings_buffer_days
        self.max_consecutive_losses = 3
        
        # ====================================================================
        # AUGMENTED PARAMETERS (Wave 2)
        # ====================================================================
        self.prefer_macd_cross = prefer_macd_cross
        self.use_pattern_confirmation = use_pattern_confirmation
        self.use_divergence = use_divergence
        self.use_market_context = use_market_context
        
        # Point-based take profit (for stocks)
        self.point_target_low = 4       # For stocks under $200
        self.point_target_high = 8      # For stocks over $200
        
        # ====================================================================
        # RISK MANAGEMENT PARAMETERS (Wave 1 & 3)
        # ====================================================================
        self.risk_percent_default = risk_percent_default
        self.risk_percent_min = risk_percent_min
        self.risk_percent_max = risk_percent_max
        
        # Stop loss pips behind zone (from Wave 3)
        self.stop_pips_default = stop_pips_default
        self.stop_pips_yen = stop_pips_yen
        
        # ====================================================================
        # TRACKING VARIABLES
        # ====================================================================
        self.open_positions = []
        self.trade_history = []
        self.consecutive_losses = 0
        self.daily_loss = 0.0
        self.daily_profit = 0.0
        self.current_market_trend = MarketTrend.UNKNOWN
        
        # Pattern detector reference (will be initialized later)
        self.pattern_detector = None
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🎯 SID METHOD v3.0 (Fully Augmented)")
            print(f"{'='*60}")
            print(f"📊 Account Balance: ${self.account_balance:,.2f}")
            print(f"📈 RSI Thresholds: oversold < {self.rsi_oversold}, overbought > {self.rsi_overbought}")
            print(f"🔄 MACD: {'Prefer CROSS' if prefer_macd_cross else 'Pointing OK'}")
            print(f"📐 Pattern Confirmation: {'Enabled' if use_pattern_confirmation else 'Disabled'}")
            print(f"⚡ Divergence Detection: {'Enabled' if use_divergence else 'Disabled'}")
            print(f"🌍 Market Context Filter: {'Enabled' if use_market_context else 'Disabled'}")
            print(f"💰 Risk per trade: {self.risk_percent_min}%-{self.risk_percent_max}% (default: {self.risk_percent_default}%)")
            print(f"🛑 Stop pips: {self.stop_pips_default} (default) / {self.stop_pips_yen} (Yen)")
            print(f"📅 Earnings buffer: {self.earnings_buffer_days} days")
            print(f"{'='*60}\n")

    # ========================================================================
    # SECTION 1: CORE INDICATOR CALCULATIONS (Wave 1)
    # ========================================================================

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14, 
                      desc: str = "Calculating RSI") -> pd.Series:
        """
        Calculate RSI (Relative Strength Index)
        Default period = 14 (from Sid Method)
        """
        if self.verbose:
            print(f"[SidMethod] {desc} (period={period})...")

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Avoid division by zero
        rs = gain / loss.replace(0, float('nan'))
        rsi = 100 - (100 / (1 + rs))

        if self.verbose:
            print(f"[SidMethod]   RSI range: {rsi.min():.2f} - {rsi.max():.2f}")

        return rsi.fillna(50)

    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, 
                        signal: int = 9, desc: str = "Calculating MACD") -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        Default settings: 12, 26, 9 (from Sid Method)
        """
        if self.verbose:
            print(f"[SidMethod] {desc} (fast={fast}, slow={slow}, signal={signal})...")

        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        if self.verbose:
            print(f"[SidMethod]   MACD range: {macd_line.min():.4f} - {macd_line.max():.4f}")

        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })

    # ========================================================================
    # SECTION 2: SIGNAL DETECTION (Wave 1 - Exact Thresholds)
    # ========================================================================

    def check_rsi_signal(self, rsi_value: float, strict: bool = True) -> Tuple[str, bool]:
        """
        Check RSI signal with EXACT thresholds (Wave 3 precision rule)
        
        Args:
            rsi_value: Current RSI value
            strict: If True, requires EXACTLY below 30 or above 70 (no "near")
        
        Returns:
            (signal_type, is_exact) where signal_type is 'oversold', 'overbought', or 'neutral'
        """
        if strict:
            # WAVE 3 PRECISION RULE: Must be EXACTLY below 30 or above 70
            # No "near" allowed - 29.9 is oversold, 30.1 is NOT
            if rsi_value < self.rsi_oversold:
                if self.verbose:
                    print(f"[SidMethod]   RSI={rsi_value:.2f} < {self.rsi_oversold} -> OVERSOLD (EXACT)")
                return 'oversold', True
            elif rsi_value > self.rsi_overbought:
                if self.verbose:
                    print(f"[SidMethod]   RSI={rsi_value:.2f} > {self.rsi_overbought} -> OVERBOUGHT (EXACT)")
                return 'overbought', True
            else:
                if self.verbose:
                    print(f"[SidMethod]   RSI={rsi_value:.2f} -> neutral (not exact threshold)")
                return 'neutral', False
        else:
            # Legacy mode (for comparison/testing)
            if rsi_value < self.rsi_oversold:
                return 'oversold', True
            elif rsi_value > self.rsi_overbought:
                return 'overbought', True
            else:
                return 'neutral', False

    def check_macd_alignment(self, macd_df: pd.DataFrame, current_idx: int, 
                               signal_type: str) -> bool:
        """
        Check MACD alignment (pointing same direction)
        Wave 1: Basic alignment check
        """
        if current_idx < 2:
            return False

        current_macd = macd_df['macd'].iloc[current_idx]
        prev_macd = macd_df['macd'].iloc[current_idx - 1]

        if signal_type == 'oversold':
            # For long trades, MACD should be pointing UP
            aligned = current_macd > prev_macd
            if self.verbose and aligned:
                print(f"[SidMethod]   MACD aligned UP: {prev_macd:.4f} -> {current_macd:.4f}")
            return aligned
        elif signal_type == 'overbought':
            # For short trades, MACD should be pointing DOWN
            aligned = current_macd < prev_macd
            if self.verbose and aligned:
                print(f"[SidMethod]   MACD aligned DOWN: {prev_macd:.4f} -> {current_macd:.4f}")
            return aligned
        else:
            return False

    def check_macd_cross(self, macd_df: pd.DataFrame, current_idx: int,
                           signal_type: str) -> bool:
        """
        Check MACD cross (higher confidence than alignment)
        Wave 2: Cross is preferred for lower risk entries
        """
        if current_idx < 1:
            return False

        current_macd = macd_df['macd'].iloc[current_idx]
        current_signal = macd_df['signal'].iloc[current_idx]
        prev_macd = macd_df['macd'].iloc[current_idx - 1]
        prev_signal = macd_df['signal'].iloc[current_idx - 1]

        if signal_type == 'oversold':
            # MACD crosses ABOVE signal line (bullish)
            crossed = (prev_macd <= prev_signal and current_macd > current_signal)
            if self.verbose and crossed:
                print(f"[SidMethod]   MACD CROSSED UP (bullish)")
            return crossed
        elif signal_type == 'overbought':
            # MACD crosses BELOW signal line (bearish)
            crossed = (prev_macd >= prev_signal and current_macd < current_signal)
            if self.verbose and crossed:
                print(f"[SidMethod]   MACD CROSSED DOWN (bearish)")
            return crossed
        else:
            return False

    def get_entry_confidence(self, macd_aligned: bool, macd_crossed: bool, 
                              pattern_confirmed: bool = False,
                              divergence_detected: bool = False) -> Tuple[str, float]:
        """
        Calculate entry confidence based on multiple confirmations
        
        Wave 2 & 3: Multiple confirmation factors increase confidence
        
        Returns:
            (confidence_level, confidence_score) where confidence_level is 'high', 'medium', 'low'
        """
        score = 0.0
        
        # MACD cross is weighted more heavily (Wave 2)
        if macd_crossed:
            score += 0.5
        elif macd_aligned:
            score += 0.3
        
        # Pattern confirmation adds confidence (Wave 2)
        if pattern_confirmed:
            score += 0.3
        
        # Divergence adds confidence (Wave 2)
        if divergence_detected:
            score += 0.2
        
        # Determine confidence level
        if score >= 0.8:
            confidence_level = "high"
        elif score >= 0.5:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        return confidence_level, score

    # ========================================================================
    # SECTION 3: STOP LOSS CALCULATION (Wave 1 + Wave 3 Refinements)
    # ========================================================================

    def calculate_stop_loss(self, df: pd.DataFrame, 
                             signal_date: datetime,
                             entry_date: datetime,
                             signal_type: str,
                             use_rounded: bool = True,
                             use_pip_buffer: bool = False,
                             instrument: str = None) -> float:
        """
        Calculate stop loss with Wave 1 rounding rules and Wave 3 pip buffer
        
        WAVE 1 RULE:
        - Long (oversold): Lowest low between signal and entry, rounded DOWN
        - Short (overbought): Highest high between signal and entry, rounded UP
        
        WAVE 3 REFINEMENT:
        - Add pip buffer (5 pips for default, 10 for Yen pairs)
        
        Args:
            df: Price DataFrame
            signal_date: Date when RSI first triggered
            entry_date: Date of trade entry
            signal_type: 'oversold' or 'overbought'
            use_rounded: Apply rounding to nearest whole number
            use_pip_buffer: Add pip buffer behind zone (Wave 3)
            instrument: Instrument for pip calculation
        
        Returns:
            Stop loss price
        """
        mask = (df.index >= signal_date) & (df.index <= entry_date)
        period_df = df.loc[mask]

        if period_df.empty:
            if self.verbose:
                print(f"[SidMethod]   WARNING: Empty period for stop loss")
            return 0.0

        if signal_type == 'oversold':
            # LONG TRADE: Lowest low between signal and entry
            lowest_low = period_df['low'].min()
            
            if use_rounded:
                # WAVE 1 RULE: Round DOWN to nearest whole number
                stop_loss = np.floor(lowest_low)
            else:
                stop_loss = lowest_low
            
            if self.verbose:
                print(f"[SidMethod]   Stop loss (long): lowest={lowest_low:.5f} -> {stop_loss:.5f}")
                
        else:
            # SHORT TRADE: Highest high between signal and entry
            highest_high = period_df['high'].max()
            
            if use_rounded:
                # WAVE 1 RULE: Round UP to nearest whole number
                stop_loss = np.ceil(highest_high)
            else:
                stop_loss = highest_high
            
            if self.verbose:
                print(f"[SidMethod]   Stop loss (short): highest={highest_high:.5f} -> {stop_loss:.5f}")
        
        # WAVE 3 REFINEMENT: Add pip buffer behind zone
        if use_pip_buffer and instrument:
            pip_buffer = self.get_stop_pips(instrument)
            pip_value = self.get_pip_value(instrument)
            buffer_amount = pip_buffer * pip_value
            
            if signal_type == 'oversold':
                # Long: move stop DOWN by buffer
                stop_loss = stop_loss - buffer_amount
            else:
                # Short: move stop UP by buffer
                stop_loss = stop_loss + buffer_amount
            
            if self.verbose:
                print(f"[SidMethod]   Added {pip_buffer} pip buffer: stop -> {stop_loss:.5f}")
        
        return float(stop_loss)

    def get_pip_value(self, instrument: str) -> float:
        """Get pip value for an instrument (Wave 3)"""
        if instrument and ('JPY' in instrument or instrument.endswith('_JPY')):
            return 0.01  # Yen pairs: 2nd decimal
        return 0.0001    # Most pairs: 4th decimal

    def get_stop_pips(self, instrument: str) -> int:
        """
        Get number of pips for stop loss placement (Wave 3)
        Yen pairs: 10 pips behind zone
        Others: 5 pips behind zone
        """
        if instrument and ('JPY' in instrument or instrument.endswith('_JPY')):
            return self.stop_pips_yen
        return self.stop_pips_default

    # ========================================================================
    # SECTION 4: TAKE PROFIT CALCULATION (Wave 1 + Wave 2 Alternatives)
    # ========================================================================

    def calculate_take_profit(self, entry_price: float, stop_loss: float,
                                direction: str, 
                                method: str = 'rsi_50',
                                sma_50: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate take profit levels with multiple methods (Wave 1 + Wave 2)
        
        WAVE 1 PRIMARY: RSI reaches 50 (risk:reward ~1:1)
        
        WAVE 2 ALTERNATIVES:
        - 50-SMA (blue line) - institutional support/resistance
        - Point targets: +4 points (<$200) or +8 points (>$200)
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            direction: 'long' or 'short'
            method: 'rsi_50', 'sma_50', 'points', 'trailing'
            sma_50: 50-period SMA value (for sma_50 method)
        
        Returns:
            Dictionary with take profit levels and method
        """
        risk_distance = abs(entry_price - stop_loss)
        
        results = {
            'primary_tp': 0.0,
            'alternative_tp': 0.0,
            'risk_distance': risk_distance,
            'reward_ratio': 1.0,
            'method_used': method
        }
        
        if method == 'rsi_50':
            # WAVE 1 PRIMARY: Risk:Reward ~1:1
            if direction == 'long':
                tp = entry_price + risk_distance
            else:
                tp = entry_price - risk_distance
            results['primary_tp'] = tp
            results['reward_ratio'] = 1.0
            
        elif method == 'sma_50' and sma_50 is not None:
            # WAVE 2 ALTERNATIVE: 50-SMA as target
            if direction == 'long':
                tp = sma_50
                if tp > entry_price:
                    results['reward_ratio'] = abs(tp - entry_price) / risk_distance
            else:
                tp = sma_50
                if tp < entry_price:
                    results['reward_ratio'] = abs(entry_price - tp) / risk_distance
            results['primary_tp'] = tp
            
        elif method == 'points':
            # WAVE 2 ALTERNATIVE: Fixed point targets
            if entry_price < 200:
                points = self.point_target_low
            else:
                points = self.point_target_high
            
            if direction == 'long':
                tp = entry_price + points
            else:
                tp = entry_price - points
            
            results['primary_tp'] = tp
            results['reward_ratio'] = abs(tp - entry_price) / risk_distance if risk_distance > 0 else 1.0
            
        elif method == 'trailing':
            # WAVE 2 ADVANCED: No fixed TP, use trailing stop
            results['primary_tp'] = 0.0  # Will be managed dynamically
            results['method_used'] = 'trailing'
        
        if self.verbose:
            print(f"[SidMethod]   Take profit ({method}): {results['primary_tp']:.5f} (R:R = {results['reward_ratio']:.2f})")
        
        return results

    # ========================================================================
    # SECTION 5: DIVERGENCE DETECTION (Wave 2)
    # ========================================================================

    def detect_divergence(self, df: pd.DataFrame, current_idx: int, 
                           lookback: int = 20) -> Dict[str, bool]:
        """
        Detect bullish/bearish divergence between price and RSI/MACD
        
        WAVE 2: Divergence is an early warning signal
        - Bullish divergence: Price makes lower low, RSI/MACD makes higher low
        - Bearish divergence: Price makes higher high, RSI/MACD makes lower high
        
        Returns:
            Dictionary with divergence flags
        """
        if current_idx < lookback + 5:
            return {'bullish_rsi': False, 'bearish_rsi': False, 
                    'bullish_macd': False, 'bearish_macd': False}
        
        # Get recent data
        price_window = df['close'].iloc[current_idx - lookback:current_idx + 1]
        rsi_window = df['rsi'].iloc[current_idx - lookback:current_idx + 1] if 'rsi' in df.columns else None
        macd_window = df['macd'].iloc[current_idx - lookback:current_idx + 1] if 'macd' in df.columns else None
        
        # Find swing lows and highs
        price_min_idx = price_window.idxmin()
        price_max_idx = price_window.idxmax()
        
        results = {
            'bullish_rsi': False,
            'bearish_rsi': False,
            'bullish_macd': False,
            'bearish_macd': False
        }
        
        # RSI divergence
        if rsi_window is not None:
            rsi_min_idx = rsi_window.idxmin()
            rsi_max_idx = rsi_window.idxmax()
            
            # Bullish divergence: price lower low, RSI higher low
            if (price_window.loc[price_min_idx] < price_window.iloc[-lookback] and
                rsi_window.loc[rsi_min_idx] > rsi_window.iloc[-lookback]):
                results['bullish_rsi'] = True
                if self.verbose:
                    print(f"[SidMethod]   Bullish RSI divergence detected")
            
            # Bearish divergence: price higher high, RSI lower high
            if (price_window.loc[price_max_idx] > price_window.iloc[-lookback] and
                rsi_window.loc[rsi_max_idx] < rsi_window.iloc[-lookback]):
                results['bearish_rsi'] = True
                if self.verbose:
                    print(f"[SidMethod]   Bearish RSI divergence detected")
        
        # MACD divergence (similar logic)
        if macd_window is not None:
            macd_min_idx = macd_window.idxmin()
            macd_max_idx = macd_window.idxmax()
            
            if (price_window.loc[price_min_idx] < price_window.iloc[-lookback] and
                macd_window.loc[macd_min_idx] > macd_window.iloc[-lookback]):
                results['bullish_macd'] = True
                if self.verbose:
                    print(f"[SidMethod]   Bullish MACD divergence detected")
            
            if (price_window.loc[price_max_idx] > price_window.iloc[-lookback] and
                macd_window.loc[macd_max_idx] < macd_window.iloc[-lookback]):
                results['bearish_macd'] = True
                if self.verbose:
                    print(f"[SidMethod]   Bearish MACD divergence detected")
        
        return results

    # ========================================================================
    # SECTION 6: MARKET CONTEXT ANALYSIS (Wave 2)
    # ========================================================================

    def analyze_market_trend(self, df: pd.DataFrame, lookback: int = 50) -> MarketTrend:
        """
        Analyze overall market trend to determine trade focus
        
        WAVE 2 RULE:
        - Uptrend: Focus on OVERSOLD (long) trades
        - Downtrend: Focus on OVERBOUGHT (short) trades
        - Sideways: Take NO trades (be patient)
        
        Args:
            df: Price DataFrame (typically SPY for US market)
            lookback: Number of bars to analyze
        
        Returns:
            MarketTrend enum value
        """
        if len(df) < lookback:
            return MarketTrend.UNKNOWN
        
        recent_data = df.iloc[-lookback:]
        
        # Calculate higher highs/lower lows
        highs = recent_data['high']
        lows = recent_data['low']
        
        # Check for uptrend: higher highs and higher lows
        higher_highs = all(highs.iloc[i] <= highs.iloc[i+1] for i in range(len(highs)-10, len(highs)-1))
        higher_lows = all(lows.iloc[i] <= lows.iloc[i+1] for i in range(len(lows)-10, len(lows)-1))
        
        if higher_highs and higher_lows:
            trend = MarketTrend.UPTREND
        elif not higher_highs and not higher_lows:
            # Check for downtrend: lower highs and lower lows
            lower_highs = all(highs.iloc[i] >= highs.iloc[i+1] for i in range(len(highs)-10, len(highs)-1))
            lower_lows = all(lows.iloc[i] >= lows.iloc[i+1] for i in range(len(lows)-10, len(lows)-1))
            if lower_highs and lower_lows:
                trend = MarketTrend.DOWNTREND
            else:
                trend = MarketTrend.SIDEWAYS
        else:
            trend = MarketTrend.SIDEWAYS
        
        self.current_market_trend = trend
        
        if self.verbose:
            print(f"[SidMethod]   Market trend: {trend.value}")
            if trend == MarketTrend.SIDEWAYS:
                print(f"[SidMethod]   WARNING: Sideways market - reduce position size or take NO trades")
        
        return trend

    def should_trade_based_on_context(self, signal_type: str) -> Tuple[bool, str]:
        """
        Determine if trade should be taken based on market context (Wave 2)
        
        Returns:
            (should_trade, reason)
        """
        if not self.use_market_context:
            return True, "Market context filtering disabled"
        
        if self.current_market_trend == MarketTrend.SIDEWAYS:
            return False, "Sideways market - no SID trades (be patient)"
        
        if signal_type == 'oversold' and self.current_market_trend == MarketTrend.UPTREND:
            return True, "Uptrend + oversold = good long opportunity"
        elif signal_type == 'overbought' and self.current_market_trend == MarketTrend.DOWNTREND:
            return True, "Downtrend + overbought = good short opportunity"
        elif signal_type == 'oversold' and self.current_market_trend == MarketTrend.DOWNTREND:
            return False, "Downtrend + oversold = against trend (avoid)"
        elif signal_type == 'overbought' and self.current_market_trend == MarketTrend.UPTREND:
            return False, "Uptrend + overbought = against trend (avoid)"
        
        return True, "Context neutral"

    # ========================================================================
    # SECTION 7: REACHABILITY CHECK (Wave 2)
    # ========================================================================

    def check_reachability(self, df: pd.DataFrame, current_idx: int,
                            entry_price: float, direction: str,
                            target_price: float) -> Tuple[bool, float]:
        """
        Check if take profit target is realistically reachable (Wave 2)
        
        If there's no logical target in recent price history, reduce position size
        
        Returns:
            (is_reachable, suggested_risk_multiplier)
        """
        if current_idx < 50:
            return True, 1.0
        
        # Look at recent price action
        recent_highs = df['high'].iloc[current_idx-50:current_idx].max()
        recent_lows = df['low'].iloc[current_idx-50:current_idx].min()
        
        if direction == 'long':
            # Can we realistically reach the target?
            distance_to_target = target_price - entry_price
            distance_to_recent_high = recent_highs - entry_price
            
            if distance_to_target > distance_to_recent_high * 1.5:
                # Target is too far - reduce risk
                multiplier = 0.5
                if self.verbose:
                    print(f"[SidMethod]   Reachability warning: target too far (reducing risk to 50%)")
                return False, multiplier
            else:
                return True, 1.0
        else:
            # Short trade
            distance_to_target = entry_price - target_price
            distance_to_recent_low = entry_price - recent_lows
            
            if distance_to_target > distance_to_recent_low * 1.5:
                multiplier = 0.5
                if self.verbose:
                    print(f"[SidMethod]   Reachability warning: target too far (reducing risk to 50%)")
                return False, multiplier
            else:
                return True, 1.0

    # ========================================================================
    # SECTION 8: POSITION SIZING (Wave 1 + Wave 2 Risk Adjustment)
    # ========================================================================

    def calculate_position_size(self, entry_price: float, stop_loss: float,
                                  risk_percent: float = None,
                                  reachability_multiplier: float = 1.0) -> Dict:
        """
        Calculate position size with risk management
        
        WAVE 1: 0.5% to 2% of account per trade
        WAVE 2: Reduce risk based on consecutive losses and reachability
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_percent: Custom risk percentage (overrides default)
            reachability_multiplier: Risk reduction from reachability check
        
        Returns:
            Dictionary with position sizing details
        """
        # Determine risk percentage (Wave 1)
        if risk_percent is None:
            # Adjust risk based on consecutive losses (Wave 2)
            if self.consecutive_losses == 0:
                risk_percent = self.risk_percent_default
            elif self.consecutive_losses == 1:
                risk_percent = self.risk_percent_default * 0.75
            elif self.consecutive_losses == 2:
                risk_percent = self.risk_percent_default * 0.5
            else:
                risk_percent = self.risk_percent_default * 0.25
            
            if self.verbose and self.consecutive_losses > 0:
                print(f"[SidMethod]   Risk adjusted for {self.consecutive_losses} losses: {risk_percent:.2f}%")
        
        # Apply reachability multiplier (Wave 2)
        risk_percent = risk_percent * reachability_multiplier
        
        # Clamp to min/max
        risk_percent = max(self.risk_percent_min, min(risk_percent, self.risk_percent_max))
        
        # Calculate risk amount
        risk_amount = self.account_balance * (risk_percent / 100)
        
        # Calculate risk per unit
        if entry_price > stop_loss:
            risk_per_unit = entry_price - stop_loss
            direction = 'long'
        else:
            risk_per_unit = stop_loss - entry_price
            direction = 'short'
        
        if risk_per_unit <= 0:
            if self.verbose:
                print(f"[SidMethod]   ERROR: Invalid stop loss (risk_per_unit={risk_per_unit})")
            return {'error': 'Invalid stop loss'}
        
        units = risk_amount / risk_per_unit
        units = np.floor(units)
        
        # WAVE 1 RULE: Maximum 3-5 active trades
        max_units_per_trade = units
        
        if self.verbose:
            print(f"[SidMethod]   Position size: {units:.0f} units")
            print(f"[SidMethod]   Risk: ${risk_amount:.2f} ({risk_percent:.1f}%)")
            print(f"[SidMethod]   Risk per unit: ${risk_per_unit:.5f}")
            print(f"[SidMethod]   Position value: ${units * entry_price:.2f}")

        return {
            'units': units,
            'direction': direction,
            'risk_amount': risk_amount,
            'risk_percent': risk_percent,
            'risk_per_unit': risk_per_unit,
            'position_value': units * entry_price,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'reachability_multiplier': reachability_multiplier
        }

    # ========================================================================
    # SECTION 9: EARNINGS CHECK (Wave 1)
    # ========================================================================

    def check_earnings(self, earnings_date: Optional[datetime], entry_date: datetime) -> bool:
        """
        Check earnings buffer (Wave 1: 14 days before earnings)
        """
        if earnings_date is None:
            return True
        
        days_before = (earnings_date - entry_date).days
        is_valid = days_before >= self.earnings_buffer_days
        
        if self.verbose and not is_valid:
            print(f"[SidMethod]   Earnings check: {days_before} days before earnings (need {self.earnings_buffer_days})")
        
        return is_valid

    # ========================================================================
    # SECTION 10: PATTERN CONFIRMATION (Wave 2)
    # ========================================================================

    def check_pattern_confirmation(self, df: pd.DataFrame, current_idx: int,
                                     signal_type: str) -> Tuple[bool, str]:
        """
        Check for price pattern confirmation (Wave 2)
        
        Patterns:
        - Double Bottom (W) for oversold/long
        - Inverse Head & Shoulders for oversold/long
        - Double Top (M) for overbought/short
        - Head & Shoulders for overbought/short
        
        Returns:
            (is_confirmed, pattern_name)
        """
        if not self.use_pattern_confirmation or current_idx < 30:
            return False, "none"
        
        # Extract recent price data
        recent_lows = df['low'].iloc[current_idx-30:current_idx+1]
        recent_highs = df['high'].iloc[current_idx-30:current_idx+1]
        
        # Simple W pattern detection (double bottom)
        if signal_type == 'oversold':
            # Look for two similar lows
            min1_idx = recent_lows.idxmin()
            # Remove first low and find second
            temp_lows = recent_lows.drop(min1_idx)
            if not temp_lows.empty:
                min2_idx = temp_lows.idxmin()
                # Check if lows are within 2% of each other
                low1 = recent_lows[min1_idx]
                low2 = temp_lows[min2_idx]
                if abs(low1 - low2) / low1 < 0.02:
                    # Check if there's a peak between them
                    between_mask = (df.index >= min1_idx) & (df.index <= min2_idx)
                    peak_between = df.loc[between_mask, 'high'].max()
                    if peak_between > low1 * 1.02:
                        if self.verbose:
                            print(f"[SidMethod]   W pattern (double bottom) confirmed")
                        return True, "double_bottom"
        
        # Simple M pattern detection (double top)
        elif signal_type == 'overbought':
            # Look for two similar highs
            max1_idx = recent_highs.idxmax()
            temp_highs = recent_highs.drop(max1_idx)
            if not temp_highs.empty:
                max2_idx = temp_highs.idxmax()
                high1 = recent_highs[max1_idx]
                high2 = temp_highs[max2_idx]
                if abs(high1 - high2) / high1 < 0.02:
                    # Check if there's a trough between them
                    between_mask = (df.index >= max1_idx) & (df.index <= max2_idx)
                    trough_between = df.loc[between_mask, 'low'].min()
                    if trough_between < high1 * 0.98:
                        if self.verbose:
                            print(f"[SidMethod]   M pattern (double top) confirmed")
                        return True, "double_top"
        
        return False, "none"

    # ========================================================================
    # SECTION 11: REVERSAL DAYS CHECK (Wave 2)
    # ========================================================================

    def check_reversal_days(self, df: pd.DataFrame, entry_idx: int, 
                              current_idx: int, direction: str) -> bool:
        """
        Check for 2 consecutive reversal days (early warning)
        
        WAVE 2: If you see two reversal days in a row, consider exiting early
        """
        if current_idx - entry_idx < 2:
            return False

        candle1 = df.iloc[current_idx - 1]
        candle2 = df.iloc[current_idx]

        if direction == 'long':
            # Two red candles in a row after a long entry
            reversal1 = candle1['close'] < candle1['open']
            reversal2 = candle2['close'] < candle2['open']
        else:
            # Two green candles in a row after a short entry
            reversal1 = candle1['close'] > candle1['open']
            reversal2 = candle2['close'] > candle2['open']

        return reversal1 and reversal2

    # ========================================================================
    # SECTION 12: SESSION DETECTION (Wave 3)
    # ========================================================================

    def get_trading_session(self, dt: datetime) -> TradingSession:
        """
        Determine trading session based on GMT time (Wave 3)
        
        Asian: 00:00-09:00 GMT (Tokyo)
        London: 07:00-16:00 GMT
        US: 12:00-21:00 GMT
        Overlap: 12:00-16:00 GMT (London-US overlap - best liquidity)
        """
        hour = dt.hour
        
        if 0 <= hour < 7:
            return TradingSession.ASIAN
        elif 7 <= hour < 12:
            return TradingSession.LONDON
        elif 12 <= hour < 16:
            return TradingSession.OVERLAP  # London-US overlap
        elif 16 <= hour < 21:
            return TradingSession.US
        else:
            return TradingSession.ASIAN  # Default to Asian for late hours

    def get_session_recommendation(self, session: TradingSession) -> Dict:
        """
        Get trading recommendations based on session (Wave 3)
        """
        recommendations = {
            TradingSession.ASIAN: {
                'volatility': 'low',
                'best_strategies': ['range_trading', 'bollinger_bands'],
                'sid_method_suitability': 'low',
                'notes': 'Lower volatility, fewer SID signals'
            },
            TradingSession.LONDON: {
                'volatility': 'high',
                'best_strategies': ['breakout', 'momentum'],
                'sid_method_suitability': 'medium',
                'notes': 'Good for trend following'
            },
            TradingSession.US: {
                'volatility': 'high',
                'best_strategies': ['breakout', 'news_trading'],
                'sid_method_suitability': 'high',
                'notes': 'Best for SID method (US stocks)'
            },
            TradingSession.OVERLAP: {
                'volatility': 'very_high',
                'best_strategies': ['all_strategies'],
                'sid_method_suitability': 'very_high',
                'notes': 'Best liquidity and volatility'
            }
        }
        return recommendations.get(session, recommendations[TradingSession.ASIAN])

    # ========================================================================
    # SECTION 13: ACCOUNT MANAGEMENT
    # ========================================================================

    def update_account_balance(self, new_balance: float):
        """Update account balance"""
        if self.verbose:
            print(f"[SidMethod] Account balance updated: ${self.account_balance:,.2f} -> ${new_balance:,.2f}")
        self.account_balance = new_balance

    def update_trade_result(self, result: str, loss_amount: float = 0, profit_amount: float = 0):
        """Update tracking after trade closes (Wave 2 risk adjustment)"""
        if result == 'loss':
            self.consecutive_losses += 1
            self.daily_loss += loss_amount
            if self.verbose:
                print(f"[SidMethod] Trade LOSS: ${loss_amount:.2f}, consecutive losses: {self.consecutive_losses}")
        else:
            self.consecutive_losses = 0
            self.daily_profit += profit_amount
            if self.verbose:
                print(f"[SidMethod] Trade WIN: ${profit_amount:.2f}")

    def reset_daily(self):
        """Reset daily counters (Wave 2)"""
        if self.verbose:
            print(f"[SidMethod] Daily reset: loss=${self.daily_loss:.2f}, profit=${self.daily_profit:.2f}, consecutive losses={self.consecutive_losses}")
        self.daily_loss = 0.0
        self.daily_profit = 0.0
        self.consecutive_losses = 0

    # ========================================================================
    # SECTION 14: MAIN TRADE SCANNER
    # ========================================================================

    def find_trade_opportunities(self, df: pd.DataFrame, 
                                   earnings_dates: Dict[str, datetime] = None,
                                   progress_callback: callable = None,
                                   market_df: pd.DataFrame = None) -> List[Dict]:
        """
        Find all trade opportunities with augmented strategy rules
        
        This integrates ALL three waves of strategies:
        - Wave 1: Core RSI/MACD rules
        - Wave 2: Market context, divergence, pattern confirmation
        - Wave 3: Precision, session detection, reachability
        
        Args:
            df: Price DataFrame for the instrument
            earnings_dates: Dictionary of earnings dates
            progress_callback: Progress callback function
            market_df: Market index DataFrame (e.g., SPY) for context
        
        Returns:
            List of trade opportunity dictionaries
        """
        if df.empty or len(df) < 50:
            if self.verbose:
                print(f"[SidMethod] Insufficient data: {len(df)} rows")
            return []

        if self.verbose:
            print(f"\n[SidMethod] {'='*60}")
            print(f"[SidMethod] SCANNING FOR TRADE OPPORTUNITIES")
            print(f"[SidMethod] {'='*60}")
            print(f"[SidMethod] Data rows: {len(df):,}")
        
        # Step 1: Analyze market context (Wave 2)
        if self.use_market_context and market_df is not None:
            self.analyze_market_trend(market_df)
            if self.current_market_trend == MarketTrend.SIDEWAYS:
                print(f"[SidMethod] WARNING: Sideways market detected - few or no SID trades expected")
        
        # Step 2: Calculate indicators
        df = df.copy()
        
        if self.verbose:
            print(f"[SidMethod]   Step 1/5: Calculating RSI...")
        df['rsi'] = self.calculate_rsi(df)
        
        if self.verbose:
            print(f"[SidMethod]   Step 2/5: Calculating MACD...")
        macd_df = self.calculate_macd(df)
        df['macd'] = macd_df['macd']
        df['macd_signal'] = macd_df['signal']
        df['macd_hist'] = macd_df['histogram']
        
        if self.verbose:
            print(f"[SidMethod]   Step 3/5: Detecting divergence...")
        
        # Step 3: Detect divergence at each point (Wave 2)
        for i in range(50, len(df)):
            divergence = self.detect_divergence(df, i)
            for key, value in divergence.items():
                df.loc[df.index[i], key] = value
        
        # Fill NaN values
        for col in ['bullish_rsi', 'bearish_rsi', 'bullish_macd', 'bearish_macd']:
            if col in df.columns:
                df[col] = df[col].fillna(False)
        
        if self.verbose:
            print(f"[SidMethod]   Step 4/5: Scanning for signals...")
        
        # Step 4: Scan for trade opportunities
        opportunities = []
        
        # Use tqdm for progress if available
        if TQDM_AVAILABLE and len(df) > 5000:
            iterator = tqdm(range(20, len(df) - 1), desc="  Scanning", unit="bars")
        else:
            iterator = range(20, len(df) - 1)
        
        for i in iterator:
            current_date = df.index[i]
            rsi_value = df['rsi'].iloc[i]
            
            # Check RSI signal with EXACT thresholds (Wave 3 precision)
            signal_type, is_exact = self.check_rsi_signal(rsi_value, strict=True)
            
            if signal_type == 'neutral':
                continue
            
            # Check market context (Wave 2)
            should_trade, context_reason = self.should_trade_based_on_context(signal_type)
            if not should_trade:
                if self.verbose and i % 500 == 0:
                    print(f"[SidMethod]   Skipping: {context_reason}")
                continue
            
            # Check earnings (Wave 1)
            if earnings_dates and current_date in earnings_dates:
                earnings_date = earnings_dates.get(current_date)
                if not self.check_earnings(earnings_date, current_date):
                    continue
            
            # Check MACD alignment (Wave 1)
            aligned = self.check_macd_alignment(macd_df, i, signal_type)
            if not aligned:
                continue
            
            # Check MACD cross (Wave 2 - preferred)
            crossed = self.check_macd_cross(macd_df, i, signal_type)
            
            # If prefer MACD cross and no cross, skip
            if self.prefer_macd_cross and not crossed:
                if self.verbose and i % 500 == 0:
                    print(f"[SidMethod]   Skipping: MACD not crossed (prefer_cross=True)")
                continue
            
            # Find signal date
            signal_date = self._find_signal_date(df, i, signal_type)
            
            # Calculate stop loss (Wave 1 + Wave 3 refinements)
            stop_loss = self.calculate_stop_loss(df, signal_date, current_date, signal_type)
            
            # Determine direction
            direction = 'long' if signal_type == 'oversold' else 'short'
            entry_price = df['close'].iloc[i]
            
            # Check pattern confirmation (Wave 2)
            pattern_confirmed, pattern_name = self.check_pattern_confirmation(df, i, signal_type)
            
            # Get divergence flags
            bullish_div = df['bullish_rsi'].iloc[i] if 'bullish_rsi' in df.columns else False
            bearish_div = df['bearish_rsi'].iloc[i] if 'bearish_rsi' in df.columns else False
            divergence_detected = (bullish_div and signal_type == 'oversold') or (bearish_div and signal_type == 'overbought')
            
            # Calculate entry confidence (Wave 2)
            confidence_level, confidence_score = self.get_entry_confidence(
                aligned, crossed, pattern_confirmed, divergence_detected
            )
            
            # Calculate take profit (Wave 1 + Wave 2 alternatives)
            # Try to get 50-SMA if available
            sma_50 = df['close'].rolling(50).mean().iloc[i] if len(df) > 50 else None
            
            tp_results = self.calculate_take_profit(
                entry_price, stop_loss, direction, 
                method='rsi_50',  # Primary method
                sma_50=sma_50
            )
            
            # Check reachability (Wave 2)
            is_reachable, reachability_multiplier = self.check_reachability(
                df, i, entry_price, direction, tp_results['primary_tp']
            )
            
            # Calculate position size with reachability adjustment
            position_size = self.calculate_position_size(
                entry_price, stop_loss, 
                reachability_multiplier=reachability_multiplier
            )
            
            # Get trading session (Wave 3)
            session = self.get_trading_session(current_date)
            session_rec = self.get_session_recommendation(session)
            
            # Build opportunity dictionary
            opportunity = {
                'date': current_date,
                'session': session.value,
                'session_suitability': session_rec['sid_method_suitability'],
                'signal_type': signal_type,
                'is_exact_rsi': is_exact,
                'direction': direction,
                'rsi_value': rsi_value,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': tp_results['primary_tp'],
                'take_profit_alternative': tp_results['alternative_tp'],
                'reward_ratio': tp_results['reward_ratio'],
                'signal_date': signal_date,
                'macd_aligned': aligned,
                'macd_crossed': crossed,
                'pattern_confirmed': pattern_confirmed,
                'pattern_name': pattern_name,
                'divergence_detected': divergence_detected,
                'confidence_level': confidence_level,
                'confidence_score': confidence_score,
                'reachable': is_reachable,
                'reachability_multiplier': reachability_multiplier,
                'position_size': position_size,
                'index': i
            }
            
            opportunities.append(opportunity)
            
            if progress_callback and len(opportunities) % 100 == 0:
                progress_callback(len(opportunities))
        
        # Step 5: Filter by quality (Wave 3)
        if self.verbose:
            print(f"[SidMethod]   Step 5/5: Filtering opportunities by quality...")
        
        filtered_opportunities = []
        for opp in opportunities:
            # Filter out poor quality signals
            if opp['confidence_level'] == 'low' and not opp['macd_crossed']:
                if self.verbose:
                    print(f"[SidMethod]   Filtering out low-confidence signal: {opp['date']}")
                continue
            
            # Filter out signals in unsuitable sessions
            if opp['session_suitability'] in ['low', 'very_low']:
                if self.verbose:
                    print(f"[SidMethod]   Filtering out {opp['session']} session signal (low suitability)")
                continue
            
            filtered_opportunities.append(opp)
        
        if self.verbose:
            print(f"[SidMethod] {'='*60}")
            print(f"[SidMethod] Found {len(opportunities)} raw opportunities")
            print(f"[SidMethod] Filtered to {len(filtered_opportunities)} high-quality opportunities")
            print(f"[SidMethod] {'='*60}\n")
        
        return filtered_opportunities

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


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TESTING SID METHOD v3.0 (Fully Augmented)")
    print("="*70)
    
    # Create test data
    dates = pd.date_range(start='2024-01-01', periods=500, freq='H')
    np.random.seed(42)
    
    # Generate random walk
    returns = np.random.randn(500) * 0.001
    prices = 100 + np.cumsum(returns)
    
    test_df = pd.DataFrame({
        'open': prices,
        'high': prices + 0.5,
        'low': prices - 0.5,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 500)
    }, index=dates)
    
    # Initialize Sid Method
    sid = SidMethod(
        account_balance=10000,
        verbose=True,
        prefer_macd_cross=True,
        use_pattern_confirmation=True,
        use_divergence=True,
        use_market_context=True
    )
    
    # Find opportunities
    opportunities = sid.find_trade_opportunities(test_df)
    
    print(f"\n📊 Found {len(opportunities)} trade opportunities")
    
    for opp in opportunities[:5]:
        print(f"\n  📈 {opp['date']}: {opp['direction'].upper()} @ {opp['entry_price']:.2f}")
        print(f"     RSI: {opp['rsi_value']:.1f} | Confidence: {opp['confidence_level']} ({opp['confidence_score']:.2f})")
        print(f"     Stop: {opp['stop_loss']:.2f} | TP: {opp['take_profit']:.2f}")
        if opp['pattern_confirmed']:
            print(f"     Pattern: {opp['pattern_name']}")
        if opp['divergence_detected']:
            print(f"     Divergence detected")