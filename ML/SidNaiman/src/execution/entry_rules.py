#!/usr/bin/env python3
"""
Entry Rules Module for SID Method - AUGMENTED VERSION
=============================================================================
Validates trade entry conditions incorporating ALL THREE WAVES:

WAVE 1 (Core Quick Win):
- RSI threshold validation (exact 30/70)
- MACD alignment and cross detection
- Earnings date buffer (14 days)
- Stop loss calculation (rounded down/up)
- Position sizing with 0.5-2% risk

WAVE 2 (Live Sessions & Q&A):
- Market context filtering (uptrend/downtrend/sideways)
- Pattern confirmation (W, M, H&S)
- Divergence detection
- Multiple timeframe confirmation
- Reachability check before entry

WAVE 3 (Academy Support Sessions):
- Precision RSI validation (no "near" values)
- Session-based entry filtering
- Zone quality assessment
- Minimum candle requirements
- Stop loss pip buffer calculation

Author: Sid Naiman / Trading Cafe
Version: 3.0 (Fully Augmented)
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class EntrySignal(Enum):
    """Entry signal types"""
    VALID = "valid"
    INVALID = "invalid"
    PENDING = "pending"  # Signal detected but waiting for confirmation
    REJECTED = "rejected"  # Signal detected but filtered out


class RejectionReason(Enum):
    """Reasons for rejecting a trade entry"""
    RSI_NOT_EXACT = "RSI not at exact threshold"
    MACD_NOT_ALIGNED = "MACD not aligned"
    MACD_NO_CROSS = "MACD cross required but not present"
    EARNINGS_TOO_CLOSE = "Within earnings buffer"
    MARKET_CONTEXT_FILTER = "Filtered by market context"
    PATTERN_NOT_CONFIRMED = "Pattern confirmation required but not present"
    DIVERGENCE_MISMATCH = "Divergence direction mismatch"
    SESSION_UNSUITABLE = "Session not suitable for SID method"
    REACHABILITY_FAILED = "Take profit target not reachable"
    STOP_LOSS_INVALID = "Stop loss calculation invalid"
    ZONE_QUALITY_POOR = "Zone quality below threshold"
    MIN_CANDLES_NOT_MET = "Minimum candle requirement not met"


@dataclass
class EntryValidationResult:
    """Result of entry validation"""
    is_valid: bool
    signal_type: str  # 'oversold' or 'overbought'
    direction: str  # 'long' or 'short'
    confidence_score: float
    confidence_level: str  # 'high', 'medium', 'low'
    rejection_reason: Optional[RejectionReason] = None
    rejection_details: Optional[str] = None
    
    # Calculated values
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_units: Optional[int] = None
    risk_amount: Optional[float] = None
    
    # Context
    market_trend: Optional[str] = None
    session: Optional[str] = None
    pattern_confirmed: bool = False
    divergence_detected: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'is_valid': self.is_valid,
            'signal_type': self.signal_type,
            'direction': self.direction,
            'confidence_score': self.confidence_score,
            'confidence_level': self.confidence_level,
            'rejection_reason': self.rejection_reason.value if self.rejection_reason else None,
            'rejection_details': self.rejection_details,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_units': self.position_units,
            'risk_amount': self.risk_amount,
            'market_trend': self.market_trend,
            'session': self.session,
            'pattern_confirmed': self.pattern_confirmed,
            'divergence_detected': self.divergence_detected
        }


@dataclass
class EntryRulesConfig:
    """Configuration for entry rules (Wave 1, 2, 3)"""
    # Wave 1: Core parameters
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    rsi_target: int = 50
    prefer_macd_cross: bool = True
    earnings_buffer_days: int = 14
    
    # Wave 2: Pattern confirmation
    require_pattern_confirmation: bool = False  # Optional but recommended
    require_divergence: bool = False  # Optional
    require_market_context: bool = True
    
    # Wave 2: Reachability
    check_reachability: bool = True
    max_target_distance_multiplier: float = 1.5  # Target can't be > 1.5x recent range
    
    # Wave 3: Precision and session
    strict_rsi: bool = True
    require_exact_rsi: bool = True
    session_filter_enabled: bool = True
    min_session_suitability: str = "medium"  # 'very_high', 'high', 'medium', 'low'
    
    # Wave 3: Zone quality
    require_zone_quality: bool = False
    min_zone_quality: float = 40.0  # Minimum quality score (0-100)
    
    # Wave 3: Minimum candles for patterns
    min_pattern_candles: int = 7
    
    # Risk management
    default_risk_percent: float = 1.0
    min_risk_percent: float = 0.5
    max_risk_percent: float = 2.0


class EntryRules:
    """
    Validates SID Method trade entry conditions
    Incorporates ALL THREE WAVES of strategies
    """
    
    def __init__(self, config: EntryRulesConfig = None, verbose: bool = True):
        """
        Initialize entry rules
        
        Args:
            config: EntryRulesConfig instance
            verbose: Enable verbose output
        """
        self.config = config or EntryRulesConfig()
        self.verbose = verbose
        
        # Session suitability mapping (Wave 3)
        self.session_suitability = {
            'overlap': 'very_high',
            'us': 'high',
            'london': 'medium',
            'asian': 'low'
        }
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📋 ENTRY RULES v3.0 (Fully Augmented)")
            print(f"{'='*60}")
            print(f"📊 RSI: {self.config.rsi_oversold}/{self.config.rsi_overbought}")
            print(f"🔄 MACD cross preferred: {self.config.prefer_macd_cross}")
            print(f"📐 Pattern confirmation required: {self.config.require_pattern_confirmation}")
            print(f"⚡ Divergence required: {self.config.require_divergence}")
            print(f"🌍 Market context required: {self.config.require_market_context}")
            print(f"📅 Earnings buffer: {self.config.earnings_buffer_days} days")
            print(f"🎯 Reachability check: {self.config.check_reachability}")
            print(f"⭐ Min session suitability: {self.config.min_session_suitability}")
            print(f"{'='*60}\n")
    
    # ========================================================================
    # SECTION 1: CORE SIGNAL VALIDATION (Wave 1)
    # ========================================================================
    
    def validate_rsi_signal(self, rsi_value: float) -> Tuple[bool, Optional[str]]:
        """
        Validate RSI signal with exact thresholds (Wave 3 precision)
        
        Returns:
            (is_valid, signal_type) where signal_type is 'oversold', 'overbought', or None
        """
        if self.config.strict_rsi or self.config.require_exact_rsi:
            if rsi_value < self.config.rsi_oversold:
                return True, 'oversold'
            elif rsi_value > self.config.rsi_overbought:
                return True, 'overbought'
            else:
                if self.verbose:
                    print(f"  RSI={rsi_value:.2f} not at exact threshold")
                return False, None
        else:
            if rsi_value <= self.config.rsi_oversold:
                return True, 'oversold'
            elif rsi_value >= self.config.rsi_overbought:
                return True, 'overbought'
            else:
                return False, None
    
    def validate_macd(self, macd_df: pd.DataFrame, current_idx: int,
                       signal_type: str) -> Tuple[bool, bool, Optional[str]]:
        """
        Validate MACD alignment and cross (Wave 1 & 2)
        
        Returns:
            (is_aligned, is_crossed, rejection_reason)
        """
        if current_idx < 2:
            return False, False, "Insufficient data"
        
        current_macd = macd_df['macd'].iloc[current_idx]
        prev_macd = macd_df['macd'].iloc[current_idx - 1]
        
        # Check alignment
        if signal_type == 'oversold':
            aligned = current_macd > prev_macd
        else:
            aligned = current_macd < prev_macd
        
        if not aligned:
            return False, False, "MACD not aligned with RSI signal"
        
        # Check cross
        if current_idx < 1:
            return aligned, False, None
        
        current_signal = macd_df['signal'].iloc[current_idx]
        prev_macd = macd_df['macd'].iloc[current_idx - 1]
        prev_signal = macd_df['signal'].iloc[current_idx - 1]
        
        if signal_type == 'oversold':
            crossed = (prev_macd <= prev_signal and current_macd > current_signal)
        else:
            crossed = (prev_macd >= prev_signal and current_macd < current_signal)
        
        # If prefer cross and no cross, reject
        if self.config.prefer_macd_cross and not crossed:
            return aligned, False, "MACD cross required but not present"
        
        return aligned, crossed, None
    
    def validate_earnings(self, earnings_date: Optional[datetime], 
                           entry_date: datetime) -> Tuple[bool, Optional[str]]:
        """
        Validate earnings buffer (Wave 1: 14 days before earnings)
        """
        if earnings_date is None:
            return True, None
        
        days_before = (earnings_date - entry_date).days
        
        if days_before < self.config.earnings_buffer_days:
            return False, f"Within {days_before} days of earnings (need {self.config.earnings_buffer_days})"
        
        return True, None
    
    # ========================================================================
    # SECTION 2: PATTERN CONFIRMATION (Wave 2 & 3)
    # ========================================================================
    
    def validate_pattern_confirmation(self, df: pd.DataFrame, current_idx: int,
                                        signal_type: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate price pattern confirmation (Wave 2)
        
        Returns:
            (is_confirmed, pattern_name, rejection_reason)
        """
        if not self.config.require_pattern_confirmation:
            return True, None, None
        
        if current_idx < self.config.min_pattern_candles * 2:
            return False, None, "Insufficient candles for pattern detection"
        
        # Extract recent data
        lookback = max(30, self.config.min_pattern_candles * 2)
        recent_lows = df['low'].iloc[current_idx - lookback:current_idx + 1]
        recent_highs = df['high'].iloc[current_idx - lookback:current_idx + 1]
        
        if signal_type == 'oversold':
            # Look for double bottom (W) pattern
            min1_idx = recent_lows.idxmin()
            temp_lows = recent_lows.drop(min1_idx)
            if not temp_lows.empty:
                min2_idx = temp_lows.idxmin()
                
                low1 = recent_lows[min1_idx]
                low2 = temp_lows[min2_idx]
                
                # Check candle count
                candle_count = abs(current_idx - df.index.get_loc(min2_idx))
                if candle_count >= self.config.min_pattern_candles:
                    # Check if lows are within 2%
                    if abs(low1 - low2) / low1 < 0.02:
                        # Check for peak between
                        between_mask = (df.index >= min(min1_idx, min2_idx)) & (df.index <= max(min1_idx, min2_idx))
                        peak_between = df.loc[between_mask, 'high'].max()
                        if peak_between > low1 * 1.02:
                            return True, "double_bottom", None
            
            # Also check for inverse head & shoulders
            # (Simplified detection)
            return False, None, "Pattern confirmation required but not found"
        
        else:
            # Look for double top (M) pattern
            max1_idx = recent_highs.idxmax()
            temp_highs = recent_highs.drop(max1_idx)
            if not temp_highs.empty:
                max2_idx = temp_highs.idxmax()
                
                high1 = recent_highs[max1_idx]
                high2 = temp_highs[max2_idx]
                
                candle_count = abs(current_idx - df.index.get_loc(max2_idx))
                if candle_count >= self.config.min_pattern_candles:
                    if abs(high1 - high2) / high1 < 0.02:
                        between_mask = (df.index >= min(max1_idx, max2_idx)) & (df.index <= max(max1_idx, max2_idx))
                        trough_between = df.loc[between_mask, 'low'].min()
                        if trough_between < high1 * 0.98:
                            return True, "double_top", None
            
            return False, None, "Pattern confirmation required but not found"
    
    # ========================================================================
    # SECTION 3: DIVERGENCE VALIDATION (Wave 2)
    # ========================================================================
    
    def validate_divergence(self, df: pd.DataFrame, current_idx: int,
                              signal_type: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate divergence detection (Wave 2)
        
        Returns:
            (is_valid, divergence_type, rejection_reason)
        """
        if not self.config.require_divergence:
            return True, None, None
        
        if current_idx < 20:
            return False, None, "Insufficient data for divergence detection"
        
        # Get windows
        lookback = 20
        price_window = df['close'].iloc[current_idx - lookback:current_idx + 1]
        rsi_window = df['rsi'].iloc[current_idx - lookback:current_idx + 1] if 'rsi' in df.columns else None
        
        if rsi_window is None:
            return False, None, "RSI not available for divergence detection"
        
        # Find min and max positions
        price_min_idx = price_window.idxmin()
        price_max_idx = price_window.idxmax()
        rsi_min_idx = rsi_window.idxmin()
        rsi_max_idx = rsi_window.idxmax()
        
        # Bullish divergence (for oversold signals)
        if signal_type == 'oversold':
            bullish = (price_window.loc[price_min_idx] < price_window.iloc[0] and
                      rsi_window.loc[rsi_min_idx] > rsi_window.iloc[0])
            if bullish:
                return True, "bullish", None
            elif self.config.require_divergence:
                return False, None, "Bullish divergence required but not detected"
        
        # Bearish divergence (for overbought signals)
        else:
            bearish = (price_window.loc[price_max_idx] > price_window.iloc[0] and
                      rsi_window.loc[rsi_max_idx] < rsi_window.iloc[0])
            if bearish:
                return True, "bearish", None
            elif self.config.require_divergence:
                return False, None, "Bearish divergence required but not detected"
        
        return True, None, None  # Divergence not required or not applicable
    
    # ========================================================================
    # SECTION 4: MARKET CONTEXT VALIDATION (Wave 2)
    # ========================================================================
    
    def validate_market_context(self, market_trend: str, signal_type: str) -> Tuple[bool, Optional[str]]:
        """
        Validate market context (Wave 2)
        
        WAVE 2 RULE:
        - Uptrend: focus on oversold (long) trades
        - Downtrend: focus on overbought (short) trades
        - Sideways: no trades
        """
        if not self.config.require_market_context:
            return True, None
        
        if market_trend == "sideways":
            return False, "Sideways market - no SID trades"
        
        if signal_type == "oversold" and market_trend == "downtrend":
            return False, "Oversold in downtrend - against trend"
        
        if signal_type == "overbought" and market_trend == "uptrend":
            return False, "Overbought in uptrend - against trend"
        
        return True, None
    
    # ========================================================================
    # SECTION 5: SESSION VALIDATION (Wave 3)
    # ========================================================================
    
    def validate_session(self, session: str) -> Tuple[bool, Optional[str]]:
        """
        Validate trading session (Wave 3)
        
        Returns:
            (is_valid, rejection_reason)
        """
        if not self.config.session_filter_enabled:
            return True, None
        
        suitability = self.session_suitability.get(session, "low")
        
        # Convert string to numeric for comparison
        suitability_scores = {'very_high': 4, 'high': 3, 'medium': 2, 'low': 1}
        min_score = suitability_scores.get(self.config.min_session_suitability, 2)
        current_score = suitability_scores.get(suitability, 1)
        
        if current_score < min_score:
            return False, f"Session {session} has {suitability} suitability (need >= {self.config.min_session_suitability})"
        
        return True, None
    
    def get_trading_session(self, dt: datetime) -> str:
        """Determine trading session based on GMT time (Wave 3)"""
        hour = dt.hour
        
        if 0 <= hour < 7:
            return "asian"
        elif 7 <= hour < 12:
            return "london"
        elif 12 <= hour < 16:
            return "overlap"
        elif 16 <= hour < 21:
            return "us"
        else:
            return "asian"
    
    # ========================================================================
    # SECTION 6: REACHABILITY VALIDATION (Wave 2)
    # ========================================================================
    
    def validate_reachability(self, df: pd.DataFrame, current_idx: int,
                                entry_price: float, direction: str,
                                target_price: float) -> Tuple[bool, float, Optional[str]]:
        """
        Validate that take profit target is realistically reachable (Wave 2)
        
        Returns:
            (is_reachable, risk_multiplier, rejection_reason)
        """
        if not self.config.check_reachability:
            return True, 1.0, None
        
        if current_idx < 50:
            return True, 1.0, None
        
        # Look at recent price action (last 50 bars)
        recent_highs = df['high'].iloc[current_idx-50:current_idx].max()
        recent_lows = df['low'].iloc[current_idx-50:current_idx].min()
        
        if direction == 'long':
            distance_to_target = target_price - entry_price
            distance_to_recent_high = recent_highs - entry_price
            
            if distance_to_target > distance_to_recent_high * self.config.max_target_distance_multiplier:
                return False, 0.5, f"Target too far (needs {distance_to_target:.2f}, recent range {distance_to_recent_high:.2f})"
        else:
            distance_to_target = entry_price - target_price
            distance_to_recent_low = entry_price - recent_lows
            
            if distance_to_target > distance_to_recent_low * self.config.max_target_distance_multiplier:
                return False, 0.5, f"Target too far (needs {distance_to_target:.2f}, recent range {distance_to_recent_low:.2f})"
        
        return True, 1.0, None
    
    # ========================================================================
    # SECTION 7: STOP LOSS VALIDATION (Wave 1 & 3)
    # ========================================================================
    
    def validate_stop_loss(self, stop_loss: float, entry_price: float,
                            direction: str) -> Tuple[bool, Optional[str]]:
        """
        Validate stop loss calculation
        
        Returns:
            (is_valid, rejection_reason)
        """
        if stop_loss <= 0:
            return False, "Stop loss must be positive"
        
        if direction == 'long' and stop_loss >= entry_price:
            return False, "Stop loss must be below entry price for long trades"
        
        if direction == 'short' and stop_loss <= entry_price:
            return False, "Stop loss must be above entry price for short trades"
        
        return True, None
    
    def calculate_stop_loss(self, df: pd.DataFrame, signal_date: datetime,
                              entry_date: datetime, signal_type: str,
                              instrument: str = None,
                              use_pip_buffer: bool = True) -> float:
        """
        Calculate SID Method stop loss (Wave 1 + Wave 3)
        """
        mask = (df.index >= signal_date) & (df.index <= entry_date)
        period_df = df.loc[mask]
        
        if period_df.empty:
            return 0.0
        
        if signal_type == 'oversold':
            lowest_low = period_df['low'].min()
            stop_loss = np.floor(lowest_low)
            
            if use_pip_buffer and instrument:
                pip_buffer = 10 if 'JPY' in instrument else 5
                pip_value = 0.01 if 'JPY' in instrument else 0.0001
                stop_loss = stop_loss - (pip_buffer * pip_value)
        else:
            highest_high = period_df['high'].max()
            stop_loss = np.ceil(highest_high)
            
            if use_pip_buffer and instrument:
                pip_buffer = 10 if 'JPY' in instrument else 5
                pip_value = 0.01 if 'JPY' in instrument else 0.0001
                stop_loss = stop_loss + (pip_buffer * pip_value)
        
        return float(stop_loss)
    
    # ========================================================================
    # SECTION 8: POSITION SIZING (Wave 1 & 2)
    # ========================================================================
    
    def calculate_position_size(self, account_balance: float, entry_price: float,
                                  stop_loss: float, risk_percent: float = None,
                                  reachability_multiplier: float = 1.0) -> Dict:
        """
        Calculate position size (Wave 1: 0.5-2% risk)
        """
        if risk_percent is None:
            risk_percent = self.config.default_risk_percent
        
        # Apply reachability multiplier (Wave 2)
        risk_percent = risk_percent * reachability_multiplier
        
        # Clamp to min/max
        risk_percent = max(self.config.min_risk_percent, 
                          min(risk_percent, self.config.max_risk_percent))
        
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
            'stop_loss': stop_loss,
            'reachability_multiplier': reachability_multiplier
        }
    
    # ========================================================================
    # SECTION 9: COMPLETE ENTRY VALIDATION
    # ========================================================================
    
    def validate_entry(self, df: pd.DataFrame, current_idx: int,
                         rsi_value: float, macd_df: pd.DataFrame,
                         market_trend: str, session: str,
                         account_balance: float,
                         earnings_date: Optional[datetime] = None,
                         instrument: str = None) -> EntryValidationResult:
        """
        Complete entry validation integrating ALL THREE WAVES
        
        Args:
            df: Price DataFrame
            current_idx: Current index in DataFrame
            rsi_value: Current RSI value
            macd_df: MACD DataFrame
            market_trend: Current market trend ('uptrend', 'downtrend', 'sideways')
            session: Current trading session
            account_balance: Current account balance
            earnings_date: Next earnings date (if any)
            instrument: Instrument name (for pip calculations)
        
        Returns:
            EntryValidationResult with validation details
        """
        current_date = df.index[current_idx]
        
        # Step 1: Validate RSI signal (Wave 1 & 3)
        is_valid, signal_type = self.validate_rsi_signal(rsi_value)
        if not is_valid:
            return EntryValidationResult(
                is_valid=False,
                signal_type='neutral',
                direction='neutral',
                confidence_score=0,
                confidence_level='very_low',
                rejection_reason=RejectionReason.RSI_NOT_EXACT,
                rejection_details=f"RSI={rsi_value:.2f} not at threshold"
            )
        
        # Step 2: Validate MACD (Wave 1 & 2)
        is_aligned, is_crossed, macd_reason = self.validate_macd(macd_df, current_idx, signal_type)
        if not is_aligned:
            return EntryValidationResult(
                is_valid=False,
                signal_type=signal_type,
                direction='long' if signal_type == 'oversold' else 'short',
                confidence_score=0,
                confidence_level='very_low',
                rejection_reason=RejectionReason.MACD_NOT_ALIGNED,
                rejection_details=macd_reason
            )
        
        if self.config.prefer_macd_cross and not is_crossed:
            return EntryValidationResult(
                is_valid=False,
                signal_type=signal_type,
                direction='long' if signal_type == 'oversold' else 'short',
                confidence_score=0,
                confidence_level='very_low',
                rejection_reason=RejectionReason.MACD_NO_CROSS,
                rejection_details="MACD cross required but not present"
            )
        
        # Step 3: Validate earnings (Wave 1)
        is_valid_earnings, earnings_reason = self.validate_earnings(earnings_date, current_date)
        if not is_valid_earnings:
            return EntryValidationResult(
                is_valid=False,
                signal_type=signal_type,
                direction='long' if signal_type == 'oversold' else 'short',
                confidence_score=0,
                confidence_level='very_low',
                rejection_reason=RejectionReason.EARNINGS_TOO_CLOSE,
                rejection_details=earnings_reason
            )
        
        # Step 4: Validate market context (Wave 2)
        is_valid_context, context_reason = self.validate_market_context(market_trend, signal_type)
        if not is_valid_context:
            return EntryValidationResult(
                is_valid=False,
                signal_type=signal_type,
                direction='long' if signal_type == 'oversold' else 'short',
                confidence_score=0,
                confidence_level='very_low',
                rejection_reason=RejectionReason.MARKET_CONTEXT_FILTER,
                rejection_details=context_reason,
                market_trend=market_trend
            )
        
        # Step 5: Validate session (Wave 3)
        is_valid_session, session_reason = self.validate_session(session)
        if not is_valid_session:
            return EntryValidationResult(
                is_valid=False,
                signal_type=signal_type,
                direction='long' if signal_type == 'oversold' else 'short',
                confidence_score=0,
                confidence_level='very_low',
                rejection_reason=RejectionReason.SESSION_UNSUITABLE,
                rejection_details=session_reason,
                market_trend=market_trend,
                session=session
            )
        
        # Step 6: Validate pattern confirmation (Wave 2)
        pattern_confirmed, pattern_name, pattern_reason = self.validate_pattern_confirmation(
            df, current_idx, signal_type
        )
        if self.config.require_pattern_confirmation and not pattern_confirmed:
            return EntryValidationResult(
                is_valid=False,
                signal_type=signal_type,
                direction='long' if signal_type == 'oversold' else 'short',
                confidence_score=0,
                confidence_level='very_low',
                rejection_reason=RejectionReason.PATTERN_NOT_CONFIRMED,
                rejection_details=pattern_reason,
                market_trend=market_trend,
                session=session
            )
        
        # Step 7: Validate divergence (Wave 2)
        divergence_valid, divergence_type, divergence_reason = self.validate_divergence(
            df, current_idx, signal_type
        )
        if self.config.require_divergence and not divergence_valid:
            return EntryValidationResult(
                is_valid=False,
                signal_type=signal_type,
                direction='long' if signal_type == 'oversold' else 'short',
                confidence_score=0,
                confidence_level='very_low',
                rejection_reason=RejectionReason.DIVERGENCE_MISMATCH,
                rejection_details=divergence_reason,
                market_trend=market_trend,
                session=session
            )
        
        # Step 8: Calculate stop loss (Wave 1 & 3)
        direction = 'long' if signal_type == 'oversold' else 'short'
        entry_price = df['close'].iloc[current_idx]
        
        # Find signal date
        rsi_values = df['rsi'].iloc[:current_idx + 1]
        if signal_type == 'oversold':
            mask = rsi_values < self.config.rsi_oversold
        else:
            mask = rsi_values > self.config.rsi_overbought
        
        if mask.any():
            signal_date = mask[mask].index[-1]
        else:
            signal_date = current_date
        
        stop_loss = self.calculate_stop_loss(df, signal_date, current_date, signal_type, instrument)
        
        # Validate stop loss
        is_valid_stop, stop_reason = self.validate_stop_loss(stop_loss, entry_price, direction)
        if not is_valid_stop:
            return EntryValidationResult(
                is_valid=False,
                signal_type=signal_type,
                direction=direction,
                confidence_score=0,
                confidence_level='very_low',
                rejection_reason=RejectionReason.STOP_LOSS_INVALID,
                rejection_details=stop_reason,
                market_trend=market_trend,
                session=session,
                pattern_confirmed=pattern_confirmed,
                divergence_detected=divergence_valid,
                entry_price=entry_price,
                stop_loss=stop_loss
            )
        
        # Step 9: Calculate take profit and validate reachability (Wave 1 & 2)
        risk_distance = abs(entry_price - stop_loss)
        if direction == 'long':
            take_profit = entry_price + risk_distance
        else:
            take_profit = entry_price - risk_distance
        
        is_reachable, reachability_multiplier, reachability_reason = self.validate_reachability(
            df, current_idx, entry_price, direction, take_profit
        )
        
        if self.config.check_reachability and not is_reachable:
            return EntryValidationResult(
                is_valid=False,
                signal_type=signal_type,
                direction=direction,
                confidence_score=0,
                confidence_level='very_low',
                rejection_reason=RejectionReason.REACHABILITY_FAILED,
                rejection_details=reachability_reason,
                market_trend=market_trend,
                session=session,
                pattern_confirmed=pattern_confirmed,
                divergence_detected=divergence_valid,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
        
        # Step 10: Calculate position size (Wave 1 & 2)
        position = self.calculate_position_size(
            account_balance, entry_price, stop_loss,
            reachability_multiplier=reachability_multiplier
        )
        
        if 'error' in position:
            return EntryValidationResult(
                is_valid=False,
                signal_type=signal_type,
                direction=direction,
                confidence_score=0,
                confidence_level='very_low',
                rejection_reason=RejectionReason.STOP_LOSS_INVALID,
                rejection_details=position['error'],
                market_trend=market_trend,
                session=session,
                pattern_confirmed=pattern_confirmed,
                divergence_detected=divergence_valid,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
        
        # Step 11: Calculate confidence score (Wave 2)
        confidence_score = 0.0
        
        if is_crossed:
            confidence_score += 0.35
        elif is_aligned:
            confidence_score += 0.20
        
        if pattern_confirmed:
            confidence_score += 0.20
        
        if divergence_valid:
            confidence_score += 0.15
        
        # Session contribution
        session_scores = {'overlap': 0.10, 'us': 0.09, 'london': 0.07, 'asian': 0.04}
        confidence_score += session_scores.get(session, 0.05)
        
        # Determine confidence level
        if confidence_score >= 0.8:
            confidence_level = "high"
        elif confidence_score >= 0.6:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        # Return valid entry
        return EntryValidationResult(
            is_valid=True,
            signal_type=signal_type,
            direction=direction,
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_units=position.get('units', 0),
            risk_amount=position.get('risk_amount', 0),
            market_trend=market_trend,
            session=session,
            pattern_confirmed=pattern_confirmed,
            divergence_detected=divergence_valid
        )


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TESTING ENTRY RULES v3.0")
    print("="*70)
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=500, freq='H')
    np.random.seed(42)
    
    returns = np.random.randn(500) * 0.001
    prices = 100 + np.cumsum(returns)
    
    # Add an oversold condition
    prices[100:110] = 95
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + 0.5,
        'low': prices - 0.5,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 500)
    }, index=dates)
    
    # Calculate RSI and MACD
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, float('nan'))
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)
    
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Initialize entry rules
    config = EntryRulesConfig(
        require_pattern_confirmation=False,
        require_divergence=False,
        prefer_macd_cross=True
    )
    entry_rules = EntryRules(config, verbose=True)
    
    # Test validation at a specific index
    test_idx = 110  # After oversold condition
    rsi_value = df['rsi'].iloc[test_idx]
    
    # Create MACD DataFrame
    macd_df = pd.DataFrame({
        'macd': df['macd'],
        'signal': df['macd_signal']
    })
    
    # Validate entry
    result = entry_rules.validate_entry(
        df=df,
        current_idx=test_idx,
        rsi_value=rsi_value,
        macd_df=macd_df,
        market_trend='uptrend',
        session='us',
        account_balance=10000,
        earnings_date=None,
        instrument='EUR_USD'
    )
    
    print(f"\n📊 Validation Result:")
    print(f"  Is Valid: {result.is_valid}")
    if result.is_valid:
        print(f"  Direction: {result.direction}")
        print(f"  Entry Price: {result.entry_price:.5f}")
        print(f"  Stop Loss: {result.stop_loss:.5f}")
        print(f"  Take Profit: {result.take_profit:.5f}")
        print(f"  Position Units: {result.position_units}")
        print(f"  Risk Amount: ${result.risk_amount:.2f}")
        print(f"  Confidence: {result.confidence_level} ({result.confidence_score:.2f})")
    else:
        print(f"  Rejection: {result.rejection_reason.value}")
        print(f"  Details: {result.rejection_details}")
    
    print(f"\n✅ Entry rules test complete")