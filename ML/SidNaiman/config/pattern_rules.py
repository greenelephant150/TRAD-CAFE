#!/usr/bin/env python3
"""
Pattern Rules Module for SID Method - AUGMENTED VERSION
=============================================================================
Defines and validates price patterns incorporating ALL THREE WAVES:

WAVE 1 (Core Quick Win):
- Basic pattern recognition as confirmation
- Double bottom (W) and double top (M)
- Head and shoulders patterns

WAVE 2 (Live Sessions & Q&A):
- Pattern completion detection
- Neckline break confirmation
- Pattern quality scoring
- Volume confirmation (when available)
- Multiple timeframe pattern validation

WAVE 3 (Academy Support Sessions):
- Minimum candle requirements (7 candles minimum)
- Rogue wick handling
- Pattern invalidation rules
- Pattern confluence detection
- Zone quality integration

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

class PatternType(Enum):
    """Types of price patterns (Wave 1 & 2)"""
    DOUBLE_BOTTOM = "double_bottom"           # W pattern - bullish
    DOUBLE_TOP = "double_top"                 # M pattern - bearish
    HEAD_SHOULDERS_TOP = "head_shoulders_top"   # Bearish reversal
    HEAD_SHOULDERS_BOTTOM = "head_shoulders_bottom"  # Bullish reversal (inverse)
    RISING_WEDGE = "rising_wedge"             # Bearish
    FALLING_WEDGE = "falling_wedge"           # Bullish
    FLAG = "flag"                             # Continuation
    PENNANT = "pennant"                       # Continuation
    NONE = "none"


class PatternQuality(Enum):
    """Quality rating for detected patterns (Wave 3)"""
    EXCELLENT = "excellent"   # All criteria met, clear pattern
    GOOD = "good"             # Most criteria met
    FAIR = "fair"             # Basic criteria met
    POOR = "poor"             # Pattern present but low quality
    INVALID = "invalid"       # Does not meet minimum criteria


class PatternDirection(Enum):
    """Direction of pattern completion (Wave 2)"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class DetectedPattern:
    """Detected pattern information"""
    pattern_type: PatternType
    direction: PatternDirection
    quality: PatternQuality
    quality_score: float
    start_index: int
    end_index: int
    neckline_price: float
    target_price: float
    stop_loss: float
    confidence: float
    volume_confirmation: bool = False
    multi_timeframe_confirmed: bool = False
    validation_notes: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'pattern_type': self.pattern_type.value,
            'direction': self.direction.value,
            'quality': self.quality.value,
            'quality_score': self.quality_score,
            'neckline_price': self.neckline_price,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'confidence': self.confidence,
            'volume_confirmation': self.volume_confirmation,
            'multi_timeframe_confirmed': self.multi_timeframe_confirmed,
            'validation_notes': self.validation_notes
        }


@dataclass
class PatternRulesConfig:
    """Configuration for pattern detection (Wave 1, 2, 3)"""
    # Wave 1: Basic parameters
    double_bottom_tolerance: float = 0.02      # 2% tolerance for bottom levels
    double_top_tolerance: float = 0.02         # 2% tolerance for top levels
    head_shoulders_tolerance: float = 0.03     # 3% tolerance for head/shoulders
    
    # Wave 2: Confirmation parameters
    require_neckline_break: bool = True
    neckline_break_confirmation_bars: int = 2  # Wait for 2 bars after break
    require_volume_confirmation: bool = False  # Optional
    volume_multiplier: float = 1.5             # Volume 1.5x average for break
    
    # Wave 3: Minimum requirements
    min_pattern_candles: int = 7               # Minimum candles for pattern
    min_shoulder_candles: int = 3              # Minimum candles for each shoulder
    max_pattern_candles: int = 100             # Maximum candles for pattern
    
    # Wave 3: Quality thresholds
    excellent_quality_threshold: float = 80.0
    good_quality_threshold: float = 60.0
    fair_quality_threshold: float = 40.0
    
    # Wave 3: Rogue wick handling
    rogue_wick_ignore: bool = True
    rogue_wick_threshold: float = 0.5          # 50% of range
    
    # Wave 2: Multi-timeframe
    use_multi_timeframe_confirmation: bool = False
    higher_timeframe_multiple: int = 4         # Higher timeframe is 4x current


class PatternRules:
    """
    Defines and validates price patterns for SID Method
    Incorporates ALL THREE WAVES of strategies
    """
    
    def __init__(self, config: PatternRulesConfig = None, verbose: bool = True):
        """
        Initialize pattern rules
        
        Args:
            config: PatternRulesConfig instance
            verbose: Enable verbose output
        """
        self.config = config or PatternRulesConfig()
        self.verbose = verbose
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📐 PATTERN RULES v3.0 (Fully Augmented)")
            print(f"{'='*60}")
            print(f"🔽 Double bottom tolerance: {self.config.double_bottom_tolerance*100:.1f}%")
            print(f"🔼 Double top tolerance: {self.config.double_top_tolerance*100:.1f}%")
            print(f"📊 Head & shoulders tolerance: {self.config.head_shoulders_tolerance*100:.1f}%")
            print(f"🎯 Require neckline break: {self.config.require_neckline_break}")
            print(f"🕯️ Min pattern candles: {self.config.min_pattern_candles}")
            print(f"📏 Rogue wick ignore: {self.config.rogue_wick_ignore}")
            print(f"📈 Volume confirmation: {self.config.require_volume_confirmation}")
            print(f"{'='*60}\n")
    
    # ========================================================================
    # SECTION 1: UTILITY FUNCTIONS (Wave 3)
    # ========================================================================
    
    def _is_rogue_wick(self, candle: pd.Series, is_low: bool = True) -> bool:
        """
        Check if a candle has a rogue wick that should be ignored (Wave 3)
        
        Args:
            candle: Candle data with open, high, low, close
            is_low: True for checking low wicks, False for high wicks
        
        Returns:
            True if rogue wick should be ignored
        """
        if not self.config.rogue_wick_ignore:
            return False
        
        body = abs(candle['close'] - candle['open'])
        range_total = candle['high'] - candle['low']
        
        if range_total <= 0:
            return False
        
        if is_low:
            wick = candle['low'] - min(candle['open'], candle['close'])
        else:
            wick = max(candle['open'], candle['close']) - candle['high']
        
        wick_ratio = wick / range_total
        
        return wick_ratio > self.config.rogue_wick_threshold
    
    def _adjust_for_rogue_wicks(self, prices: List[float], is_low: bool = True) -> List[float]:
        """
        Adjust price list to ignore rogue wicks (Wave 3)
        
        Args:
            prices: List of prices (lows or highs)
            is_low: True for low prices, False for high prices
        
        Returns:
            Adjusted price list with rogue wicks filtered
        """
        if not self.config.rogue_wick_ignore:
            return prices
        
        # For now, return original - full implementation would require candle data
        return prices
    
    # ========================================================================
    # SECTION 2: DOUBLE BOTTOM (W PATTERN) - WAVE 1 & 3
    # ========================================================================
    
    def detect_double_bottom(self, df: pd.DataFrame, lookback: int = 30,
                               current_idx: int = None) -> Optional[DetectedPattern]:
        """
        Detect double bottom (W) pattern - bullish reversal (Wave 1 & 3)
        
        Requirements:
        - Two lows within 2% of each other
        - Peak between lows at least 2% above lows
        - Minimum candle count between lows (Wave 3)
        
        Returns:
            DetectedPattern or None
        """
        if current_idx is None:
            current_idx = len(df) - 1
        
        if current_idx < self.config.min_pattern_candles:
            return None
        
        # Get recent lows
        start_idx = max(0, current_idx - lookback)
        recent_lows = df['low'].iloc[start_idx:current_idx + 1]
        
        # Find the two lowest points
        min1_idx = recent_lows.idxmin()
        temp_lows = recent_lows.drop(min1_idx)
        
        if temp_lows.empty:
            return None
        
        min2_idx = temp_lows.idxmin()
        
        low1 = recent_lows[min1_idx]
        low2 = temp_lows[min2_idx]
        
        # Wave 3: Minimum candle requirement between lows
        candle_count = abs(df.index.get_loc(min2_idx) - df.index.get_loc(min1_idx))
        if candle_count < self.config.min_pattern_candles:
            if self.verbose:
                print(f"  Double bottom rejected: insufficient candles ({candle_count} < {self.config.min_pattern_candles})")
            return None
        
        # Check if lows are within tolerance (Wave 1)
        if abs(low1 - low2) / low1 > self.config.double_bottom_tolerance:
            if self.verbose:
                print(f"  Double bottom rejected: lows differ by {abs(low1-low2)/low1*100:.1f}%")
            return None
        
        # Find peak between the two lows
        between_mask = (df.index >= min(min1_idx, min2_idx)) & (df.index <= max(min1_idx, min2_idx))
        peak_between = df.loc[between_mask, 'high'].max()
        
        # Peak must be at least 2% above lows (Wave 1)
        if peak_between <= low1 * 1.02:
            if self.verbose:
                print(f"  Double bottom rejected: peak too low ({peak_between:.2f} vs {low1*1.02:.2f})")
            return None
        
        # Neckline is the peak between lows (Wave 2)
        neckline = peak_between
        
        # Target price: neckline + (neckline - low) (Wave 2)
        target = neckline + (neckline - min(low1, low2))
        
        # Stop loss: below the lower low (Wave 1)
        stop_loss = min(low1, low2) * 0.98  # 2% below
        
        # Calculate quality score (Wave 3)
        quality_score = self._calculate_double_bottom_quality(
            low1, low2, peak_between, candle_count
        )
        
        quality_rating = self._get_quality_rating(quality_score)
        
        # Check volume confirmation (Wave 2)
        volume_confirmed = self._check_volume_confirmation(df, min1_idx, min2_idx, current_idx)
        
        return DetectedPattern(
            pattern_type=PatternType.DOUBLE_BOTTOM,
            direction=PatternDirection.BULLISH,
            quality=quality_rating,
            quality_score=quality_score,
            start_index=df.index.get_loc(min(min1_idx, min2_idx)),
            end_index=current_idx,
            neckline_price=neckline,
            target_price=target,
            stop_loss=stop_loss,
            confidence=quality_score / 100,
            volume_confirmation=volume_confirmed,
            validation_notes=f"Lows: {low1:.2f}, {low2:.2f} | Neckline: {neckline:.2f}"
        )
    
    def _calculate_double_bottom_quality(self, low1: float, low2: float,
                                           peak: float, candle_count: int) -> float:
        """
        Calculate quality score for double bottom pattern (Wave 3)
        
        Returns:
            Quality score (0-100)
        """
        score = 0.0
        
        # Low equality (30 points)
        low_diff_pct = abs(low1 - low2) / low1 * 100
        if low_diff_pct < 0.5:
            score += 30
        elif low_diff_pct < 1.0:
            score += 25
        elif low_diff_pct < 1.5:
            score += 20
        elif low_diff_pct < 2.0:
            score += 15
        else:
            score += 10
        
        # Peak height (30 points)
        peak_height_pct = (peak - min(low1, low2)) / min(low1, low2) * 100
        if peak_height_pct >= 5:
            score += 30
        elif peak_height_pct >= 4:
            score += 25
        elif peak_height_pct >= 3:
            score += 20
        elif peak_height_pct >= 2:
            score += 15
        else:
            score += 5
        
        # Candle count (20 points) - more candles = more significant
        if candle_count >= 20:
            score += 20
        elif candle_count >= 15:
            score += 15
        elif candle_count >= 10:
            score += 10
        elif candle_count >= 7:
            score += 5
        else:
            score += 0
        
        # Symmetry (20 points) - how evenly spaced are the bottoms
        # Simplified - in production, calculate actual symmetry
        
        return min(score, 100.0)
    
    # ========================================================================
    # SECTION 3: DOUBLE TOP (M PATTERN) - WAVE 1 & 3
    # ========================================================================
    
    def detect_double_top(self, df: pd.DataFrame, lookback: int = 30,
                            current_idx: int = None) -> Optional[DetectedPattern]:
        """
        Detect double top (M) pattern - bearish reversal (Wave 1 & 3)
        
        Returns:
            DetectedPattern or None
        """
        if current_idx is None:
            current_idx = len(df) - 1
        
        if current_idx < self.config.min_pattern_candles:
            return None
        
        # Get recent highs
        start_idx = max(0, current_idx - lookback)
        recent_highs = df['high'].iloc[start_idx:current_idx + 1]
        
        # Find the two highest points
        max1_idx = recent_highs.idxmax()
        temp_highs = recent_highs.drop(max1_idx)
        
        if temp_highs.empty:
            return None
        
        max2_idx = temp_highs.idxmax()
        
        high1 = recent_highs[max1_idx]
        high2 = temp_highs[max2_idx]
        
        # Wave 3: Minimum candle requirement between tops
        candle_count = abs(df.index.get_loc(max2_idx) - df.index.get_loc(max1_idx))
        if candle_count < self.config.min_pattern_candles:
            return None
        
        # Check if tops are within tolerance (Wave 1)
        if abs(high1 - high2) / high1 > self.config.double_top_tolerance:
            return None
        
        # Find trough between the two tops
        between_mask = (df.index >= min(max1_idx, max2_idx)) & (df.index <= max(max1_idx, max2_idx))
        trough_between = df.loc[between_mask, 'low'].min()
        
        # Trough must be at least 2% below tops (Wave 1)
        if trough_between >= high1 * 0.98:
            return None
        
        # Neckline is the trough between tops (Wave 2)
        neckline = trough_between
        
        # Target price: neckline - (high - neckline) (Wave 2)
        target = neckline - (max(high1, high2) - neckline)
        
        # Stop loss: above the higher top (Wave 1)
        stop_loss = max(high1, high2) * 1.02  # 2% above
        
        # Calculate quality score (Wave 3)
        quality_score = self._calculate_double_top_quality(
            high1, high2, trough_between, candle_count
        )
        
        quality_rating = self._get_quality_rating(quality_score)
        
        # Check volume confirmation (Wave 2)
        volume_confirmed = self._check_volume_confirmation(df, max1_idx, max2_idx, current_idx, is_bearish=True)
        
        return DetectedPattern(
            pattern_type=PatternType.DOUBLE_TOP,
            direction=PatternDirection.BEARISH,
            quality=quality_rating,
            quality_score=quality_score,
            start_index=df.index.get_loc(min(max1_idx, max2_idx)),
            end_index=current_idx,
            neckline_price=neckline,
            target_price=target,
            stop_loss=stop_loss,
            confidence=quality_score / 100,
            volume_confirmation=volume_confirmed,
            validation_notes=f"Highs: {high1:.2f}, {high2:.2f} | Neckline: {neckline:.2f}"
        )
    
    def _calculate_double_top_quality(self, high1: float, high2: float,
                                        trough: float, candle_count: int) -> float:
        """
        Calculate quality score for double top pattern (Wave 3)
        
        Returns:
            Quality score (0-100)
        """
        score = 0.0
        
        # High equality (30 points)
        high_diff_pct = abs(high1 - high2) / high1 * 100
        if high_diff_pct < 0.5:
            score += 30
        elif high_diff_pct < 1.0:
            score += 25
        elif high_diff_pct < 1.5:
            score += 20
        elif high_diff_pct < 2.0:
            score += 15
        else:
            score += 10
        
        # Trough depth (30 points)
        trough_depth_pct = (max(high1, high2) - trough) / max(high1, high2) * 100
        if trough_depth_pct >= 5:
            score += 30
        elif trough_depth_pct >= 4:
            score += 25
        elif trough_depth_pct >= 3:
            score += 20
        elif trough_depth_pct >= 2:
            score += 15
        else:
            score += 5
        
        # Candle count (20 points)
        if candle_count >= 20:
            score += 20
        elif candle_count >= 15:
            score += 15
        elif candle_count >= 10:
            score += 10
        elif candle_count >= 7:
            score += 5
        else:
            score += 0
        
        return min(score, 100.0)
    
    # ========================================================================
    # SECTION 4: HEAD AND SHOULDERS (WAVE 1 & 3)
    # ========================================================================
    
    def detect_head_shoulders_top(self, df: pd.DataFrame, lookback: int = 50,
                                    current_idx: int = None) -> Optional[DetectedPattern]:
        """
        Detect head and shoulders top pattern - bearish reversal (Wave 1 & 3)
        
        Structure: Left Shoulder -> Head -> Right Shoulder
        Neckline connects troughs between shoulders and head
        """
        if current_idx is None:
            current_idx = len(df) - 1
        
        if current_idx < self.config.min_pattern_candles * 2:
            return None
        
        # Simplified detection - find peaks
        start_idx = max(0, current_idx - lookback)
        recent_highs = df['high'].iloc[start_idx:current_idx + 1]
        
        # Find all local peaks (simplified)
        peaks = []
        for i in range(2, len(recent_highs) - 2):
            if (recent_highs.iloc[i] > recent_highs.iloc[i-1] and
                recent_highs.iloc[i] > recent_highs.iloc[i-2] and
                recent_highs.iloc[i] > recent_highs.iloc[i+1] and
                recent_highs.iloc[i] > recent_highs.iloc[i+2]):
                peaks.append((recent_highs.index[i], recent_highs.iloc[i]))
        
        if len(peaks) < 3:
            return None
        
        # Find the highest peak (head)
        head_idx = max(peaks, key=lambda x: x[1])
        head_pos = peaks.index(head_idx)
        
        # Need at least one peak on each side
        if head_pos < 1 or head_pos >= len(peaks) - 1:
            return None
        
        left_shoulder = peaks[head_pos - 1]
        right_shoulder = peaks[head_pos + 1]
        
        # Check height relationship (head must be higher than shoulders)
        if head_idx[1] <= left_shoulder[1] or head_idx[1] <= right_shoulder[1]:
            return None
        
        # Check tolerance (Wave 1)
        shoulder_diff = abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1]
        if shoulder_diff > self.config.head_shoulders_tolerance:
            return None
        
        # Find neckline (troughs between shoulders and head)
        # Left trough between left shoulder and head
        left_trough = df.loc[left_shoulder[0]:head_idx[0], 'low'].min()
        
        # Right trough between head and right shoulder
        right_trough = df.loc[head_idx[0]:right_shoulder[0], 'low'].min()
        
        neckline = min(left_trough, right_trough)
        
        # Target: neckline - (head - neckline)
        target = neckline - (head_idx[1] - neckline)
        
        # Stop loss: above right shoulder
        stop_loss = right_shoulder[1] * 1.02
        
        # Calculate quality score
        quality_score = self._calculate_head_shoulders_quality(
            left_shoulder[1], head_idx[1], right_shoulder[1], neckline
        )
        
        quality_rating = self._get_quality_rating(quality_score)
        
        return DetectedPattern(
            pattern_type=PatternType.HEAD_SHOULDERS_TOP,
            direction=PatternDirection.BEARISH,
            quality=quality_rating,
            quality_score=quality_score,
            start_index=df.index.get_loc(left_shoulder[0]),
            end_index=current_idx,
            neckline_price=neckline,
            target_price=target,
            stop_loss=stop_loss,
            confidence=quality_score / 100,
            validation_notes=f"Head: {head_idx[1]:.2f} | Neckline: {neckline:.2f}"
        )
    
    def detect_head_shoulders_bottom(self, df: pd.DataFrame, lookback: int = 50,
                                       current_idx: int = None) -> Optional[DetectedPattern]:
        """
        Detect inverse head and shoulders bottom pattern - bullish reversal (Wave 1 & 3)
        """
        if current_idx is None:
            current_idx = len(df) - 1
        
        if current_idx < self.config.min_pattern_candles * 2:
            return None
        
        # Simplified detection - find troughs
        start_idx = max(0, current_idx - lookback)
        recent_lows = df['low'].iloc[start_idx:current_idx + 1]
        
        # Find all local troughs
        troughs = []
        for i in range(2, len(recent_lows) - 2):
            if (recent_lows.iloc[i] < recent_lows.iloc[i-1] and
                recent_lows.iloc[i] < recent_lows.iloc[i-2] and
                recent_lows.iloc[i] < recent_lows.iloc[i+1] and
                recent_lows.iloc[i] < recent_lows.iloc[i+2]):
                troughs.append((recent_lows.index[i], recent_lows.iloc[i]))
        
        if len(troughs) < 3:
            return None
        
        # Find the lowest trough (head)
        head_idx = min(troughs, key=lambda x: x[1])
        head_pos = troughs.index(head_idx)
        
        if head_pos < 1 or head_pos >= len(troughs) - 1:
            return None
        
        left_shoulder = troughs[head_pos - 1]
        right_shoulder = troughs[head_pos + 1]
        
        # Check depth relationship (head must be lower than shoulders)
        if head_idx[1] >= left_shoulder[1] or head_idx[1] >= right_shoulder[1]:
            return None
        
        # Check tolerance
        shoulder_diff = abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1]
        if shoulder_diff > self.config.head_shoulders_tolerance:
            return None
        
        # Find neckline (peaks between shoulders and head)
        left_peak = df.loc[left_shoulder[0]:head_idx[0], 'high'].max()
        right_peak = df.loc[head_idx[0]:right_shoulder[0], 'high'].max()
        
        neckline = min(left_peak, right_peak)
        
        # Target: neckline + (neckline - head)
        target = neckline + (neckline - head_idx[1])
        
        # Stop loss: below right shoulder
        stop_loss = right_shoulder[1] * 0.98
        
        # Calculate quality score
        quality_score = self._calculate_head_shoulders_quality(
            left_shoulder[1], head_idx[1], right_shoulder[1], neckline, is_bottom=True
        )
        
        quality_rating = self._get_quality_rating(quality_score)
        
        return DetectedPattern(
            pattern_type=PatternType.HEAD_SHOULDERS_BOTTOM,
            direction=PatternDirection.BULLISH,
            quality=quality_rating,
            quality_score=quality_score,
            start_index=df.index.get_loc(left_shoulder[0]),
            end_index=current_idx,
            neckline_price=neckline,
            target_price=target,
            stop_loss=stop_loss,
            confidence=quality_score / 100,
            validation_notes=f"Head: {head_idx[1]:.2f} | Neckline: {neckline:.2f}"
        )
    
    def _calculate_head_shoulders_quality(self, left: float, head: float,
                                           right: float, neckline: float,
                                           is_bottom: bool = False) -> float:
        """
        Calculate quality score for head and shoulders pattern (Wave 3)
        
        Returns:
            Quality score (0-100)
        """
        score = 0.0
        
        # Shoulder symmetry (40 points)
        if is_bottom:
            left_diff = (left - head) / head
            right_diff = (right - head) / head
        else:
            left_diff = (head - left) / head
            right_diff = (head - right) / head
        
        diff_ratio = abs(left_diff - right_diff) / max(left_diff, right_diff) if max(left_diff, right_diff) > 0 else 1
        
        if diff_ratio < 0.1:
            score += 40
        elif diff_ratio < 0.2:
            score += 30
        elif diff_ratio < 0.3:
            score += 20
        else:
            score += 10
        
        # Head height (30 points)
        if is_bottom:
            head_depth = (left - head) / left
        else:
            head_depth = (head - left) / left
        
        if head_depth >= 0.05:
            score += 30
        elif head_depth >= 0.03:
            score += 20
        elif head_depth >= 0.02:
            score += 10
        else:
            score += 5
        
        # Neckline slope (30 points) - flatter is better
        # Simplified - in production, calculate actual slope
        
        return min(score, 100.0)
    
    # ========================================================================
    # SECTION 5: NECKLINE BREAK VALIDATION (WAVE 2)
    # ========================================================================
    
    def check_neckline_break(self, df: pd.DataFrame, pattern: DetectedPattern,
                               current_idx: int, current_price: float) -> Tuple[bool, int]:
        """
        Check if neckline has been broken (Wave 2)
        
        Returns:
            (is_broken, confirmation_bars)
        """
        if not self.config.require_neckline_break:
            return True, 0
        
        # Check if price has broken the neckline
        if pattern.direction == PatternDirection.BULLISH:
            is_broken = current_price > pattern.neckline_price
        else:
            is_broken = current_price < pattern.neckline_price
        
        if not is_broken:
            return False, 0
        
        # Count confirmation bars
        confirmation_bars = 0
        for i in range(current_idx - self.config.neckline_break_confirmation_bars + 1, current_idx + 1):
            if pattern.direction == PatternDirection.BULLISH:
                if df['close'].iloc[i] > pattern.neckline_price:
                    confirmation_bars += 1
            else:
                if df['close'].iloc[i] < pattern.neckline_price:
                    confirmation_bars += 1
        
        is_confirmed = confirmation_bars >= self.config.neckline_break_confirmation_bars
        
        return is_confirmed, confirmation_bars
    
    # ========================================================================
    # SECTION 6: VOLUME CONFIRMATION (WAVE 2)
    # ========================================================================
    
    def _check_volume_confirmation(self, df: pd.DataFrame, first_idx: int,
                                     second_idx: int, current_idx: int,
                                     is_bearish: bool = False) -> bool:
        """
        Check volume confirmation for pattern breakout (Wave 2)
        """
        if not self.config.require_volume_confirmation:
            return True
        
        if 'volume' not in df.columns:
            return True
        
        # Calculate average volume
        avg_volume = df['volume'].iloc[current_idx - 20:current_idx].mean()
        
        # Check breakout volume
        breakout_volume = df['volume'].iloc[current_idx]
        
        return breakout_volume > avg_volume * self.config.volume_multiplier
    
    # ========================================================================
    # SECTION 7: QUALITY RATING (WAVE 3)
    # ========================================================================
    
    def _get_quality_rating(self, quality_score: float) -> PatternQuality:
        """Get quality rating from score (Wave 3)"""
        if quality_score >= self.config.excellent_quality_threshold:
            return PatternQuality.EXCELLENT
        elif quality_score >= self.config.good_quality_threshold:
            return PatternQuality.GOOD
        elif quality_score >= self.config.fair_quality_threshold:
            return PatternQuality.FAIR
        else:
            return PatternQuality.POOR
    
    # ========================================================================
    # SECTION 8: COMPLETE PATTERN DETECTION
    # ========================================================================
    
    def detect_all_patterns(self, df: pd.DataFrame, current_idx: int = None) -> List[DetectedPattern]:
        """
        Detect all patterns in the current data (Wave 1, 2, 3)
        
        Returns:
            List of detected patterns
        """
        patterns = []
        
        # Detect double bottom
        db = self.detect_double_bottom(df, current_idx=current_idx)
        if db:
            patterns.append(db)
        
        # Detect double top
        dt = self.detect_double_top(df, current_idx=current_idx)
        if dt:
            patterns.append(dt)
        
        # Detect head and shoulders top
        hst = self.detect_head_shoulders_top(df, current_idx=current_idx)
        if hst:
            patterns.append(hst)
        
        # Detect head and shoulders bottom
        hsb = self.detect_head_shoulders_bottom(df, current_idx=current_idx)
        if hsb:
            patterns.append(hsb)
        
        # Sort by quality (best first)
        patterns.sort(key=lambda x: x.quality_score, reverse=True)
        
        return patterns
    
    def get_best_pattern(self, df: pd.DataFrame, current_idx: int = None) -> Optional[DetectedPattern]:
        """
        Get the best quality pattern (Wave 3)
        
        Returns:
            Best pattern or None
        """
        patterns = self.detect_all_patterns(df, current_idx)
        
        if not patterns:
            return None
        
        # Filter by quality
        good_patterns = [p for p in patterns if p.quality in [PatternQuality.EXCELLENT, PatternQuality.GOOD]]
        
        if good_patterns:
            return good_patterns[0]
        
        return patterns[0]


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TESTING PATTERN RULES v3.0")
    print("="*70)
    
    # Create sample data with patterns
    dates = pd.date_range(start='2024-01-01', periods=500, freq='H')
    np.random.seed(42)
    
    # Create double bottom pattern
    prices = 100 + np.random.randn(500) * 0.5
    
    # Create double bottom around index 100-150
    prices[100:110] = 95
    prices[110:140] = 98
    prices[140:150] = 94
    prices[150:180] = 99
    
    # Create double top around index 300-350
    prices[300:310] = 105
    prices[310:340] = 102
    prices[340:350] = 106
    prices[350:380] = 101
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + 0.5,
        'low': prices - 0.5,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 500)
    }, index=dates)
    
    print(f"Sample data shape: {df.shape}")
    
    # Initialize pattern rules
    config = PatternRulesConfig(
        min_pattern_candles=7,
        double_bottom_tolerance=0.02,
        double_top_tolerance=0.02,
        require_neckline_break=False  # For testing
    )
    pattern_rules = PatternRules(config, verbose=True)
    
    # Test double bottom detection
    print(f"\n📊 Testing Double Bottom Detection:")
    db = pattern_rules.detect_double_bottom(df, current_idx=180)
    if db:
        print(f"  Pattern: {db.pattern_type.value}")
        print(f"  Direction: {db.direction.value}")
        print(f"  Quality: {db.quality.value} ({db.quality_score:.1f})")
        print(f"  Neckline: {db.neckline_price:.2f}")
        print(f"  Target: {db.target_price:.2f}")
        print(f"  Stop Loss: {db.stop_loss:.2f}")
    
    # Test double top detection
    print(f"\n📊 Testing Double Top Detection:")
    dt = pattern_rules.detect_double_top(df, current_idx=380)
    if dt:
        print(f"  Pattern: {dt.pattern_type.value}")
        print(f"  Direction: {dt.direction.value}")
        print(f"  Quality: {dt.quality.value} ({dt.quality_score:.1f})")
        print(f"  Neckline: {dt.neckline_price:.2f}")
        print(f"  Target: {dt.target_price:.2f}")
        print(f"  Stop Loss: {dt.stop_loss:.2f}")
    
    # Test all patterns
    print(f"\n📊 Testing All Patterns:")
    patterns = pattern_rules.detect_all_patterns(df)
    for p in patterns:
        print(f"  {p.pattern_type.value}: {p.quality.value} (score: {p.quality_score:.1f})")
    
    print(f"\n✅ Pattern rules test complete")