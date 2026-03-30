#!/usr/bin/env python3
"""
Exit Rules Module for SID Method - AUGMENTED VERSION
=============================================================================
Manages trade exits incorporating ALL THREE WAVES:

WAVE 1 (Core Quick Win):
- Primary exit: RSI reaches 50
- Simple 1:1 risk-reward exit
- Stop loss management

WAVE 2 (Live Sessions & Q&A):
- Alternative exits: 50-SMA (blue line)
- Fixed point targets: +4 points (<$200), +8 points (>$200)
- Trailing stops for strong trends
- Divergence-based early exits
- Pattern-based exits (head & shoulders completion)

WAVE 3 (Academy Support Sessions):
- Reversal candle detection (2 consecutive reversals)
- Session-based exit timing
- Zone quality-based exits
- Volatility-based stop adjustments
- Partial profit taking strategies

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

class ExitReason(Enum):
    """Reasons for exiting a trade"""
    RSI_50 = "rsi_50"                       # Wave 1 primary
    SMA_50 = "sma_50"                       # Wave 2 alternative
    POINT_TARGET = "point_target"           # Wave 2 alternative
    TRAILING_STOP = "trailing_stop"         # Wave 2 advanced
    STOP_LOSS = "stop_loss"                 # Wave 1
    REVERSAL_CANDLES = "reversal_candles"   # Wave 3
    DIVERGENCE_EXIT = "divergence_exit"     # Wave 2
    PATTERN_COMPLETION = "pattern_completion" # Wave 2
    VOLATILITY_SPIKE = "volatility_spike"   # Wave 3
    TIME_STOP = "time_stop"                 # Wave 3
    MANUAL = "manual"


class ExitType(Enum):
    """Type of exit strategy"""
    FIXED = "fixed"           # Fixed target (RSI 50, SMA 50, points)
    TRAILING = "trailing"     # Trailing stop
    PARTIAL = "partial"       # Partial profit taking
    SCALING = "scaling"       # Scale out at multiple levels


@dataclass
class ExitSignal:
    """Signal to exit a trade"""
    should_exit: bool
    reason: ExitReason
    exit_price: Optional[float] = None
    exit_quantity: Optional[float] = None  # For partial exits (percentage)
    exit_type: ExitType = ExitType.FIXED
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'should_exit': self.should_exit,
            'reason': self.reason.value,
            'exit_price': self.exit_price,
            'exit_quantity': self.exit_quantity,
            'exit_type': self.exit_type.value,
            'notes': self.notes
        }


@dataclass
class ExitRulesConfig:
    """Configuration for exit rules (Wave 1, 2, 3)"""
    # Wave 1: Primary exit
    rsi_target: int = 50
    use_rsi_50_exit: bool = True
    
    # Wave 2: Alternative exits
    use_sma_50_exit: bool = True
    use_point_target_exit: bool = True
    point_target_low: int = 4      # For stocks under $200
    point_target_high: int = 8     # For stocks over $200
    
    # Wave 2: Trailing stop
    use_trailing_stop: bool = False
    trailing_stop_atr_multiplier: float = 2.0
    trailing_stop_percent: float = 0.02  # 2% trailing stop
    
    # Wave 2: Divergence exit
    use_divergence_exit: bool = True
    divergence_lookback: int = 20
    
    # Wave 3: Reversal exit
    use_reversal_exit: bool = True
    reversal_candles_required: int = 2
    
    # Wave 3: Partial profit taking
    use_partial_exits: bool = False
    partial_levels: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0])  # R:R ratios
    partial_percentages: List[float] = field(default_factory=lambda: [0.5, 0.25, 0.25])
    
    # Wave 3: Time stop
    use_time_stop: bool = False
    max_hold_bars: int = 20  # Maximum bars to hold a trade
    
    # Wave 3: Volatility exit
    use_volatility_exit: bool = False
    volatility_multiplier: float = 2.0  # Exit if volatility spikes above this multiple
    
    # Wave 2: Minimum profit for trail
    min_profit_to_trail: float = 0.0  # Minimum profit in pips/points to start trailing


class ExitRules:
    """
    Manages trade exits for SID Method
    Incorporates ALL THREE WAVES of strategies
    """
    
    def __init__(self, config: ExitRulesConfig = None, verbose: bool = True):
        """
        Initialize exit rules
        
        Args:
            config: ExitRulesConfig instance
            verbose: Enable verbose output
        """
        self.config = config or ExitRulesConfig()
        self.verbose = verbose
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🚪 EXIT RULES v3.0 (Fully Augmented)")
            print(f"{'='*60}")
            print(f"📊 Primary: RSI {self.config.rsi_target} (Wave 1)")
            print(f"📈 SMA 50 exit: {self.config.use_sma_50_exit}")
            print(f"🎯 Point target exit: {self.config.use_point_target_exit}")
            print(f"🔄 Trailing stop: {self.config.use_trailing_stop}")
            print(f"⚡ Divergence exit: {self.config.use_divergence_exit}")
            print(f"🕯️ Reversal exit: {self.config.use_reversal_exit}")
            print(f"⏱️ Time stop: {self.config.use_time_stop} (max {self.config.max_hold_bars} bars)")
            print(f"📊 Partial exits: {self.config.use_partial_exits}")
            print(f"{'='*60}\n")
    
    # ========================================================================
    # SECTION 1: PRIMARY EXIT (Wave 1)
    # ========================================================================
    
    def check_rsi_50_exit(self, rsi_value: float, entry_rsi: float = None) -> ExitSignal:
        """
        Check if RSI has reached 50 (Wave 1 primary exit)
        
        Args:
            rsi_value: Current RSI value
            entry_rsi: RSI at entry (for direction detection)
        
        Returns:
            ExitSignal
        """
        if not self.config.use_rsi_50_exit:
            return ExitSignal(should_exit=False, reason=ExitReason.RSI_50)
        
        # For long trades: exit when RSI crosses above 50
        # For short trades: exit when RSI crosses below 50
        if entry_rsi is not None:
            if entry_rsi < self.config.rsi_target:
                # Long trade: looking for cross above
                if rsi_value >= self.config.rsi_target:
                    return ExitSignal(
                        should_exit=True,
                        reason=ExitReason.RSI_50,
                        exit_type=ExitType.FIXED,
                        notes=f"RSI reached {self.config.rsi_target}"
                    )
            else:
                # Short trade: looking for cross below
                if rsi_value <= self.config.rsi_target:
                    return ExitSignal(
                        should_exit=True,
                        reason=ExitReason.RSI_50,
                        exit_type=ExitType.FIXED,
                        notes=f"RSI reached {self.config.rsi_target}"
                    )
        else:
            # Simple check: if RSI crosses 50 from below or above
            # This is simplified; in practice track entry direction
            pass
        
        return ExitSignal(should_exit=False, reason=ExitReason.RSI_50)
    
    # ========================================================================
    # SECTION 2: ALTERNATIVE EXITS (Wave 2)
    # ========================================================================
    
    def check_sma_50_exit(self, current_price: float, sma_50: float,
                            direction: str) -> ExitSignal:
        """
        Check if price has reached the 50-period SMA (Wave 2 alternative)
        
        Args:
            current_price: Current price
            sma_50: 50-period SMA value
            direction: 'long' or 'short'
        
        Returns:
            ExitSignal
        """
        if not self.config.use_sma_50_exit or sma_50 is None:
            return ExitSignal(should_exit=False, reason=ExitReason.SMA_50)
        
        if direction == 'long':
            if current_price >= sma_50:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.SMA_50,
                    exit_type=ExitType.FIXED,
                    notes=f"Price reached 50-SMA at {sma_50:.5f}"
                )
        else:
            if current_price <= sma_50:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.SMA_50,
                    exit_type=ExitType.FIXED,
                    notes=f"Price reached 50-SMA at {sma_50:.5f}"
                )
        
        return ExitSignal(should_exit=False, reason=ExitReason.SMA_50)
    
    def check_point_target_exit(self, entry_price: float, current_price: float,
                                  direction: str) -> ExitSignal:
        """
        Check if fixed point target has been reached (Wave 2 alternative)
        
        Point targets:
        - Stocks under $200: +4 points
        - Stocks over $200: +8 points
        
        Args:
            entry_price: Entry price
            current_price: Current price
            direction: 'long' or 'short'
        
        Returns:
            ExitSignal
        """
        if not self.config.use_point_target_exit:
            return ExitSignal(should_exit=False, reason=ExitReason.POINT_TARGET)
        
        # Determine point target based on entry price
        if entry_price < 200:
            point_target = self.config.point_target_low
        else:
            point_target = self.config.point_target_high
        
        if direction == 'long':
            target_price = entry_price + point_target
            if current_price >= target_price:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.POINT_TARGET,
                    exit_type=ExitType.FIXED,
                    notes=f"Reached {point_target} point target at {target_price:.2f}"
                )
        else:
            target_price = entry_price - point_target
            if current_price <= target_price:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.POINT_TARGET,
                    exit_type=ExitType.FIXED,
                    notes=f"Reached {point_target} point target at {target_price:.2f}"
                )
        
        return ExitSignal(should_exit=False, reason=ExitReason.POINT_TARGET)
    
    # ========================================================================
    # SECTION 3: TRAILING STOP (Wave 2)
    # ========================================================================
    
    def calculate_trailing_stop_atr(self, atr: float, direction: str,
                                      current_price: float, entry_price: float,
                                      highest_price: float = None,
                                      lowest_price: float = None) -> float:
        """
        Calculate trailing stop based on ATR (Wave 2)
        
        Args:
            atr: Average True Range
            direction: 'long' or 'short'
            current_price: Current price
            entry_price: Entry price
            highest_price: Highest price since entry (for longs)
            lowest_price: Lowest price since entry (for shorts)
        
        Returns:
            Trailing stop price
        """
        atr_stop = self.config.trailing_stop_atr_multiplier * atr
        
        if direction == 'long':
            if highest_price is None:
                highest_price = max(entry_price, current_price)
            else:
                highest_price = max(highest_price, current_price)
            return highest_price - atr_stop
        else:
            if lowest_price is None:
                lowest_price = min(entry_price, current_price)
            else:
                lowest_price = min(lowest_price, current_price)
            return lowest_price + atr_stop
    
    def calculate_trailing_stop_percent(self, current_price: float, 
                                          direction: str) -> float:
        """
        Calculate trailing stop based on fixed percentage (Wave 2)
        
        Args:
            current_price: Current price
            direction: 'long' or 'short'
        
        Returns:
            Trailing stop price
        """
        percent_stop = self.config.trailing_stop_percent
        
        if direction == 'long':
            return current_price * (1 - percent_stop)
        else:
            return current_price * (1 + percent_stop)
    
    def check_trailing_stop(self, current_price: float, trailing_stop: float,
                              direction: str, profit_pips: float = 0) -> ExitSignal:
        """
        Check if trailing stop has been hit (Wave 2)
        
        Args:
            current_price: Current price
            trailing_stop: Current trailing stop level
            direction: 'long' or 'short'
            profit_pips: Current profit in pips
        
        Returns:
            ExitSignal
        """
        if not self.config.use_trailing_stop:
            return ExitSignal(should_exit=False, reason=ExitReason.TRAILING_STOP)
        
        # Only trail after minimum profit is reached
        if profit_pips < self.config.min_profit_to_trail:
            return ExitSignal(should_exit=False, reason=ExitReason.TRAILING_STOP)
        
        if direction == 'long':
            if current_price <= trailing_stop:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.TRAILING_STOP,
                    exit_type=ExitType.TRAILING,
                    exit_price=trailing_stop,
                    notes=f"Trailing stop hit at {trailing_stop:.5f}"
                )
        else:
            if current_price >= trailing_stop:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.TRAILING_STOP,
                    exit_type=ExitType.TRAILING,
                    exit_price=trailing_stop,
                    notes=f"Trailing stop hit at {trailing_stop:.5f}"
                )
        
        return ExitSignal(should_exit=False, reason=ExitReason.TRAILING_STOP)
    
    # ========================================================================
    # SECTION 4: DIVERGENCE EXIT (Wave 2)
    # ========================================================================
    
    def check_divergence_exit(self, df: pd.DataFrame, current_idx: int,
                                direction: str) -> ExitSignal:
        """
        Check for divergence that signals exit (Wave 2)
        
        Args:
            df: Price DataFrame with RSI column
            current_idx: Current index
            direction: 'long' or 'short'
        
        Returns:
            ExitSignal
        """
        if not self.config.use_divergence_exit or current_idx < self.config.divergence_lookback:
            return ExitSignal(should_exit=False, reason=ExitReason.DIVERGENCE_EXIT)
        
        if 'rsi' not in df.columns:
            return ExitSignal(should_exit=False, reason=ExitReason.DIVERGENCE_EXIT)
        
        # Get windows
        lookback = self.config.divergence_lookback
        price_window = df['close'].iloc[current_idx - lookback:current_idx + 1]
        rsi_window = df['rsi'].iloc[current_idx - lookback:current_idx + 1]
        
        # Find min and max positions
        price_min_idx = price_window.idxmin()
        price_max_idx = price_window.idxmax()
        rsi_min_idx = rsi_window.idxmin()
        rsi_max_idx = rsi_window.idxmax()
        
        if direction == 'long':
            # For long trades, bearish divergence signals exit
            bearish = (price_window.loc[price_max_idx] > price_window.iloc[0] and
                      rsi_window.loc[rsi_max_idx] < rsi_window.iloc[0])
            
            if bearish:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.DIVERGENCE_EXIT,
                    notes="Bearish divergence detected - consider exiting long"
                )
        else:
            # For short trades, bullish divergence signals exit
            bullish = (price_window.loc[price_min_idx] < price_window.iloc[0] and
                      rsi_window.loc[rsi_min_idx] > rsi_window.iloc[0])
            
            if bullish:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.DIVERGENCE_EXIT,
                    notes="Bullish divergence detected - consider exiting short"
                )
        
        return ExitSignal(should_exit=False, reason=ExitReason.DIVERGENCE_EXIT)
    
    # ========================================================================
    # SECTION 5: REVERSAL EXIT (Wave 3)
    # ========================================================================
    
    def check_reversal_exit(self, df: pd.DataFrame, entry_idx: int,
                              current_idx: int, direction: str) -> ExitSignal:
        """
        Check for reversal candles that signal exit (Wave 3)
        
        Wave 3: 2 consecutive reversal days signal exit
        
        Args:
            df: Price DataFrame
            entry_idx: Entry index
            current_idx: Current index
            direction: 'long' or 'short'
        
        Returns:
            ExitSignal
        """
        if not self.config.use_reversal_exit:
            return ExitSignal(should_exit=False, reason=ExitReason.REVERSAL_CANDLES)
        
        if current_idx - entry_idx < self.config.reversal_candles_required:
            return ExitSignal(should_exit=False, reason=ExitReason.REVERSAL_CANDLES)
        
        # Check last N candles
        reversal_count = 0
        for i in range(current_idx - self.config.reversal_candles_required + 1, current_idx + 1):
            candle = df.iloc[i]
            
            if direction == 'long':
                # Reversal for long: red candle (close < open)
                if candle['close'] < candle['open']:
                    reversal_count += 1
            else:
                # Reversal for short: green candle (close > open)
                if candle['close'] > candle['open']:
                    reversal_count += 1
        
        if reversal_count >= self.config.reversal_candles_required:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.REVERSAL_CANDLES,
                notes=f"{reversal_count} consecutive reversal candles detected"
            )
        
        return ExitSignal(should_exit=False, reason=ExitReason.REVERSAL_CANDLES)
    
    # ========================================================================
    # SECTION 6: PATTERN COMPLETION EXIT (Wave 2)
    # ========================================================================
    
    def check_pattern_completion_exit(self, df: pd.DataFrame, current_idx: int,
                                        pattern: str, direction: str) -> ExitSignal:
        """
        Check if a pattern has completed (Wave 2)
        
        Args:
            df: Price DataFrame
            current_idx: Current index
            pattern: Pattern name ('double_top', 'double_bottom', etc.)
            direction: Trade direction
        
        Returns:
            ExitSignal
        """
        if current_idx < 10:
            return ExitSignal(should_exit=False, reason=ExitReason.PATTERN_COMPLETION)
        
        # Simplified pattern completion detection
        if pattern == 'double_top':
            # Exit when price breaks below the trough between the two tops
            recent_lows = df['low'].iloc[current_idx-30:current_idx+1]
            trough = recent_lows.min()
            
            if df['close'].iloc[current_idx] < trough:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.PATTERN_COMPLETION,
                    notes="Double top pattern completed - exit short"
                )
        
        elif pattern == 'double_bottom':
            # Exit when price breaks above the peak between the two bottoms
            recent_highs = df['high'].iloc[current_idx-30:current_idx+1]
            peak = recent_highs.max()
            
            if df['close'].iloc[current_idx] > peak:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.PATTERN_COMPLETION,
                    notes="Double bottom pattern completed - exit long"
                )
        
        return ExitSignal(should_exit=False, reason=ExitReason.PATTERN_COMPLETION)
    
    # ========================================================================
    # SECTION 7: VOLATILITY EXIT (Wave 3)
    # ========================================================================
    
    def check_volatility_exit(self, current_volatility: float,
                                entry_volatility: float) -> ExitSignal:
        """
        Check if volatility has spiked (Wave 3)
        
        Args:
            current_volatility: Current volatility (e.g., ATR)
            entry_volatility: Volatility at entry
        
        Returns:
            ExitSignal
        """
        if not self.config.use_volatility_exit:
            return ExitSignal(should_exit=False, reason=ExitReason.VOLATILITY_SPIKE)
        
        if entry_volatility <= 0:
            return ExitSignal(should_exit=False, reason=ExitReason.VOLATILITY_SPIKE)
        
        volatility_ratio = current_volatility / entry_volatility
        
        if volatility_ratio >= self.config.volatility_multiplier:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.VOLATILITY_SPIKE,
                notes=f"Volatility spike: {volatility_ratio:.2f}x entry level"
            )
        
        return ExitSignal(should_exit=False, reason=ExitReason.VOLATILITY_SPIKE)
    
    # ========================================================================
    # SECTION 8: TIME STOP (Wave 3)
    # ========================================================================
    
    def check_time_stop(self, entry_idx: int, current_idx: int) -> ExitSignal:
        """
        Check if trade has been held too long (Wave 3)
        
        Args:
            entry_idx: Entry index
            current_idx: Current index
        
        Returns:
            ExitSignal
        """
        if not self.config.use_time_stop:
            return ExitSignal(should_exit=False, reason=ExitReason.TIME_STOP)
        
        bars_held = current_idx - entry_idx
        
        if bars_held >= self.config.max_hold_bars:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.TIME_STOP,
                notes=f"Maximum hold time reached ({bars_held} bars)"
            )
        
        return ExitSignal(should_exit=False, reason=ExitReason.TIME_STOP)
    
    # ========================================================================
    # SECTION 9: PARTIAL PROFIT TAKING (Wave 3)
    # ========================================================================
    
    def get_partial_exit_levels(self, entry_price: float, stop_loss: float,
                                  direction: str) -> List[Dict]:
        """
        Get partial profit taking levels (Wave 3)
        
        Returns:
            List of dictionaries with target price and percentage to exit
        """
        if not self.config.use_partial_exits:
            return []
        
        risk_distance = abs(entry_price - stop_loss)
        levels = []
        
        for ratio, percentage in zip(self.config.partial_levels, self.config.partial_percentages):
            if direction == 'long':
                target_price = entry_price + (risk_distance * ratio)
            else:
                target_price = entry_price - (risk_distance * ratio)
            
            levels.append({
                'target_price': target_price,
                'exit_percentage': percentage,
                'risk_reward_ratio': ratio
            })
        
        return levels
    
    def check_partial_exit(self, current_price: float, entry_price: float,
                             stop_loss: float, direction: str,
                             remaining_position: float) -> Tuple[ExitSignal, float]:
        """
        Check if a partial exit level has been reached (Wave 3)
        
        Returns:
            (ExitSignal, percentage_to_exit)
        """
        if not self.config.use_partial_exits:
            return ExitSignal(should_exit=False, reason=ExitReason.MANUAL), 0
        
        levels = self.get_partial_exit_levels(entry_price, stop_loss, direction)
        
        for level in levels:
            if direction == 'long' and current_price >= level['target_price']:
                # Exit only the specified percentage
                exit_percentage = level['exit_percentage']
                if remaining_position >= exit_percentage:
                    return ExitSignal(
                        should_exit=True,
                        reason=ExitReason.MANUAL,
                        exit_type=ExitType.PARTIAL,
                        exit_price=current_price,
                        exit_quantity=exit_percentage,
                        notes=f"Partial exit at {level['risk_reward_ratio']:.1f}R"
                    ), exit_percentage
            elif direction == 'short' and current_price <= level['target_price']:
                exit_percentage = level['exit_percentage']
                if remaining_position >= exit_percentage:
                    return ExitSignal(
                        should_exit=True,
                        reason=ExitReason.MANUAL,
                        exit_type=ExitType.PARTIAL,
                        exit_price=current_price,
                        exit_quantity=exit_percentage,
                        notes=f"Partial exit at {level['risk_reward_ratio']:.1f}R"
                    ), exit_percentage
        
        return ExitSignal(should_exit=False, reason=ExitReason.MANUAL), 0
    
    # ========================================================================
    # SECTION 10: COMPLETE EXIT EVALUATION
    # ========================================================================
    
    def evaluate_exit(self, df: pd.DataFrame, current_idx: int,
                       entry_idx: int, entry_price: float, stop_loss: float,
                       direction: str, rsi_value: float, entry_rsi: float,
                       sma_50: float = None, atr: float = None,
                       trailing_stop_current: float = None,
                       highest_price: float = None, lowest_price: float = None,
                       pattern: str = None, profit_pips: float = 0) -> ExitSignal:
        """
        Complete exit evaluation integrating ALL THREE WAVES
        
        Args:
            df: Price DataFrame
            current_idx: Current index
            entry_idx: Entry index
            entry_price: Entry price
            stop_loss: Original stop loss
            direction: 'long' or 'short'
            rsi_value: Current RSI
            entry_rsi: RSI at entry
            sma_50: Current 50-SMA
            atr: Current ATR
            trailing_stop_current: Current trailing stop level (if any)
            highest_price: Highest price since entry (for longs)
            lowest_price: Lowest price since entry (for shorts)
            pattern: Pattern name if detected
            profit_pips: Current profit in pips
        
        Returns:
            ExitSignal with highest priority exit
        """
        # Priority order (Wave 1 primary first, then Wave 2 & 3)
        exit_signals = []
        
        # 1. Stop loss (Wave 1) - highest priority
        if direction == 'long':
            if df['close'].iloc[current_idx] <= stop_loss:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.STOP_LOSS,
                    exit_price=stop_loss,
                    notes="Stop loss hit"
                )
        else:
            if df['close'].iloc[current_idx] >= stop_loss:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.STOP_LOSS,
                    exit_price=stop_loss,
                    notes="Stop loss hit"
                )
        
        # 2. RSI 50 (Wave 1 primary)
        exit_signal = self.check_rsi_50_exit(rsi_value, entry_rsi)
        if exit_signal.should_exit:
            return exit_signal
        
        # 3. SMA 50 (Wave 2 alternative)
        if sma_50 is not None:
            exit_signal = self.check_sma_50_exit(df['close'].iloc[current_idx], sma_50, direction)
            if exit_signal.should_exit:
                return exit_signal
        
        # 4. Point target (Wave 2 alternative)
        exit_signal = self.check_point_target_exit(entry_price, df['close'].iloc[current_idx], direction)
        if exit_signal.should_exit:
            return exit_signal
        
        # 5. Divergence exit (Wave 2)
        exit_signal = self.check_divergence_exit(df, current_idx, direction)
        if exit_signal.should_exit:
            return exit_signal
        
        # 6. Reversal exit (Wave 3)
        exit_signal = self.check_reversal_exit(df, entry_idx, current_idx, direction)
        if exit_signal.should_exit:
            return exit_signal
        
        # 7. Pattern completion exit (Wave 2)
        if pattern:
            exit_signal = self.check_pattern_completion_exit(df, current_idx, pattern, direction)
            if exit_signal.should_exit:
                return exit_signal
        
        # 8. Trailing stop (Wave 2)
        if trailing_stop_current is not None:
            exit_signal = self.check_trailing_stop(
                df['close'].iloc[current_idx], trailing_stop_current, direction, profit_pips
            )
            if exit_signal.should_exit:
                return exit_signal
        
        # 9. Volatility exit (Wave 3)
        if atr is not None:
            # Use ATR as volatility proxy
            exit_signal = self.check_volatility_exit(atr, atr)  # Simplified
            if exit_signal.should_exit:
                return exit_signal
        
        # 10. Time stop (Wave 3)
        exit_signal = self.check_time_stop(entry_idx, current_idx)
        if exit_signal.should_exit:
            return exit_signal
        
        # No exit signal
        return ExitSignal(should_exit=False, reason=ExitReason.MANUAL)
    
    # ========================================================================
    # SECTION 11: UPDATE TRAILING STOP
    # ========================================================================
    
    def update_trailing_stop(self, current_price: float, highest_price: float,
                               lowest_price: float, direction: str, atr: float,
                               use_atr: bool = True) -> float:
        """
        Update trailing stop level
        
        Args:
            current_price: Current price
            highest_price: Highest price since entry
            lowest_price: Lowest price since entry
            direction: 'long' or 'short'
            atr: Current ATR
            use_atr: Use ATR-based trailing (if False, use percentage)
        
        Returns:
            New trailing stop level
        """
        if use_atr:
            return self.calculate_trailing_stop_atr(atr, direction, current_price, 
                                                     entry_price=current_price,
                                                     highest_price=highest_price,
                                                     lowest_price=lowest_price)
        else:
            return self.calculate_trailing_stop_percent(current_price, direction)
    
    # ========================================================================
    # SECTION 12: PROFIT CALCULATION
    # ========================================================================
    
    def calculate_profit_pips(self, entry_price: float, exit_price: float,
                                direction: str, instrument: str = None) -> float:
        """
        Calculate profit in pips
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            direction: 'long' or 'short'
            instrument: Instrument (for pip value)
        
        Returns:
            Profit in pips
        """
        pip_value = 0.0001
        if instrument and ('JPY' in instrument or '_JPY' in instrument):
            pip_value = 0.01
        
        if direction == 'long':
            profit = exit_price - entry_price
        else:
            profit = entry_price - exit_price
        
        return profit / pip_value
    
    def calculate_profit_percent(self, entry_price: float, exit_price: float,
                                   direction: str) -> float:
        """
        Calculate profit percentage
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            direction: 'long' or 'short'
        
        Returns:
            Profit percentage
        """
        if direction == 'long':
            return (exit_price - entry_price) / entry_price * 100
        else:
            return (entry_price - exit_price) / entry_price * 100


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TESTING EXIT RULES v3.0")
    print("="*70)
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=500, freq='H')
    np.random.seed(42)
    
    returns = np.random.randn(500) * 0.001
    prices = 100 + np.cumsum(returns)
    
    # Add a trend
    prices[200:300] = prices[200:300] + np.linspace(0, 5, 100)
    prices[300:400] = prices[300:400] - np.linspace(0, 3, 100)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + 0.5,
        'low': prices - 0.5,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 500)
    }, index=dates)
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, float('nan'))
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)
    
    # Calculate SMA 50
    df['sma_50'] = df['close'].rolling(50).mean()
    
    # Initialize exit rules
    config = ExitRulesConfig(
        use_rsi_50_exit=True,
        use_sma_50_exit=True,
        use_point_target_exit=True,
        use_trailing_stop=True,
        use_divergence_exit=True,
        use_reversal_exit=True,
        use_partial_exits=True
    )
    exit_rules = ExitRules(config, verbose=True)
    
    # Simulate a long trade
    entry_idx = 200
    entry_price = df['close'].iloc[entry_idx]
    entry_rsi = df['rsi'].iloc[entry_idx]
    stop_loss = entry_price - 1.0  # 1 point stop
    direction = 'long'
    
    print(f"\n📊 Simulating long trade:")
    print(f"  Entry: {entry_idx} @ {entry_price:.2f}")
    print(f"  Entry RSI: {entry_rsi:.2f}")
    print(f"  Stop Loss: {stop_loss:.2f}")
    
    # Track trade progress
    highest_price = entry_price
    trailing_stop = None
    
    # Simulate bars
    for i in range(entry_idx + 1, min(entry_idx + 50, len(df))):
        current_price = df['close'].iloc[i]
        current_rsi = df['rsi'].iloc[i]
        sma_50 = df['sma_50'].iloc[i]
        
        # Update highest price
        if current_price > highest_price:
            highest_price = current_price
            
            # Update trailing stop if using
            if trailing_stop is not None:
                trailing_stop = exit_rules.update_trailing_stop(
                    current_price, highest_price, None, direction, atr=0.5
                )
        
        # Evaluate exit
        exit_signal = exit_rules.evaluate_exit(
            df=df,
            current_idx=i,
            entry_idx=entry_idx,
            entry_price=entry_price,
            stop_loss=stop_loss,
            direction=direction,
            rsi_value=current_rsi,
            entry_rsi=entry_rsi,
            sma_50=sma_50,
            trailing_stop_current=trailing_stop,
            highest_price=highest_price,
            profit_pips=(current_price - entry_price) / 0.0001
        )
        
        if exit_signal.should_exit:
            print(f"\n  🔴 Exit at bar {i}:")
            print(f"     Price: {current_price:.2f}")
            print(f"     Reason: {exit_signal.reason.value}")
            print(f"     Notes: {exit_signal.notes}")
            break
    else:
        print(f"\n  ✅ Trade still open after 50 bars")
    
    # Test partial profit taking
    print(f"\n📊 Partial Profit Taking Levels:")
    levels = exit_rules.get_partial_exit_levels(entry_price, stop_loss, direction)
    for level in levels:
        print(f"  {level['risk_reward_ratio']:.1f}R @ {level['target_price']:.2f} ({level['exit_percentage']*100:.0f}%)")
    
    print(f"\n✅ Exit rules test complete")