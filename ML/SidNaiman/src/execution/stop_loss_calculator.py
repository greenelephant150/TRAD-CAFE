#!/usr/bin/env python3
"""
Stop Loss Calculator Module for SID Method - AUGMENTED VERSION
=============================================================================
Calculates and manages stop losses incorporating ALL THREE WAVES:

WAVE 1 (Core Quick Win):
- Stop loss based on lowest low (longs) or highest high (shorts)
- Rounded DOWN for longs, rounded UP for shorts
- Stop placed between RSI signal date and entry date

WAVE 2 (Live Sessions & Q&A):
- Dynamic stop adjustment based on volatility
- ATR-based stops
- Support/resistance-based stops
- Alert-based stops (no hard stop, just alert)

WAVE 3 (Academy Support Sessions):
- Pip buffer behind zone (5 pips for default, 10 for Yen pairs)
- Aggressive vs passive stop selection
- Zone quality-based stop placement
- Minimum distance requirements (broker constraints)

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

class StopLossType(Enum):
    """Types of stop loss calculation methods"""
    SID_METHOD = "sid_method"           # Wave 1: lowest low / highest high
    ATR_BASED = "atr_based"             # Wave 2: based on Average True Range
    SUPPORT_RESISTANCE = "support_resistance"  # Wave 2: key S/R levels
    PIP_BUFFER = "pip_buffer"           # Wave 3: fixed pips behind zone
    PERCENT_BASED = "percent_based"     # Fixed percentage of entry


class StopLossStyle(Enum):
    """Style of stop loss placement (Wave 3)"""
    AGGRESSIVE = "aggressive"   # Tighter stop, higher risk/reward
    PASSIVE = "passive"         # Wider stop, lower chance of being stopped out
    NORMAL = "normal"           # Standard SID method stop


@dataclass
class StopLossCalculation:
    """Result of stop loss calculation"""
    stop_price: float
    stop_type: StopLossType
    stop_style: StopLossStyle
    risk_distance: float
    risk_percent: float
    risk_amount: float
    is_valid: bool
    validation_notes: Optional[str] = None
    
    # Additional data
    signal_date: Optional[datetime] = None
    entry_date: Optional[datetime] = None
    lowest_low: Optional[float] = None
    highest_high: Optional[float] = None
    pip_buffer_applied: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {
            'stop_price': self.stop_price,
            'stop_type': self.stop_type.value,
            'stop_style': self.stop_style.value,
            'risk_distance': self.risk_distance,
            'risk_percent': self.risk_percent,
            'risk_amount': self.risk_amount,
            'is_valid': self.is_valid,
            'validation_notes': self.validation_notes
        }


@dataclass
class StopLossConfig:
    """Configuration for stop loss calculator (Wave 1, 2, 3)"""
    # Wave 1: Core parameters
    use_sid_method_stops: bool = True
    
    # Wave 2: ATR-based stops
    use_atr_stops: bool = False
    atr_multiplier: float = 2.0
    atr_period: int = 14
    
    # Wave 2: Support/resistance stops
    use_support_resistance: bool = False
    support_resistance_lookback: int = 50
    
    # Wave 3: Pip buffer
    use_pip_buffer: bool = True
    pip_buffer_default: int = 5      # For non-Yen pairs
    pip_buffer_yen: int = 10         # For Yen pairs
    
    # Wave 3: Stop style
    default_stop_style: str = "normal"  # 'aggressive', 'normal', 'passive'
    
    # Wave 3: Minimum distance requirements
    min_distance_pips: int = 5      # Minimum stop distance in pips
    enforce_min_distance: bool = True
    
    # Wave 3: Broker constraints
    broker_min_distance: int = 2     # Minimum distance in pips allowed by broker
    use_broker_constraints: bool = True
    
    # Wave 2: Alert-based stops
    use_alert_stops: bool = False    # Place alert instead of hard stop
    alert_trigger_offset: float = 0.0  # Offset from stop level


class StopLossCalculator:
    """
    Calculates and manages stop losses for SID Method
    Incorporates ALL THREE WAVES of strategies
    """
    
    def __init__(self, config: StopLossConfig = None, verbose: bool = True):
        """
        Initialize stop loss calculator
        
        Args:
            config: StopLossConfig instance
            verbose: Enable verbose output
        """
        self.config = config or StopLossConfig()
        self.verbose = verbose
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🛑 STOP LOSS CALCULATOR v3.0 (Fully Augmented)")
            print(f"{'='*60}")
            print(f"📐 SID Method stops: {self.config.use_sid_method_stops}")
            print(f"📊 ATR-based stops: {self.config.use_atr_stops}")
            print(f"🏔️ S/R-based stops: {self.config.use_support_resistance}")
            print(f"💧 Pip buffer: {self.config.use_pip_buffer} (default {self.config.pip_buffer_default}p, Yen {self.config.pip_buffer_yen}p)")
            print(f"⚡ Stop style: {self.config.default_stop_style}")
            print(f"📏 Min distance: {self.config.min_distance_pips} pips")
            print(f"🏦 Broker constraints: {self.config.use_broker_constraints}")
            print(f"🔔 Alert stops: {self.config.use_alert_stops}")
            print(f"{'='*60}\n")
    
    # ========================================================================
    # SECTION 1: PIP VALUE UTILITIES (Wave 3)
    # ========================================================================
    
    def get_pip_value(self, instrument: str) -> float:
        """
        Get pip value for an instrument (Wave 3)
        
        Args:
            instrument: Instrument symbol (e.g., 'EUR_USD', 'USD_JPY')
        
        Returns:
            Pip value (0.0001 for most, 0.01 for Yen pairs)
        """
        if instrument and ('JPY' in instrument or '_JPY' in instrument):
            return 0.01
        return 0.0001
    
    def get_stop_pips(self, instrument: str) -> int:
        """
        Get recommended stop distance in pips (Wave 3)
        
        From Academy sessions:
        - Yen pairs: 10 pips behind zone
        - Others: 5 pips behind zone
        
        Args:
            instrument: Instrument symbol
        
        Returns:
            Stop distance in pips
        """
        if instrument and ('JPY' in instrument or '_JPY' in instrument):
            return self.config.pip_buffer_yen
        return self.config.pip_buffer_default
    
    def pips_to_price(self, instrument: str, pips: float) -> float:
        """
        Convert pips to price difference
        
        Args:
            instrument: Instrument symbol
            pips: Number of pips
        
        Returns:
            Price difference
        """
        pip_value = self.get_pip_value(instrument)
        return pips * pip_value
    
    def price_to_pips(self, instrument: str, price_diff: float) -> float:
        """
        Convert price difference to pips
        
        Args:
            instrument: Instrument symbol
            price_diff: Price difference
        
        Returns:
            Number of pips
        """
        pip_value = self.get_pip_value(instrument)
        return price_diff / pip_value
    
    # ========================================================================
    # SECTION 2: WAVE 1 - SID METHOD STOP LOSS
    # ========================================================================
    
    def calculate_sid_stop_loss(self, df: pd.DataFrame, 
                                  signal_date: datetime,
                                  entry_date: datetime,
                                  signal_type: str,
                                  instrument: str = None,
                                  use_rounding: bool = True,
                                  use_pip_buffer: bool = None) -> StopLossCalculation:
        """
        Calculate SID Method stop loss (Wave 1 core)
        
        WAVE 1 RULE:
        - Long (oversold): lowest low between signal and entry, rounded DOWN
        - Short (overbought): highest high between signal and entry, rounded UP
        
        WAVE 3 REFINEMENT:
        - Add pip buffer behind zone
        
        Args:
            df: Price DataFrame
            signal_date: Date when RSI first triggered
            entry_date: Date of trade entry
            signal_type: 'oversold' or 'overbought'
            instrument: Instrument for pip calculation
            use_rounding: Apply rounding to nearest whole number
            use_pip_buffer: Apply pip buffer (default from config)
        
        Returns:
            StopLossCalculation
        """
        if use_pip_buffer is None:
            use_pip_buffer = self.config.use_pip_buffer
        
        mask = (df.index >= signal_date) & (df.index <= entry_date)
        period_df = df.loc[mask]
        
        if period_df.empty:
            return StopLossCalculation(
                stop_price=0.0,
                stop_type=StopLossType.SID_METHOD,
                stop_style=StopLossStyle.NORMAL,
                risk_distance=0.0,
                risk_percent=0.0,
                risk_amount=0.0,
                is_valid=False,
                validation_notes="Empty period between signal and entry"
            )
        
        if signal_type == 'oversold':
            # LONG TRADE: Lowest low rounded DOWN
            lowest_low = period_df['low'].min()
            
            if use_rounding:
                stop_price = np.floor(lowest_low)
            else:
                stop_price = lowest_low
            
            # WAVE 3: Add pip buffer behind zone
            if use_pip_buffer and instrument:
                pip_buffer = self.get_stop_pips(instrument)
                pip_value = self.get_pip_value(instrument)
                stop_price = stop_price - (pip_buffer * pip_value)
                pip_buffer_applied = pip_buffer
            else:
                pip_buffer_applied = 0
            
            risk_distance = stop_price  # Will be adjusted in validation
            validation_notes = f"Lowest low: {lowest_low:.5f}, stop: {stop_price:.5f}"
            
        else:
            # SHORT TRADE: Highest high rounded UP
            highest_high = period_df['high'].max()
            
            if use_rounding:
                stop_price = np.ceil(highest_high)
            else:
                stop_price = highest_high
            
            # WAVE 3: Add pip buffer behind zone
            if use_pip_buffer and instrument:
                pip_buffer = self.get_stop_pips(instrument)
                pip_value = self.get_pip_value(instrument)
                stop_price = stop_price + (pip_buffer * pip_value)
                pip_buffer_applied = pip_buffer
            else:
                pip_buffer_applied = 0
            
            risk_distance = stop_price
            validation_notes = f"Highest high: {highest_high:.5f}, stop: {stop_price:.5f}"
        
        return StopLossCalculation(
            stop_price=stop_price,
            stop_type=StopLossType.SID_METHOD,
            stop_style=StopLossStyle.NORMAL,
            risk_distance=abs(stop_price - df['close'].loc[entry_date]) if entry_date in df.index else 0,
            risk_percent=0.0,  # To be filled by caller
            risk_amount=0.0,    # To be filled by caller
            is_valid=stop_price > 0,
            validation_notes=validation_notes,
            signal_date=signal_date,
            entry_date=entry_date,
            lowest_low=period_df['low'].min() if signal_type == 'oversold' else None,
            highest_high=period_df['high'].max() if signal_type == 'overbought' else None,
            pip_buffer_applied=pip_buffer_applied
        )
    
    # ========================================================================
    # SECTION 3: WAVE 2 - ATR-BASED STOP LOSS
    # ========================================================================
    
    def calculate_atr_stop_loss(self, df: pd.DataFrame, current_idx: int,
                                  entry_price: float, direction: str,
                                  instrument: str = None) -> StopLossCalculation:
        """
        Calculate ATR-based stop loss (Wave 2)
        
        Args:
            df: Price DataFrame with ATR column
            current_idx: Current index
            entry_price: Entry price
            direction: 'long' or 'short'
            instrument: Instrument for pip calculation
        
        Returns:
            StopLossCalculation
        """
        if not self.config.use_atr_stops:
            return None
        
        if 'atr' not in df.columns:
            # Calculate ATR if not present
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(window=self.config.atr_period).mean()
        
        atr = df['atr'].iloc[current_idx]
        atr_stop = self.config.atr_multiplier * atr
        
        if direction == 'long':
            stop_price = entry_price - atr_stop
        else:
            stop_price = entry_price + atr_stop
        
        # Apply pip buffer if requested (Wave 3)
        if self.config.use_pip_buffer and instrument:
            pip_buffer = self.get_stop_pips(instrument)
            pip_value = self.get_pip_value(instrument)
            if direction == 'long':
                stop_price = stop_price - (pip_buffer * pip_value)
            else:
                stop_price = stop_price + (pip_buffer * pip_value)
        
        return StopLossCalculation(
            stop_price=stop_price,
            stop_type=StopLossType.ATR_BASED,
            stop_style=StopLossStyle.NORMAL,
            risk_distance=abs(entry_price - stop_price),
            risk_percent=0.0,
            risk_amount=0.0,
            is_valid=stop_price > 0,
            validation_notes=f"ATR={atr:.5f}, multiplier={self.config.atr_multiplier}"
        )
    
    # ========================================================================
    # SECTION 4: WAVE 2 - SUPPORT/RESISTANCE STOP LOSS
    # ========================================================================
    
    def find_nearest_support(self, df: pd.DataFrame, current_idx: int,
                               current_price: float, lookback: int = 50) -> Optional[float]:
        """
        Find nearest support level (Wave 2)
        
        Args:
            df: Price DataFrame
            current_idx: Current index
            current_price: Current price
            lookback: Number of bars to look back
        
        Returns:
            Support price or None
        """
        recent_lows = df['low'].iloc[current_idx - lookback:current_idx]
        # Find swing lows
        supports = []
        for i in range(2, len(recent_lows) - 2):
            if (recent_lows.iloc[i] < recent_lows.iloc[i-1] and 
                recent_lows.iloc[i] < recent_lows.iloc[i-2] and
                recent_lows.iloc[i] < recent_lows.iloc[i+1] and
                recent_lows.iloc[i] < recent_lows.iloc[i+2]):
                supports.append(recent_lows.iloc[i])
        
        if not supports:
            return None
        
        # Find nearest support below current price
        supports_below = [s for s in supports if s < current_price]
        if supports_below:
            return max(supports_below)  # Closest support below
        
        return None
    
    def find_nearest_resistance(self, df: pd.DataFrame, current_idx: int,
                                  current_price: float, lookback: int = 50) -> Optional[float]:
        """
        Find nearest resistance level (Wave 2)
        
        Args:
            df: Price DataFrame
            current_idx: Current index
            current_price: Current price
            lookback: Number of bars to look back
        
        Returns:
            Resistance price or None
        """
        recent_highs = df['high'].iloc[current_idx - lookback:current_idx]
        # Find swing highs
        resistances = []
        for i in range(2, len(recent_highs) - 2):
            if (recent_highs.iloc[i] > recent_highs.iloc[i-1] and 
                recent_highs.iloc[i] > recent_highs.iloc[i-2] and
                recent_highs.iloc[i] > recent_highs.iloc[i+1] and
                recent_highs.iloc[i] > recent_highs.iloc[i+2]):
                resistances.append(recent_highs.iloc[i])
        
        if not resistances:
            return None
        
        # Find nearest resistance above current price
        resistances_above = [r for r in resistances if r > current_price]
        if resistances_above:
            return min(resistances_above)  # Closest resistance above
        
        return None
    
    def calculate_support_resistance_stop(self, df: pd.DataFrame, current_idx: int,
                                            entry_price: float, direction: str,
                                            instrument: str = None) -> StopLossCalculation:
        """
        Calculate stop loss based on support/resistance levels (Wave 2)
        
        Args:
            df: Price DataFrame
            current_idx: Current index
            entry_price: Entry price
            direction: 'long' or 'short'
            instrument: Instrument for pip calculation
        
        Returns:
            StopLossCalculation
        """
        if not self.config.use_support_resistance:
            return None
        
        lookback = self.config.support_resistance_lookback
        
        if direction == 'long':
            support = self.find_nearest_support(df, current_idx, entry_price, lookback)
            if support:
                stop_price = support
            else:
                # Fallback to SID method
                return None
        else:
            resistance = self.find_nearest_resistance(df, current_idx, entry_price, lookback)
            if resistance:
                stop_price = resistance
            else:
                return None
        
        # Apply pip buffer (Wave 3)
        if self.config.use_pip_buffer and instrument:
            pip_buffer = self.get_stop_pips(instrument)
            pip_value = self.get_pip_value(instrument)
            if direction == 'long':
                stop_price = stop_price - (pip_buffer * pip_value)
            else:
                stop_price = stop_price + (pip_buffer * pip_value)
        
        return StopLossCalculation(
            stop_price=stop_price,
            stop_type=StopLossType.SUPPORT_RESISTANCE,
            stop_style=StopLossStyle.NORMAL,
            risk_distance=abs(entry_price - stop_price),
            risk_percent=0.0,
            risk_amount=0.0,
            is_valid=stop_price > 0,
            validation_notes=f"Support/resistance based stop"
        )
    
    # ========================================================================
    # SECTION 5: WAVE 3 - PIP BUFFER STOP LOSS
    # ========================================================================
    
    def calculate_pip_buffer_stop(self, entry_price: float, direction: str,
                                    instrument: str, pips: int = None) -> StopLossCalculation:
        """
        Calculate simple pip buffer stop loss (Wave 3)
        
        Args:
            entry_price: Entry price
            direction: 'long' or 'short'
            instrument: Instrument symbol
            pips: Number of pips (if None, uses default)
        
        Returns:
            StopLossCalculation
        """
        if pips is None:
            pips = self.get_stop_pips(instrument)
        
        pip_value = self.get_pip_value(instrument)
        stop_distance = pips * pip_value
        
        if direction == 'long':
            stop_price = entry_price - stop_distance
        else:
            stop_price = entry_price + stop_distance
        
        return StopLossCalculation(
            stop_price=stop_price,
            stop_type=StopLossType.PIP_BUFFER,
            stop_style=StopLossStyle.NORMAL,
            risk_distance=stop_distance,
            risk_percent=0.0,
            risk_amount=0.0,
            is_valid=stop_price > 0,
            validation_notes=f"{pips} pips behind entry",
            pip_buffer_applied=pips
        )
    
    # ========================================================================
    # SECTION 6: WAVE 3 - AGGRESSIVE VS PASSIVE STOPS
    # ========================================================================
    
    def apply_stop_style(self, stop_calc: StopLossCalculation, entry_price: float,
                           direction: str, style: str = None) -> StopLossCalculation:
        """
        Apply aggressive or passive style to stop loss (Wave 3)
        
        Aggressive: Tighter stop (closer to entry)
        Passive: Wider stop (further from entry)
        
        Args:
            stop_calc: Original stop calculation
            entry_price: Entry price
            direction: 'long' or 'short'
            style: 'aggressive', 'normal', or 'passive'
        
        Returns:
            Modified StopLossCalculation
        """
        if style is None:
            style = self.config.default_stop_style
        
        if style == 'normal':
            return stop_calc
        
        risk_distance = stop_calc.risk_distance
        
        if style == 'aggressive':
            # Reduce risk distance by 30%
            new_risk_distance = risk_distance * 0.7
        else:  # passive
            # Increase risk distance by 30%
            new_risk_distance = risk_distance * 1.3
        
        if direction == 'long':
            new_stop_price = entry_price - new_risk_distance
        else:
            new_stop_price = entry_price + new_risk_distance
        
        new_stop_calc = StopLossCalculation(
            stop_price=new_stop_price,
            stop_type=stop_calc.stop_type,
            stop_style=StopLossStyle.AGGRESSIVE if style == 'aggressive' else StopLossStyle.PASSIVE,
            risk_distance=new_risk_distance,
            risk_percent=stop_calc.risk_percent,
            risk_amount=stop_calc.risk_amount,
            is_valid=stop_calc.is_valid,
            validation_notes=f"{style} style applied: {stop_calc.validation_notes}",
            pip_buffer_applied=stop_calc.pip_buffer_applied
        )
        
        return new_stop_calc
    
    # ========================================================================
    # SECTION 7: VALIDATION (Wave 3)
    # ========================================================================
    
    def validate_stop_distance(self, stop_calc: StopLossCalculation, entry_price: float,
                                 instrument: str) -> Tuple[bool, str]:
        """
        Validate stop loss meets minimum distance requirements (Wave 3)
        
        Args:
            stop_calc: Stop loss calculation
            entry_price: Entry price
            instrument: Instrument symbol
        
        Returns:
            (is_valid, message)
        """
        if not self.config.enforce_min_distance:
            return True, "Min distance not enforced"
        
        pips_distance = self.price_to_pips(instrument, abs(entry_price - stop_calc.stop_price))
        
        if pips_distance < self.config.min_distance_pips:
            return False, f"Stop too tight: {pips_distance:.1f} pips (need {self.config.min_distance_pips})"
        
        # Broker constraints
        if self.config.use_broker_constraints and pips_distance < self.config.broker_min_distance:
            return False, f"Stop violates broker minimum: {pips_distance:.1f} pips (broker min {self.config.broker_min_distance})"
        
        return True, f"Stop distance OK: {pips_distance:.1f} pips"
    
    def validate_stop_direction(self, stop_calc: StopLossCalculation, entry_price: float,
                                  direction: str) -> Tuple[bool, str]:
        """
        Validate stop loss is on correct side of entry (Wave 1)
        
        Args:
            stop_calc: Stop loss calculation
            entry_price: Entry price
            direction: 'long' or 'short'
        
        Returns:
            (is_valid, message)
        """
        if direction == 'long':
            if stop_calc.stop_price >= entry_price:
                return False, "Stop loss must be below entry price for long trades"
        else:
            if stop_calc.stop_price <= entry_price:
                return False, "Stop loss must be above entry price for short trades"
        
        return True, "Stop direction OK"
    
    # ========================================================================
    # SECTION 8: ALERT-BASED STOPS (Wave 2)
    # ========================================================================
    
    def create_stop_alert(self, stop_calc: StopLossCalculation, instrument: str,
                            alert_message: str = None) -> Dict:
        """
        Create an alert instead of a hard stop (Wave 2)
        
        Args:
            stop_calc: Stop loss calculation
            instrument: Instrument symbol
            alert_message: Custom alert message
        
        Returns:
            Alert configuration dictionary
        """
        if not self.config.use_alert_stops:
            return {'enabled': False}
        
        alert_trigger = stop_calc.stop_price
        
        # Add offset if configured
        if self.config.alert_trigger_offset > 0:
            if stop_calc.stop_style == StopLossStyle.AGGRESSIVE:
                alert_trigger = alert_trigger + self.config.alert_trigger_offset
            else:
                alert_trigger = alert_trigger - self.config.alert_trigger_offset
        
        if alert_message is None:
            alert_message = f"⚠️ Stop loss alert for {instrument}: {stop_calc.stop_price:.5f}"
        
        return {
            'enabled': True,
            'trigger_price': alert_trigger,
            'original_stop': stop_calc.stop_price,
            'message': alert_message,
            'offset': self.config.alert_trigger_offset
        }
    
    # ========================================================================
    # SECTION 9: COMPLETE STOP LOSS CALCULATION
    # ========================================================================
    
    def calculate_stop_loss(self, df: pd.DataFrame, entry_price: float,
                              entry_date: datetime, signal_date: datetime,
                              signal_type: str, direction: str,
                              instrument: str = None,
                              stop_style: str = None,
                              account_balance: float = None,
                              risk_percent: float = None) -> StopLossCalculation:
        """
        Complete stop loss calculation using all available methods (Wave 1, 2, 3)
        
        Priority order:
        1. SID Method (Wave 1)
        2. Support/Resistance (Wave 2) - if enabled and available
        3. ATR-based (Wave 2) - if enabled
        4. Pip buffer fallback (Wave 3)
        
        Args:
            df: Price DataFrame
            entry_price: Entry price
            entry_date: Entry date
            signal_date: RSI signal date
            signal_type: 'oversold' or 'overbought'
            direction: 'long' or 'short'
            instrument: Instrument symbol
            stop_style: 'aggressive', 'normal', or 'passive'
            account_balance: Account balance (for risk calculation)
            risk_percent: Risk percentage
        
        Returns:
            StopLossCalculation
        """
        if stop_style is None:
            stop_style = self.config.default_stop_style
        
        stop_calc = None
        
        # 1. Try SID Method (Wave 1)
        if self.config.use_sid_method_stops:
            stop_calc = self.calculate_sid_stop_loss(
                df, signal_date, entry_date, signal_type, instrument
            )
        
        # 2. Try Support/Resistance (Wave 2) - if SID method not available or as alternative
        if (stop_calc is None or not stop_calc.is_valid) and self.config.use_support_resistance:
            entry_idx = df.index.get_loc(entry_date)
            sr_stop = self.calculate_support_resistance_stop(df, entry_idx, entry_price, direction, instrument)
            if sr_stop and sr_stop.is_valid:
                stop_calc = sr_stop
        
        # 3. Try ATR-based (Wave 2)
        if (stop_calc is None or not stop_calc.is_valid) and self.config.use_atr_stops:
            entry_idx = df.index.get_loc(entry_date)
            atr_stop = self.calculate_atr_stop_loss(df, entry_idx, entry_price, direction, instrument)
            if atr_stop and atr_stop.is_valid:
                stop_calc = atr_stop
        
        # 4. Fallback to pip buffer (Wave 3)
        if stop_calc is None or not stop_calc.is_valid:
            stop_calc = self.calculate_pip_buffer_stop(entry_price, direction, instrument)
        
        # Apply stop style (aggressive/passive)
        if stop_style != 'normal':
            stop_calc = self.apply_stop_style(stop_calc, entry_price, direction, stop_style)
        
        # Validate stop direction
        is_valid, direction_msg = self.validate_stop_direction(stop_calc, entry_price, direction)
        if not is_valid:
            stop_calc.is_valid = False
            stop_calc.validation_notes = direction_msg
            return stop_calc
        
        # Validate stop distance
        is_valid, distance_msg = self.validate_stop_distance(stop_calc, entry_price, instrument)
        if not is_valid:
            stop_calc.is_valid = False
            stop_calc.validation_notes = distance_msg
            return stop_calc
        
        # Calculate risk metrics if account balance provided
        if account_balance and risk_percent:
            risk_distance = abs(entry_price - stop_calc.stop_price)
            risk_amount = account_balance * (risk_percent / 100)
            stop_calc.risk_percent = risk_percent
            stop_calc.risk_amount = risk_amount
            
            # Validate risk amount
            if risk_amount <= 0:
                stop_calc.is_valid = False
                stop_calc.validation_notes = "Invalid risk amount"
        
        stop_calc.is_valid = True
        return stop_calc
    
    # ========================================================================
    # SECTION 10: DYNAMIC STOP ADJUSTMENT (Wave 2)
    # ========================================================================
    
    def adjust_stop_to_breakeven(self, current_stop: float, entry_price: float,
                                   direction: str, profit_pips: float,
                                   breakeven_threshold_pips: float = 10) -> Tuple[float, bool]:
        """
        Adjust stop to breakeven when profit threshold is reached (Wave 2)
        
        Args:
            current_stop: Current stop price
            entry_price: Entry price
            direction: 'long' or 'short'
            profit_pips: Current profit in pips
            breakeven_threshold_pips: Minimum profit to move to breakeven
        
        Returns:
            (new_stop_price, was_adjusted)
        """
        if profit_pips >= breakeven_threshold_pips:
            if direction == 'long' and current_stop < entry_price:
                return entry_price, True
            elif direction == 'short' and current_stop > entry_price:
                return entry_price, True
        
        return current_stop, False
    
    def adjust_stop_to_partial(self, current_stop: float, entry_price: float,
                                 direction: str, highest_price: float,
                                 lowest_price: float, atr: float,
                                 trail_percentage: float = 0.5) -> Tuple[float, bool]:
        """
        Adjust stop to trail after reaching a profit target (Wave 2)
        
        Args:
            current_stop: Current stop price
            entry_price: Entry price
            direction: 'long' or 'short'
            highest_price: Highest price since entry (for longs)
            lowest_price: Lowest price since entry (for shorts)
            atr: Current ATR
            trail_percentage: Percentage of profit to trail (0-1)
        
        Returns:
            (new_stop_price, was_adjusted)
        """
        if direction == 'long':
            profit = highest_price - entry_price
            if profit > 0:
                new_stop = entry_price + (profit * trail_percentage)
                if new_stop > current_stop:
                    return new_stop, True
        else:
            profit = entry_price - lowest_price
            if profit > 0:
                new_stop = entry_price - (profit * trail_percentage)
                if new_stop < current_stop:
                    return new_stop, True
        
        return current_stop, False
    
    # ========================================================================
    # SECTION 11: STOP LOSS FOR ALERT-BASED TRADING
    # ========================================================================
    
    def get_stop_alert_config(self, stop_calc: StopLossCalculation, 
                                instrument: str) -> Dict:
        """
        Get alert configuration for stop loss (Wave 2)
        
        Returns:
            Alert configuration dictionary
        """
        return self.create_stop_alert(stop_calc, instrument)


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TESTING STOP LOSS CALCULATOR v3.0")
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
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, float('nan'))
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)
    
    # Find a signal date
    oversold_mask = df['rsi'] < 30
    if oversold_mask.any():
        signal_idx = oversold_mask[oversold_mask].index[-1]
        signal_date = signal_idx
        
        # Find entry date (when MACD crosses up)
        # Simplified: use a later date
        entry_idx = df.index.get_loc(signal_idx) + 5
        if entry_idx < len(df):
            entry_date = df.index[entry_idx]
            entry_price = df['close'].iloc[entry_idx]
            
            print(f"📊 Test Data:")
            print(f"  Signal Date: {signal_date}")
            print(f"  Entry Date: {entry_date}")
            print(f"  Entry Price: {entry_price:.2f}")
            
            # Initialize calculator
            config = StopLossConfig(
                use_sid_method_stops=True,
                use_atr_stops=True,
                use_support_resistance=True,
                use_pip_buffer=True,
                enforce_min_distance=True
            )
            calculator = StopLossCalculator(config, verbose=True)
            
            # Calculate SID stop loss
            print(f"\n📊 SID Method Stop Loss:")
            stop_calc = calculator.calculate_sid_stop_loss(
                df, signal_date, entry_date, 'oversold', 'EUR_USD'
            )
            print(f"  Stop Price: {stop_calc.stop_price:.2f}")
            print(f"  Risk Distance: {stop_calc.risk_distance:.2f}")
            print(f"  Valid: {stop_calc.is_valid}")
            print(f"  Notes: {stop_calc.validation_notes}")
            
            # Calculate with aggressive style
            print(f"\n📊 Aggressive Stop Loss:")
            aggressive_stop = calculator.apply_stop_style(
                stop_calc, entry_price, 'long', 'aggressive'
            )
            print(f"  Stop Price: {aggressive_stop.stop_price:.2f}")
            print(f"  Risk Distance: {aggressive_stop.risk_distance:.2f}")
            
            # Calculate with passive style
            print(f"\n📊 Passive Stop Loss:")
            passive_stop = calculator.apply_stop_style(
                stop_calc, entry_price, 'long', 'passive'
            )
            print(f"  Stop Price: {passive_stop.stop_price:.2f}")
            print(f"  Risk Distance: {passive_stop.risk_distance:.2f}")
            
            # Test validation
            print(f"\n📊 Validation:")
            is_valid, msg = calculator.validate_stop_distance(stop_calc, entry_price, 'EUR_USD')
            print(f"  Distance validation: {msg}")
    
    print(f"\n✅ Stop loss calculator test complete")