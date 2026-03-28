#!/usr/bin/env python3
"""
Danislav Dantev - Institutional Order Flow Detection
Core Concepts:
- Order Blocks (Bullish/Bearish)
- Fair Value Gaps (FVG)
- Liquidity Sweeps
- Break of Structure (BOS)
- Change of Character (CHoCH)
- Premium/Discount Arrays
- Optimal Trade Entry (OTE)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class OrderBlock:
    """Institutional Order Block"""
    type: str  # 'bullish' or 'bearish'
    start_idx: int
    end_idx: int
    high: float
    low: float
    close: float
    open_price: float
    volume: float
    strength: float
    mitigated: bool = False
    mitigation_level: float = 0.0
    mitigation_price: Optional[float] = None
    created_at: datetime = None
    touched: bool = False
    touch_count: int = 0
    
    def __hash__(self):
        """Make OrderBlock hashable for set operations"""
        return hash((self.start_idx, self.end_idx, self.type))
    
    def __eq__(self, other):
        if not isinstance(other, OrderBlock):
            return False
        return (self.start_idx == other.start_idx and 
                self.end_idx == other.end_idx and 
                self.type == other.type)


@dataclass
class FairValueGap:
    """Fair Value Gap (Imbalance)"""
    start_idx: int
    end_idx: int
    gap_top: float
    gap_bottom: float
    gap_size_pips: float
    age_bars: int
    filled: bool = False
    fill_price: Optional[float] = None
    fill_idx: Optional[int] = None
    strength: float = 0.0
    volume_imbalance: float = 0.0
    
    def __hash__(self):
        """Make FairValueGap hashable for set operations"""
        return hash((self.start_idx, self.end_idx))
    
    def __eq__(self, other):
        if not isinstance(other, FairValueGap):
            return False
        return (self.start_idx == other.start_idx and 
                self.end_idx == other.end_idx)


@dataclass
class LiquidityLevel:
    """Liquidity level (Swing High/Low)"""
    type: str  # 'high' or 'low'
    price: float
    idx: int
    volume: float
    swept: bool = False
    sweep_price: Optional[float] = None
    sweep_idx: Optional[int] = None
    sweep_time: Optional[datetime] = None
    strength: float = 0.0
    
    def __hash__(self):
        """Make LiquidityLevel hashable for set operations"""
        return hash((self.idx, self.type, self.price))
    
    def __eq__(self, other):
        if not isinstance(other, LiquidityLevel):
            return False
        return (self.idx == other.idx and 
                self.type == other.type and 
                self.price == other.price)


@dataclass
class BreakOfStructure:
    """Break of Structure"""
    type: str  # 'bos' or 'choch'
    direction: str  # 'bullish' or 'bearish'
    break_idx: int
    break_price: float
    previous_structure: float
    volume_spike: bool
    confirmed: bool = False
    confirmation_candles: int = 0
    retest: bool = False
    retest_price: Optional[float] = None
    strength: float = 0.0
    
    def __hash__(self):
        """Make BreakOfStructure hashable for set operations"""
        return hash((self.break_idx, self.type, self.direction))
    
    def __eq__(self, other):
        if not isinstance(other, BreakOfStructure):
            return False
        return (self.break_idx == other.break_idx and 
                self.type == other.type)


class InstitutionalOrderFlow:
    """
    Danislav Dantev's Institutional Order Flow Analysis
    Identifies where smart money is placing orders
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ob_config = config.get('ORDER_BLOCK_CONFIG', {})
        self.fvg_config = config.get('FVG_CONFIG', {})
        self.liquidity_config = config.get('LIQUIDITY_CONFIG', {})
        self.bos_config = config.get('BOS_CONFIG', {})
        self.choch_config = config.get('CHOCH_CONFIG', {})
        self.fib_config = config.get('FIBONACCI_CONFIG', {})
        self.pd_config = config.get('PD_ARRAY_CONFIG', {})
        
    def detect_order_blocks(self, df: pd.DataFrame) -> List[OrderBlock]:
        """
        Detect Institutional Order Blocks
        Bullish OB: Last down candle before a strong up move
        Bearish OB: Last up candle before a strong down move
        """
        order_blocks = []
        volume_threshold = self.ob_config.get('volume_spike_threshold', 1.5)
        strength_threshold = self.ob_config.get('strength_threshold', 0.6)
        
        for i in range(3, len(df) - 5):
            current = df.iloc[i]
            next_candles = df.iloc[i+1:i+5]
            
            if len(next_candles) < 3:
                continue
            
            # Calculate average volume
            avg_volume = df['volume'].iloc[max(0, i-20):i].mean()
            volume_spike = current['volume'] > avg_volume * volume_threshold
            
            # Bullish Order Block detection
            if current['close'] < current['open']:  # Red candle
                # Check if following candles are bullish
                bullish_following = (next_candles['close'] > next_candles['open']).sum()
                if bullish_following >= 3:  # At least 3 bullish candles
                    # Calculate move size
                    move_start = next_candles['close'].iloc[0]
                    move_end = next_candles['close'].iloc[-1]
                    move_size = (move_end - move_start) / move_start
                    
                    if move_size > 0.003:  # 0.3% minimum move
                        # Calculate strength
                        strength = min(1.0, move_size * 100)
                        if volume_spike:
                            strength += 0.2
                        strength = min(strength, 1.0)
                        
                        if strength >= strength_threshold:
                            ob = OrderBlock(
                                type='bullish',
                                start_idx=i,
                                end_idx=i,
                                high=current['high'],
                                low=current['low'],
                                close=current['close'],
                                open_price=current['open'],
                                volume=current['volume'],
                                strength=strength,
                                created_at=df.index[i]
                            )
                            order_blocks.append(ob)
            
            # Bearish Order Block detection
            elif current['close'] > current['open']:  # Green candle
                bearish_following = (next_candles['close'] < next_candles['open']).sum()
                if bearish_following >= 3:
                    move_start = next_candles['close'].iloc[0]
                    move_end = next_candles['close'].iloc[-1]
                    move_size = (move_start - move_end) / move_start
                    
                    if move_size > 0.003:
                        strength = min(1.0, move_size * 100)
                        if volume_spike:
                            strength += 0.2
                        strength = min(strength, 1.0)
                        
                        if strength >= strength_threshold:
                            ob = OrderBlock(
                                type='bearish',
                                start_idx=i,
                                end_idx=i,
                                high=current['high'],
                                low=current['low'],
                                close=current['close'],
                                open_price=current['open'],
                                volume=current['volume'],
                                strength=strength,
                                created_at=df.index[i]
                            )
                            order_blocks.append(ob)
        
        return order_blocks
    
    def detect_fair_value_gaps(self, df: pd.DataFrame) -> List[FairValueGap]:
        """
        Detect Fair Value Gaps (Imbalances)
        FVG = gap between candles where price didn't trade
        """
        fvgs = []
        min_gap_pips = self.fvg_config.get('min_gap_pips', 10)
        pip_value = 0.0001  # Default for non-JPY
        
        for i in range(2, len(df)):
            candle1 = df.iloc[i-2]
            candle2 = df.iloc[i-1]
            candle3 = df.iloc[i]
            
            # Calculate gap size in pips
            def gap_to_pips(gap):
                return gap / pip_value if gap < 1 else gap / 0.01
            
            # Bullish FVG (gap up)
            if candle2['low'] > candle1['high']:
                gap_bottom = candle1['high']
                gap_top = candle2['low']
                gap_size = gap_top - gap_bottom
                gap_pips = gap_to_pips(gap_size)
                
                if gap_pips >= min_gap_pips:
                    # Calculate volume imbalance
                    avg_volume = df['volume'].iloc[max(0, i-20):i].mean()
                    volume_imbalance = candle2['volume'] / avg_volume if avg_volume > 0 else 1.0
                    
                    # Calculate strength
                    size_factor = min(gap_pips / 50, 1.0)
                    volume_factor = min(volume_imbalance / 1.5, 1.0)
                    strength = (size_factor * 0.5 + volume_factor * 0.5)
                    
                    fvg = FairValueGap(
                        start_idx=i-2,
                        end_idx=i,
                        gap_bottom=gap_bottom,
                        gap_top=gap_top,
                        gap_size_pips=gap_pips,
                        age_bars=0,
                        strength=strength,
                        volume_imbalance=volume_imbalance
                    )
                    fvgs.append(fvg)
            
            # Bearish FVG (gap down)
            elif candle2['high'] < candle1['low']:
                gap_bottom = candle2['high']
                gap_top = candle1['low']
                gap_size = gap_top - gap_bottom
                gap_pips = gap_to_pips(gap_size)
                
                if gap_pips >= min_gap_pips:
                    avg_volume = df['volume'].iloc[max(0, i-20):i].mean()
                    volume_imbalance = candle2['volume'] / avg_volume if avg_volume > 0 else 1.0
                    
                    size_factor = min(gap_pips / 50, 1.0)
                    volume_factor = min(volume_imbalance / 1.5, 1.0)
                    strength = (size_factor * 0.5 + volume_factor * 0.5)
                    
                    fvg = FairValueGap(
                        start_idx=i-2,
                        end_idx=i,
                        gap_bottom=gap_bottom,
                        gap_top=gap_top,
                        gap_size_pips=gap_pips,
                        age_bars=0,
                        strength=strength,
                        volume_imbalance=volume_imbalance
                    )
                    fvgs.append(fvg)
        
        # Update age and check if filled
        current_idx = len(df) - 1
        for fvg in fvgs:
            fvg.age_bars = current_idx - fvg.end_idx
            
            # Check if FVG has been filled
            for j in range(fvg.end_idx + 1, min(fvg.end_idx + fvg.age_bars + 1, len(df))):
                candle = df.iloc[j]
                if (candle['low'] <= fvg.gap_top and candle['high'] >= fvg.gap_bottom):
                    fvg.filled = True
                    fvg.fill_idx = j
                    fvg.fill_price = candle['close']
                    break
        
        return fvgs
    
    def detect_liquidity_levels(self, df: pd.DataFrame) -> Dict[str, List[LiquidityLevel]]:
        """
        Detect liquidity levels (swing highs/lows)
        Smart money hunts these levels
        """
        lookback = self.liquidity_config.get('swing_high_lookback', 20)
        tolerance = self.liquidity_config.get('equal_highs_tolerance', 0.001)
        
        highs = []
        lows = []
        
        # Find swing highs
        for i in range(lookback, len(df) - lookback):
            current_high = df['high'].iloc[i]
            is_swing_high = True
            
            # Check left side
            for j in range(i - lookback, i):
                if df['high'].iloc[j] >= current_high:
                    is_swing_high = False
                    break
            
            # Check right side
            if is_swing_high:
                for j in range(i + 1, i + lookback + 1):
                    if df['high'].iloc[j] >= current_high:
                        is_swing_high = False
                        break
            
            if is_swing_high:
                # Calculate strength based on volume and surrounding context
                avg_volume = df['volume'].iloc[max(0, i-lookback):i].mean()
                volume_ratio = df['volume'].iloc[i] / avg_volume if avg_volume > 0 else 1.0
                strength = min(1.0, volume_ratio * 0.5 + 0.5)
                
                highs.append(LiquidityLevel(
                    type='high',
                    price=current_high,
                    idx=i,
                    volume=df['volume'].iloc[i],
                    strength=strength
                ))
        
        # Find swing lows
        for i in range(lookback, len(df) - lookback):
            current_low = df['low'].iloc[i]
            is_swing_low = True
            
            for j in range(i - lookback, i):
                if df['low'].iloc[j] <= current_low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                for j in range(i + 1, i + lookback + 1):
                    if df['low'].iloc[j] <= current_low:
                        is_swing_low = False
                        break
            
            if is_swing_low:
                avg_volume = df['volume'].iloc[max(0, i-lookback):i].mean()
                volume_ratio = df['volume'].iloc[i] / avg_volume if avg_volume > 0 else 1.0
                strength = min(1.0, volume_ratio * 0.5 + 0.5)
                
                lows.append(LiquidityLevel(
                    type='low',
                    price=current_low,
                    idx=i,
                    volume=df['volume'].iloc[i],
                    strength=strength
                ))
        
        # Check for liquidity sweeps
        current_price = df['close'].iloc[-1]
        
        for high in highs:
            # Check if price swept this high
            for j in range(high.idx, len(df)):
                if df['high'].iloc[j] >= high.price:
                    high.swept = True
                    high.sweep_idx = j
                    high.sweep_price = df['high'].iloc[j]
                    break
        
        for low in lows:
            for j in range(low.idx, len(df)):
                if df['low'].iloc[j] <= low.price:
                    low.swept = True
                    low.sweep_idx = j
                    low.sweep_price = df['low'].iloc[j]
                    break
        
        # Find equal highs (potential double tops) - using dictionary for deduplication
        equal_highs_dict = {}
        for i, high1 in enumerate(highs):
            for high2 in highs[i+1:]:
                if abs(high1.price - high2.price) / high1.price < tolerance:
                    high1.swept = True
                    high2.swept = True
                    # Use price as key for deduplication
                    equal_highs_dict[high1.idx] = high1
                    equal_highs_dict[high2.idx] = high2
        
        # Find equal lows (potential double bottoms) - using dictionary for deduplication
        equal_lows_dict = {}
        for i, low1 in enumerate(lows):
            for low2 in lows[i+1:]:
                if abs(low1.price - low2.price) / low1.price < tolerance:
                    low1.swept = True
                    low2.swept = True
                    equal_lows_dict[low1.idx] = low1
                    equal_lows_dict[low2.idx] = low2
        
        return {
            'swing_highs': highs,
            'swing_lows': lows,
            'equal_highs': list(equal_highs_dict.values()),
            'equal_lows': list(equal_lows_dict.values())
        }
    
    def detect_break_of_structure(self, df: pd.DataFrame, 
                                    liquidity_levels: Dict) -> List[BreakOfStructure]:
        """
        Detect Break of Structure (BOS)
        Price breaking previous swing high/low
        """
        bos_list = []
        threshold_pips = self.bos_config.get('bos_threshold_pips', 15)
        pip_value = 0.0001
        confirmation_candles = self.bos_config.get('confirmation_candles', 2)
        
        # Function to convert price difference to pips
        def to_pips(diff):
            return diff / pip_value if diff < 1 else diff / 0.01
        
        # Check bullish BOS (break above swing high)
        for level in liquidity_levels.get('swing_highs', []):
            for i in range(level.idx + 1, min(level.idx + 20, len(df))):
                if df['high'].iloc[i] > level.price:
                    break_size = df['high'].iloc[i] - level.price
                    break_pips = to_pips(break_size)
                    
                    if break_pips >= threshold_pips:
                        # Check if confirmed
                        confirmed = False
                        if i + confirmation_candles < len(df):
                            closes_above = all(
                                df['close'].iloc[i + j] > level.price 
                                for j in range(confirmation_candles)
                            )
                            confirmed = closes_above
                        
                        # Check volume spike
                        avg_volume = df['volume'].iloc[max(0, level.idx-20):level.idx].mean()
                        volume_spike = df['volume'].iloc[i] > avg_volume * 1.5
                        
                        # Calculate strength
                        strength = min(1.0, break_pips / 50 + (0.2 if volume_spike else 0))
                        
                        bos = BreakOfStructure(
                            type='bos',
                            direction='bullish',
                            break_idx=i,
                            break_price=df['high'].iloc[i],
                            previous_structure=level.price,
                            volume_spike=volume_spike,
                            confirmed=confirmed,
                            confirmation_candles=confirmation_candles if confirmed else 0,
                            strength=strength
                        )
                        bos_list.append(bos)
                        break
        
        # Check bearish BOS (break below swing low)
        for level in liquidity_levels.get('swing_lows', []):
            for i in range(level.idx + 1, min(level.idx + 20, len(df))):
                if df['low'].iloc[i] < level.price:
                    break_size = level.price - df['low'].iloc[i]
                    break_pips = to_pips(break_size)
                    
                    if break_pips >= threshold_pips:
                        confirmed = False
                        if i + confirmation_candles < len(df):
                            closes_below = all(
                                df['close'].iloc[i + j] < level.price 
                                for j in range(confirmation_candles)
                            )
                            confirmed = closes_below
                        
                        avg_volume = df['volume'].iloc[max(0, level.idx-20):level.idx].mean()
                        volume_spike = df['volume'].iloc[i] > avg_volume * 1.5
                        
                        strength = min(1.0, break_pips / 50 + (0.2 if volume_spike else 0))
                        
                        bos = BreakOfStructure(
                            type='bos',
                            direction='bearish',
                            break_idx=i,
                            break_price=df['low'].iloc[i],
                            previous_structure=level.price,
                            volume_spike=volume_spike,
                            confirmed=confirmed,
                            confirmation_candles=confirmation_candles if confirmed else 0,
                            strength=strength
                        )
                        bos_list.append(bos)
                        break
        
        return bos_list
    
    def detect_change_of_character(self, df: pd.DataFrame, 
                                     bos_list: List[BreakOfStructure]) -> List[BreakOfStructure]:
        """
        Detect Change of Character (CHoCH)
        When price breaks trend and retraces to confirm
        """
        choch_list = []
        retracement_threshold = self.choch_config.get('retracement_threshold', 0.382)
        confirmation_candles = self.choch_config.get('confirmation_candles', 3)
        
        if len(bos_list) < 2:
            return choch_list
        
        for i in range(1, len(bos_list)):
            prev_bos = bos_list[i-1]
            curr_bos = bos_list[i]
            
            # Check if direction changed
            if prev_bos.direction != curr_bos.direction:
                # Calculate retracement
                if curr_bos.direction == 'bullish':
                    retracement = (curr_bos.break_price - prev_bos.break_price) / prev_bos.break_price
                else:
                    retracement = (prev_bos.break_price - curr_bos.break_price) / prev_bos.break_price
                
                if retracement >= retracement_threshold:
                    # Check confirmation
                    confirmed = False
                    start_idx = curr_bos.break_idx
                    if start_idx + confirmation_candles < len(df):
                        if curr_bos.direction == 'bullish':
                            confirmed = all(
                                df['close'].iloc[start_idx + j] > df['close'].iloc[start_idx - 1]
                                for j in range(confirmation_candles)
                            )
                        else:
                            confirmed = all(
                                df['close'].iloc[start_idx + j] < df['close'].iloc[start_idx - 1]
                                for j in range(confirmation_candles)
                            )
                    
                    strength = min(1.0, retracement * 2 + (0.2 if confirmed else 0))
                    
                    choch = BreakOfStructure(
                        type='choch',
                        direction=curr_bos.direction,
                        break_idx=curr_bos.break_idx,
                        break_price=curr_bos.break_price,
                        previous_structure=prev_bos.break_price,
                        volume_spike=curr_bos.volume_spike,
                        confirmed=confirmed,
                        confirmation_candles=confirmation_candles if confirmed else 0,
                        strength=strength
                    )
                    choch_list.append(choch)
        
        return choch_list
    
    def calculate_premium_discount(self, df: pd.DataFrame, 
                                    lookback: int = None) -> pd.Series:
        """
        Calculate Premium/Discount Array
        Premium = price above 61.8% of range
        Discount = price below 38.2% of range
        """
        if lookback is None:
            lookback = self.pd_config.get('range_bars', 50)
        
        smoothing = self.pd_config.get('premium_discount_smoothing', 5)
        
        premium_discount = pd.Series(index=df.index, dtype=float)
        
        for i in range(lookback, len(df)):
            range_high = df['high'].iloc[i-lookback:i].max()
            range_low = df['low'].iloc[i-lookback:i].min()
            range_size = range_high - range_low
            
            if range_size > 0:
                current_price = df['close'].iloc[i]
                position = (current_price - range_low) / range_size
                premium_discount.iloc[i] = position
        
        # Apply smoothing
        premium_discount = premium_discount.rolling(window=smoothing, min_periods=1).mean()
        
        return premium_discount
    
    def calculate_ote_levels(self, high: float, low: float) -> Dict[str, float]:
        """
        Calculate Optimal Trade Entry (OTE) Levels
        Based on Fibonacci retracements
        """
        range_size = high - low
        fib_levels = self.fib_config.get('fib_levels', [0.236, 0.382, 0.5, 0.618, 0.705, 0.786, 0.886])
        
        levels = {}
        for level in fib_levels:
            levels[f'fib_{int(level*1000)}'] = low + range_size * level
        
        levels['premium'] = low + range_size * 0.382
        levels['discount'] = low + range_size * 0.618
        levels['fair_value'] = low + range_size * 0.5
        levels['golden_ratio'] = low + range_size * 0.618
        levels['deep_retracement'] = low + range_size * 0.705
        levels['extreme_retracement'] = low + range_size * 0.79
        
        return levels
    
    def calculate_trend_strength(self, df: pd.DataFrame, lookback: int = 20) -> float:
        """
        Calculate overall trend strength using ADX-like calculation
        """
        if len(df) < lookback:
            return 0.5
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Calculate directional movement
        plus_dm = np.zeros(len(df))
        minus_dm = np.zeros(len(df))
        
        for i in range(1, len(df)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
        
        # Calculate true range
        tr = np.zeros(len(df))
        for i in range(1, len(df)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Smooth using Wilder's method
        period = lookback
        atr = np.zeros(len(df))
        plus_di = np.zeros(len(df))
        minus_di = np.zeros(len(df))
        
        for i in range(period, len(df)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period if i > period else tr[i-period:i].mean()
            plus_di[i] = (plus_di[i-1] * (period - 1) + plus_dm[i]) / period if i > period else plus_dm[i-period:i].mean()
            minus_di[i] = (minus_di[i-1] * (period - 1) + minus_dm[i]) / period if i > period else minus_dm[i-period:i].mean()
        
        # Calculate DX and ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
        adx = np.zeros(len(df))
        
        for i in range(period, len(df)):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period if i > period else dx[i-period:i].mean()
        
        # Normalize to 0-1
        adx_norm = np.clip(adx[-1] / 100, 0, 1) if len(adx) > 0 else 0.5
        
        return float(adx_norm)
    
    def analyze(self, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        Complete institutional order flow analysis
        """
        # Detect all components
        order_blocks = self.detect_order_blocks(df)
        fvgs = self.detect_fair_value_gaps(df)
        liquidity = self.detect_liquidity_levels(df)
        bos_list = self.detect_break_of_structure(df, liquidity)
        choch_list = self.detect_change_of_character(df, bos_list)
        premium_discount = self.calculate_premium_discount(df)
        
        # Calculate OTE levels from recent range
        recent_high = df['high'].iloc[-50:].max()
        recent_low = df['low'].iloc[-50:].min()
        ote_levels = self.calculate_ote_levels(recent_high, recent_low)
        
        # Determine market structure
        current_pd = premium_discount.iloc[-1] if not premium_discount.empty else 0.5
        is_premium = current_pd > 0.618
        is_discount = current_pd < 0.382
        
        # Calculate trend strength
        trend_strength = self.calculate_trend_strength(df)
        
        # Find nearest order block
        nearest_ob = None
        min_distance = float('inf')
        for ob in order_blocks:
            distance = abs(current_price - ob.close) / current_price
            if distance < min_distance and not ob.mitigated:
                min_distance = distance
                nearest_ob = ob
        
        # Find unfilled FVG
        unfilled_fvgs = [f for f in fvgs if not f.filled]
        nearest_fvg = min(unfilled_fvgs, key=lambda f: abs(current_price - f.gap_top)) if unfilled_fvgs else None
        
        # Calculate confluence score
        confluence_score = 0
        
        if nearest_ob and nearest_ob.strength > 0.7:
            confluence_score += 25
        elif nearest_ob and nearest_ob.strength > 0.5:
            confluence_score += 15
        
        if nearest_fvg and nearest_fvg.strength > 0.7:
            confluence_score += 20
        elif nearest_fvg and nearest_fvg.strength > 0.5:
            confluence_score += 10
        
        if bos_list and bos_list[-1].confirmed:
            confluence_score += 15
        
        if choch_list and choch_list[-1].confirmed:
            confluence_score += 20
        
        if is_discount:
            confluence_score += 10
        if is_premium:
            confluence_score += 10
        
        if trend_strength > 0.7:
            confluence_score += 10
        
        # Determine trade direction
        trade_direction = None
        
        if is_discount and nearest_ob and nearest_ob.type == 'bullish':
            trade_direction = 'long'
        elif is_premium and nearest_ob and nearest_ob.type == 'bearish':
            trade_direction = 'short'
        
        # Calculate entry, stop, target
        entry_price = None
        stop_loss = None
        take_profit = None
        risk_reward = 0
        
        if trade_direction == 'long' and nearest_ob:
            entry_price = nearest_ob.close
            stop_loss = nearest_ob.low * 0.998  # Just below order block
            target_risk = entry_price - stop_loss
            
            # Check OTE alignment
            if entry_price <= ote_levels['golden_ratio']:
                take_profit = entry_price + (target_risk * 5)  # 5:1 for OTE entry
            else:
                take_profit = entry_price + (target_risk * 3)  # 3:1 default
            
            risk_reward = (take_profit - entry_price) / (entry_price - stop_loss) if (entry_price - stop_loss) > 0 else 0
        
        elif trade_direction == 'short' and nearest_ob:
            entry_price = nearest_ob.close
            stop_loss = nearest_ob.high * 1.002  # Just above order block
            target_risk = stop_loss - entry_price
            
            if entry_price >= ote_levels['golden_ratio']:
                take_profit = entry_price - (target_risk * 5)
            else:
                take_profit = entry_price - (target_risk * 3)
            
            risk_reward = (entry_price - take_profit) / (stop_loss - entry_price) if (stop_loss - entry_price) > 0 else 0
        
        # Determine signal strength
        if confluence_score >= 70 and risk_reward >= 2.0:
            signal_strength = "STRONG"
        elif confluence_score >= 50 and risk_reward >= 2.0:
            signal_strength = "MODERATE"
        elif confluence_score >= 30:
            signal_strength = "WEAK"
        else:
            signal_strength = "AVOID"
        
        return {
            'timestamp': datetime.now(),
            'current_price': current_price,
            'order_blocks': order_blocks,
            'fair_value_gaps': fvgs,
            'liquidity_levels': liquidity,
            'break_of_structure': bos_list,
            'change_of_character': choch_list,
            'premium_discount': current_pd,
            'is_premium': is_premium,
            'is_discount': is_discount,
            'ote_levels': ote_levels,
            'trend_strength': trend_strength,
            'confluence_score': confluence_score,
            'signal_strength': signal_strength,
            'trade_direction': trade_direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward': risk_reward,
            'nearest_order_block': nearest_ob,
            'nearest_fvg': nearest_fvg
        }