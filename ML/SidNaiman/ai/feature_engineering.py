#!/usr/bin/env python3
"""
Feature Engineering Module for SID Method - AUGMENTED VERSION
=============================================================================
Creates comprehensive features for ML models incorporating ALL THREE WAVES:

WAVE 1 (Core Quick Win):
- RSI with exact thresholds (30/70)
- MACD line, signal line, histogram
- RSI oversold/overbought binary signals
- MACD alignment and cross detection
- RSI 50 target proximity

WAVE 2 (Live Sessions & Q&A):
- Divergence detection (price vs RSI, price vs MACD)
- Pattern recognition (W, M, Head & Shoulders)
- Market context features (trend strength, volatility regime)
- Alternative take profit targets (50-SMA, points)
- Reachability metrics

WAVE 3 (Academy Support Sessions):
- Session-based features (Asian, London, US, Overlap)
- Zone quality metrics (tightness, violence)
- Precision RSI crossing detection
- Stop loss pip buffer features
- Minimum candle requirements

Author: Sid Naiman / Trading Cafe
Version: 3.0 (Fully Augmented)
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Try to import tqdm
try:
    from tqdm import tqdm
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

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering (Wave 1, 2, 3)"""
    # Wave 1: Core SID parameters
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    rsi_target: int = 50
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Wave 2: Advanced features
    divergence_lookback: int = 20
    pattern_lookback: int = 30
    volatility_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    momentum_periods: List[int] = field(default_factory=lambda: [5, 10, 20])
    
    # Wave 2: Market context
    trend_lookback: int = 50
    sma_periods: List[int] = field(default_factory=lambda: [20, 50, 200])
    
    # Wave 3: Session and quality features
    use_session_features: bool = True
    use_zone_quality: bool = True
    strict_rsi: bool = True
    min_pattern_candles: int = 7
    
    # Wave 3: Stop loss features
    stop_pips_default: int = 5
    stop_pips_yen: int = 10
    
    # Target horizons (bars)
    target_horizons: List[int] = field(default_factory=lambda: [5, 10, 20])


class FeatureEngineer:
    """
    Comprehensive feature engineering for SID Method
    Creates features from raw price data for ML model training
    """
    
    def __init__(self, config: FeatureEngineeringConfig = None, verbose: bool = True):
        """
        Initialize the feature engineer with configuration
        
        Args:
            config: FeatureEngineeringConfig instance
            verbose: Enable verbose output
        """
        self.config = config or FeatureEngineeringConfig()
        self.verbose = verbose
        self.feature_names = []
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🔧 FEATURE ENGINEERING v3.0 (Fully Augmented)")
            print(f"{'='*60}")
            print(f"📊 RSI: period={self.config.rsi_period}, thresholds={self.config.rsi_oversold}/{self.config.rsi_overbought}")
            print(f"📈 MACD: {self.config.macd_fast}/{self.config.macd_slow}/{self.config.macd_signal}")
            print(f"🔄 Divergence lookback: {self.config.divergence_lookback}")
            print(f"📐 Pattern lookback: {self.config.pattern_lookback}")
            print(f"🌍 Session features: {'Enabled' if self.config.use_session_features else 'Disabled'}")
            print(f"⭐ Zone quality: {'Enabled' if self.config.use_zone_quality else 'Disabled'}")
            print(f"{'='*60}\n")
    
    # ========================================================================
    # SECTION 1: CORE SID INDICATORS (Wave 1)
    # ========================================================================
    
    def add_rsi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add RSI features (Wave 1 - Core)
        
        Features:
        - rsi: RSI value
        - rsi_oversold: binary (1 if RSI < 30)
        - rsi_overbought: binary (1 if RSI > 70)
        - rsi_mid: binary (1 if RSI between 45-55)
        - rsi_change: period-over-period change
        - rsi_change_X: multi-period changes
        """
        df = df.copy()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_period).mean()
        rs = gain / loss.replace(0, float('nan'))
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
        
        # WAVE 3: Strict RSI thresholds (exact, no "near")
        if self.config.strict_rsi:
            df['rsi_oversold'] = (df['rsi'] < self.config.rsi_oversold).astype(int)
            df['rsi_overbought'] = (df['rsi'] > self.config.rsi_overbought).astype(int)
        else:
            df['rsi_oversold'] = (df['rsi'] <= self.config.rsi_oversold).astype(int)
            df['rsi_overbought'] = (df['rsi'] >= self.config.rsi_overbought).astype(int)
        
        # WAVE 3: Exact crossing detection
        df['rsi_crossed_below_30'] = ((df['rsi'].shift(1) >= self.config.rsi_oversold) & 
                                       (df['rsi'] < self.config.rsi_oversold)).astype(int)
        df['rsi_crossed_above_70'] = ((df['rsi'].shift(1) <= self.config.rsi_overbought) & 
                                       (df['rsi'] > self.config.rsi_overbought)).astype(int)
        
        # RSI mid-range (45-55) - potential consolidation
        df['rsi_mid'] = ((df['rsi'] >= 45) & (df['rsi'] <= 55)).astype(int)
        
        # RSI changes
        df['rsi_change'] = df['rsi'].diff()
        df['rsi_change_3'] = df['rsi'].diff(3)
        df['rsi_change_5'] = df['rsi'].diff(5)
        df['rsi_change_10'] = df['rsi'].diff(10)
        
        # RSI distance to target (50)
        df['rsi_distance_to_50'] = abs(df['rsi'] - self.config.rsi_target)
        df['rsi_above_50'] = (df['rsi'] > self.config.rsi_target).astype(int)
        df['rsi_below_50'] = (df['rsi'] < self.config.rsi_target).astype(int)
        
        # RSI momentum (rate of change)
        df['rsi_momentum'] = df['rsi'].diff(5) / df['rsi'].shift(5).replace(0, 1)
        
        if self.verbose:
            print(f"  ✅ Added RSI features: {len([c for c in df.columns if 'rsi' in c])} columns")
        
        return df
    
    def add_macd_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add MACD features (Wave 1 - Core)
        
        Features:
        - macd: MACD line
        - macd_signal: Signal line
        - macd_hist: Histogram
        - macd_aligned_up/down: direction alignment
        - macd_cross_above/below: cross detection
        - macd_hist_turning: histogram turning points
        """
        df = df.copy()
        
        # Calculate MACD
        exp1 = df['close'].ewm(span=self.config.macd_fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=self.config.macd_slow, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=self.config.macd_signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # WAVE 1: MACD alignment (pointing direction)
        df['macd_aligned_up'] = (df['macd'] > df['macd'].shift(1)).astype(int)
        df['macd_aligned_down'] = (df['macd'] < df['macd'].shift(1)).astype(int)
        
        # WAVE 1 & 2: MACD cross detection
        df['macd_cross_above'] = ((df['macd'].shift(1) <= df['macd_signal'].shift(1)) & 
                                   (df['macd'] > df['macd_signal'])).astype(int)
        df['macd_cross_below'] = ((df['macd'].shift(1) >= df['macd_signal'].shift(1)) & 
                                   (df['macd'] < df['macd_signal'])).astype(int)
        
        # WAVE 2: Histogram turning points (momentum shifts)
        df['macd_hist_positive'] = (df['macd_hist'] > 0).astype(int)
        df['macd_hist_negative'] = (df['macd_hist'] < 0).astype(int)
        df['macd_hist_turning_up'] = ((df['macd_hist'].shift(1) < 0) & (df['macd_hist'] > 0)).astype(int)
        df['macd_hist_turning_down'] = ((df['macd_hist'].shift(1) > 0) & (df['macd_hist'] < 0)).astype(int)
        
        # MACD divergence from zero
        df['macd_distance_to_zero'] = abs(df['macd'])
        df['macd_above_zero'] = (df['macd'] > 0).astype(int)
        df['macd_below_zero'] = (df['macd'] < 0).astype(int)
        
        # MACD slope (rate of change)
        df['macd_slope'] = df['macd'].diff(3) / df['macd'].shift(3).replace(0, 1)
        df['macd_signal_slope'] = df['macd_signal'].diff(3) / df['macd_signal'].shift(3).replace(0, 1)
        
        if self.verbose:
            print(f"  ✅ Added MACD features: {len([c for c in df.columns if 'macd' in c])} columns")
        
        return df
    
    # ========================================================================
    # SECTION 2: DIVERGENCE DETECTION (Wave 2)
    # ========================================================================
    
    def add_divergence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add divergence detection features (Wave 2)
        
        Features:
        - bullish_divergence: price lower low, RSI higher low
        - bearish_divergence: price higher high, RSI lower high
        - macd_bullish_divergence: same for MACD
        - macd_bearish_divergence: same for MACD
        """
        df = df.copy()
        lookback = self.config.divergence_lookback
        
        # Initialize divergence columns
        df['bullish_divergence'] = 0
        df['bearish_divergence'] = 0
        df['macd_bullish_divergence'] = 0
        df['macd_bearish_divergence'] = 0
        
        if len(df) < lookback + 5:
            return df
        
        # Vectorized rolling window divergence detection
        for i in range(lookback, len(df)):
            # Get windows
            price_window = df['close'].iloc[i - lookback:i + 1]
            rsi_window = df['rsi'].iloc[i - lookback:i + 1]
            macd_window = df['macd'].iloc[i - lookback:i + 1]
            
            # Find min and max positions
            price_min_idx = price_window.idxmin()
            price_max_idx = price_window.idxmax()
            rsi_min_idx = rsi_window.idxmin()
            rsi_max_idx = rsi_window.idxmax()
            macd_min_idx = macd_window.idxmin()
            macd_max_idx = macd_window.idxmax()
            
            # Bullish divergence: price lower low, RSI higher low
            if (price_window.loc[price_min_idx] < price_window.iloc[0] and
                rsi_window.loc[rsi_min_idx] > rsi_window.iloc[0]):
                df.loc[df.index[i], 'bullish_divergence'] = 1
            
            # Bearish divergence: price higher high, RSI lower high
            if (price_window.loc[price_max_idx] > price_window.iloc[0] and
                rsi_window.loc[rsi_max_idx] < rsi_window.iloc[0]):
                df.loc[df.index[i], 'bearish_divergence'] = 1
            
            # MACD bullish divergence
            if (price_window.loc[price_min_idx] < price_window.iloc[0] and
                macd_window.loc[macd_min_idx] > macd_window.iloc[0]):
                df.loc[df.index[i], 'macd_bullish_divergence'] = 1
            
            # MACD bearish divergence
            if (price_window.loc[price_max_idx] > price_window.iloc[0] and
                macd_window.loc[macd_max_idx] < macd_window.iloc[0]):
                df.loc[df.index[i], 'macd_bearish_divergence'] = 1
        
        # Rolling divergence counts (how many divergences in recent history)
        for window in [5, 10, 20]:
            df[f'bullish_divergence_count_{window}'] = df['bullish_divergence'].rolling(window).sum()
            df[f'bearish_divergence_count_{window}'] = df['bearish_divergence'].rolling(window).sum()
        
        if self.verbose:
            bullish_pct = df['bullish_divergence'].mean() * 100
            bearish_pct = df['bearish_divergence'].mean() * 100
            print(f"  ✅ Added divergence features: bullish={bullish_pct:.1f}%, bearish={bearish_pct:.1f}%")
        
        return df
    
    # ========================================================================
    # SECTION 3: PATTERN DETECTION (Wave 2 & 3)
    # ========================================================================
    
    def add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price pattern detection features (Wave 2 & 3)
        
        Features:
        - double_bottom (W): bullish reversal pattern
        - double_top (M): bearish reversal pattern
        - head_shoulders_top: bearish reversal
        - head_shoulders_bottom (inverse): bullish reversal
        - pattern_quality: quality score for detected patterns
        """
        df = df.copy()
        lookback = self.config.pattern_lookback
        
        # Initialize pattern columns
        df['double_bottom'] = 0
        df['double_top'] = 0
        df['head_shoulders_top'] = 0
        df['head_shoulders_bottom'] = 0
        df['pattern_quality'] = 0
        
        if len(df) < lookback:
            return df
        
        for i in range(lookback, len(df)):
            window = df.iloc[i - lookback:i + 1]
            
            # Double Bottom (W pattern) detection
            lows = window['low']
            min1_idx = lows.idxmin()
            temp_lows = lows.drop(min1_idx)
            if not temp_lows.empty:
                min2_idx = temp_lows.idxmin()
                low1 = lows[min1_idx]
                low2 = temp_lows[min2_idx]
                
                # WAVE 3: Minimum candle requirement
                candle_count = abs(window.index.get_loc(min2_idx) - window.index.get_loc(min1_idx))
                if candle_count >= self.config.min_pattern_candles:
                    if abs(low1 - low2) / low1 < 0.02:  # Within 2%
                        # Check for peak between
                        between_mask = (window.index >= min(min1_idx, min2_idx)) & (window.index <= max(min1_idx, min2_idx))
                        peak_between = window.loc[between_mask, 'high'].max()
                        if peak_between > low1 * 1.02:
                            df.loc[df.index[i], 'double_bottom'] = 1
                            df.loc[df.index[i], 'pattern_quality'] += 50
            
            # Double Top (M pattern) detection
            highs = window['high']
            max1_idx = highs.idxmax()
            temp_highs = highs.drop(max1_idx)
            if not temp_highs.empty:
                max2_idx = temp_highs.idxmax()
                high1 = highs[max1_idx]
                high2 = temp_highs[max2_idx]
                
                candle_count = abs(window.index.get_loc(max2_idx) - window.index.get_loc(max1_idx))
                if candle_count >= self.config.min_pattern_candles:
                    if abs(high1 - high2) / high1 < 0.02:
                        between_mask = (window.index >= min(max1_idx, max2_idx)) & (window.index <= max(max1_idx, max2_idx))
                        trough_between = window.loc[between_mask, 'low'].min()
                        if trough_between < high1 * 0.98:
                            df.loc[df.index[i], 'double_top'] = 1
                            df.loc[df.index[i], 'pattern_quality'] += 50
        
        # Rolling pattern counts
        for window in [5, 10, 20]:
            df[f'double_bottom_count_{window}'] = df['double_bottom'].rolling(window).sum()
            df[f'double_top_count_{window}'] = df['double_top'].rolling(window).sum()
        
        if self.verbose:
            db_pct = df['double_bottom'].mean() * 100
            dt_pct = df['double_top'].mean() * 100
            print(f"  ✅ Added pattern features: double_bottom={db_pct:.1f}%, double_top={dt_pct:.1f}%")
        
        return df
    
    # ========================================================================
    # SECTION 4: PRICE ACTION FEATURES (Wave 1 & 2)
    # ========================================================================
    
    def add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price action features (Wave 1 & 2)
        
        Features:
        - body_size, upper_wick, lower_wick
        - body_ratio (body/total range)
        - hammer, shooting_star, engulfing candles
        - returns and absolute returns
        """
        df = df.copy()
        
        # Candle components
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
        df['total_range'] = df['high'] - df['low']
        df['body_ratio'] = df['body_size'] / df['total_range'].replace(0, 1)
        
        # Candlestick patterns
        df['hammer'] = ((df['lower_wick'] > df['body_size'] * 2) & 
                        (df['upper_wick'] < df['body_size'])).astype(int)
        df['shooting_star'] = ((df['upper_wick'] > df['body_size'] * 2) & 
                               (df['lower_wick'] < df['body_size'])).astype(int)
        df['doji'] = (df['body_size'] < df['total_range'] * 0.1).astype(int)
        
        # Engulfing pattern
        df['bullish_engulfing'] = ((df['close'] > df['open']) & 
                                   (df['close'].shift(1) < df['open'].shift(1)) & 
                                   (df['close'] > df['open'].shift(1)) & 
                                   (df['open'] < df['close'].shift(1))).astype(int)
        df['bearish_engulfing'] = ((df['close'] < df['open']) & 
                                   (df['close'].shift(1) > df['open'].shift(1)) & 
                                   (df['close'] < df['open'].shift(1)) & 
                                   (df['open'] > df['close'].shift(1))).astype(int)
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['returns_abs'] = abs(df['returns'])
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Momentum
        for period in self.config.momentum_periods:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        if self.verbose:
            print(f"  ✅ Added price action features: {len([c for c in df.columns if 'wick' in c or 'body' in c or 'engulf' in c])} columns")
        
        return df
    
    # ========================================================================
    # SECTION 5: VOLATILITY FEATURES (Wave 2)
    # ========================================================================
    
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility features (Wave 2)
        
        Features:
        - rolling standard deviation of returns
        - Average True Range (ATR)
        - Bollinger Bands (upper, lower, width)
        - volatility regime classification
        """
        df = df.copy()
        
        # Rolling volatility
        for period in self.config.volatility_periods:
            df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()
        
        # Average True Range (ATR)
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        for period in [7, 14, 20]:
            df[f'atr_{period}'] = df['true_range'].rolling(window=period).mean()
            df[f'atr_ratio_{period}'] = df[f'atr_{period}'] / df['close']
        
        # Bollinger Bands
        for period in [20]:
            sma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            df[f'bb_upper_{period}'] = sma + (std * 2)
            df[f'bb_lower_{period}'] = sma - (std * 2)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
        
        # Volatility regime (low/medium/high)
        vol_median = df['volatility_20'].median()
        df['volatility_regime'] = 0  # low
        df.loc[df['volatility_20'] > vol_median, 'volatility_regime'] = 1  # medium
        df.loc[df['volatility_20'] > vol_median * 2, 'volatility_regime'] = 2  # high
        
        if self.verbose:
            print(f"  ✅ Added volatility features: ATR, Bollinger Bands, volatility regime")
        
        return df
    
    # ========================================================================
    # SECTION 6: TREND AND MOVING AVERAGE FEATURES (Wave 2)
    # ========================================================================
    
    def add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend and moving average features (Wave 2)
        
        Features:
        - SMA (Simple Moving Average) for multiple periods
        - Price vs SMA position
        - SMA slopes (trend direction)
        - Golden cross / death cross signals
        """
        df = df.copy()
        
        # Simple Moving Averages
        for period in self.config.sma_periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'price_vs_sma_{period}'] = (df['close'] > df[f'sma_{period}']).astype(int)
            df[f'distance_to_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
            df[f'sma_slope_{period}'] = df[f'sma_{period}'].diff(5) / df[f'sma_{period}'].shift(5).replace(0, 1)
        
        # Exponential Moving Averages
        for period in [9, 21, 50]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'price_vs_ema_{period}'] = (df['close'] > df[f'ema_{period}']).astype(int)
        
        # Golden Cross (50 SMA crosses above 200 SMA)
        df['golden_cross'] = ((df['sma_50'].shift(1) <= df['sma_200'].shift(1)) & 
                              (df['sma_50'] > df['sma_200'])).astype(int)
        
        # Death Cross (50 SMA crosses below 200 SMA)
        df['death_cross'] = ((df['sma_50'].shift(1) >= df['sma_200'].shift(1)) & 
                             (df['sma_50'] < df['sma_200'])).astype(int)
        
        # Trend strength (ADX approximation)
        # Simplified: difference between +DI and -DI
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        df['plus_dm'] = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        df['minus_dm'] = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        atr = df['true_range'].rolling(14).mean()
        df['plus_di'] = 100 * (df['plus_dm'].rolling(14).mean() / atr.replace(0, 1))
        df['minus_di'] = 100 * (df['minus_dm'].rolling(14).mean() / atr.replace(0, 1))
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']).replace(0, 1)
        df['adx'] = df['dx'].rolling(14).mean()
        
        # Trend regime based on ADX
        df['trend_regime'] = 0  # ranging
        df.loc[df['adx'] > 25, 'trend_regime'] = 1  # trending
        df.loc[df['adx'] > 40, 'trend_regime'] = 2  # strong trending
        
        if self.verbose:
            print(f"  ✅ Added trend features: SMAs, EMAs, cross signals, ADX")
        
        return df
    
    # ========================================================================
    # SECTION 7: SESSION FEATURES (Wave 3)
    # ========================================================================
    
    def add_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trading session features (Wave 3)
        
        Features:
        - hour, day_of_week
        - is_asian_session, is_london_session, is_us_session, is_overlap_session
        - is_weekend
        - session_volatility (historical volatility by session)
        """
        if not self.config.use_session_features:
            return df
        
        df = df.copy()
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # Session indicators (GMT times)
        df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 7)).astype(int)
        df['is_london_session'] = ((df['hour'] >= 7) & (df['hour'] < 12)).astype(int)
        df['is_us_session'] = ((df['hour'] >= 12) & (df['hour'] < 21)).astype(int)
        df['is_overlap_session'] = ((df['hour'] >= 12) & (df['hour'] < 16)).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Session-based volatility (rolling by session)
        for session in ['asian', 'london', 'us', 'overlap']:
            col = f'is_{session}_session'
            if col in df.columns:
                # Volatility during this session only
                mask = df[col] == 1
                df[f'{session}_session_volatility'] = df['returns'].rolling(50).std()
                df.loc[~mask, f'{session}_session_volatility'] = 0
        
        if self.verbose:
            asian_pct = df['is_asian_session'].mean() * 100
            london_pct = df['is_london_session'].mean() * 100
            us_pct = df['is_us_session'].mean() * 100
            print(f"  ✅ Added session features: Asian={asian_pct:.1f}%, London={london_pct:.1f}%, US={us_pct:.1f}%")
        
        return df
    
    # ========================================================================
    # SECTION 8: TARGET VARIABLES (Wave 1 & 2)
    # ========================================================================
    
    def add_target_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add target variables for ML training (Wave 1 & 2)
        
        Targets:
        - target_rsi_50_N: RSI reaches 50 within N bars
        - target_direction_N: Price direction (up/down) after N bars
        - target_sma_50_N: Price reaches 50-SMA within N bars
        - target_pattern_completion: Pattern completes within N bars
        """
        df = df.copy()
        
        # WAVE 1: RSI 50 target
        for horizon in self.config.target_horizons:
            df[f'target_rsi_50_{horizon}'] = 0
            for i in range(len(df) - horizon):
                if any(df['rsi'].iloc[i+1:i+horizon+1] >= self.config.rsi_target):
                    df.loc[df.index[i], f'target_rsi_50_{horizon}'] = 1
        
        # WAVE 2: Price direction target
        for horizon in self.config.target_horizons:
            df[f'target_direction_{horizon}'] = (df['close'].shift(-horizon) > df['close']).astype(int)
        
        # WAVE 2: 50-SMA touch target
        df['sma_50'] = df['close'].rolling(50).mean()
        for horizon in self.config.target_horizons:
            df[f'target_sma_50_{horizon}'] = 0
            for i in range(len(df) - horizon):
                if any(df['high'].iloc[i+1:i+horizon+1] >= df['sma_50'].iloc[i]):
                    df.loc[df.index[i], f'target_sma_50_{horizon}'] = 1
        
        # WAVE 2: Return magnitude target (regression)
        for horizon in self.config.target_horizons:
            df[f'target_return_{horizon}'] = (df['close'].shift(-horizon) / df['close'] - 1)
        
        # WAVE 3: Pattern completion target
        for horizon in [5, 10]:
            df[f'target_pattern_bullish_{horizon}'] = 0
            df[f'target_pattern_bearish_{horizon}'] = 0
            for i in range(len(df) - horizon):
                # Check if a bullish pattern completes within horizon
                if any(df['double_bottom'].iloc[i+1:i+horizon+1] == 1):
                    df.loc[df.index[i], f'target_pattern_bullish_{horizon}'] = 1
                if any(df['double_top'].iloc[i+1:i+horizon+1] == 1):
                    df.loc[df.index[i], f'target_pattern_bearish_{horizon}'] = 1
        
        if self.verbose:
            for horizon in self.config.target_horizons:
                pos_pct = df[f'target_direction_{horizon}'].mean() * 100
                print(f"  ✅ Target {horizon} bars: direction positive={pos_pct:.1f}%")
        
        return df
    
    # ========================================================================
    # SECTION 9: ZONE QUALITY FEATURES (Wave 3)
    # ========================================================================
    
    def add_zone_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add supply/demand zone quality features (Wave 3)
        
        Features:
        - zone_tightness: consolidation tightness score
        - zone_violence: breakout violence score
        - zone_quality: overall quality rating
        - zone_distance: distance to nearest zone
        """
        if not self.config.use_zone_quality:
            return df
        
        df = df.copy()
        
        # Initialize zone quality columns
        df['zone_tightness'] = 0
        df['zone_violence'] = 0
        df['zone_quality'] = 0
        df['zone_distance'] = 0
        
        lookback = 20
        
        for i in range(lookback, len(df)):
            window = df.iloc[i - lookback:i + 1]
            
            # Find consolidation zone
            recent_highs = window['high'].iloc[-10:]
            recent_lows = window['low'].iloc[-10:]
            
            zone_high = recent_highs.max()
            zone_low = recent_lows.min()
            zone_range = zone_high - zone_low
            
            if zone_range > 0:
                # Tightness: how small is the range relative to price
                avg_price = window['close'].mean()
                tightness = 100 - min(100, (zone_range / avg_price * 100))
                df.loc[df.index[i], 'zone_tightness'] = tightness
                
                # Violence: breakout candle size
                if i + 1 < len(df):
                    breakout = df.iloc[i + 1]
                    prev = df.iloc[i]
                    breakout_size = abs(breakout['close'] - breakout['open'])
                    prev_size = abs(prev['close'] - prev['open'])
                    violence = min(100, (breakout_size / prev_size * 100) if prev_size > 0 else 50)
                    df.loc[df.index[i], 'zone_violence'] = violence
                
                # Overall quality
                df.loc[df.index[i], 'zone_quality'] = (tightness * 0.6 + violence * 0.4)
        
        # Zone quality rating categorical
        df['zone_quality_rating'] = 0  # poor
        df.loc[df['zone_quality'] >= 40, 'zone_quality_rating'] = 1  # fair
        df.loc[df['zone_quality'] >= 60, 'zone_quality_rating'] = 2  # good
        df.loc[df['zone_quality'] >= 80, 'zone_quality_rating'] = 3  # excellent
        
        if self.verbose:
            excellent_pct = (df['zone_quality_rating'] == 3).mean() * 100
            print(f"  ✅ Added zone quality features: excellent zones={excellent_pct:.1f}%")
        
        return df
    
    # ========================================================================
    # SECTION 10: COMPLETE FEATURE PIPELINE
    # ========================================================================
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features for ML training
        Runs the complete feature engineering pipeline
        
        Args:
            df: Raw price DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        
        Returns:
            DataFrame with all engineered features
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🔧 CREATING ALL FEATURES")
            print(f"{'='*60}")
            print(f"Input shape: {df.shape}")
        
        start_time = datetime.now()
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Copy to avoid modifying original
        df = df.copy()
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Clean data
        for col in required:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=required)
        
        # Run feature engineering steps
        steps = [
            ("RSI Features", self.add_rsi_features),
            ("MACD Features", self.add_macd_features),
            ("Divergence Features", self.add_divergence_features),
            ("Pattern Features", self.add_pattern_features),
            ("Price Action Features", self.add_price_action_features),
            ("Volatility Features", self.add_volatility_features),
            ("Trend Features", self.add_trend_features),
            ("Session Features", self.add_session_features),
            ("Zone Quality Features", self.add_zone_quality_features),
            ("Target Features", self.add_target_features),
        ]
        
        for step_name, step_func in steps:
            if self.verbose:
                print(f"\n  📊 {step_name}...")
            df = step_func(df)
        
        # Drop rows with NaN
        initial_rows = len(df)
        df = df.dropna()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"✅ FEATURE ENGINEERING COMPLETE")
            print(f"{'='*60}")
            print(f"Final shape: {df.shape}")
            print(f"Rows dropped: {initial_rows - len(df)} ({100*(initial_rows - len(df))/initial_rows:.1f}%)")
            print(f"Total features: {len(df.columns)}")
            print(f"Time elapsed: {elapsed:.1f}s")
            print(f"{'='*60}\n")
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame, exclude_targets: bool = True) -> List[str]:
        """
        Get list of feature column names
        
        Args:
            df: Feature DataFrame
            exclude_targets: Exclude target columns starting with 'target_'
        
        Returns:
            List of feature column names
        """
        feature_cols = []
        for col in df.columns:
            if exclude_targets and col.startswith('target_'):
                continue
            if col not in ['open', 'high', 'low', 'close', 'volume']:
                feature_cols.append(col)
        return feature_cols


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TESTING FEATURE ENGINEERING v3.0")
    print("="*70)
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=5000, freq='H')
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.randn(5000) * 0.001
    prices = 100 + np.cumsum(returns)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(5000) * 0.5),
        'low': prices - np.abs(np.random.randn(5000) * 0.5),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 5000)
    }, index=dates)
    
    # Add some patterns
    # Add a double bottom at index 1000-1100
    df.iloc[1000:1010, df.columns.get_loc('low')] = 98
    df.iloc[1080:1090, df.columns.get_loc('low')] = 98
    
    # Add a double top at index 2000-2100
    df.iloc[2000:2010, df.columns.get_loc('high')] = 102
    df.iloc[2080:2090, df.columns.get_loc('high')] = 102
    
    print(f"Sample data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Initialize feature engineer
    config = FeatureEngineeringConfig()
    engineer = FeatureEngineer(config, verbose=True)
    
    # Create all features
    df_features = engineer.create_all_features(df)
    
    # Get feature names
    feature_names = engineer.get_feature_names(df_features)
    
    print(f"\n📊 Feature Summary:")
    print(f"  Total features: {len(feature_names)}")
    print(f"  Feature columns: {feature_names[:10]}...")
    
    # Print target distributions
    print(f"\n🎯 Target Distributions:")
    for col in df_features.columns:
        if col.startswith('target_'):
            positive_pct = df_features[col].mean() * 100
            print(f"  {col}: {positive_pct:.1f}% positive")
    
    print(f"\n✅ Feature engineering test complete")