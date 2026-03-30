#!/usr/bin/env python3
"""
Signal Predictor Module for SID Method - AUGMENTED VERSION
=============================================================================
Predicts trade signals and provides confidence scoring incorporating ALL THREE WAVES:

WAVE 1 (Core Quick Win):
- RSI threshold detection (exact 30/70)
- MACD alignment and cross detection
- Stop loss and take profit calculation
- Position sizing with 0.5-2% risk

WAVE 2 (Live Sessions & Q&A):
- Confidence scoring based on multiple confirmations
- Pattern confirmation (W, M, H&S)
- Divergence detection
- Market context filtering
- Reachability check

WAVE 3 (Academy Support Sessions):
- Quality filtering (excellent/good/fair/poor)
- Session-based suitability scoring
- Minimum candle requirements
- Zone quality assessment
- Stop loss pip buffer

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
import pickle
import json
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

# ============================================================================
# ENUMS FOR SIGNAL QUALITY (Wave 2 & 3)
# ============================================================================

class SignalQuality(Enum):
    """Quality grade for trade signals (Wave 3)"""
    EXCELLENT = "excellent"   # 80%+ quality score - high confidence
    GOOD = "good"             # 60-79% quality score - medium confidence
    FAIR = "fair"             # 40-59% quality score - low confidence
    POOR = "poor"             # Below 40% quality score - avoid
    INVALID = "invalid"       # Does not meet minimum criteria

class MarketTrend(Enum):
    """Market trend direction (Wave 2)"""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"

class TradingSession(Enum):
    """Trading sessions (Wave 3)"""
    ASIAN = "asian"
    LONDON = "london"
    US = "us"
    OVERLAP = "overlap"

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TradeSignal:
    """Represents a SID Method trade signal (Wave 1-3)"""
    # Core signal data (Wave 1)
    date: datetime
    instrument: str
    direction: str  # 'long' or 'short'
    signal_type: str  # 'oversold' or 'overbought'
    rsi_value: float
    entry_price: float
    stop_loss: float
    take_profit: float
    take_profit_alt: float
    
    # MACD data (Wave 1 & 2)
    macd_aligned: bool
    macd_crossed: bool
    
    # Pattern data (Wave 2)
    pattern_confirmed: bool
    pattern_name: str
    
    # Divergence data (Wave 2)
    divergence_detected: bool
    divergence_type: str  # 'bullish' or 'bearish'
    
    # Quality metrics (Wave 2 & 3)
    confidence_score: float
    confidence_level: str  # 'high', 'medium', 'low'
    quality: str  # SignalQuality value
    quality_score: float
    
    # Context data (Wave 2 & 3)
    market_trend: str
    session: str
    session_suitability: str
    
    # Risk metrics (Wave 1 & 2)
    reward_ratio: float
    risk_amount: float
    position_units: int
    
    # Reachability (Wave 2)
    reachable: bool
    reachability_multiplier: float
    
    # ML prediction (if available)
    ml_prediction: Optional[float] = None
    ml_confidence: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'date': self.date.isoformat(),
            'instrument': self.instrument,
            'direction': self.direction,
            'signal_type': self.signal_type,
            'rsi_value': self.rsi_value,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'take_profit_alt': self.take_profit_alt,
            'macd_aligned': self.macd_aligned,
            'macd_crossed': self.macd_crossed,
            'pattern_confirmed': self.pattern_confirmed,
            'pattern_name': self.pattern_name,
            'divergence_detected': self.divergence_detected,
            'divergence_type': self.divergence_type,
            'confidence_score': self.confidence_score,
            'confidence_level': self.confidence_level,
            'quality': self.quality,
            'quality_score': self.quality_score,
            'market_trend': self.market_trend,
            'session': self.session,
            'session_suitability': self.session_suitability,
            'reward_ratio': self.reward_ratio,
            'risk_amount': self.risk_amount,
            'position_units': self.position_units,
            'reachable': self.reachable,
            'reachability_multiplier': self.reachability_multiplier,
            'ml_prediction': self.ml_prediction,
            'ml_confidence': self.ml_confidence
        }


@dataclass
class SignalPredictorConfig:
    """Configuration for signal predictor (Wave 1, 2, 3)"""
    # Wave 1: Core parameters
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    rsi_target: int = 50
    prefer_macd_cross: bool = True
    
    # Wave 2: Confidence scoring weights
    macd_cross_weight: float = 0.35
    macd_aligned_weight: float = 0.20
    pattern_weight: float = 0.20
    divergence_weight: float = 0.15
    session_weight: float = 0.10
    
    # Wave 2: Market context
    use_market_context: bool = True
    market_trend_lookback: int = 50
    
    # Wave 3: Quality thresholds
    excellent_threshold: float = 80.0
    good_threshold: float = 60.0
    fair_threshold: float = 40.0
    
    # Wave 3: Session suitability
    session_scores: Dict[str, float] = field(default_factory=lambda: {
        'overlap': 1.0,   # London-US overlap - best
        'us': 0.9,        # US session - very good
        'london': 0.7,    # London session - good
        'asian': 0.4      # Asian session - lower suitability
    })
    
    # Wave 2: Minimum requirements
    min_confidence_score: float = 0.5
    min_quality_score: float = 40.0
    
    # Wave 3: Minimum pattern candles
    min_pattern_candles: int = 7
    
    # Wave 1: Risk management
    default_risk_percent: float = 1.0
    min_risk_percent: float = 0.5
    max_risk_percent: float = 2.0


class SignalPredictor:
    """
    SID Method Signal Predictor with confidence scoring
    Integrates ALL THREE WAVES of strategies
    """
    
    def __init__(self, config: SignalPredictorConfig = None, 
                 model_dir: str = 'ai/trained_models/',
                 verbose: bool = True):
        """
        Initialize the signal predictor
        
        Args:
            config: SignalPredictorConfig instance
            model_dir: Directory containing trained ML models
            verbose: Enable verbose output
        """
        self.config = config or SignalPredictorConfig()
        self.model_dir = model_dir
        self.verbose = verbose
        
        # Loaded models
        self.models = {}
        self.model_metadata = {}
        
        # Load available models
        self._load_models()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🎯 SIGNAL PREDICTOR v3.0 (Fully Augmented)")
            print(f"{'='*60}")
            print(f"📊 RSI: {self.config.rsi_oversold}/{self.config.rsi_overbought}")
            print(f"🔄 MACD cross preferred: {self.config.prefer_macd_cross}")
            print(f"⭐ Min confidence: {self.config.min_confidence_score}")
            print(f"📐 Min quality: {self.config.min_quality_score}")
            print(f"🤖 Loaded models: {len(self.models)}")
            print(f"{'='*60}\n")
    
    # ========================================================================
    # MODEL LOADING
    # ========================================================================
    
    def _load_models(self):
        """Load trained ML models from disk"""
        import os
        import glob
        
        if not os.path.exists(self.model_dir):
            if self.verbose:
                print(f"  Model directory not found: {self.model_dir}")
            return
        
        # Find all model files
        model_files = glob.glob(os.path.join(self.model_dir, "*.pkl"))
        
        for model_file in model_files:
            try:
                model_name = os.path.basename(model_file).replace('.pkl', '')
                
                # Load model
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                
                # Load metadata if available
                metadata_file = model_file.replace('.pkl', '_metadata.json')
                metadata = {}
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                
                self.models[model_name] = model
                self.model_metadata[model_name] = metadata
                
                if self.verbose:
                    print(f"  Loaded model: {model_name}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"  Failed to load {model_file}: {e}")
    
    def get_model_for_instrument(self, instrument: str) -> Tuple[Any, Dict]:
        """
        Get the best model for a specific instrument
        
        Args:
            instrument: Trading instrument (e.g., 'EUR_USD')
        
        Returns:
            (model, metadata) tuple
        """
        # Try exact match first
        for model_name, model in self.models.items():
            if instrument in model_name:
                return model, self.model_metadata.get(model_name, {})
        
        # Try general model
        for model_name, model in self.models.items():
            if 'general' in model_name.lower() or 'default' in model_name.lower():
                return model, self.model_metadata.get(model_name, {})
        
        return None, {}
    
    # ========================================================================
    # SIGNAL DETECTION (Wave 1)
    # ========================================================================
    
    def detect_rsi_signal(self, rsi_value: float, strict: bool = True) -> Tuple[str, bool]:
        """
        Detect RSI signal with exact thresholds (Wave 3 precision)
        
        Returns:
            (signal_type, is_exact) where signal_type is 'oversold', 'overbought', or 'neutral'
        """
        if strict:
            if rsi_value < self.config.rsi_oversold:
                return 'oversold', True
            elif rsi_value > self.config.rsi_overbought:
                return 'overbought', True
            else:
                return 'neutral', False
        else:
            if rsi_value <= self.config.rsi_oversold:
                return 'oversold', True
            elif rsi_value >= self.config.rsi_overbought:
                return 'overbought', True
            else:
                return 'neutral', False
    
    def detect_macd_alignment(self, macd_df: pd.DataFrame, current_idx: int, 
                                signal_type: str) -> bool:
        """Detect MACD alignment (Wave 1)"""
        if current_idx < 2:
            return False
        
        current_macd = macd_df['macd'].iloc[current_idx]
        prev_macd = macd_df['macd'].iloc[current_idx - 1]
        
        if signal_type == 'oversold':
            return current_macd > prev_macd
        elif signal_type == 'overbought':
            return current_macd < prev_macd
        return False
    
    def detect_macd_cross(self, macd_df: pd.DataFrame, current_idx: int,
                            signal_type: str) -> bool:
        """Detect MACD cross (Wave 2 - preferred)"""
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
    # CONFIDENCE SCORING (Wave 2 & 3)
    # ========================================================================
    
    def calculate_confidence_score(self, 
                                     macd_aligned: bool,
                                     macd_crossed: bool,
                                     pattern_confirmed: bool,
                                     divergence_detected: bool,
                                     session: str,
                                     is_exact_rsi: bool) -> Tuple[float, str]:
        """
        Calculate confidence score based on multiple confirmations (Wave 2)
        
        Returns:
            (confidence_score, confidence_level)
        """
        score = 0.0
        
        # MACD cross (highest weight)
        if macd_crossed:
            score += self.config.macd_cross_weight
        elif macd_aligned:
            score += self.config.macd_aligned_weight
        
        # Pattern confirmation
        if pattern_confirmed:
            score += self.config.pattern_weight
        
        # Divergence
        if divergence_detected:
            score += self.config.divergence_weight
        
        # Session suitability
        session_score = self.config.session_scores.get(session, 0.5)
        score += session_score * self.config.session_weight
        
        # Exact RSI (bonus)
        if is_exact_rsi:
            score += 0.05
        
        # Normalize to 0-1 range
        score = min(score, 1.0)
        
        # Determine confidence level
        if score >= 0.8:
            confidence_level = "high"
        elif score >= 0.6:
            confidence_level = "medium"
        elif score >= 0.4:
            confidence_level = "low"
        else:
            confidence_level = "very_low"
        
        return score, confidence_level
    
    def calculate_quality_score(self, 
                                  confidence_score: float,
                                  reward_ratio: float,
                                  reachable: bool,
                                  session_suitability: str) -> float:
        """
        Calculate overall quality score (Wave 3)
        
        Returns:
            quality_score (0-100)
        """
        quality = 0.0
        
        # Confidence contributes 40%
        quality += confidence_score * 40
        
        # Reward ratio contributes 20%
        reward_score = min(reward_ratio, 3.0) / 3.0 * 20
        quality += reward_score
        
        # Reachability contributes 20%
        quality += 20 if reachable else 0
        
        # Session suitability contributes 20%
        session_scores = {'very_high': 20, 'high': 15, 'medium': 10, 'low': 5}
        quality += session_scores.get(session_suitability, 10)
        
        return quality
    
    def get_quality_rating(self, quality_score: float) -> SignalQuality:
        """Get quality rating from score (Wave 3)"""
        if quality_score >= self.config.excellent_threshold:
            return SignalQuality.EXCELLENT
        elif quality_score >= self.config.good_threshold:
            return SignalQuality.GOOD
        elif quality_score >= self.config.fair_threshold:
            return SignalQuality.FAIR
        else:
            return SignalQuality.POOR
    
    # ========================================================================
    # MARKET CONTEXT (Wave 2)
    # ========================================================================
    
    def analyze_market_trend(self, df: pd.DataFrame) -> MarketTrend:
        """Analyze market trend using multiple timeframes (Wave 2)"""
        if df.empty or len(df) < self.config.market_trend_lookback:
            return MarketTrend.UNKNOWN
        
        recent_data = df.iloc[-self.config.market_trend_lookback:]
        highs = recent_data['high']
        lows = recent_data['low']
        
        # Check for higher highs and higher lows
        higher_highs = all(highs.iloc[i] <= highs.iloc[i+1] 
                          for i in range(len(highs)-10, len(highs)-1))
        higher_lows = all(lows.iloc[i] <= lows.iloc[i+1] 
                         for i in range(len(lows)-10, len(lows)-1))
        
        if higher_highs and higher_lows:
            return MarketTrend.UPTREND
        
        # Check for lower highs and lower lows
        lower_highs = all(highs.iloc[i] >= highs.iloc[i+1] 
                         for i in range(len(highs)-10, len(highs)-1))
        lower_lows = all(lows.iloc[i] >= lows.iloc[i+1] 
                        for i in range(len(lows)-10, len(lows)-1))
        
        if lower_highs and lower_lows:
            return MarketTrend.DOWNTREND
        
        return MarketTrend.SIDEWAYS
    
    def should_filter_by_context(self, signal_type: str, market_trend: MarketTrend) -> Tuple[bool, str]:
        """
        Determine if signal should be filtered based on market context (Wave 2)
        
        Returns:
            (should_filter, reason)
        """
        if not self.config.use_market_context:
            return False, "Context filtering disabled"
        
        if market_trend == MarketTrend.SIDEWAYS:
            return True, "Sideways market - no SID trades"
        
        if signal_type == 'oversold' and market_trend == MarketTrend.DOWNTREND:
            return True, "Oversold in downtrend - against trend"
        
        if signal_type == 'overbought' and market_trend == MarketTrend.UPTREND:
            return True, "Overbought in uptrend - against trend"
        
        return False, "Context OK"
    
    # ========================================================================
    # REACHABILITY CHECK (Wave 2)
    # ========================================================================
    
    def check_reachability(self, df: pd.DataFrame, current_idx: int,
                            entry_price: float, direction: str,
                            target_price: float) -> Tuple[bool, float]:
        """
        Check if take profit target is realistically reachable (Wave 2)
        
        Returns:
            (is_reachable, risk_multiplier)
        """
        if current_idx < 50:
            return True, 1.0
        
        # Look at recent price action (last 50 bars)
        recent_highs = df['high'].iloc[current_idx-50:current_idx].max()
        recent_lows = df['low'].iloc[current_idx-50:current_idx].min()
        
        if direction == 'long':
            distance_to_target = target_price - entry_price
            distance_to_recent_high = recent_highs - entry_price
            
            if distance_to_target > distance_to_recent_high * 1.5:
                return False, 0.5
            else:
                return True, 1.0
        else:
            distance_to_target = entry_price - target_price
            distance_to_recent_low = entry_price - recent_lows
            
            if distance_to_target > distance_to_recent_low * 1.5:
                return False, 0.5
            else:
                return True, 1.0
    
    # ========================================================================
    # PATTERN DETECTION (Wave 2 & 3)
    # ========================================================================
    
    def detect_double_bottom(self, df: pd.DataFrame, lookback: int = 30) -> Tuple[bool, Optional[float]]:
        """Detect double bottom (W) pattern (Wave 2 & 3)"""
        if len(df) < lookback:
            return False, None
        
        recent_lows = df['low'].iloc[-lookback:]
        
        # Find two lowest points
        min1_idx = recent_lows.idxmin()
        temp_lows = recent_lows.drop(min1_idx)
        if temp_lows.empty:
            return False, None
        
        min2_idx = temp_lows.idxmin()
        
        low1 = recent_lows[min1_idx]
        low2 = temp_lows[min2_idx]
        
        # Wave 3: Minimum candle requirement
        candle_count = abs(df.index.get_loc(min2_idx) - df.index.get_loc(min1_idx))
        if candle_count < self.config.min_pattern_candles:
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
        """Detect double top (M) pattern (Wave 2 & 3)"""
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
        if candle_count < self.config.min_pattern_candles:
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
    
    def detect_divergence(self, df: pd.DataFrame, lookback: int = 20) -> Tuple[bool, str]:
        """Detect RSI divergence (Wave 2)"""
        if len(df) < lookback + 5:
            return False, ''
        
        price_window = df['close'].iloc[-lookback:]
        rsi_window = df['rsi'].iloc[-lookback:] if 'rsi' in df.columns else None
        
        if rsi_window is None:
            return False, ''
        
        price_min_idx = price_window.idxmin()
        rsi_min_idx = rsi_window.idxmin()
        price_max_idx = price_window.idxmax()
        rsi_max_idx = rsi_window.idxmax()
        
        # Bullish divergence
        if (price_window.loc[price_min_idx] < price_window.iloc[0] and
            rsi_window.loc[rsi_min_idx] > rsi_window.iloc[0]):
            return True, 'bullish'
        
        # Bearish divergence
        if (price_window.loc[price_max_idx] > price_window.iloc[0] and
            rsi_window.loc[rsi_max_idx] < rsi_window.iloc[0]):
            return True, 'bearish'
        
        return False, ''
    
    # ========================================================================
    # SESSION DETECTION (Wave 3)
    # ========================================================================
    
    def get_trading_session(self, dt: datetime) -> str:
        """Determine trading session (Wave 3)"""
        hour = dt.hour
        
        if 0 <= hour < 7:
            return 'asian'
        elif 7 <= hour < 12:
            return 'london'
        elif 12 <= hour < 16:
            return 'overlap'
        elif 16 <= hour < 21:
            return 'us'
        else:
            return 'asian'
    
    def get_session_suitability(self, session: str) -> str:
        """Get SID method suitability for session (Wave 3)"""
        suitability = {
            'overlap': 'very_high',
            'us': 'high',
            'london': 'medium',
            'asian': 'low'
        }
        return suitability.get(session, 'medium')
    
    # ========================================================================
    # STOP LOSS CALCULATION (Wave 1 + Wave 3)
    # ========================================================================
    
    def calculate_stop_loss(self, df: pd.DataFrame, signal_date: datetime,
                              entry_date: datetime, signal_type: str,
                              instrument: str = None,
                              use_pip_buffer: bool = True) -> float:
        """Calculate SID Method stop loss (Wave 1 + Wave 3)"""
        mask = (df.index >= signal_date) & (df.index <= entry_date)
        period_df = df.loc[mask]
        
        if period_df.empty:
            return 0.0
        
        if signal_type == 'oversold':
            lowest_low = period_df['low'].min()
            stop_loss = np.floor(lowest_low)
            
            if use_pip_buffer and instrument:
                # Add pip buffer (Wave 3)
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
    # TAKE PROFIT CALCULATION (Wave 1 & 2)
    # ========================================================================
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float,
                                direction: str, sma_50: Optional[float] = None) -> Dict:
        """Calculate take profit (Wave 1 primary, Wave 2 alternatives)"""
        risk_distance = abs(entry_price - stop_loss)
        
        # Primary: RSI 50 (1:1 risk-reward)
        if direction == 'long':
            primary_tp = entry_price + risk_distance
        else:
            primary_tp = entry_price - risk_distance
        
        # Alternative: 50-SMA (if available and logical)
        alternative_tp = primary_tp
        if sma_50 is not None:
            if direction == 'long' and sma_50 > entry_price:
                alternative_tp = sma_50
            elif direction == 'short' and sma_50 < entry_price:
                alternative_tp = sma_50
        
        reward_ratio = abs(alternative_tp - entry_price) / risk_distance if risk_distance > 0 else 1.0
        
        return {
            'primary_tp': primary_tp,
            'alternative_tp': alternative_tp,
            'reward_ratio': reward_ratio,
            'risk_distance': risk_distance
        }
    
    # ========================================================================
    # POSITION SIZING (Wave 1 & 2)
    # ========================================================================
    
    def calculate_position_size(self, account_balance: float, entry_price: float,
                                  stop_loss: float, risk_percent: float = None,
                                  reachability_multiplier: float = 1.0) -> Dict:
        """Calculate position size (Wave 1: 0.5-2% risk)"""
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
    # ML PREDICTION (Wave 2)
    # ========================================================================
    
    def predict_with_ml(self, features: np.ndarray, instrument: str) -> Tuple[float, float]:
        """
        Get ML prediction for a signal (Wave 2)
        
        Returns:
            (prediction_probability, confidence)
        """
        model, metadata = self.get_model_for_instrument(instrument)
        
        if model is None:
            return 0.5, 0.0
        
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features)
                if len(proba.shape) > 1 and proba.shape[1] > 1:
                    prediction = proba[0, 1]
                else:
                    prediction = proba[0]
            elif hasattr(model, 'predict'):
                prediction = model.predict(features)[0]
            else:
                return 0.5, 0.0
            
            # Calculate confidence based on distance from 0.5
            confidence = abs(prediction - 0.5) * 2
            
            return float(prediction), float(confidence)
            
        except Exception as e:
            if self.verbose:
                print(f"  ML prediction failed: {e}")
            return 0.5, 0.0
    
    # ========================================================================
    # COMPLETE SIGNAL DETECTION PIPELINE
    # ========================================================================
    
    def scan_for_signals(self, df: pd.DataFrame, instrument: str,
                           account_balance: float = 10000,
                           market_df: pd.DataFrame = None,
                           use_ml: bool = True) -> List[TradeSignal]:
        """
        Scan for SID Method signals (Wave 1, 2, 3)
        
        Args:
            df: Price DataFrame for the instrument
            instrument: Instrument name
            account_balance: Current account balance
            market_df: Market index DataFrame for context
            use_ml: Use ML predictions if available
        
        Returns:
            List of TradeSignal objects
        """
        if df.empty or len(df) < 50:
            if self.verbose:
                print(f"  Insufficient data: {len(df)} rows")
            return []
        
        if self.verbose:
            print(f"\n🔍 Scanning for signals: {instrument}")
            print(f"  Data rows: {len(df):,}")
        
        # Calculate indicators
        df = df.copy()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, float('nan'))
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # SMA 50 for take profit
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # Market trend (Wave 2)
        market_trend = MarketTrend.UNKNOWN
        if market_df is not None:
            market_trend = self.analyze_market_trend(market_df)
        
        signals = []
        
        # Scan for signals (using tqdm for progress)
        iterator = tqdm(range(20, len(df) - 1), desc="  Scanning", 
                       disable=not TQDM_AVAILABLE or not self.verbose)
        
        for i in iterator:
            current_date = df.index[i]
            rsi_value = df['rsi'].iloc[i]
            
            # Detect RSI signal (Wave 1 & 3)
            signal_type, is_exact = self.detect_rsi_signal(rsi_value, strict=True)
            if signal_type == 'neutral':
                continue
            
            # Filter by market context (Wave 2)
            should_filter, filter_reason = self.should_filter_by_context(signal_type, market_trend)
            if should_filter:
                continue
            
            # Detect MACD (Wave 1 & 2)
            macd_aligned = self.detect_macd_alignment(df, i, signal_type)
            if not macd_aligned:
                continue
            
            macd_crossed = self.detect_macd_cross(df, i, signal_type)
            
            # If prefer cross and no cross, skip (Wave 2)
            if self.config.prefer_macd_cross and not macd_crossed:
                continue
            
            # Find signal date
            rsi_values = df['rsi'].iloc[:i+1]
            if signal_type == 'oversold':
                mask = rsi_values < self.config.rsi_oversold
            else:
                mask = rsi_values > self.config.rsi_overbought
            
            if mask.any():
                signal_idx = mask[mask].index[-1]
                signal_date = signal_idx
            else:
                signal_date = current_date
            
            # Detect patterns (Wave 2 & 3)
            pattern_confirmed = False
            pattern_name = ''
            
            if signal_type == 'oversold':
                detected, _ = self.detect_double_bottom(df.iloc[:i+1])
                pattern_confirmed = detected
                pattern_name = 'double_bottom' if detected else ''
            else:
                detected, _ = self.detect_double_top(df.iloc[:i+1])
                pattern_confirmed = detected
                pattern_name = 'double_top' if detected else ''
            
            # Detect divergence (Wave 2)
            divergence_detected, divergence_type = self.detect_divergence(df.iloc[:i+1])
            
            # Calculate stop loss (Wave 1 + Wave 3)
            stop_loss = self.calculate_stop_loss(df, signal_date, current_date, 
                                                  signal_type, instrument)
            
            direction = 'long' if signal_type == 'oversold' else 'short'
            entry_price = df['close'].iloc[i]
            sma_50_value = df['sma_50'].iloc[i] if not pd.isna(df['sma_50'].iloc[i]) else None
            
            # Calculate take profit (Wave 1 & 2)
            tp_result = self.calculate_take_profit(entry_price, stop_loss, direction, sma_50_value)
            
            # Get session (Wave 3)
            session = self.get_trading_session(current_date)
            session_suitability = self.get_session_suitability(session)
            
            # Calculate confidence (Wave 2 & 3)
            confidence_score, confidence_level = self.calculate_confidence_score(
                macd_aligned, macd_crossed, pattern_confirmed, 
                divergence_detected, session, is_exact
            )
            
            # Check reachability (Wave 2)
            reachable, reachability_multiplier = self.check_reachability(
                df, i, entry_price, direction, tp_result['alternative_tp']
            )
            
            # Calculate position size (Wave 1 & 2)
            position = self.calculate_position_size(
                account_balance, entry_price, stop_loss,
                reachability_multiplier=reachability_multiplier
            )
            
            # Calculate quality score (Wave 3)
            quality_score = self.calculate_quality_score(
                confidence_score, tp_result['reward_ratio'], 
                reachable, session_suitability
            )
            quality_rating = self.get_quality_rating(quality_score)
            
            # Filter by minimum quality (Wave 3)
            if quality_score < self.config.min_quality_score:
                continue
            
            # ML prediction (Wave 2)
            ml_prediction = None
            ml_confidence = None
            if use_ml and len(self.models) > 0:
                # Create feature vector (simplified for now)
                feature_vector = np.array([[
                    rsi_value,
                    df['macd'].iloc[i],
                    df['macd_signal'].iloc[i],
                    macd_crossed,
                    pattern_confirmed,
                    divergence_detected,
                    tp_result['reward_ratio']
                ]])
                ml_prediction, ml_confidence = self.predict_with_ml(feature_vector, instrument)
            
            # Create signal object
            signal = TradeSignal(
                date=current_date,
                instrument=instrument,
                direction=direction,
                signal_type=signal_type,
                rsi_value=rsi_value,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=tp_result['primary_tp'],
                take_profit_alt=tp_result['alternative_tp'],
                macd_aligned=macd_aligned,
                macd_crossed=macd_crossed,
                pattern_confirmed=pattern_confirmed,
                pattern_name=pattern_name,
                divergence_detected=divergence_detected,
                divergence_type=divergence_type,
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                quality=quality_rating.value,
                quality_score=quality_score,
                market_trend=market_trend.value if market_trend != MarketTrend.UNKNOWN else 'unknown',
                session=session,
                session_suitability=session_suitability,
                reward_ratio=tp_result['reward_ratio'],
                risk_amount=position.get('risk_amount', 0),
                position_units=position.get('units', 0),
                reachable=reachable,
                reachability_multiplier=reachability_multiplier,
                ml_prediction=ml_prediction,
                ml_confidence=ml_confidence
            )
            
            signals.append(signal)
        
        # Sort by quality score (best first)
        signals.sort(key=lambda x: x.quality_score, reverse=True)
        
        if self.verbose:
            print(f"\n  ✅ Found {len(signals)} signals")
            if signals:
                print(f"     Best: {signals[0].direction} @ {signals[0].entry_price:.5f}")
                print(f"     Quality: {signals[0].quality} ({signals[0].quality_score:.1f})")
                print(f"     Confidence: {signals[0].confidence_level} ({signals[0].confidence_score:.2f})")
        
        return signals
    
    def get_best_signal(self, signals: List[TradeSignal], 
                         min_quality: str = 'good') -> Optional[TradeSignal]:
        """
        Get the best signal from a list, filtered by minimum quality
        
        Args:
            signals: List of TradeSignal objects
            min_quality: Minimum quality ('excellent', 'good', 'fair', 'poor')
        
        Returns:
            Best signal or None
        """
        quality_order = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1}
        min_quality_value = quality_order.get(min_quality, 2)
        
        filtered = [s for s in signals 
                   if quality_order.get(s.quality, 0) >= min_quality_value]
        
        if not filtered:
            return None
        
        # Return highest quality (already sorted)
        return filtered[0]
    
    def get_signal_summary(self, signals: List[TradeSignal]) -> Dict:
        """Get summary statistics for signals"""
        if not signals:
            return {'total': 0}
        
        qualities = [s.quality for s in signals]
        directions = [s.direction for s in signals]
        
        return {
            'total': len(signals),
            'excellent': qualities.count('excellent'),
            'good': qualities.count('good'),
            'fair': qualities.count('fair'),
            'poor': qualities.count('poor'),
            'long': directions.count('long'),
            'short': directions.count('short'),
            'avg_confidence': np.mean([s.confidence_score for s in signals]),
            'avg_quality': np.mean([s.quality_score for s in signals]),
            'avg_reward_ratio': np.mean([s.reward_ratio for s in signals if s.reward_ratio > 0])
        }


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TESTING SIGNAL PREDICTOR v3.0")
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
    
    # Initialize predictor
    config = SignalPredictorConfig()
    predictor = SignalPredictor(config, verbose=True)
    
    # Scan for signals
    signals = predictor.scan_for_signals(df, 'EUR_USD', account_balance=10000)
    
    print(f"\n📊 Signal Summary:")
    summary = predictor.get_signal_summary(signals)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Get best signal
    best = predictor.get_best_signal(signals)
    if best:
        print(f"\n🏆 Best Signal:")
        print(f"  Date: {best.date}")
        print(f"  Direction: {best.direction}")
        print(f"  Entry: {best.entry_price:.5f}")
        print(f"  Stop: {best.stop_loss:.5f}")
        print(f"  TP: {best.take_profit:.5f}")
        print(f"  Quality: {best.quality} ({best.quality_score:.1f})")
        print(f"  Confidence: {best.confidence_level} ({best.confidence_score:.2f})")
        if best.pattern_confirmed:
            print(f"  Pattern: {best.pattern_name}")
        if best.divergence_detected:
            print(f"  Divergence: {best.divergence_type}")
    
    print(f"\n✅ Signal predictor test complete")