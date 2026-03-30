"""
Simon Pullen M-Top and W-Bottom Pattern Detection
Rules:
- Minimum 7 candles, maximum 30 candles in pattern
- Neckline based on bodies only (ignore wicks)
- Entry on break of neckline (no retest needed)
- Stop loss: behind pattern, can ignore single rogue wicks
- Take profit: 1:1 risk-to-reward minimum
- Price action must be impulsive going in
- Take profit must be within last impulsive leg
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class MWPattern:
    """Represents an M-Top or W-Bottom pattern"""
    pattern_type: str  # 'M' or 'W'
    instrument: str
    timeframe: str
    left_peak_idx: int
    left_peak_price: float
    right_peak_idx: int
    right_peak_price: float
    neckline_idx: int  # Index of lowest body between peaks
    neckline_price: float
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    candle_count: int
    valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    confluence_score: float = 0.0
    detected_at: Optional[pd.Timestamp] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'pattern_type': self.pattern_type,
            'instrument': self.instrument,
            'timeframe': self.timeframe,
            'left_peak_idx': self.left_peak_idx,
            'left_peak_price': self.left_peak_price,
            'right_peak_idx': self.right_peak_idx,
            'right_peak_price': self.right_peak_price,
            'neckline_idx': self.neckline_idx,
            'neckline_price': self.neckline_price,
            'entry_price': self.entry_price,
            'stop_loss_price': self.stop_loss_price,
            'take_profit_price': self.take_profit_price,
            'candle_count': self.candle_count,
            'valid': self.valid,
            'validation_errors': self.validation_errors,
            'confluence_score': self.confluence_score,
            'detected_at': str(self.detected_at) if self.detected_at else None
        }


class MWPatternDetector:
    """
    Detects M-Tops and W-Bottoms following Simon Pullen's strict rules
    
    Key rules implemented:
    1. Impulsive move into pattern (visual check - can add momentum filter)
    2. 7-30 candles between left peak and right peak
    3. Peaks within similarity tolerance (configurable)
    4. Neckline based on bodies, not wicks
    5. Stop loss ignores single rogue wicks
    6. Take profit must be within last impulsive leg
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.peak_similarity_tolerance = config.get('peak_similarity_tolerance', 0.02)  # 2%
        self.min_candles = config.get('min_mw_candles', 7)
        self.max_candles = config.get('max_mw_candles', 30)
        self.min_impulsive_move_pips = config.get('min_impulsive_move_pips', 50)
        self.ignore_rogue_wicks = config.get('ignore_rogue_wicks', True)
        
    def detect_m_top(self, df: pd.DataFrame, instrument: str = "", timeframe: str = "1h") -> List[MWPattern]:
        """
        Detect M-Top patterns in the dataframe
        
        Process:
        1. Identify peaks (local maxima)
        2. Check for pairs of peaks within similarity tolerance
        3. Validate candle count between peaks
        4. Calculate neckline (lowest body between peaks)
        5. Check if break would have occurred
        6. Validate take profit within last impulsive leg
        """
        peaks = self._find_peaks(df)
        patterns = []
        
        for i in range(len(peaks) - 1):
            left_peak = peaks[i]
            right_peak = peaks[i + 1]
            
            # Check peak similarity
            if not self._peaks_similar(df, left_peak, right_peak):
                continue
                
            # Get candles between peaks (use integer positions)
            left_pos = df.index.get_loc(left_peak['timestamp'])
            right_pos = df.index.get_loc(right_peak['timestamp'])
            
            candle_count = right_pos - left_pos + 1
            
            # Check candle count
            if not (self.min_candles <= candle_count <= self.max_candles):
                continue
                
            # Find neckline (lowest body between peaks)
            neckline_idx, neckline_price = self._find_neckline(df, left_pos, right_pos)
            
            # Check if pattern is valid
            pattern = self._validate_m_pattern(
                df, left_peak, right_peak, left_pos, right_pos, neckline_idx, neckline_price,
                instrument, timeframe
            )
            
            if pattern:
                patterns.append(pattern)
                
        return patterns
    
    def detect_w_bottom(self, df: pd.DataFrame, instrument: str = "", timeframe: str = "1h") -> List[MWPattern]:
        """Detect W-Bottom patterns (inverse of M-Top)"""
        troughs = self._find_troughs(df)
        patterns = []
        
        for i in range(len(troughs) - 1):
            left_trough = troughs[i]
            right_trough = troughs[i + 1]
            
            if not self._troughs_similar(df, left_trough, right_trough):
                continue
                
            left_pos = df.index.get_loc(left_trough['timestamp'])
            right_pos = df.index.get_loc(right_trough['timestamp'])
            candle_count = right_pos - left_pos + 1
            
            if not (self.min_candles <= candle_count <= self.max_candles):
                continue
                
            # Find neckline (highest body between troughs)
            neckline_idx, neckline_price = self._find_neckline_w(df, left_pos, right_pos)
            
            pattern = self._validate_w_pattern(
                df, left_trough, right_trough, left_pos, right_pos, neckline_idx, neckline_price,
                instrument, timeframe
            )
            
            if pattern:
                patterns.append(pattern)
                
        return patterns
    
    def _find_peaks(self, df: pd.DataFrame, window: int = 5) -> List[Dict]:
        """Find local maxima (peaks) in price"""
        peaks = []
        
        for i in range(window, len(df) - window):
            current_high = df.iloc[i]['high']
            
            # Check if this is a local maximum
            left_max = df.iloc[i-window:i]['high'].max()
            right_max = df.iloc[i+1:i+window+1]['high'].max()
            
            if current_high > left_max and current_high > right_max:
                peaks.append({
                    'idx': i,
                    'price': current_high,
                    'timestamp': df.index[i]
                })
        
        return peaks
    
    def _find_troughs(self, df: pd.DataFrame, window: int = 5) -> List[Dict]:
        """Find local minima (troughs) in price"""
        troughs = []
        
        for i in range(window, len(df) - window):
            current_low = df.iloc[i]['low']
            
            # Check if this is a local minimum
            left_min = df.iloc[i-window:i]['low'].min()
            right_min = df.iloc[i+1:i+window+1]['low'].min()
            
            if current_low < left_min and current_low < right_min:
                troughs.append({
                    'idx': i,
                    'price': current_low,
                    'timestamp': df.index[i]
                })
        
        return troughs
    
    def _peaks_similar(self, df: pd.DataFrame, left: Dict, right: Dict) -> bool:
        """Check if two peaks are within similarity tolerance"""
        left_price = left['price']
        right_price = right['price']
        
        price_diff_pct = abs(right_price - left_price) / left_price
        return price_diff_pct <= self.peak_similarity_tolerance
    
    def _troughs_similar(self, df: pd.DataFrame, left: Dict, right: Dict) -> bool:
        """Check if two troughs are within similarity tolerance"""
        left_price = left['price']
        right_price = right['price']
        
        price_diff_pct = abs(right_price - left_price) / left_price
        return price_diff_pct <= self.peak_similarity_tolerance
    
    def _find_neckline(self, df: pd.DataFrame, left_pos: int, right_pos: int) -> Tuple[int, float]:
        """
        Find neckline for M-Top (lowest body between peaks)
        Uses bodies only, ignores wicks
        """
        between_df = df.iloc[left_pos:right_pos + 1]
        # Use low as proxy for bottom of body
        min_low_idx = between_df['low'].idxmin()
        # Convert back to integer position
        neckline_idx = df.index.get_loc(min_low_idx)
        neckline_price = df.loc[min_low_idx, 'low']
        return neckline_idx, neckline_price
    
    def _find_neckline_w(self, df: pd.DataFrame, left_pos: int, right_pos: int) -> Tuple[int, float]:
        """
        Find neckline for W-Bottom (highest body between troughs)
        Uses bodies only, ignores wicks
        """
        between_df = df.iloc[left_pos:right_pos + 1]
        # Use high as proxy for top of body
        max_high_idx = between_df['high'].idxmax()
        neckline_idx = df.index.get_loc(max_high_idx)
        neckline_price = df.loc[max_high_idx, 'high']
        return neckline_idx, neckline_price
    
    def _calculate_stop_loss(self, df: pd.DataFrame, peak: Dict, pattern_type: str = 'M') -> float:
        """
        Calculate stop loss position
        For M-Top: above right peak (or highest point)
        Can ignore single rogue wicks
        """
        idx = peak['idx']
        
        if self.ignore_rogue_wicks:
            # Look at nearby candles to see if this is a rogue wick
            window = min(3, len(df) - idx - 1)
            surrounding_highs = []
            
            for i in range(max(0, idx - window), min(len(df), idx + window + 1)):
                surrounding_highs.append(df.iloc[i]['high'])
            
            # If this peak is significantly higher than neighbors, it's a rogue wick
            avg_surrounding = np.mean([h for h in surrounding_highs if h != peak['price']])
            if avg_surrounding > 0 and peak['price'] > avg_surrounding * 1.02:  # 2% higher than average
                # Use next highest point
                next_highest = max([h for h in surrounding_highs if h != peak['price']])
                logger.debug(f"Rogue wick detected at {df.index[idx]}, using {next_highest} instead of {peak['price']}")
                return next_highest
        
        return peak['price']
    
    def _check_neckline_break(self, df: pd.DataFrame, neckline_idx: int, neckline_price: float, pattern_type: str) -> bool:
        """
        Check if price actually broke the neckline after pattern formed
        For M-Top: break below neckline
        """
        if neckline_idx >= len(df) - 1:
            return False
            
        for i in range(neckline_idx + 1, min(neckline_idx + 10, len(df))):
            if pattern_type == 'M':
                if df.iloc[i]['low'] < neckline_price:
                    logger.debug(f"Neckline break at {df.index[i]}: low={df.iloc[i]['low']} < neckline={neckline_price}")
                    return True
            else:  # W
                if df.iloc[i]['high'] > neckline_price:
                    logger.debug(f"Neckline break at {df.index[i]}: high={df.iloc[i]['high']} > neckline={neckline_price}")
                    return True
        
        return False
    
    def _check_tp_within_leg(self, df: pd.DataFrame, left_peak_pos: int, tp_price: float, pattern_type: str) -> bool:
        """
        Check if take profit is within the last impulsive leg
        Critical Simon rule: TP must be within the price range of the move into pattern
        """
        # Look back up to 20 candles before pattern
        lookback = min(20, left_peak_pos)
        leg_df = df.iloc[left_peak_pos - lookback:left_peak_pos]
        
        if pattern_type == 'M':
            # For M-Top, TP is lower, must be within leg's low range
            leg_low = leg_df['low'].min()
            valid = tp_price >= leg_low
            if not valid:
                logger.debug(f"TP {tp_price} not within leg low {leg_low}")
            return valid
        else:
            # For W-Bottom, TP is higher, must be within leg's high range
            leg_high = leg_df['high'].max()
            valid = tp_price <= leg_high
            if not valid:
                logger.debug(f"TP {tp_price} not within leg high {leg_high}")
            return valid
    
    def _check_impulsive_move(self, df: pd.DataFrame, left_peak_pos: int, lookback: int = 20) -> bool:
        """
        Check if price move into pattern was impulsive
        Simon: "Beautiful clean price action going into it"
        """
        if left_peak_pos < lookback:
            lookback = left_peak_pos
            
        pre_df = df.iloc[left_peak_pos - lookback:left_peak_pos]
        
        # Calculate metrics
        price_change = (pre_df['close'].iloc[-1] - pre_df['close'].iloc[0]) / pre_df['close'].iloc[0]
        
        # Count pullbacks (candles opposite to trend)
        trend_direction = 1 if price_change > 0 else -1
        pullbacks = ((pre_df['close'] - pre_df['open']) * trend_direction < 0).sum()
        pullback_ratio = pullbacks / len(pre_df) if len(pre_df) > 0 else 1.0
        
        # Simon's rule: impulsive if minimal pullbacks (<30%) and decent move
        is_impulsive = pullback_ratio < 0.3 and abs(price_change) > 0.005
        if not is_impulsive:
            logger.debug(f"Not impulsive: pullback_ratio={pullback_ratio:.2f}, price_change={price_change:.4f}")
        return is_impulsive
    
    def _validate_m_pattern(self, df, left_peak, right_peak, left_pos, right_pos, 
                           neckline_idx, neckline_price, instrument, timeframe) -> Optional[MWPattern]:
        """
        Validate M-Top pattern with all Simon's rules
        """
        validation_errors = []
        
        # Check impulsive move into pattern (optional but preferred)
        impulsive = self._check_impulsive_move(df, left_pos)
        if not impulsive:
            validation_errors.append("Insufficient impulsive move into pattern")
            
        # Check if neckline break would have occurred
        break_occurred = self._check_neckline_break(df, neckline_idx, neckline_price, 'M')
        if not break_occurred:
            validation_errors.append("Pattern never broke neckline")
            # If no break, pattern is invalid
            return None
            
        # Calculate stop loss (behind right peak, ignoring rogue wicks)
        stop_loss = self._calculate_stop_loss(df, right_peak, pattern_type='M')
        if stop_loss <= 0:
            validation_errors.append(f"Invalid stop loss: {stop_loss}")
            return None
            
        # Entry price is neckline price
        entry_price = neckline_price
        
        # Calculate take profit (1:1 from entry)
        risk_distance = stop_loss - entry_price
        if risk_distance <= 0:
            validation_errors.append(f"Invalid risk distance: {risk_distance}")
            return None
            
        take_profit = entry_price - risk_distance
        
        # Check if take profit is within last impulsive leg
        tp_valid = self._check_tp_within_leg(df, left_pos, take_profit, pattern_type='M')
        if not tp_valid:
            validation_errors.append("Take profit not within last impulsive leg")
            
        # Create pattern
        pattern = MWPattern(
            pattern_type='M',
            instrument=instrument,
            timeframe=timeframe,
            left_peak_idx=left_peak['idx'],
            left_peak_price=left_peak['price'],
            right_peak_idx=right_peak['idx'],
            right_peak_price=right_peak['price'],
            neckline_idx=neckline_idx,
            neckline_price=neckline_price,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            candle_count=right_pos - left_pos + 1,
            valid=len(validation_errors) == 0,
            validation_errors=validation_errors,
            detected_at=df.index[-1] if len(df) > 0 else None
        )
        
        if pattern.valid:
            logger.debug(f"Valid M-Top found: entry={entry_price:.5f}, stop={stop_loss:.5f}, tp={take_profit:.5f}")
        else:
            logger.debug(f"Invalid M-Top: {validation_errors}")
            
        return pattern if pattern.valid else None
    
    def _validate_w_pattern(self, df, left_trough, right_trough, left_pos, right_pos,
                           neckline_idx, neckline_price, instrument, timeframe) -> Optional[MWPattern]:
        """Validate W-Bottom pattern"""
        validation_errors = []
        
        # Check impulsive move into pattern
        impulsive = self._check_impulsive_move(df, left_pos)
        if not impulsive:
            validation_errors.append("Insufficient impulsive move into pattern")
            
        # Check if neckline break would have occurred
        break_occurred = self._check_neckline_break(df, neckline_idx, neckline_price, 'W')
        if not break_occurred:
            validation_errors.append("Pattern never broke neckline")
            return None
            
        # Calculate stop loss (below right trough, ignoring rogue wicks)
        stop_loss = self._calculate_stop_loss(df, right_trough, pattern_type='W')
        if stop_loss <= 0:
            validation_errors.append(f"Invalid stop loss: {stop_loss}")
            return None
            
        entry_price = neckline_price
        risk_distance = entry_price - stop_loss
        if risk_distance <= 0:
            validation_errors.append(f"Invalid risk distance: {risk_distance}")
            return None
            
        take_profit = entry_price + risk_distance
        
        # Check if take profit is within last impulsive leg
        tp_valid = self._check_tp_within_leg(df, left_pos, take_profit, pattern_type='W')
        if not tp_valid:
            validation_errors.append("Take profit not within last impulsive leg")
            
        pattern = MWPattern(
            pattern_type='W',
            instrument=instrument,
            timeframe=timeframe,
            left_peak_idx=left_trough['idx'],
            left_peak_price=left_trough['price'],
            right_peak_idx=right_trough['idx'],
            right_peak_price=right_trough['price'],
            neckline_idx=neckline_idx,
            neckline_price=neckline_price,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            candle_count=right_pos - left_pos + 1,
            valid=len(validation_errors) == 0,
            validation_errors=validation_errors,
            detected_at=df.index[-1] if len(df) > 0 else None
        )
        
        if pattern.valid:
            logger.debug(f"Valid W-Bottom found: entry={entry_price:.5f}, stop={stop_loss:.5f}, tp={take_profit:.5f}")
        else:
            logger.debug(f"Invalid W-Bottom: {validation_errors}")
            
        return pattern if pattern.valid else None
