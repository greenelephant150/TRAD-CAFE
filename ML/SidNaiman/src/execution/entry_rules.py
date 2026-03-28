"""
Entry rule engine for Simon Pullen patterns
Handles the different entry rules for M&W vs Head & Shoulders
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging

from src.core.mw_pattern import MWPattern
from src.core.head_shoulders import HeadShouldersPattern

logger = logging.getLogger(__name__)


@dataclass
class EntrySignal:
    """Represents an entry signal"""
    instrument: str
    pattern_type: str  # 'M', 'W', 'HS_normal', 'HS_inverted'
    direction: str  # 'long' or 'short'
    entry_price: float
    stop_loss: float
    take_profit: float
    stop_loss_type: str  # 'conservative', 'moderate', 'aggressive'
    confidence: float  # 0-1
    confluence_score: float
    detected_at: pd.Timestamp
    pattern_ref: Any  # Reference to original pattern


class EntryRuleEngine:
    """
    Manages entry rules for different pattern types
    
    M&W: Entry on break of neckline (no retest needed)
    H&S: Must have break + close + retest + entry candle
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mw_entry_requires_retest = config.get('mw_entry_requires_retest', False)
        self.hs_requires_retest = config.get('hs_requires_retest', True)
        self.hs_requires_entry_candle = config.get('hs_requires_entry_candle', True)
        self.default_stop_type = config.get('default_stop_type', 'moderate')
        
    def check_mw_entry(self, df: pd.DataFrame, pattern: MWPattern, current_idx: int = -1) -> Optional[EntrySignal]:
        """
        Check if M or W pattern has triggered entry
        Simon: Entry on break of neckline, no retest needed
        """
        if not pattern.valid:
            return None
            
        if current_idx < 0:
            current_idx = len(df) - 1
            
        # Validate stop loss
        if pattern.stop_loss_price <= 0:
            logger.debug(f"Invalid stop loss for {pattern.pattern_type}: {pattern.stop_loss_price}")
            return None
            
        # Check if price has broken neckline
        if pattern.pattern_type == 'M':
            # For M-Top, break below neckline
            for i in range(pattern.neckline_idx, current_idx + 1):
                if df.iloc[i]['low'] <= pattern.neckline_price:
                    logger.debug(f"M-Top entry triggered at {df.index[i]}")
                    return EntrySignal(
                        instrument=pattern.instrument,
                        pattern_type='M',
                        direction='short',
                        entry_price=pattern.neckline_price,
                        stop_loss=pattern.stop_loss_price,
                        take_profit=pattern.take_profit_price,
                        stop_loss_type='moderate',  # M&W uses fixed stop
                        confidence=0.7 + (pattern.confluence_score * 0.3),
                        confluence_score=pattern.confluence_score,
                        detected_at=df.index[current_idx],
                        pattern_ref=pattern
                    )
        else:  # W-Bottom
            # For W, break above neckline
            for i in range(pattern.neckline_idx, current_idx + 1):
                if df.iloc[i]['high'] >= pattern.neckline_price:
                    logger.debug(f"W-Bottom entry triggered at {df.index[i]}")
                    return EntrySignal(
                        instrument=pattern.instrument,
                        pattern_type='W',
                        direction='long',
                        entry_price=pattern.neckline_price,
                        stop_loss=pattern.stop_loss_price,
                        take_profit=pattern.take_profit_price,
                        stop_loss_type='moderate',
                        confidence=0.7 + (pattern.confluence_score * 0.3),
                        confluence_score=pattern.confluence_score,
                        detected_at=df.index[current_idx],
                        pattern_ref=pattern
                    )
        
        return None
    
    def check_hs_entry(self, df: pd.DataFrame, pattern: HeadShouldersPattern, 
                        current_idx: int = -1, stop_type: str = None) -> Optional[EntrySignal]:
        """
        Check if Head & Shoulders pattern has triggered entry
        Simon: Must have break + close + retest + entry candle
        """
        if not pattern.valid:
            return None
            
        if stop_type is None:
            stop_type = self.default_stop_type
            
        if current_idx < 0:
            current_idx = len(df) - 1
            
        # Check if we have all required components
        if pattern.break_idx is None or pattern.retest_idx is None or pattern.entry_candle_idx is None:
            return None
            
        # Get stop loss from options
        if stop_type not in pattern.stop_loss_options:
            logger.debug(f"Stop type {stop_type} not available, using moderate")
            stop_type = 'moderate'
            
        stop_loss = pattern.stop_loss_options.get(stop_type, 0)
        if stop_loss <= 0:
            logger.debug(f"Invalid stop loss for {pattern.pattern_type}: {stop_loss}")
            return None
            
        # Check if entry candle has been triggered
        if pattern.pattern_type == 'normal':
            # For normal H&S, entry on break below entry candle
            if pattern.entry_candle_idx > current_idx:
                return None
                
            for i in range(pattern.entry_candle_idx, current_idx + 1):
                if df.iloc[i]['low'] <= pattern.entry_price:
                    logger.debug(f"H&S normal entry triggered at {df.index[i]}")
                    return EntrySignal(
                        instrument=pattern.instrument,
                        pattern_type='HS_normal',
                        direction='short',
                        entry_price=pattern.entry_price,
                        stop_loss=stop_loss,
                        take_profit=pattern.take_profit_price,
                        stop_loss_type=stop_type,
                        confidence=0.75 + (pattern.confluence_score * 0.25),
                        confluence_score=pattern.confluence_score,
                        detected_at=df.index[current_idx],
                        pattern_ref=pattern
                    )
        else:  # inverted
            if pattern.entry_candle_idx > current_idx:
                return None
                
            for i in range(pattern.entry_candle_idx, current_idx + 1):
                if df.iloc[i]['high'] >= pattern.entry_price:
                    logger.debug(f"H&S inverted entry triggered at {df.index[i]}")
                    return EntrySignal(
                        instrument=pattern.instrument,
                        pattern_type='HS_inverted',
                        direction='long',
                        entry_price=pattern.entry_price,
                        stop_loss=stop_loss,
                        take_profit=pattern.take_profit_price,
                        stop_loss_type=stop_type,
                        confidence=0.75 + (pattern.confluence_score * 0.25),
                        confluence_score=pattern.confluence_score,
                        detected_at=df.index[current_idx],
                        pattern_ref=pattern
                    )
        
        return None
    
    def should_enter_now(self, df: pd.DataFrame, pattern, current_idx: int = -1, **kwargs) -> bool:
        """Convenience method to check if we should enter now"""
        if isinstance(pattern, MWPattern):
            return self.check_mw_entry(df, pattern, current_idx) is not None
        elif isinstance(pattern, HeadShouldersPattern):
            return self.check_hs_entry(df, pattern, current_idx, **kwargs) is not None
        else:
            return False
