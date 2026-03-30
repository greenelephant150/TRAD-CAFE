"""
Position sizing following Simon Pullen's 1% rule
- 1% per trade standard
- 2% max with high confluence
- Max 2% across correlated pairs
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

from src.risk.correlation_manager import CorrelationManager

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Position size recommendation"""
    instrument: str
    direction: str
    units: float
    risk_percent: float
    risk_amount: float
    account_risk_percent: float
    correlation_adjusted: bool = False


class PositionSizer:
    """
    Calculates position sizes based on Simon's risk rules
    """
    
    def __init__(self, config: Dict[str, Any], account_balance: float):
        self.config = config
        self.account_balance = account_balance
        self.default_risk_pct = config.get('default_risk_percent', 1.0) / 100
        self.max_risk_pct = config.get('max_risk_percent', 2.0) / 100
        
        # Initialize correlation manager
        self.correlation_manager = CorrelationManager(config)
        
        # Track open positions
        self.open_positions: List[Dict] = []
        
    def calculate_position_size(self, instrument: str, direction: str,
                                 entry_price: float, stop_loss: float,
                                 confidence: float = 0.7,
                                 instrument_info: Optional[Dict] = None) -> Optional[PositionSize]:
        """
        Calculate position size based on risk rules
        Returns None if risk per unit is invalid
        """
        # Validate inputs
        if entry_price <= 0 or stop_loss <= 0:
            logger.warning(f"Invalid price values: entry={entry_price}, stop={stop_loss}")
            return None
            
        # Calculate risk per unit
        if direction == 'long':
            if stop_loss >= entry_price:
                logger.warning(f"Invalid stop for long: stop={stop_loss} >= entry={entry_price}")
                return None
            risk_per_unit = entry_price - stop_loss
        else:  # short
            if stop_loss <= entry_price:
                logger.warning(f"Invalid stop for short: stop={stop_loss} <= entry={entry_price}")
                return None
            risk_per_unit = stop_loss - entry_price
        
        if risk_per_unit <= 0:
            logger.warning(f"Zero or negative risk per unit: {risk_per_unit}")
            return None
        
        # Determine risk percentage for this trade
        base_risk_pct = self.default_risk_pct
        
        # Increase risk for high-confidence trades
        if confidence > 0.85:
            base_risk_pct = min(base_risk_pct * 1.5, self.max_risk_pct)
        elif confidence > 0.75:
            base_risk_pct = min(base_risk_pct * 1.2, self.max_risk_pct)
        
        # Check correlation limits
        can_add, reason = self.correlation_manager.can_add_to_group(instrument, base_risk_pct * 100)
        correlation_adjusted = False
        
        if not can_add:
            # Try with reduced risk
            available_risk = self._get_available_risk(instrument)
            if available_risk <= 0:
                logger.debug(f"Cannot add position: {reason}")
                return None
            
            base_risk_pct = min(base_risk_pct, available_risk / 100)
            correlation_adjusted = True
        
        # Calculate position size
        risk_amount = self.account_balance * base_risk_pct
        units = risk_amount / risk_per_unit
        
        # Round to instrument step size
        if instrument_info and 'step_size' in instrument_info:
            step = instrument_info['step_size']
            units = round(units / step) * step
        
        return PositionSize(
            instrument=instrument,
            direction=direction,
            units=units,
            risk_percent=base_risk_pct * 100,
            risk_amount=risk_amount,
            account_risk_percent=base_risk_pct * 100,
            correlation_adjusted=correlation_adjusted
        )
    
    def _get_available_risk(self, instrument: str) -> float:
        """Get available risk percentage for an instrument considering correlations"""
        groups = self.correlation_manager.get_all_groups_for_instrument(instrument)
        
        if not groups:
            return self.max_risk_pct * 100
            
        # Find the most restrictive group
        max_allowed = self.max_risk_pct * 100
        for group in groups:
            current = self.correlation_manager.get_group_risk(group)
            group_max = self.correlation_manager.groups[group].max_risk_percent
            available = group_max - current
            max_allowed = min(max_allowed, available)
        
        return max(0, max_allowed)
    
    def add_position(self, position: Dict):
        """Add a position to tracking"""
        self.open_positions.append(position)
        self.correlation_manager.update_risk(
            position['instrument'], 
            position.get('risk_percent', 0),
            add=True
        )
    
    def remove_position(self, instrument: str, risk_percent: float):
        """Remove a position from tracking"""
        self.open_positions = [p for p in self.open_positions if p['instrument'] != instrument]
        self.correlation_manager.update_risk(instrument, risk_percent, add=False)
    
    def get_account_risk(self) -> float:
        """Get total account risk from all open positions"""
        if not self.open_positions:
            return 0.0
            
        total = sum(p.get('risk_percent', 0) for p in self.open_positions)
        
        # Get group risks
        group_risks = self.correlation_manager.get_all_group_risks()
        
        # Account risk is max of total or any group
        return max(total, max(group_risks.values()) if group_risks else total)
    
    def can_take_trade(self, instrument: str, confidence: float = 0.7) -> Tuple[bool, str]:
        """
        Check if we can take a new trade based on risk limits
        """
        # Determine risk for this trade
        trade_risk_pct = self.default_risk_pct * 100
        if confidence > 0.85:
            trade_risk_pct = min(trade_risk_pct * 1.5, self.max_risk_pct * 100)
        
        # Check correlation limits
        can_add, reason = self.correlation_manager.can_add_to_group(instrument, trade_risk_pct)
        if not can_add:
            return False, reason
        
        # Check total account risk
        current_risk = self.get_account_risk()
        if current_risk + trade_risk_pct > self.max_risk_pct * 100:
            return False, f"Would exceed max account risk ({current_risk:.1f}% + {trade_risk_pct:.1f}% > {self.max_risk_pct*100:.1f}%)"
        
        return True, "OK"
    
    def update_account_balance(self, new_balance: float):
        """Update account balance"""
        self.account_balance = new_balance
