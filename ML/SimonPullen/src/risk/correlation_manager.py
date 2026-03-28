"""
Correlation management for Simon Pullen's risk rules
Tracks correlations between instruments to limit exposure
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CorrelationGroup:
    """Represents a group of correlated instruments"""
    name: str
    instruments: List[str]
    max_risk_percent: float = 2.0


class CorrelationManager:
    """
    Manages correlation between instruments to prevent overexposure
    Simon: Max 2% across correlated pairs (JPY, EUR, GBP groups)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.groups = self._init_groups()
        self.current_risk_by_group: Dict[str, float] = {}
        
    def _init_groups(self) -> Dict[str, CorrelationGroup]:
        """Initialize correlation groups from config"""
        groups = {}
        
        # Get correlation groups from config
        correlation_config = self.config.get('correlation_groups', {})
        max_risk = self.config.get('max_correlation_risk', 2.0)
        
        for group_name, instruments in correlation_config.items():
            groups[group_name] = CorrelationGroup(
                name=group_name,
                instruments=instruments,
                max_risk_percent=max_risk
            )
        
        # Add default groups if none in config
        if not groups:
            groups = {
                'jpy_pairs': CorrelationGroup('jpy_pairs', 
                    ['USD/JPY', 'EUR/JPY', 'GBP/JPY', 'AUD/JPY', 'CAD/JPY', 'NZD/JPY', 'CHF/JPY'], max_risk),
                'eur_pairs': CorrelationGroup('eur_pairs',
                    ['EUR/USD', 'EUR/GBP', 'EUR/JPY', 'EUR/CHF', 'EUR/AUD', 'EUR/CAD', 'EUR/NZD'], max_risk),
                'gbp_pairs': CorrelationGroup('gbp_pairs',
                    ['GBP/USD', 'GBP/JPY', 'GBP/CHF', 'GBP/AUD', 'GBP/CAD', 'GBP/NZD'], max_risk),
                'aud_pairs': CorrelationGroup('aud_pairs',
                    ['AUD/USD', 'AUD/JPY', 'AUD/CHF', 'AUD/CAD', 'AUD/NZD'], max_risk),
                'usd_pairs': CorrelationGroup('usd_pairs',
                    ['USD/JPY', 'USD/CHF', 'USD/CAD', 'EUR/USD', 'GBP/USD', 'AUD/USD', 'NZD/USD'], max_risk)
            }
        
        return groups
    
    def get_group_for_instrument(self, instrument: str) -> Optional[str]:
        """Get correlation group for an instrument"""
        for group_name, group in self.groups.items():
            if instrument in group.instruments:
                return group_name
        return None
    
    def get_all_groups_for_instrument(self, instrument: str) -> List[str]:
        """Get all correlation groups that contain this instrument"""
        groups = []
        for group_name, group in self.groups.items():
            if instrument in group.instruments:
                groups.append(group_name)
        return groups
    
    def update_risk(self, instrument: str, risk_percent: float, add: bool = True):
        """Update risk tracking for an instrument's groups"""
        groups = self.get_all_groups_for_instrument(instrument)
        
        for group in groups:
            if group not in self.current_risk_by_group:
                self.current_risk_by_group[group] = 0
                
            if add:
                self.current_risk_by_group[group] += risk_percent
            else:
                self.current_risk_by_group[group] -= risk_percent
                
            logger.debug(f"Group {group} risk: {self.current_risk_by_group[group]:.2f}%")
    
    def can_add_to_group(self, instrument: str, risk_percent: float) -> Tuple[bool, str]:
        """
        Check if we can add risk to all groups containing this instrument
        Returns (can_add, reason)
        """
        groups = self.get_all_groups_for_instrument(instrument)
        
        for group in groups:
            current = self.current_risk_by_group.get(group, 0)
            max_risk = self.groups[group].max_risk_percent
            
            if current + risk_percent > max_risk:
                return False, f"Group {group} would exceed max risk ({current:.1f}% + {risk_percent:.1f}% > {max_risk:.1f}%)"
        
        return True, "OK"
    
    def get_group_risk(self, group: str) -> float:
        """Get current risk for a group"""
        return self.current_risk_by_group.get(group, 0.0)
    
    def reset(self):
        """Reset all risk tracking"""
        self.current_risk_by_group = {}
    
    def get_all_group_risks(self) -> Dict[str, float]:
        """Get current risks for all groups"""
        return self.current_risk_by_group.copy()
