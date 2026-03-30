"""
OANDA API utilities for Simon Pullen trading system
"""

import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class OandaUtils:
    """Utility functions for OANDA API operations"""
    
    @staticmethod
    def pair_to_oanda_format(pair: str) -> str:
        """Convert EUR_USD to OANDA format"""
        return pair.replace('_', '')
    
    @staticmethod
    def oanda_to_pair_format(oanda_pair: str) -> str:
        """Convert EURUSD to EUR_USD format"""
        if len(oanda_pair) == 6:
            return f"{oanda_pair[:3]}_{oanda_pair[3:]}"
        return oanda_pair
    
    @staticmethod
    def get_pip_value(instrument: str) -> float:
        """Get pip value for instrument"""
        if 'JPY' in instrument:
            return 0.01
        elif 'XAU' in instrument or 'XAG' in instrument:
            return 0.01  # Gold/Silver
        else:
            return 0.0001
    
    @staticmethod
    def calculate_pips(price1: float, price2: float, instrument: str) -> float:
        """Calculate pips between two prices"""
        pip_value = OandaUtils.get_pip_value(instrument)
        return abs(price1 - price2) / pip_value
    
    @staticmethod
    def price_to_pips(price: float, instrument: str) -> float:
        """Convert price to pips"""
        pip_value = OandaUtils.get_pip_value(instrument)
        return price / pip_value
    
    @staticmethod
    def pips_to_price(pips: float, instrument: str) -> float:
        """Convert pips to price"""
        pip_value = OandaUtils.get_pip_value(instrument)
        return pips * pip_value
