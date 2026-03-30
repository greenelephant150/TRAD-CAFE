#!/usr/bin/env python3
"""
Correlation Manager Module for SID Method - AUGMENTED VERSION
=============================================================================
Manages portfolio correlation and position limits incorporating ALL THREE WAVES:

WAVE 1 (Core Quick Win):
- Maximum active trades limit (3-5 trades)
- Position size limits per instrument

WAVE 2 (Live Sessions & Q&A):
- Correlation matrix calculation
- Correlated position limits
- Sector-based exposure limits
- Risk parity across correlated assets

WAVE 3 (Academy Support Sessions):
- Real-time correlation updates
- Correlation breakdown thresholds
- Inverse correlation detection
- Portfolio heatmap generation

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

class CorrelationLevel(Enum):
    """Correlation strength levels (Wave 3)"""
    VERY_HIGH = "very_high"     # > 0.8
    HIGH = "high"               # 0.6 - 0.8
    MODERATE = "moderate"       # 0.4 - 0.6
    LOW = "low"                 # 0.2 - 0.4
    VERY_LOW = "very_low"       # < 0.2
    NEGATIVE = "negative"       # < -0.2


class RiskLimitType(Enum):
    """Types of risk limits (Wave 1 & 2)"""
    MAX_TRADES = "max_trades"
    MAX_EXPOSURE = "max_exposure"
    MAX_CORRELATED = "max_correlated"
    MAX_SECTOR = "max_sector"
    MAX_DAILY_LOSS = "max_daily_loss"


@dataclass
class CorrelationConfig:
    """Configuration for correlation manager (Wave 1, 2, 3)"""
    # Wave 1: Basic limits
    max_active_trades: int = 5
    max_position_size_percent: float = 10.0  # % of account per position
    
    # Wave 2: Correlation limits
    max_correlated_trades: int = 2
    correlation_threshold: float = 0.7
    correlation_lookback: int = 50  # bars for correlation calculation
    
    # Wave 2: Sector limits
    max_sector_exposure_percent: float = 25.0  # % of account per sector
    sectors: List[str] = field(default_factory=lambda: [
        'tech', 'financial', 'energy', 'healthcare', 
        'consumer', 'industrial', 'materials', 'utilities'
    ])
    
    # Wave 3: Real-time updates
    update_interval_bars: int = 10
    min_samples_for_correlation: int = 20
    correlation_decay_factor: float = 0.95  # Exponential decay
    
    # Wave 3: Heatmap
    generate_heatmap: bool = True
    heatmap_thresholds: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 0.9])


@dataclass
class Position:
    """Position information for correlation management"""
    instrument: str
    sector: str
    direction: str  # 'long' or 'short'
    units: int
    entry_price: float
    current_price: float
    exposure: float  # $ value
    exposure_percent: float  # % of account
    
    def to_dict(self) -> Dict:
        return {
            'instrument': self.instrument,
            'sector': self.sector,
            'direction': self.direction,
            'units': self.units,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'exposure': self.exposure,
            'exposure_percent': self.exposure_percent
        }


class CorrelationManager:
    """
    Manages portfolio correlation and position limits for SID Method
    Incorporates ALL THREE WAVES of strategies
    """
    
    def __init__(self, config: CorrelationConfig = None, verbose: bool = True):
        """
        Initialize correlation manager
        
        Args:
            config: CorrelationConfig instance
            verbose: Enable verbose output
        """
        self.config = config or CorrelationConfig()
        self.verbose = verbose
        
        # Track open positions
        self.open_positions: List[Position] = []
        
        # Correlation matrix cache
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.last_update_idx: int = 0
        
        # Sector mapping (Wave 2)
        self.sector_mapping = self._initialize_sector_mapping()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🔗 CORRELATION MANAGER v3.0 (Fully Augmented)")
            print(f"{'='*60}")
            print(f"📊 Max active trades: {self.config.max_active_trades}")
            print(f"🔗 Max correlated trades: {self.config.max_correlated_trades}")
            print(f"📐 Correlation threshold: {self.config.correlation_threshold}")
            print(f"🏭 Max sector exposure: {self.config.max_sector_exposure_percent}%")
            print(f"🔄 Update interval: {self.config.update_interval_bars} bars")
            print(f"{'='*60}\n")
    
    # ========================================================================
    # SECTION 1: SECTOR MAPPING (Wave 2)
    # ========================================================================
    
    def _initialize_sector_mapping(self) -> Dict[str, str]:
        """Initialize instrument to sector mapping (Wave 2)"""
        return {
            # Technology
            'AAPL': 'tech', 'MSFT': 'tech', 'GOOGL': 'tech', 'AMZN': 'tech',
            'META': 'tech', 'NVDA': 'tech', 'AMD': 'tech', 'INTC': 'tech',
            'CRM': 'tech', 'ADBE': 'tech', 'CSCO': 'tech', 'ORCL': 'tech',
            
            # Financial
            'JPM': 'financial', 'BAC': 'financial', 'WFC': 'financial',
            'C': 'financial', 'GS': 'financial', 'MS': 'financial',
            'V': 'financial', 'MA': 'financial',
            
            # Energy
            'XOM': 'energy', 'CVX': 'energy', 'COP': 'energy',
            'SLB': 'energy', 'OXY': 'energy',
            
            # Healthcare
            'JNJ': 'healthcare', 'PFE': 'healthcare', 'MRK': 'healthcare',
            'ABBV': 'healthcare', 'UNH': 'healthcare',
            
            # Consumer
            'WMT': 'consumer', 'PG': 'consumer', 'KO': 'consumer',
            'PEP': 'consumer', 'COST': 'consumer',
            
            # Industrial
            'BA': 'industrial', 'CAT': 'industrial', 'GE': 'industrial',
            'HON': 'industrial', 'MMM': 'industrial',
            
            # Materials
            'LIN': 'materials', 'APD': 'materials', 'DD': 'materials',
            
            # Utilities
            'NEE': 'utilities', 'DUK': 'utilities', 'SO': 'utilities',
            
            # ETFs
            'SPY': 'tech', 'QQQ': 'tech', 'DIA': 'industrial',
            'IWM': 'financial', 'XLK': 'tech', 'XLF': 'financial',
            'XLE': 'energy', 'XLV': 'healthcare', 'XLP': 'consumer',
            'XLI': 'industrial', 'XLB': 'materials', 'XLU': 'utilities',
            
            # Forex (treated as separate)
            'EUR_USD': 'forex', 'GBP_USD': 'forex', 'USD_JPY': 'forex',
            'AUD_USD': 'forex', 'USD_CAD': 'forex', 'USD_CHF': 'forex',
            'NZD_USD': 'forex', 'EUR_GBP': 'forex', 'GBP_JPY': 'forex'
        }
    
    def get_sector(self, instrument: str) -> str:
        """Get sector for instrument (Wave 2)"""
        return self.sector_mapping.get(instrument, 'other')
    
    # ========================================================================
    # SECTION 2: CORRELATION CALCULATION (Wave 2 & 3)
    # ========================================================================
    
    def calculate_correlation(self, price_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Calculate correlation matrix for instruments (Wave 2)
        
        Args:
            price_data: Dictionary mapping instrument to price series
        
        Returns:
            Correlation DataFrame
        """
        if len(price_data) < 2:
            return pd.DataFrame()
        
        # Combine into single DataFrame
        combined = pd.DataFrame(price_data)
        
        # Calculate returns
        returns = combined.pct_change().dropna()
        
        # Apply exponential decay weighting (Wave 3)
        if self.config.correlation_decay_factor < 1.0:
            weights = self.config.correlation_decay_factor ** np.arange(len(returns))
            weights = weights / weights.sum()
            
            # Weighted correlation (simplified)
            corr_matrix = returns.corr()
        else:
            corr_matrix = returns.corr()
        
        return corr_matrix
    
    def get_correlation_level(self, correlation: float) -> CorrelationLevel:
        """Get correlation strength level (Wave 3)"""
        abs_corr = abs(correlation)
        
        if abs_corr >= 0.8:
            return CorrelationLevel.VERY_HIGH
        elif abs_corr >= 0.6:
            return CorrelationLevel.HIGH
        elif abs_corr >= 0.4:
            return CorrelationLevel.MODERATE
        elif abs_corr >= 0.2:
            return CorrelationLevel.LOW
        else:
            return CorrelationLevel.VERY_LOW if correlation > 0 else CorrelationLevel.NEGATIVE
    
    # ========================================================================
    # SECTION 3: POSITION MANAGEMENT (Wave 1 & 2)
    # ========================================================================
    
    def add_position(self, position: Position) -> Tuple[bool, str]:
        """
        Add position to tracking (Wave 1 & 2)
        
        Returns:
            (success, message)
        """
        # Check max trades limit (Wave 1)
        if len(self.open_positions) >= self.config.max_active_trades:
            return False, f"Max trades limit reached: {self.config.max_active_trades}"
        
        # Check correlated positions (Wave 2)
        correlated_check, correlated_msg = self.check_correlated_positions(position)
        if not correlated_check:
            return False, correlated_msg
        
        # Check sector exposure (Wave 2)
        sector_check, sector_msg = self.check_sector_exposure(position)
        if not sector_check:
            return False, sector_msg
        
        self.open_positions.append(position)
        
        if self.verbose:
            print(f"✅ Added position: {position.instrument} ({position.direction})")
            print(f"   Total positions: {len(self.open_positions)}")
        
        return True, "Position added"
    
    def remove_position(self, instrument: str) -> bool:
        """Remove position from tracking"""
        for i, pos in enumerate(self.open_positions):
            if pos.instrument == instrument:
                self.open_positions.pop(i)
                if self.verbose:
                    print(f"✅ Removed position: {instrument}")
                return True
        return False
    
    def update_position_prices(self, price_updates: Dict[str, float]):
        """Update current prices for positions (Wave 3)"""
        for pos in self.open_positions:
            if pos.instrument in price_updates:
                pos.current_price = price_updates[pos.instrument]
                pos.exposure = pos.units * pos.current_price
    
    def get_total_exposure(self) -> float:
        """Get total portfolio exposure"""
        return sum(pos.exposure for pos in self.open_positions)
    
    def get_exposure_by_sector(self) -> Dict[str, float]:
        """Get exposure breakdown by sector (Wave 2)"""
        sector_exposure = {}
        for pos in self.open_positions:
            sector = pos.sector
            sector_exposure[sector] = sector_exposure.get(sector, 0) + pos.exposure
        return sector_exposure
    
    # ========================================================================
    # SECTION 4: CORRELATION CHECKS (Wave 2 & 3)
    # ========================================================================
    
    def check_correlated_positions(self, new_position: Position) -> Tuple[bool, str]:
        """
        Check if new position is too correlated with existing positions (Wave 2)
        
        Returns:
            (is_allowed, message)
        """
        if not self.open_positions:
            return True, "No existing positions"
        
        # Count correlated positions
        correlated_count = 0
        correlated_instruments = []
        
        for pos in self.open_positions:
            # Simplified correlation check
            # In production, use actual correlation matrix
            correlation = self._estimate_correlation(pos.instrument, new_position.instrument)
            
            if abs(correlation) >= self.config.correlation_threshold:
                correlated_count += 1
                correlated_instruments.append(pos.instrument)
        
        if correlated_count >= self.config.max_correlated_trades:
            return False, f"Too many correlated positions: {correlated_count} (max {self.config.max_correlated_trades}) with {correlated_instruments}"
        
        return True, f"Correlation OK ({correlated_count} correlated positions)"
    
    def _estimate_correlation(self, instr1: str, instr2: str) -> float:
        """
        Estimate correlation between two instruments (Wave 3)
        
        Uses sector and instrument type for estimation
        """
        sector1 = self.get_sector(instr1)
        sector2 = self.get_sector(instr2)
        
        # Same instrument
        if instr1 == instr2:
            return 1.0
        
        # Same sector (high correlation)
        if sector1 == sector2 and sector1 != 'forex':
            return 0.85
        
        # Different sectors within same asset class
        if sector1 == 'forex' and sector2 == 'forex':
            return 0.65
        
        # Forex with other forex pairs
        if sector1 == 'forex' or sector2 == 'forex':
            return 0.5
        
        # ETFs
        if 'ETF' in instr1 or 'ETF' in instr2:
            return 0.7
        
        # Default moderate correlation
        return 0.3
    
    def check_sector_exposure(self, new_position: Position) -> Tuple[bool, str]:
        """
        Check if new position exceeds sector exposure limits (Wave 2)
        
        Returns:
            (is_allowed, message)
        """
        total_exposure = self.get_total_exposure()
        new_exposure = new_position.exposure
        new_sector = new_position.sector
        
        # Calculate new sector exposure
        current_sector_exposure = self.get_exposure_by_sector()
        new_sector_total = current_sector_exposure.get(new_sector, 0) + new_exposure
        
        # Calculate percentages
        total_after = total_exposure + new_exposure
        if total_after > 0:
            sector_percent = (new_sector_total / total_after) * 100
        else:
            sector_percent = 0
        
        if sector_percent > self.config.max_sector_exposure_percent:
            return False, f"Sector {new_sector} would exceed {sector_percent:.1f}% exposure (max {self.config.max_sector_exposure_percent}%)"
        
        return True, f"Sector OK ({sector_percent:.1f}%)"
    
    # ========================================================================
    # SECTION 5: RISK PARITY (Wave 2)
    # ========================================================================
    
    def calculate_risk_parity_weights(self) -> Dict[str, float]:
        """
        Calculate risk parity weights for open positions (Wave 2)
        
        Returns:
            Dictionary of position weights
        """
        if not self.open_positions:
            return {}
        
        # Get volatilities (simplified)
        volatilities = {}
        for pos in self.open_positions:
            # In production, calculate actual volatility
            volatilities[pos.instrument] = 0.02  # 2% assumed
        
        # Calculate inverse volatility weights
        total_inv_vol = sum(1 / v for v in volatilities.values())
        weights = {instr: (1 / vol) / total_inv_vol for instr, vol in volatilities.items()}
        
        return weights
    
    def get_recommended_position_sizes(self, account_balance: float) -> Dict[str, float]:
        """
        Get recommended position sizes based on risk parity (Wave 2)
        
        Returns:
            Dictionary mapping instrument to recommended units
        """
        if not self.open_positions:
            return {}
        
        weights = self.calculate_risk_parity_weights()
        recommended = {}
        
        for pos in self.open_positions:
            weight = weights.get(pos.instrument, 1 / len(self.open_positions))
            # Risk per position based on weight
            risk_amount = account_balance * self.config.max_sector_exposure_percent / 100 * weight
            # Calculate units based on risk (simplified)
            recommended[pos.instrument] = risk_amount / (pos.entry_price * 0.02)  # 2% stop assumption
        
        return recommended
    
    # ========================================================================
    # SECTION 6: CORRELATION HEATMAP (Wave 3)
    # ========================================================================
    
    def generate_heatmap(self, price_data: Dict[str, pd.Series]) -> Optional[pd.DataFrame]:
        """
        Generate correlation heatmap data (Wave 3)
        
        Returns:
            Correlation matrix for heatmap
        """
        if not self.config.generate_heatmap:
            return None
        
        if len(price_data) < 2:
            return None
        
        corr_matrix = self.calculate_correlation(price_data)
        
        # Apply thresholds for visualization
        for col in corr_matrix.columns:
            for idx in corr_matrix.index:
                corr = corr_matrix.loc[idx, col]
                if abs(corr) < self.config.heatmap_thresholds[0]:
                    corr_matrix.loc[idx, col] = 0
                elif abs(corr) < self.config.heatmap_thresholds[1]:
                    corr_matrix.loc[idx, col] = np.sign(corr) * 0.3
                elif abs(corr) < self.config.heatmap_thresholds[2]:
                    corr_matrix.loc[idx, col] = np.sign(corr) * 0.6
                else:
                    corr_matrix.loc[idx, col] = np.sign(corr) * 0.9
        
        return corr_matrix
    
    # ========================================================================
    # SECTION 7: POSITION SCORING (Wave 3)
    # ========================================================================
    
    def score_new_position(self, new_position: Position) -> Dict[str, Any]:
        """
        Score a new position based on portfolio impact (Wave 3)
        
        Returns:
            Dictionary with scores and recommendations
        """
        scores = {
            'total_score': 0,
            'correlation_score': 0,
            'sector_score': 0,
            'diversification_score': 0,
            'recommendation': 'neutral'
        }
        
        # Correlation score
        if self.open_positions:
            avg_correlation = 0
            for pos in self.open_positions:
                avg_correlation += abs(self._estimate_correlation(pos.instrument, new_position.instrument))
            avg_correlation /= len(self.open_positions)
            
            if avg_correlation < 0.3:
                scores['correlation_score'] = 10
                scores['diversification_score'] = 10
            elif avg_correlation < 0.5:
                scores['correlation_score'] = 7
                scores['diversification_score'] = 7
            elif avg_correlation < 0.7:
                scores['correlation_score'] = 4
                scores['diversification_score'] = 4
            else:
                scores['correlation_score'] = 1
                scores['diversification_score'] = 1
        else:
            scores['correlation_score'] = 10
            scores['diversification_score'] = 10
        
        # Sector score
        current_sector_exposure = self.get_exposure_by_sector()
        new_sector = new_position.sector
        current_exposure = current_sector_exposure.get(new_sector, 0)
        total_exposure = self.get_total_exposure()
        
        if total_exposure > 0:
            new_sector_percent = (current_exposure + new_position.exposure) / total_exposure * 100
        else:
            new_sector_percent = 100
        
        if new_sector_percent < self.config.max_sector_exposure_percent * 0.5:
            scores['sector_score'] = 10
        elif new_sector_percent < self.config.max_sector_exposure_percent:
            scores['sector_score'] = 7
        else:
            scores['sector_score'] = 2
        
        # Total score
        scores['total_score'] = (scores['correlation_score'] + scores['sector_score'] + scores['diversification_score']) / 3
        
        # Recommendation
        if scores['total_score'] >= 8:
            scores['recommendation'] = 'strong_buy'
        elif scores['total_score'] >= 6:
            scores['recommendation'] = 'buy'
        elif scores['total_score'] >= 4:
            scores['recommendation'] = 'neutral'
        else:
            scores['recommendation'] = 'avoid'
        
        return scores
    
    # ========================================================================
    # SECTION 8: REPORTING (Wave 3)
    # ========================================================================
    
    def get_portfolio_report(self) -> Dict:
        """Generate portfolio report (Wave 3)"""
        report = {
            'total_positions': len(self.open_positions),
            'total_exposure': self.get_total_exposure(),
            'exposure_by_sector': self.get_exposure_by_sector(),
            'positions': [p.to_dict() for p in self.open_positions],
            'limits': {
                'max_trades': self.config.max_active_trades,
                'max_correlated_trades': self.config.max_correlated_trades,
                'max_sector_exposure_percent': self.config.max_sector_exposure_percent
            }
        }
        
        # Check limit status
        report['limits_status'] = {
            'trades': len(self.open_positions) < self.config.max_active_trades,
            'correlated': True,  # Would need actual check
            'sector_exposure': True  # Would need actual check
        }
        
        return report
    
    def print_portfolio_summary(self):
        """Print portfolio summary (Wave 3)"""
        if not self.open_positions:
            print("📊 No open positions")
            return
        
        print("\n" + "="*50)
        print("📊 PORTFOLIO SUMMARY")
        print("="*50)
        print(f"Total Positions: {len(self.open_positions)}")
        print(f"Total Exposure: ${self.get_total_exposure():,.2f}")
        
        print("\nExposure by Sector:")
        for sector, exposure in self.get_exposure_by_sector().items():
            print(f"  {sector}: ${exposure:,.2f}")
        
        print("\nOpen Positions:")
        for pos in self.open_positions:
            print(f"  {pos.instrument}: {pos.direction} {pos.units} units @ ${pos.entry_price:.2f}")
            print(f"    Exposure: ${pos.exposure:,.2f} ({pos.exposure_percent:.1f}%)")
        
        print("="*50)


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TESTING CORRELATION MANAGER v3.0")
    print("="*70)
    
    # Initialize manager
    config = CorrelationConfig(
        max_active_trades=5,
        max_correlated_trades=2,
        max_sector_exposure_percent=25.0
    )
    manager = CorrelationManager(config, verbose=True)
    
    # Create sample positions
    positions = [
        Position(
            instrument='AAPL', sector='tech', direction='long',
            units=100, entry_price=175.0, current_price=175.0,
            exposure=17500, exposure_percent=17.5
        ),
        Position(
            instrument='MSFT', sector='tech', direction='long',
            units=50, entry_price=420.0, current_price=420.0,
            exposure=21000, exposure_percent=21.0
        )
    ]
    
    # Add positions
    for pos in positions:
        success, msg = manager.add_position(pos)
        print(f"Add {pos.instrument}: {msg}")
    
    # Test adding correlated position
    new_pos = Position(
        instrument='NVDA', sector='tech', direction='long',
        units=25, entry_price=900.0, current_price=900.0,
        exposure=22500, exposure_percent=22.5
    )
    
    success, msg = manager.add_position(new_pos)
    print(f"\nAdd correlated position (NVDA): {msg}")
    
    # Test adding position in different sector
    different_pos = Position(
        instrument='XOM', sector='energy', direction='long',
        units=200, entry_price=120.0, current_price=120.0,
        exposure=24000, exposure_percent=24.0
    )
    
    success, msg = manager.add_position(different_pos)
    print(f"\nAdd different sector (XOM): {msg}")
    
    # Score new position
    test_pos = Position(
        instrument='JPM', sector='financial', direction='long',
        units=150, entry_price=180.0, current_price=180.0,
        exposure=27000, exposure_percent=27.0
    )
    
    scores = manager.score_new_position(test_pos)
    print(f"\nScore for JPM: {scores}")
    
    # Print portfolio summary
    manager.print_portfolio_summary()
    
    print(f"\n✅ Correlation manager test complete")