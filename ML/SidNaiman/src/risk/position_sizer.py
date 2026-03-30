#!/usr/bin/env python3
"""
Position Sizer Module for SID Method - AUGMENTED VERSION
=============================================================================
Calculates optimal position sizes incorporating ALL THREE WAVES:

WAVE 1 (Core Quick Win):
- Risk per trade: 0.5% to 2% of account
- Fixed percentage risk model
- Maximum 3-5 active trades
- Round down position sizes

WAVE 2 (Live Sessions & Q&A):
- Consecutive loss adjustment (reduce risk after losses)
- Reachability multiplier (reduce risk if target is hard to reach)
- Account equity curve-based sizing
- Kelly Criterion for optimal sizing
- Correlation-based position limits

WAVE 3 (Academy Support Sessions):
- Session-based risk adjustment
- Volatility-based position scaling
- Maximum drawdown protection
- Risk of ruin calculations
- Partial position scaling for multiple entries

Author: Sid Naiman / Trading Cafe
Version: 3.0 (Fully Augmented)
=============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class RiskModel(Enum):
    """Types of risk models (Wave 1 & 2)"""
    FIXED_PERCENT = "fixed_percent"           # Wave 1: fixed % of account
    KELLY_CRITERION = "kelly_criterion"       # Wave 2: optimal sizing
    EQUITY_CURVE = "equity_curve"             # Wave 2: based on equity curve
    VOLATILITY_ADJUSTED = "volatility_adjusted"  # Wave 3: adjust for volatility


class RiskAdjustmentReason(Enum):
    """Reasons for adjusting risk (Wave 2 & 3)"""
    CONSECUTIVE_LOSSES = "consecutive_losses"
    REACHABILITY = "reachability"
    VOLATILITY = "volatility"
    SESSION = "session"
    CORRELATION = "correlation"
    MAX_DRAWDOWN = "max_drawdown"
    NONE = "none"


@dataclass
class PositionSizeResult:
    """Result of position size calculation"""
    units: int
    risk_amount: float
    risk_percent: float
    base_risk_percent: float
    adjustment_reason: RiskAdjustmentReason
    adjustment_multiplier: float
    position_value: float
    entry_price: float
    stop_loss: float
    is_valid: bool
    validation_notes: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'units': self.units,
            'risk_amount': self.risk_amount,
            'risk_percent': self.risk_percent,
            'base_risk_percent': self.base_risk_percent,
            'adjustment_reason': self.adjustment_reason.value,
            'adjustment_multiplier': self.adjustment_multiplier,
            'position_value': self.position_value,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'is_valid': self.is_valid,
            'validation_notes': self.validation_notes
        }


@dataclass
class PositionSizerConfig:
    """Configuration for position sizer (Wave 1, 2, 3)"""
    # Wave 1: Core risk parameters
    default_risk_percent: float = 1.0
    min_risk_percent: float = 0.5
    max_risk_percent: float = 2.0
    max_active_trades: int = 5
    min_account_balance: float = 500.0
    
    # Wave 2: Consecutive loss adjustment
    use_consecutive_loss_adjustment: bool = True
    consecutive_loss_multipliers: Dict[int, float] = field(default_factory=lambda: {
        1: 0.75,
        2: 0.50,
        3: 0.35,
        4: 0.25,
        5: 0.15
    })
    max_consecutive_losses_before_pause: int = 5
    
    # Wave 2: Reachability adjustment
    use_reachability_adjustment: bool = True
    reachability_multiplier: float = 0.5  # Reduce risk by 50% if not reachable
    
    # Wave 2: Kelly Criterion
    use_kelly: bool = False
    kelly_fraction: float = 0.25  # Use 25% of Kelly for conservative sizing
    
    # Wave 3: Volatility adjustment
    use_volatility_adjustment: bool = False
    volatility_period: int = 20
    volatility_target: float = 0.01  # Target daily volatility
    
    # Wave 3: Session adjustment
    use_session_adjustment: bool = False
    session_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'overlap': 1.0,
        'us': 0.9,
        'london': 0.8,
        'asian': 0.6
    })
    
    # Wave 3: Correlation adjustment
    use_correlation_adjustment: bool = False
    max_correlation: float = 0.7  # Reduce risk if correlation above this
    
    # Wave 3: Maximum drawdown protection
    max_daily_loss_percent: float = 5.0
    max_weekly_loss_percent: float = 10.0
    max_monthly_loss_percent: float = 20.0
    pause_on_daily_loss: bool = True
    
    # Wave 3: Risk of ruin protection
    risk_of_ruin_threshold: float = 0.05  # 5% risk of ruin triggers reduction
    min_sharpe_for_full_risk: float = 0.5
    
    # Wave 3: Rounding
    round_units_down: bool = True
    minimum_units: int = 1


class PositionSizer:
    """
    Calculates optimal position sizes for SID Method trades
    Incorporates ALL THREE WAVES of strategies
    """
    
    def __init__(self, config: PositionSizerConfig = None, verbose: bool = True):
        """
        Initialize position sizer
        
        Args:
            config: PositionSizerConfig instance
            verbose: Enable verbose output
        """
        self.config = config or PositionSizerConfig()
        self.verbose = verbose
        
        # Track trading statistics
        self.trade_history: List[Dict] = []
        self.consecutive_losses = 0
        self.daily_loss = 0.0
        self.weekly_loss = 0.0
        self.monthly_loss = 0.0
        self.peak_balance = 0.0
        self.drawdown = 0.0
        
        # Kelly criterion tracking
        self.win_rate = 0.5
        self.avg_win = 0.0
        self.avg_loss = 0.0
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📐 POSITION SIZER v3.0 (Fully Augmented)")
            print(f"{'='*60}")
            print(f"💰 Default risk: {self.config.default_risk_percent}%")
            print(f"📊 Risk range: {self.config.min_risk_percent}% - {self.config.max_risk_percent}%")
            print(f"📈 Max active trades: {self.config.max_active_trades}")
            print(f"🔄 Consecutive loss adj: {self.config.use_consecutive_loss_adjustment}")
            print(f"🎯 Reachability adj: {self.config.use_reachability_adjustment}")
            print(f"📊 Kelly Criterion: {self.config.use_kelly}")
            print(f"🌊 Volatility adj: {self.config.use_volatility_adjustment}")
            print(f"🌍 Session adj: {self.config.use_session_adjustment}")
            print(f"🔗 Correlation adj: {self.config.use_correlation_adjustment}")
            print(f"🛡️ Max daily loss: {self.config.max_daily_loss_percent}%")
            print(f"{'='*60}\n")
    
    # ========================================================================
    # SECTION 1: CORE POSITION SIZING (Wave 1)
    # ========================================================================
    
    def calculate_fixed_percent_size(self, account_balance: float, entry_price: float,
                                       stop_loss: float, risk_percent: float = None) -> Dict:
        """
        Calculate position size using fixed percentage risk model (Wave 1)
        
        Args:
            account_balance: Current account balance
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_percent: Risk percentage (if None, uses default)
        
        Returns:
            Dictionary with position size details
        """
        if risk_percent is None:
            risk_percent = self.config.default_risk_percent
        
        # Clamp risk to min/max
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
            return {
                'error': 'Invalid stop loss',
                'units': 0,
                'risk_amount': risk_amount,
                'risk_percent': risk_percent
            }
        
        units = risk_amount / risk_per_unit
        
        if self.config.round_units_down:
            units = np.floor(units)
        else:
            units = round(units)
        
        units = max(units, self.config.minimum_units)
        
        return {
            'units': int(units),
            'direction': direction,
            'risk_amount': risk_amount,
            'risk_percent': risk_percent,
            'risk_per_unit': risk_per_unit,
            'position_value': units * entry_price,
            'entry_price': entry_price,
            'stop_loss': stop_loss
        }
    
    # ========================================================================
    # SECTION 2: CONSECUTIVE LOSS ADJUSTMENT (Wave 2)
    # ========================================================================
    
    def get_consecutive_loss_multiplier(self, consecutive_losses: int) -> float:
        """
        Get risk multiplier based on consecutive losses (Wave 2)
        
        Args:
            consecutive_losses: Number of consecutive losing trades
        
        Returns:
            Risk multiplier (1.0 = no reduction)
        """
        if not self.config.use_consecutive_loss_adjustment:
            return 1.0
        
        # Find the appropriate multiplier
        for losses, multiplier in sorted(self.config.consecutive_loss_multipliers.items()):
            if consecutive_losses >= losses:
                return multiplier
        
        # If more losses than configured, use smallest multiplier
        if consecutive_losses > 0:
            return min(self.config.consecutive_loss_multipliers.values(), default=0.1)
        
        return 1.0
    
    def update_consecutive_losses(self, was_win: bool):
        """
        Update consecutive loss counter (Wave 2)
        
        Args:
            was_win: True if the trade was a win
        """
        if was_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            
            # Pause trading if too many consecutive losses
            if self.consecutive_losses >= self.config.max_consecutive_losses_before_pause:
                if self.verbose:
                    print(f"⚠️ {self.consecutive_losses} consecutive losses - consider pausing")
    
    # ========================================================================
    # SECTION 3: REACHABILITY ADJUSTMENT (Wave 2)
    # ========================================================================
    
    def get_reachability_multiplier(self, reachable: bool, 
                                      reachability_multiplier: float = 0.5) -> float:
        """
        Get risk multiplier based on reachability (Wave 2)
        
        Args:
            reachable: Is the target realistically reachable?
            reachability_multiplier: Multiplier if not reachable
        
        Returns:
            Risk multiplier
        """
        if not self.config.use_reachability_adjustment:
            return 1.0
        
        if reachable:
            return 1.0
        else:
            return reachability_multiplier
    
    # ========================================================================
    # SECTION 4: KELLY CRITERION (Wave 2)
    # ========================================================================
    
    def update_kelly_stats(self, trade_result: Dict):
        """
        Update statistics for Kelly Criterion calculation (Wave 2)
        
        Args:
            trade_result: Dictionary with 'profit', 'risk_amount', 'was_win'
        """
        self.trade_history.append(trade_result)
        
        # Keep only last 100 trades for Kelly calculation
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]
        
        # Calculate win rate
        wins = [t for t in self.trade_history if t.get('was_win', False)]
        self.win_rate = len(wins) / len(self.trade_history) if self.trade_history else 0.5
        
        # Calculate average win/loss
        win_amounts = [t.get('profit', 0) for t in wins]
        loss_amounts = [t.get('loss', 0) for t in self.trade_history if not t.get('was_win', False)]
        
        self.avg_win = np.mean(win_amounts) if win_amounts else 0
        self.avg_loss = np.mean(loss_amounts) if loss_amounts else 0
    
    def calculate_kelly_percentage(self) -> float:
        """
        Calculate optimal risk percentage using Kelly Criterion (Wave 2)
        
        Kelly % = (Win Rate * Average Win - Loss Rate * Average Loss) / Average Win
        
        Returns:
            Optimal risk percentage (capped at 25% of account)
        """
        if not self.config.use_kelly:
            return self.config.default_risk_percent
        
        if self.avg_win <= 0:
            return self.config.default_risk_percent
        
        # Kelly formula
        kelly = (self.win_rate * self.avg_win - (1 - self.win_rate) * self.avg_loss) / self.avg_win
        
        # Apply fraction (conservative)
        kelly = kelly * self.config.kelly_fraction
        
        # Clamp to reasonable range
        kelly = max(0.01, min(kelly, 0.25))  # 1% to 25%
        
        # Convert to percentage
        return kelly * 100
    
    # ========================================================================
    # SECTION 5: VOLATILITY ADJUSTMENT (Wave 3)
    # ========================================================================
    
    def calculate_volatility_multiplier(self, df: pd.DataFrame, current_idx: int) -> float:
        """
        Calculate risk multiplier based on current volatility (Wave 3)
        
        Args:
            df: Price DataFrame
            current_idx: Current index
        
        Returns:
            Risk multiplier
        """
        if not self.config.use_volatility_adjustment:
            return 1.0
        
        if current_idx < self.config.volatility_period:
            return 1.0
        
        # Calculate recent volatility
        returns = df['close'].pct_change().iloc[current_idx - self.config.volatility_period:current_idx]
        current_volatility = returns.std()
        
        # Target volatility
        target_volatility = self.config.volatility_target
        
        if current_volatility > 0:
            multiplier = target_volatility / current_volatility
            multiplier = max(0.5, min(multiplier, 2.0))  # Clamp to 0.5-2.0
            return multiplier
        
        return 1.0
    
    # ========================================================================
    # SECTION 6: SESSION ADJUSTMENT (Wave 3)
    # ========================================================================
    
    def get_session_multiplier(self, session: str) -> float:
        """
        Get risk multiplier based on trading session (Wave 3)
        
        Args:
            session: Trading session ('asian', 'london', 'us', 'overlap')
        
        Returns:
            Risk multiplier
        """
        if not self.config.use_session_adjustment:
            return 1.0
        
        return self.config.session_multipliers.get(session, 0.8)
    
    # ========================================================================
    # SECTION 7: CORRELATION ADJUSTMENT (Wave 3)
    # ========================================================================
    
    def calculate_correlation_multiplier(self, instrument: str, 
                                           current_positions: List[Dict]) -> float:
        """
        Calculate risk multiplier based on correlation with existing positions (Wave 3)
        
        Args:
            instrument: New instrument
            current_positions: List of current open positions
        
        Returns:
            Risk multiplier
        """
        if not self.config.use_correlation_adjustment:
            return 1.0
        
        if not current_positions:
            return 1.0
        
        # Simplified correlation check
        # In production, use actual correlation matrix
        correlated_pairs = [
            ('EUR_USD', 'GBP_USD'), ('EUR_USD', 'USD_CHF'),
            ('GBP_USD', 'USD_CHF'), ('USD_JPY', 'EUR_JPY'),
            ('AUD_USD', 'NZD_USD'), ('EUR_GBP', 'GBP_USD')
        ]
        
        correlation_count = 0
        for pos in current_positions:
            pos_instr = pos.get('instrument', '')
            if (instrument, pos_instr) in correlated_pairs or (pos_instr, instrument) in correlated_pairs:
                correlation_count += 1
        
        if correlation_count > 0:
            # Reduce risk for each correlated position
            multiplier = max(0.5, 1.0 - (correlation_count * 0.2))
            return multiplier
        
        return 1.0
    
    # ========================================================================
    # SECTION 8: DRAWDOWN PROTECTION (Wave 3)
    # ========================================================================
    
    def check_drawdown_protection(self, account_balance: float, 
                                    starting_balance: float) -> Tuple[bool, float]:
        """
        Check if drawdown exceeds thresholds (Wave 3)
        
        Args:
            account_balance: Current account balance
            starting_balance: Starting balance (for drawdown calculation)
        
        Returns:
            (should_reduce_risk, risk_multiplier)
        """
        if self.peak_balance < account_balance:
            self.peak_balance = account_balance
        
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - account_balance) / self.peak_balance
        else:
            drawdown = 0
        
        self.drawdown = drawdown
        
        if drawdown > 0.2:  # 20% drawdown
            return True, 0.5
        elif drawdown > 0.15:
            return True, 0.7
        elif drawdown > 0.1:
            return True, 0.8
        elif drawdown > 0.05:
            return True, 0.9
        
        return False, 1.0
    
    def check_daily_loss_protection(self, daily_loss: float, 
                                      account_balance: float) -> Tuple[bool, str]:
        """
        Check if daily loss limit is exceeded (Wave 3)
        
        Args:
            daily_loss: Loss for the current day
            account_balance: Current account balance
        
        Returns:
            (should_pause, message)
        """
        if daily_loss <= 0:
            return False, ""
        
        daily_loss_percent = daily_loss / account_balance * 100
        
        if daily_loss_percent >= self.config.max_daily_loss_percent:
            return True, f"Daily loss limit reached: {daily_loss_percent:.1f}% (max {self.config.max_daily_loss_percent}%)"
        
        return False, ""
    
    # ========================================================================
    # SECTION 9: RISK OF RUIN CALCULATION (Wave 3)
    # ========================================================================
    
    def calculate_risk_of_ruin(self, win_rate: float, risk_percent: float, 
                                 reward_ratio: float = 1.0) -> float:
        """
        Calculate risk of ruin (probability of losing entire account) (Wave 3)
        
        Args:
            win_rate: Probability of winning trade
            risk_percent: Risk per trade as percentage
            reward_ratio: Risk-reward ratio (default 1:1)
        
        Returns:
            Risk of ruin probability (0-1)
        """
        if win_rate <= 0 or win_rate >= 1:
            return 1.0
        
        # Simplified risk of ruin formula
        # Assumes fixed risk per trade
        max_trades = 100 / risk_percent  # Number of trades to lose entire account
        
        # Probability of losing N trades in a row
        loss_prob = 1 - win_rate
        ruin_prob = loss_prob ** max_trades
        
        return min(ruin_prob, 1.0)
    
    def get_risk_of_ruin_multiplier(self, win_rate: float, risk_percent: float) -> float:
        """
        Get risk multiplier based on risk of ruin (Wave 3)
        
        Args:
            win_rate: Current win rate
            risk_percent: Proposed risk percentage
        
        Returns:
            Risk multiplier (1.0 = no reduction)
        """
        ruin_prob = self.calculate_risk_of_ruin(win_rate, risk_percent)
        
        if ruin_prob > self.config.risk_of_ruin_threshold:
            # Reduce risk proportionally
            multiplier = self.config.risk_of_ruin_threshold / ruin_prob
            return max(0.25, min(multiplier, 1.0))
        
        return 1.0
    
    # ========================================================================
    # SECTION 10: COMPLETE POSITION SIZE CALCULATION
    # ========================================================================
    
    def calculate_position_size(self, account_balance: float, entry_price: float,
                                  stop_loss: float, risk_percent: float = None,
                                  consecutive_losses: int = None,
                                  reachable: bool = True,
                                  reachability_multiplier: float = 0.5,
                                  session: str = 'us',
                                  instrument: str = None,
                                  current_positions: List[Dict] = None,
                                  df: pd.DataFrame = None,
                                  current_idx: int = None,
                                  starting_balance: float = None) -> PositionSizeResult:
        """
        Complete position size calculation incorporating ALL THREE WAVES
        
        Args:
            account_balance: Current account balance
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_percent: Base risk percentage (if None, uses default)
            consecutive_losses: Number of consecutive losses (if None, uses internal)
            reachable: Is target reachable?
            reachability_multiplier: Multiplier if not reachable
            session: Trading session
            instrument: Instrument symbol
            current_positions: Current open positions
            df: Price DataFrame (for volatility adjustment)
            current_idx: Current index (for volatility adjustment)
            starting_balance: Starting balance (for drawdown calculation)
        
        Returns:
            PositionSizeResult
        """
        # Start with base risk
        base_risk = risk_percent if risk_percent is not None else self.config.default_risk_percent
        
        # Track adjustments
        adjustment_multiplier = 1.0
        adjustment_reason = RiskAdjustmentReason.NONE
        
        # 1. Consecutive loss adjustment (Wave 2)
        if consecutive_losses is None:
            consecutive_losses = self.consecutive_losses
        
        loss_multiplier = self.get_consecutive_loss_multiplier(consecutive_losses)
        if loss_multiplier < 1.0:
            adjustment_multiplier *= loss_multiplier
            adjustment_reason = RiskAdjustmentReason.CONSECUTIVE_LOSSES
        
        # 2. Reachability adjustment (Wave 2)
        reach_multiplier = self.get_reachability_multiplier(reachable, reachability_multiplier)
        if reach_multiplier < 1.0:
            adjustment_multiplier *= reach_multiplier
            adjustment_reason = RiskAdjustmentReason.REACHABILITY
        
        # 3. Kelly Criterion (Wave 2)
        if self.config.use_kelly and len(self.trade_history) > 20:
            kelly_risk = self.calculate_kelly_percentage()
            if kelly_risk < base_risk:
                kelly_multiplier = kelly_risk / base_risk
                adjustment_multiplier *= kelly_multiplier
                if adjustment_reason == RiskAdjustmentReason.NONE:
                    adjustment_reason = RiskAdjustmentReason.NONE  # Keep as base
        
        # 4. Volatility adjustment (Wave 3)
        if df is not None and current_idx is not None:
            vol_multiplier = self.calculate_volatility_multiplier(df, current_idx)
            if vol_multiplier < 1.0:
                adjustment_multiplier *= vol_multiplier
                if adjustment_reason == RiskAdjustmentReason.NONE:
                    adjustment_reason = RiskAdjustmentReason.VOLATILITY
        
        # 5. Session adjustment (Wave 3)
        session_multiplier = self.get_session_multiplier(session)
        if session_multiplier < 1.0:
            adjustment_multiplier *= session_multiplier
            if adjustment_reason == RiskAdjustmentReason.NONE:
                adjustment_reason = RiskAdjustmentReason.SESSION
        
        # 6. Correlation adjustment (Wave 3)
        if current_positions:
            corr_multiplier = self.calculate_correlation_multiplier(instrument, current_positions)
            if corr_multiplier < 1.0:
                adjustment_multiplier *= corr_multiplier
                if adjustment_reason == RiskAdjustmentReason.NONE:
                    adjustment_reason = RiskAdjustmentReason.CORRELATION
        
        # 7. Drawdown protection (Wave 3)
        if starting_balance is not None:
            should_reduce, dd_multiplier = self.check_drawdown_protection(account_balance, starting_balance)
            if should_reduce:
                adjustment_multiplier *= dd_multiplier
                if adjustment_reason == RiskAdjustmentReason.NONE:
                    adjustment_reason = RiskAdjustmentReason.MAX_DRAWDOWN
        
        # 8. Risk of ruin adjustment (Wave 3)
        if len(self.trade_history) > 20:
            ruin_multiplier = self.get_risk_of_ruin_multiplier(self.win_rate, base_risk * adjustment_multiplier)
            if ruin_multiplier < 1.0:
                adjustment_multiplier *= ruin_multiplier
        
        # Apply adjustment
        adjusted_risk = base_risk * adjustment_multiplier
        
        # Clamp to min/max
        adjusted_risk = max(self.config.min_risk_percent, 
                           min(adjusted_risk, self.config.max_risk_percent))
        
        # Check if we should pause trading
        should_pause, pause_msg = self.check_daily_loss_protection(self.daily_loss, account_balance)
        if should_pause:
            return PositionSizeResult(
                units=0,
                risk_amount=0,
                risk_percent=0,
                base_risk_percent=base_risk,
                adjustment_reason=RiskAdjustmentReason.MAX_DRAWDOWN,
                adjustment_multiplier=0,
                position_value=0,
                entry_price=entry_price,
                stop_loss=stop_loss,
                is_valid=False,
                validation_notes=f"Trading paused: {pause_msg}"
            )
        
        # Check minimum account balance
        if account_balance < self.config.min_account_balance:
            return PositionSizeResult(
                units=0,
                risk_amount=0,
                risk_percent=0,
                base_risk_percent=base_risk,
                adjustment_reason=RiskAdjustmentReason.NONE,
                adjustment_multiplier=adjustment_multiplier,
                position_value=0,
                entry_price=entry_price,
                stop_loss=stop_loss,
                is_valid=False,
                validation_notes=f"Account balance below minimum: ${account_balance:.2f} < ${self.config.min_account_balance:.2f}"
            )
        
        # Calculate position size
        position = self.calculate_fixed_percent_size(
            account_balance, entry_price, stop_loss, adjusted_risk
        )
        
        if 'error' in position:
            return PositionSizeResult(
                units=0,
                risk_amount=position.get('risk_amount', 0),
                risk_percent=adjusted_risk,
                base_risk_percent=base_risk,
                adjustment_reason=adjustment_reason,
                adjustment_multiplier=adjustment_multiplier,
                position_value=0,
                entry_price=entry_price,
                stop_loss=stop_loss,
                is_valid=False,
                validation_notes=position['error']
            )
        
        return PositionSizeResult(
            units=position['units'],
            risk_amount=position['risk_amount'],
            risk_percent=adjusted_risk,
            base_risk_percent=base_risk,
            adjustment_reason=adjustment_reason,
            adjustment_multiplier=adjustment_multiplier,
            position_value=position['position_value'],
            entry_price=entry_price,
            stop_loss=stop_loss,
            is_valid=True,
            validation_notes=f"Risk: {adjusted_risk:.2f}% (base {base_risk:.2f}%, adj: {adjustment_multiplier:.2f}x)"
        )
    
    # ========================================================================
    # SECTION 11: MULTIPLE ENTRY POSITION SIZING (Wave 3)
    # ========================================================================
    
    def calculate_scaled_position(self, account_balance: float, entry_prices: List[float],
                                    stop_loss: float, total_risk_percent: float = 1.0,
                                    weights: List[float] = None) -> List[PositionSizeResult]:
        """
        Calculate position sizes for multiple scaled entries (Wave 3)
        
        Args:
            account_balance: Current account balance
            entry_prices: List of entry prices for scaling
            stop_loss: Stop loss price (same for all entries)
            total_risk_percent: Total risk percentage for the position
            weights: Weights for each entry (default equal)
        
        Returns:
            List of PositionSizeResult for each entry
        """
        if weights is None:
            weights = [1.0 / len(entry_prices)] * len(entry_prices)
        
        results = []
        
        for i, (entry_price, weight) in enumerate(zip(entry_prices, weights)):
            # Risk allocated to this entry
            entry_risk_percent = total_risk_percent * weight
            
            result = self.calculate_position_size(
                account_balance=account_balance,
                entry_price=entry_price,
                stop_loss=stop_loss,
                risk_percent=entry_risk_percent
            )
            
            results.append(result)
        
        return results
    
    # ========================================================================
    # SECTION 12: UTILITY METHODS
    # ========================================================================
    
    def reset_daily(self):
        """Reset daily loss counter (Wave 3)"""
        if self.verbose:
            print(f"🔄 Resetting daily loss: ${self.daily_loss:.2f}")
        self.daily_loss = 0.0
    
    def reset_weekly(self):
        """Reset weekly loss counter (Wave 3)"""
        if self.verbose:
            print(f"🔄 Resetting weekly loss: ${self.weekly_loss:.2f}")
        self.weekly_loss = 0.0
    
    def reset_monthly(self):
        """Reset monthly loss counter (Wave 3)"""
        if self.verbose:
            print(f"🔄 Resetting monthly loss: ${self.monthly_loss:.2f}")
        self.monthly_loss = 0.0
    
    def update_loss(self, loss_amount: float):
        """Update loss counters (Wave 3)"""
        self.daily_loss += loss_amount
        self.weekly_loss += loss_amount
        self.monthly_loss += loss_amount
    
    def get_stats(self) -> Dict:
        """Get current position sizer statistics"""
        return {
            'consecutive_losses': self.consecutive_losses,
            'daily_loss': self.daily_loss,
            'weekly_loss': self.weekly_loss,
            'monthly_loss': self.monthly_loss,
            'peak_balance': self.peak_balance,
            'drawdown': self.drawdown,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'trade_count': len(self.trade_history)
        }


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TESTING POSITION SIZER v3.0")
    print("="*70)
    
    # Initialize position sizer
    config = PositionSizerConfig(
        default_risk_percent=1.0,
        use_consecutive_loss_adjustment=True,
        use_reachability_adjustment=True,
        use_session_adjustment=True
    )
    sizer = PositionSizer(config, verbose=True)
    
    # Test 1: Basic position sizing
    print(f"\n📊 Test 1: Basic Position Sizing")
    result = sizer.calculate_position_size(
        account_balance=10000,
        entry_price=100.0,
        stop_loss=99.0,
        risk_percent=1.0
    )
    print(f"  Units: {result.units}")
    print(f"  Risk Amount: ${result.risk_amount:.2f}")
    print(f"  Position Value: ${result.position_value:.2f}")
    print(f"  Valid: {result.is_valid}")
    print(f"  Notes: {result.validation_notes}")
    
    # Test 2: Consecutive losses
    print(f"\n📊 Test 2: After 3 Consecutive Losses")
    sizer.update_consecutive_losses(was_win=False)
    sizer.update_consecutive_losses(was_win=False)
    sizer.update_consecutive_losses(was_win=False)
    
    result = sizer.calculate_position_size(
        account_balance=10000,
        entry_price=100.0,
        stop_loss=99.0,
        risk_percent=1.0,
        consecutive_losses=sizer.consecutive_losses
    )
    print(f"  Consecutive Losses: {sizer.consecutive_losses}")
    print(f"  Units: {result.units}")
    print(f"  Risk Percent: {result.risk_percent:.2f}%")
    print(f"  Adjustment Multiplier: {result.adjustment_multiplier:.2f}x")
    print(f"  Reason: {result.adjustment_reason.value}")
    
    # Test 3: Not reachable
    print(f"\n📊 Test 3: Target Not Reachable")
    result = sizer.calculate_position_size(
        account_balance=10000,
        entry_price=100.0,
        stop_loss=99.0,
        risk_percent=1.0,
        reachable=False,
        reachability_multiplier=0.5
    )
    print(f"  Units: {result.units}")
    print(f"  Risk Percent: {result.risk_percent:.2f}%")
    print(f"  Adjustment Multiplier: {result.adjustment_multiplier:.2f}x")
    print(f"  Reason: {result.adjustment_reason.value}")
    
    # Test 4: Session adjustment
    print(f"\n📊 Test 4: Asian Session")
    result = sizer.calculate_position_size(
        account_balance=10000,
        entry_price=100.0,
        stop_loss=99.0,
        risk_percent=1.0,
        session='asian'
    )
    print(f"  Session: asian")
    print(f"  Units: {result.units}")
    print(f"  Risk Percent: {result.risk_percent:.2f}%")
    print(f"  Adjustment Multiplier: {result.adjustment_multiplier:.2f}x")
    
    # Test 5: Combined adjustments
    print(f"\n📊 Test 5: Combined (3 losses + Asian session + not reachable)")
    result = sizer.calculate_position_size(
        account_balance=10000,
        entry_price=100.0,
        stop_loss=99.0,
        risk_percent=1.0,
        consecutive_losses=3,
        reachable=False,
        session='asian'
    )
    print(f"  Units: {result.units}")
    print(f"  Risk Percent: {result.risk_percent:.2f}%")
    print(f"  Adjustment Multiplier: {result.adjustment_multiplier:.2f}x")
    print(f"  Reason: {result.adjustment_reason.value}")
    
    # Test 6: Kelly Criterion (with trade history)
    print(f"\n📊 Test 6: Kelly Criterion")
    # Simulate some trades
    for _ in range(10):
        sizer.update_kelly_stats({'was_win': True, 'profit': 100, 'risk_amount': 100})
    for _ in range(3):
        sizer.update_kelly_stats({'was_win': False, 'loss': 100, 'risk_amount': 100})
    
    result = sizer.calculate_position_size(
        account_balance=10000,
        entry_price=100.0,
        stop_loss=99.0
    )
    print(f"  Win Rate: {sizer.win_rate:.2f}")
    print(f"  Units: {result.units}")
    print(f"  Risk Percent: {result.risk_percent:.2f}%")
    
    # Test 7: Scaled entries
    print(f"\n📊 Test 7: Scaled Entries (3 entries)")
    results = sizer.calculate_scaled_position(
        account_balance=10000,
        entry_prices=[100.0, 101.0, 102.0],
        stop_loss=99.0,
        total_risk_percent=1.0,
        weights=[0.5, 0.3, 0.2]
    )
    for i, r in enumerate(results):
        print(f"  Entry {i+1}: {r.units} units at ${r.entry_price:.2f}")
    
    print(f"\n✅ Position sizer test complete")