#!/usr/bin/env python3
"""
Time Filter Module for SID Method - AUGMENTED VERSION
=============================================================================
Filters trades based on time-based criteria incorporating ALL THREE WAVES:

WAVE 1 (Core Quick Win):
- Market hours filtering (regular trading hours)
- Session-based trading

WAVE 2 (Live Sessions & Q&A):
- London-Overlap optimization
- Asian session avoidance
- Volatility time windows
- News time blocks

WAVE 3 (Academy Support Sessions):
- Optimal entry time windows
- Session transition periods
- Intraday seasonality
- Holiday calendars
- Time-of-day performance analysis

Author: Sid Naiman / Trading Cafe
Version: 3.0 (Fully Augmented)
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta, time
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class TradingSession(Enum):
    """Trading sessions (Wave 1 & 3)"""
    ASIAN = "asian"       # Tokyo: 00:00-09:00 GMT
    LONDON = "london"     # London: 07:00-16:00 GMT
    US = "us"             # New York: 12:00-21:00 GMT
    OVERLAP = "overlap"   # London-US overlap: 12:00-16:00 GMT
    ASIAN_LONDON = "asian_london"  # Transition: 07:00-09:00
    LONDON_US = "london_us"        # Transition: 16:00-17:00


class MarketCondition(Enum):
    """Market conditions based on time (Wave 2 & 3)"""
    HIGH_VOLATILITY = "high_volatility"     # London open, US open, overlap
    MEDIUM_VOLATILITY = "medium_volatility" # London close, US afternoon
    LOW_VOLATILITY = "low_volatility"       # Asian session, late US
    NEWS_TIME = "news_time"                 # Major news releases
    CLOSE_TIME = "close_time"               # End of session


@dataclass
class TimeFilterConfig:
    """Configuration for time filter (Wave 1, 2, 3)"""
    # Wave 1: Basic session filters
    allowed_sessions: List[str] = field(default_factory=lambda: ['overlap', 'us', 'london'])
    blocked_sessions: List[str] = field(default_factory=lambda: ['asian'])
    
    # Wave 2: Volatility windows
    high_volatility_start: time = field(default_factory=lambda: time(12, 0))  # 12:00 GMT (US open)
    high_volatility_end: time = field(default_factory=lambda: time(16, 0))     # 16:00 GMT (US close)
    medium_volatility_start: time = field(default_factory=lambda: time(7, 0))   # 07:00 GMT (London open)
    medium_volatility_end: time = field(default_factory=lambda: time(12, 0))    # 12:00 GMT
    
    # Wave 2: News blocks
    news_block_start: time = field(default_factory=lambda: time(8, 30))  # 08:30 GMT (US data)
    news_block_end: time = field(default_factory=lambda: time(9, 30))     # 09:30 GMT
    avoid_news_blocks: bool = True
    
    # Wave 3: Optimal entry windows
    optimal_entry_start: time = field(default_factory=lambda: time(12, 30))  # After US open
    optimal_entry_end: time = field(default_factory=lambda: time(15, 0))     # Before US close
    
    # Wave 3: Session transition periods
    transition_buffer_minutes: int = 30
    
    # Wave 3: Holiday calendar
    holidays: List[str] = field(default_factory=lambda: [
        '01-01',  # New Year's Day
        '07-04',  # Independence Day
        '12-25',  # Christmas Day
        '12-26'   # Boxing Day
    ])
    
    # Wave 3: Weekend filter
    allow_weekends: bool = False
    
    # Wave 3: Time-of-day performance
    use_performance_data: bool = False
    performance_data: Dict[str, Dict] = field(default_factory=dict)


class TimeFilter:
    """
    Filters trades based on time criteria for SID Method
    Incorporates ALL THREE WAVES of strategies
    """
    
    def __init__(self, config: TimeFilterConfig = None, verbose: bool = True):
        """
        Initialize time filter
        
        Args:
            config: TimeFilterConfig instance
            verbose: Enable verbose output
        """
        self.config = config or TimeFilterConfig()
        self.verbose = verbose
        
        # Session time boundaries (GMT)
        self.session_boundaries = {
            TradingSession.ASIAN: (time(0, 0), time(7, 0)),
            TradingSession.LONDON: (time(7, 0), time(16, 0)),
            TradingSession.US: (time(12, 0), time(21, 0)),
            TradingSession.OVERLAP: (time(12, 0), time(16, 0)),
            TradingSession.ASIAN_LONDON: (time(7, 0), time(9, 0)),
            TradingSession.LONDON_US: (time(16, 0), time(17, 0))
        }
        
        # Load performance data if available
        self._load_performance_data()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"⏰ TIME FILTER v3.0 (Fully Augmented)")
            print(f"{'='*60}")
            print(f"📊 Allowed sessions: {self.config.allowed_sessions}")
            print(f"🚫 Blocked sessions: {self.config.blocked_sessions}")
            print(f"⚡ High volatility: {self.config.high_volatility_start.strftime('%H:%M')}-{self.config.high_volatility_end.strftime('%H:%M')} GMT")
            print(f"🎯 Optimal entry: {self.config.optimal_entry_start.strftime('%H:%M')}-{self.config.optimal_entry_end.strftime('%H:%M')} GMT")
            print(f"📰 News blocks: {self.config.avoid_news_blocks}")
            print(f"📅 Weekends: {'Allowed' if self.config.allow_weekends else 'Blocked'}")
            print(f"{'='*60}\n")
    
    # ========================================================================
    # SECTION 1: SESSION DETECTION (Wave 1 & 3)
    # ========================================================================
    
    def get_trading_session(self, dt: datetime) -> TradingSession:
        """
        Determine trading session (Wave 1)
        
        Args:
            dt: Datetime in GMT
        
        Returns:
            TradingSession
        """
        t = dt.time()
        
        # Check overlap first
        if self.session_boundaries[TradingSession.OVERLAP][0] <= t < self.session_boundaries[TradingSession.OVERLAP][1]:
            return TradingSession.OVERLAP
        
        # Check Asian-London transition
        if self.session_boundaries[TradingSession.ASIAN_LONDON][0] <= t < self.session_boundaries[TradingSession.ASIAN_LONDON][1]:
            return TradingSession.ASIAN_LONDON
        
        # Check London-US transition
        if self.session_boundaries[TradingSession.LONDON_US][0] <= t < self.session_boundaries[TradingSession.LONDON_US][1]:
            return TradingSession.LONDON_US
        
        # Check regular sessions
        for session in [TradingSession.ASIAN, TradingSession.LONDON, TradingSession.US]:
            start, end = self.session_boundaries[session]
            if start <= t < end:
                return session
        
        return TradingSession.ASIAN  # Default
    
    def get_market_condition(self, dt: datetime) -> MarketCondition:
        """
        Determine market condition based on time (Wave 2 & 3)
        
        Returns:
            MarketCondition
        """
        t = dt.time()
        
        # Check news block
        if (self.config.avoid_news_blocks and 
            self.config.news_block_start <= t < self.config.news_block_end):
            return MarketCondition.NEWS_TIME
        
        # Check high volatility (US session)
        if self.config.high_volatility_start <= t < self.config.high_volatility_end:
            return MarketCondition.HIGH_VOLATILITY
        
        # Check medium volatility (London session)
        if self.config.medium_volatility_start <= t < self.config.medium_volatility_end:
            return MarketCondition.MEDIUM_VOLATILITY
        
        # Check close time (end of sessions)
        if t >= time(20, 0) or t < time(1, 0):
            return MarketCondition.CLOSE_TIME
        
        return MarketCondition.LOW_VOLATILITY
    
    # ========================================================================
    # SECTION 2: SESSION VALIDATION (Wave 1)
    # ========================================================================
    
    def is_session_allowed(self, session: TradingSession) -> bool:
        """
        Check if session is allowed for trading (Wave 1)
        
        Returns:
            True if session is allowed
        """
        session_name = session.value
        
        if session_name in self.config.blocked_sessions:
            return False
        
        if session_name in self.config.allowed_sessions:
            return True
        
        return False
    
    def validate_session(self, dt: datetime) -> Tuple[bool, str]:
        """
        Validate trade time based on session rules (Wave 1)
        
        Returns:
            (is_valid, message)
        """
        session = self.get_trading_session(dt)
        
        if not self.is_session_allowed(session):
            return False, f"Session {session.value} is not allowed (allowed: {self.config.allowed_sessions})"
        
        return True, f"Session {session.value} is allowed"
    
    # ========================================================================
    # SECTION 3: VOLATILITY-BASED FILTERING (Wave 2)
    # ========================================================================
    
    def validate_volatility_time(self, dt: datetime) -> Tuple[bool, str]:
        """
        Validate trade time based on volatility windows (Wave 2)
        
        Returns:
            (is_valid, message)
        """
        condition = self.get_market_condition(dt)
        
        if condition == MarketCondition.NEWS_TIME:
            return False, "News time - avoid trading"
        
        if condition == MarketCondition.CLOSE_TIME:
            return False, "Session close time - reduced liquidity"
        
        if condition == MarketCondition.LOW_VOLATILITY:
            return False, "Low volatility session - SID method less effective"
        
        return True, f"Condition: {condition.value}"
    
    # ========================================================================
    # SECTION 4: OPTIMAL ENTRY WINDOWS (Wave 3)
    # ========================================================================
    
    def is_optimal_entry_time(self, dt: datetime) -> Tuple[bool, str]:
        """
        Check if time is within optimal entry window (Wave 3)
        
        Returns:
            (is_optimal, message)
        """
        t = dt.time()
        
        if (self.config.optimal_entry_start <= t < self.config.optimal_entry_end):
            return True, f"Optimal entry window: {self.config.optimal_entry_start.strftime('%H:%M')}-{self.config.optimal_entry_end.strftime('%H:%M')} GMT"
        
        return False, f"Not in optimal window (optimal: {self.config.optimal_entry_start.strftime('%H:%M')}-{self.config.optimal_entry_end.strftime('%H:%M')} GMT)"
    
    # ========================================================================
    # SECTION 5: HOLIDAY AND WEEKEND FILTERS (Wave 3)
    # ========================================================================
    
    def is_holiday(self, dt: datetime) -> bool:
        """
        Check if date is a holiday (Wave 3)
        
        Returns:
            True if holiday
        """
        date_str = dt.strftime('%m-%d')
        return date_str in self.config.holidays
    
    def is_weekend(self, dt: datetime) -> bool:
        """
        Check if date is weekend (Wave 3)
        
        Returns:
            True if weekend
        """
        return dt.weekday() >= 5  # 5 = Saturday, 6 = Sunday
    
    def validate_date(self, dt: datetime) -> Tuple[bool, str]:
        """
        Validate trade date (Wave 3)
        
        Returns:
            (is_valid, message)
        """
        if self.is_holiday(dt):
            return False, f"Holiday: {dt.strftime('%Y-%m-%d')}"
        
        if not self.config.allow_weekends and self.is_weekend(dt):
            return False, "Weekend trading not allowed"
        
        return True, "Date valid"
    
    # ========================================================================
    # SECTION 6: TRANSITION PERIODS (Wave 3)
    # ========================================================================
    
    def is_transition_period(self, dt: datetime) -> Tuple[bool, str]:
        """
        Check if time is in a session transition period (Wave 3)
        
        Returns:
            (is_transition, message)
        """
        t = dt.time()
        
        # London open transition (Asian-London)
        if time(7, 0) <= t < time(7, self.config.transition_buffer_minutes):
            return True, "Asian-London transition period - increased volatility"
        
        # London-US overlap transition
        if time(12, 0) <= t < time(12, self.config.transition_buffer_minutes):
            return True, "London-US overlap start - high volatility"
        
        # US close transition
        if time(20, self.config.transition_buffer_minutes) <= t < time(21, 0):
            return True, "US close transition - reducing liquidity"
        
        return False, "Not in transition period"
    
    # ========================================================================
    # SECTION 7: TIME-OF-DAY PERFORMANCE (Wave 3)
    # ========================================================================
    
    def _load_performance_data(self):
        """Load time-of-day performance data (Wave 3)"""
        if not self.config.use_performance_data:
            return
        
        # Default performance data based on historical analysis
        # Higher scores = better for SID method
        self.config.performance_data = {
            '00': {'score': 0.3, 'description': 'Asian session - low liquidity'},
            '01': {'score': 0.3, 'description': 'Asian session - low liquidity'},
            '02': {'score': 0.3, 'description': 'Asian session - low liquidity'},
            '03': {'score': 0.3, 'description': 'Asian session - low liquidity'},
            '04': {'score': 0.3, 'description': 'Asian session - low liquidity'},
            '05': {'score': 0.4, 'description': 'Asian session end - slight increase'},
            '06': {'score': 0.5, 'description': 'Pre-London - moderate liquidity'},
            '07': {'score': 0.7, 'description': 'London open - good liquidity'},
            '08': {'score': 0.8, 'description': 'London session - good liquidity'},
            '09': {'score': 0.8, 'description': 'London session - good liquidity'},
            '10': {'score': 0.8, 'description': 'London session - good liquidity'},
            '11': {'score': 0.8, 'description': 'London session - good liquidity'},
            '12': {'score': 0.9, 'description': 'London-US overlap - excellent liquidity'},
            '13': {'score': 0.9, 'description': 'London-US overlap - excellent liquidity'},
            '14': {'score': 0.9, 'description': 'London-US overlap - excellent liquidity'},
            '15': {'score': 0.9, 'description': 'London-US overlap - excellent liquidity'},
            '16': {'score': 0.8, 'description': 'US session - good liquidity'},
            '17': {'score': 0.8, 'description': 'US session - good liquidity'},
            '18': {'score': 0.7, 'description': 'US session - moderate liquidity'},
            '19': {'score': 0.6, 'description': 'US session end - reduced liquidity'},
            '20': {'score': 0.5, 'description': 'US close - low liquidity'},
            '21': {'score': 0.4, 'description': 'Post-US - low liquidity'},
            '22': {'score': 0.3, 'description': 'Post-US - low liquidity'},
            '23': {'score': 0.3, 'description': 'Post-US - low liquidity'}
        }
    
    def get_time_performance(self, dt: datetime) -> Dict:
        """
        Get performance metrics for a specific time (Wave 3)
        
        Returns:
            Dictionary with performance data
        """
        hour_key = dt.strftime('%H')
        default = {'score': 0.5, 'description': 'Unknown'}
        return self.config.performance_data.get(hour_key, default)
    
    # ========================================================================
    # SECTION 8: COMPREHENSIVE TIME VALIDATION (Wave 1, 2, 3)
    # ========================================================================
    
    def validate_trade_time(self, dt: datetime) -> Tuple[bool, str, Dict]:
        """
        Comprehensive time validation (Wave 1, 2, 3)
        
        Returns:
            (is_valid, message, details)
        """
        details = {
            'datetime': dt.isoformat(),
            'session': None,
            'condition': None,
            'is_holiday': False,
            'is_weekend': False,
            'performance_score': 0,
            'optimal_window': False,
            'transition_period': False
        }
        
        # Check date (holidays, weekends)
        date_valid, date_msg = self.validate_date(dt)
        if not date_valid:
            details['is_holiday'] = self.is_holiday(dt)
            details['is_weekend'] = self.is_weekend(dt)
            return False, date_msg, details
        
        # Check session (Wave 1)
        session = self.get_trading_session(dt)
        details['session'] = session.value
        session_valid, session_msg = self.validate_session(dt)
        if not session_valid:
            return False, session_msg, details
        
        # Check volatility time (Wave 2)
        condition = self.get_market_condition(dt)
        details['condition'] = condition.value
        volatility_valid, volatility_msg = self.validate_volatility_time(dt)
        if not volatility_valid:
            return False, volatility_msg, details
        
        # Check optimal window (Wave 3)
        is_optimal, optimal_msg = self.is_optimal_entry_time(dt)
        details['optimal_window'] = is_optimal
        
        # Check transition period (Wave 3)
        is_transition, transition_msg = self.is_transition_period(dt)
        details['transition_period'] = is_transition
        
        # Get performance score (Wave 3)
        performance = self.get_time_performance(dt)
        details['performance_score'] = performance.get('score', 0.5)
        
        # Build message
        messages = [session_msg, volatility_msg]
        if is_optimal:
            messages.append(optimal_msg)
        if is_transition:
            messages.append(transition_msg)
        
        return True, " | ".join(messages), details
    
    # ========================================================================
    # SECTION 9: OPTIMAL TRADE TIMES (Wave 3)
    # ========================================================================
    
    def get_optimal_trade_times(self, date: datetime) -> List[Tuple[time, float]]:
        """
        Get optimal trade times for a given date (Wave 3)
        
        Returns:
            List of (time, performance_score) tuples
        """
        optimal_times = []
        
        for hour in range(12, 16):  # 12:00-16:00 GMT
            for minute in [0, 30]:
                t = time(hour, minute)
                performance = self.config.performance_data.get(str(hour).zfill(2), {}).get('score', 0.5)
                optimal_times.append((t, performance))
        
        # Sort by performance score
        optimal_times.sort(key=lambda x: x[1], reverse=True)
        
        return optimal_times
    
    def get_next_optimal_time(self, dt: datetime) -> Tuple[datetime, float]:
        """
        Get the next optimal trading time (Wave 3)
        
        Returns:
            (next_optimal_time, performance_score)
        """
        current_hour = dt.hour
        
        for hour in range(current_hour + 1, 24):
            performance = self.config.performance_data.get(str(hour).zfill(2), {}).get('score', 0.5)
            if performance >= 0.8:  # High performance threshold
                next_time = dt.replace(hour=hour, minute=0, second=0, microsecond=0)
                if next_time > dt:
                    return next_time, performance
        
        # Next day
        next_day = dt + timedelta(days=1)
        return self.get_next_optimal_time(next_day.replace(hour=0, minute=0, second=0)), 0
    
    # ========================================================================
    # SECTION 10: REPORTING (Wave 3)
    # ========================================================================
    
    def get_time_analysis(self, dt: datetime) -> Dict:
        """
        Get detailed time analysis for a specific datetime (Wave 3)
        
        Returns:
            Dictionary with time analysis
        """
        is_valid, message, details = self.validate_trade_time(dt)
        
        analysis = {
            'datetime': dt.isoformat(),
            'is_valid': is_valid,
            'message': message,
            'session': details.get('session'),
            'condition': details.get('condition'),
            'performance_score': details.get('performance_score', 0),
            'optimal_window': details.get('optimal_window', False),
            'transition_period': details.get('transition_period', False),
            'is_holiday': details.get('is_holiday', False),
            'is_weekend': details.get('is_weekend', False)
        }
        
        return analysis
    
    def print_time_summary(self, dt: datetime):
        """Print time analysis summary (Wave 3)"""
        analysis = self.get_time_analysis(dt)
        
        print("\n" + "="*50)
        print(f"⏰ TIME ANALYSIS: {analysis['datetime']}")
        print("="*50)
        print(f"Session: {analysis['session']}")
        print(f"Condition: {analysis['condition']}")
        print(f"Performance Score: {analysis['performance_score']:.2f}")
        print(f"Optimal Window: {'✓' if analysis['optimal_window'] else '✗'}")
        print(f"Transition Period: {'✓' if analysis['transition_period'] else '✗'}")
        print(f"Trade Allowed: {'✓' if analysis['is_valid'] else '✗'}")
        if not analysis['is_valid']:
            print(f"Reason: {analysis['message']}")
        print("="*50)


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TESTING TIME FILTER v3.0")
    print("="*70)
    
    # Initialize filter
    config = TimeFilterConfig(
        allowed_sessions=['overlap', 'us', 'london'],
        blocked_sessions=['asian'],
        avoid_news_blocks=True,
        allow_weekends=False
    )
    time_filter = TimeFilter(config, verbose=True)
    
    # Test various times
    test_times = [
        datetime(2024, 2, 1, 2, 0),   # Asian session
        datetime(2024, 2, 1, 8, 0),   # London session
        datetime(2024, 2, 1, 12, 30), # Overlap session
        datetime(2024, 2, 1, 15, 0),  # Overlap session
        datetime(2024, 2, 1, 20, 30), # US close
        datetime(2024, 2, 1, 8, 45),  # News block
        datetime(2024, 2, 3, 12, 0),  # Saturday (weekend)
        datetime(2024, 12, 25, 12, 0) # Christmas Day
    ]
    
    for dt in test_times:
        print(f"\n📊 Testing: {dt}")
        is_valid, msg, details = time_filter.validate_trade_time(dt)
        print(f"  Valid: {is_valid}")
        print(f"  Message: {msg}")
        print(f"  Session: {details.get('session')}")
        print(f"  Condition: {details.get('condition')}")
        print(f"  Performance: {details.get('performance_score')}")
    
    # Test optimal entry times
    print(f"\n📊 Optimal Entry Times:")
    current = datetime(2024, 2, 1, 10, 0)
    next_time, score = time_filter.get_next_optimal_time(current)
    print(f"  Current: {current}")
    print(f"  Next optimal: {next_time} (score: {score:.2f})")
    
    # Print detailed analysis
    test_dt = datetime(2024, 2, 1, 13, 30)
    time_filter.print_time_summary(test_dt)
    
    print(f"\n✅ Time filter test complete")