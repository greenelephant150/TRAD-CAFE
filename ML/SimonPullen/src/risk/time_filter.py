"""
Time filter for Simon Pullen's risk management
Filters trades based on day of week and session
Simon's best days: Tue, Wed, Thu (avoid Mon/Fri)
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pytz
import logging

logger = logging.getLogger(__name__)


class TimeFilter:
    """
    Filters trades based on time of day and day of week
    Simon: Best days are Tuesday, Wednesday, Thursday
    Avoid Mondays and Fridays, especially Friday afternoons
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preferred_days = config.get('trading_days', {}).get('preferred', ['Tuesday', 'Wednesday', 'Thursday'])
        self.avoid_days = config.get('trading_days', {}).get('avoid', ['Monday', 'Friday'])
        self.avoid_friday_afternoon = config.get('trading_days', {}).get('friday_afternoon_avoid', True)
        self.timezone = pytz.timezone('Europe/London')  # Simon's timezone
        
    def is_preferred_day(self, dt: datetime = None) -> bool:
        """Check if current day is a preferred trading day"""
        if dt is None:
            dt = datetime.now(self.timezone)
        elif dt.tzinfo is None:
            dt = self.timezone.localize(dt)
            
        day_name = dt.strftime('%A')
        return day_name in self.preferred_days
    
    def should_avoid(self, dt: datetime = None) -> Tuple[bool, str]:
        """Check if we should avoid trading at this time"""
        if dt is None:
            dt = datetime.now(self.timezone)
        elif dt.tzinfo is None:
            dt = self.timezone.localize(dt)
            
        day_name = dt.strftime('%A')
        
        # Check if day is in avoid list
        if day_name in self.avoid_days:
            return True, f"{day_name} - avoid trading"
            
        # Check Friday afternoon (after 12:00 UK time)
        if day_name == 'Friday' and self.avoid_friday_afternoon and dt.hour >= 12:
            return True, "Friday afternoon - institutional squaring up"
            
        return False, "OK"
    
    def get_current_session(self, dt: datetime = None) -> str:
        """Get current trading session"""
        if dt is None:
            dt = datetime.now(self.timezone)
        elif dt.tzinfo is None:
            dt = self.timezone.localize(dt)
            
        hour = dt.hour
        
        # Rough session times (UTC)
        if 23 <= hour or hour < 8:
            return "asia"
        elif 8 <= hour < 16:
            return "london"
        elif 16 <= hour < 23:
            return "ny"
        else:
            return "unknown"
    
    def is_session_active(self, session: str, dt: datetime = None) -> bool:
        """Check if a specific trading session is active"""
        current_session = self.get_current_session(dt)
        return current_session == session
    
    def get_next_valid_trading_time(self, from_dt: datetime = None) -> datetime:
        """Get the next valid trading time"""
        if from_dt is None:
            from_dt = datetime.now(self.timezone)
        elif from_dt.tzinfo is None:
            from_dt = self.timezone.localize(from_dt)
            
        check_dt = from_dt.replace(hour=8, minute=0, second=0, microsecond=0)  # Start at London open
        
        # Check next 7 days
        for days in range(8):
            candidate = check_dt + timedelta(days=days)
            avoid, _ = self.should_avoid(candidate)
            if not avoid:
                return candidate
        
        return check_dt + timedelta(days=7)  # Default to next week
    
    def get_trading_hours_remaining(self, dt: datetime = None) -> float:
        """Get hours remaining in current trading session"""
        if dt is None:
            dt = datetime.now(self.timezone)
        elif dt.tzinfo is None:
            dt = self.timezone.localize(dt)
            
        session = self.get_current_session(dt)
        
        if session == "asia":
            end_hour = 8
        elif session == "london":
            end_hour = 16
        elif session == "ny":
            end_hour = 23
        else:
            return 0
            
        # If current hour is past end, return 0
        if dt.hour >= end_hour:
            return 0
            
        return end_hour - dt.hour - (dt.minute / 60)
    
    def would_complete_before_session_end(self, entry_time: datetime, 
                                          bars_needed: int, timeframe: str) -> bool:
        """
        Check if trade would complete before current session ends
        """
        hours_needed = bars_needed * self._timeframe_to_hours(timeframe)
        hours_remaining = self.get_trading_hours_remaining(entry_time)
        
        return hours_needed <= hours_remaining
    
    def _timeframe_to_hours(self, timeframe: str) -> float:
        """Convert timeframe string to hours"""
        if timeframe == '15m':
            return 0.25
        elif timeframe == '1h':
            return 1.0
        elif timeframe == '4h':
            return 4.0
        elif timeframe == '1d':
            return 24.0
        else:
            return 1.0
