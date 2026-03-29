"""
News filter for Simon Pullen's risk management
Filters out trades around high-impact news events
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class NewsFilter:
    """
    Filters out trades around high-impact news events
    Simon: Never hold through high-impact news, avoid 2 hours before
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.avoid_hours = config.get('news_avoid_hours', 2)
        self.news_data = self._load_news_data()
        
    def _load_news_data(self) -> List[Dict]:
        """
        Load news data (would connect to economic calendar API in production)
        For now, return empty list
        """
        return []
    
    def set_news_data(self, news: List[Dict]):
        """Set news data from external source"""
        self.news_data = news
    
    def is_safe_to_trade(self, current_time: datetime = None) -> Tuple[bool, Optional[Dict]]:
        """
        Check if it's safe to enter a trade now
        Returns (safe, upcoming_news)
        """
        if current_time is None:
            current_time = datetime.now()
            
        for news in self.news_data:
            news_time = news.get('time')
            if not news_time:
                continue
                
            impact = news.get('impact', '').lower()
            if impact not in ['high', 'medium']:
                continue
                
            time_diff = (news_time - current_time).total_seconds() / 3600
            
            # If news is within avoid_hours, it's not safe
            if 0 < time_diff < self.avoid_hours:
                return False, news
                
            # If news just passed, also wait a bit (15 min)
            if -0.25 < time_diff < 0:
                return False, news
        
        return True, None
    
    def can_hold_through_news(self, entry_time: datetime, current_time: datetime, 
                              pattern_type: str) -> Tuple[bool, str]:
        """
        Check if we can hold a position through upcoming news
        Simon: Never hold through high-impact news on lower timeframes
        """
        # For M&W on 1h, never hold through news
        if pattern_type in ['M', 'W']:
            return False, "M&W patterns should never be held through news"
            
        # For H&S, depends on timeframe
        for news in self.news_data:
            news_time = news.get('time')
            if not news_time:
                continue
                
            impact = news.get('impact', '').lower()
            if impact != 'high':
                continue
                
            # If news is between entry and now+2h, close position
            if entry_time < news_time < current_time + timedelta(hours=2):
                return False, f"High-impact news at {news_time}"
        
        return True, "OK"
    
    def add_manual_news(self, news_time: datetime, impact: str, description: str = ""):
        """Add a manual news entry for testing"""
        self.news_data.append({
            'time': news_time,
            'impact': impact,
            'description': description,
            'currency': 'USD'  # Default
        })
    
    def clear_news(self):
        """Clear all news data"""
        self.news_data = []
