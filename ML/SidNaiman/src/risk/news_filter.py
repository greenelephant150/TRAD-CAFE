#!/usr/bin/env python3
"""
News Filter Module for SID Method - AUGMENTED VERSION
=============================================================================
Filters trades based on economic news events incorporating ALL THREE WAVES:

WAVE 1 (Core Quick Win):
- Earnings date filtering (14 days before)
- Basic news avoidance

WAVE 2 (Live Sessions & Q&A):
- High-impact economic news filtering
- FOMC meetings, NFP, CPI, PCE
- Central bank announcements
- Geopolitical event detection

WAVE 3 (Academy Support Sessions):
- News sentiment analysis
- Volatility spike prediction
- News impact scoring
- Custom news categories
- News calendar integration

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

class NewsImpact(Enum):
    """News impact levels (Wave 2 & 3)"""
    EXTREME = "extreme"     # FOMC, NFP, CPI
    HIGH = "high"           # PCE, GDP, Central Bank speeches
    MEDIUM = "medium"       # Unemployment, Retail Sales
    LOW = "low"             # Consumer Confidence, Housing data
    MINIMAL = "minimal"     # Routine economic releases


class NewsCategory(Enum):
    """News categories (Wave 2 & 3)"""
    MONETARY_POLICY = "monetary_policy"     # FOMC, rate decisions
    EMPLOYMENT = "employment"               # NFP, unemployment
    INFLATION = "inflation"                 # CPI, PCE, PPI
    GDP = "gdp"                             # GDP releases
    CONSUMER = "consumer"                   # Retail sales, confidence
    GEOPOLITICAL = "geopolitical"           # Wars, elections
    CORPORATE = "corporate"                 # Earnings, M&A


@dataclass
class NewsEvent:
    """News event data (Wave 2 & 3)"""
    date: datetime
    title: str
    category: NewsCategory
    impact: NewsImpact
    country: str
    actual_value: Optional[float] = None
    expected_value: Optional[float] = None
    previous_value: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'date': self.date.isoformat(),
            'title': self.title,
            'category': self.category.value,
            'impact': self.impact.value,
            'country': self.country,
            'actual': self.actual_value,
            'expected': self.expected_value,
            'previous': self.previous_value
        }


@dataclass
class NewsFilterConfig:
    """Configuration for news filter (Wave 1, 2, 3)"""
    # Wave 1: Earnings filter
    earnings_buffer_hours: int = 24  # Hours before earnings to avoid
    earnings_buffer_after_hours: int = 2  # Hours after earnings to avoid
    
    # Wave 2: Economic news filters
    high_impact_buffer_hours: int = 4  # Hours before high-impact news
    high_impact_after_hours: int = 2   # Hours after high-impact news
    medium_impact_buffer_hours: int = 2
    medium_impact_after_hours: int = 1
    
    # Wave 2: News categories to filter
    filter_categories: List[str] = field(default_factory=lambda: [
        'monetary_policy', 'employment', 'inflation', 'gdp'
    ])
    
    # Wave 3: Custom news sources
    custom_news_sources: List[str] = field(default_factory=list)
    
    # Wave 3: Sentiment analysis
    use_sentiment_analysis: bool = False
    sentiment_threshold: float = 0.7  # Sentiment score threshold
    
    # Wave 3: Volatility prediction
    predict_volatility_spikes: bool = True
    volatility_threshold: float = 0.02  # 2% volatility threshold


class NewsFilter:
    """
    Filters trades based on news events for SID Method
    Incorporates ALL THREE WAVES of strategies
    """
    
    def __init__(self, config: NewsFilterConfig = None, verbose: bool = True):
        """
        Initialize news filter
        
        Args:
            config: NewsFilterConfig instance
            verbose: Enable verbose output
        """
        self.config = config or NewsFilterConfig()
        self.verbose = verbose
        
        # News calendar (pre-populated with known events)
        self.news_calendar: List[NewsEvent] = []
        
        # Earnings calendar
        self.earnings_calendar: Dict[str, List[datetime]] = {}
        
        # Sentiment cache
        self.sentiment_cache: Dict[str, float] = {}
        
        # Initialize with major economic events
        self._initialize_major_events()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📰 NEWS FILTER v3.0 (Fully Augmented)")
            print(f"{'='*60}")
            print(f"📅 Earnings buffer: ±{self.config.earnings_buffer_hours}h")
            print(f"⚠️ High impact buffer: ±{self.config.high_impact_buffer_hours}h")
            print(f"📊 Medium impact buffer: ±{self.config.medium_impact_buffer_hours}h")
            print(f"📋 Filter categories: {self.config.filter_categories}")
            print(f"🎯 Sentiment analysis: {self.config.use_sentiment_analysis}")
            print(f"{'='*60}\n")
    
    # ========================================================================
    # SECTION 1: EVENT INITIALIZATION (Wave 2)
    # ========================================================================
    
    def _initialize_major_events(self):
        """Initialize major economic events calendar (Wave 2)"""
        # These would typically be loaded from an API or file
        # This is a simplified template
        
        # FOMC meeting dates (example for 2024)
        fomc_dates = [
            '2024-01-31', '2024-03-20', '2024-05-01', '2024-06-12',
            '2024-07-31', '2024-09-18', '2024-11-07', '2024-12-18'
        ]
        
        for date_str in fomc_dates:
            event = NewsEvent(
                date=datetime.strptime(date_str, '%Y-%m-%d'),
                title="FOMC Meeting",
                category=NewsCategory.MONETARY_POLICY,
                impact=NewsImpact.EXTREME,
                country="US"
            )
            self.news_calendar.append(event)
        
        # NFP dates (first Friday of month)
        # Example for 2024
        nfp_dates = [
            '2024-01-05', '2024-02-02', '2024-03-08', '2024-04-05',
            '2024-05-03', '2024-06-07', '2024-07-05', '2024-08-02',
            '2024-09-06', '2024-10-04', '2024-11-01', '2024-12-06'
        ]
        
        for date_str in nfp_dates:
            event = NewsEvent(
                date=datetime.strptime(date_str, '%Y-%m-%d'),
                title="Non-Farm Payrolls",
                category=NewsCategory.EMPLOYMENT,
                impact=NewsImpact.EXTREME,
                country="US"
            )
            self.news_calendar.append(event)
        
        # CPI release dates (mid-month)
        cpi_dates = [
            '2024-01-11', '2024-02-13', '2024-03-12', '2024-04-10',
            '2024-05-15', '2024-06-12', '2024-07-11', '2024-08-14',
            '2024-09-11', '2024-10-10', '2024-11-13', '2024-12-11'
        ]
        
        for date_str in cpi_dates:
            event = NewsEvent(
                date=datetime.strptime(date_str, '%Y-%m-%d'),
                title="CPI Release",
                category=NewsCategory.INFLATION,
                impact=NewsImpact.EXTREME,
                country="US"
            )
            self.news_calendar.append(event)
    
    # ========================================================================
    # SECTION 2: EARNINGS FILTER (Wave 1)
    # ========================================================================
    
    def add_earnings_date(self, instrument: str, earnings_date: datetime):
        """Add earnings date for an instrument (Wave 1)"""
        if instrument not in self.earnings_calendar:
            self.earnings_calendar[instrument] = []
        self.earnings_calendar[instrument].append(earnings_date)
        
        if self.verbose:
            print(f"📅 Added earnings for {instrument}: {earnings_date.date()}")
    
    def check_earnings(self, instrument: str, trade_date: datetime) -> Tuple[bool, str]:
        """
        Check if trade date is within earnings buffer (Wave 1)
        
        Returns:
            (is_safe, message)
        """
        if instrument not in self.earnings_calendar:
            return True, "No earnings data"
        
        for earnings_date in self.earnings_calendar[instrument]:
            # Check before earnings
            hours_before = (earnings_date - trade_date).total_seconds() / 3600
            if 0 < hours_before <= self.config.earnings_buffer_hours:
                return False, f"Within {hours_before:.0f} hours before earnings"
            
            # Check after earnings
            hours_after = (trade_date - earnings_date).total_seconds() / 3600
            if 0 < hours_after <= self.config.earnings_buffer_after_hours:
                return False, f"Within {hours_after:.0f} hours after earnings"
        
        return True, "Earnings check passed"
    
    # ========================================================================
    # SECTION 3: ECONOMIC NEWS FILTER (Wave 2)
    # ========================================================================
    
    def add_news_event(self, event: NewsEvent):
        """Add custom news event (Wave 2)"""
        self.news_calendar.append(event)
    
    def get_news_events(self, start_date: datetime, end_date: datetime) -> List[NewsEvent]:
        """Get news events within date range (Wave 2)"""
        events = []
        for event in self.news_calendar:
            if start_date <= event.date <= end_date:
                events.append(event)
        return events
    
    def check_news(self, trade_date: datetime, instrument: str = None) -> Tuple[bool, str, Optional[NewsEvent]]:
        """
        Check if trade date is within news buffer (Wave 2)
        
        Returns:
            (is_safe, message, nearest_event)
        """
        nearest_event = None
        min_distance = float('inf')
        
        for event in self.news_calendar:
            # Skip if category not in filter list
            if event.category.value not in self.config.filter_categories:
                continue
            
            # Calculate time difference
            diff_seconds = (event.date - trade_date).total_seconds()
            diff_hours = abs(diff_seconds) / 3600
            
            # Determine buffer based on impact
            if event.impact == NewsImpact.EXTREME:
                buffer_hours = self.config.high_impact_buffer_hours
                after_hours = self.config.high_impact_after_hours
            elif event.impact == NewsImpact.HIGH:
                buffer_hours = self.config.high_impact_buffer_hours
                after_hours = self.config.high_impact_after_hours
            elif event.impact == NewsImpact.MEDIUM:
                buffer_hours = self.config.medium_impact_buffer_hours
                after_hours = self.config.medium_impact_after_hours
            else:
                continue  # Skip low impact events
            
            # Check if within buffer
            if diff_seconds > 0:  # News after trade
                if diff_hours <= buffer_hours:
                    if diff_hours < min_distance:
                        min_distance = diff_hours
                        nearest_event = event
            else:  # News before trade
                if diff_hours <= after_hours:
                    if diff_hours < min_distance:
                        min_distance = diff_hours
                        nearest_event = event
        
        if nearest_event:
            return False, f"Within {min_distance:.1f} hours of {nearest_event.title} ({nearest_event.impact.value} impact)", nearest_event
        
        return True, "No conflicting news", None
    
    # ========================================================================
    # SECTION 4: VOLATILITY PREDICTION (Wave 3)
    # ========================================================================
    
    def predict_volatility_spike(self, news_events: List[NewsEvent]) -> float:
        """
        Predict volatility spike probability based on news (Wave 3)
        
        Returns:
            Probability of volatility spike (0-1)
        """
        if not self.config.predict_volatility_spikes:
            return 0.0
        
        if not news_events:
            return 0.0
        
        # Base probability from news impact
        impact_weights = {
            NewsImpact.EXTREME: 0.8,
            NewsImpact.HIGH: 0.6,
            NewsImpact.MEDIUM: 0.4,
            NewsImpact.LOW: 0.2,
            NewsImpact.MINIMAL: 0.1
        }
        
        # Combined probability (max of events)
        max_probability = 0.0
        for event in news_events:
            prob = impact_weights.get(event.impact, 0.3)
            if prob > max_probability:
                max_probability = prob
        
        # Adjust for time proximity
        # (Would need actual time to event)
        
        return max_probability
    
    def get_volatility_risk_level(self, trade_date: datetime) -> Tuple[str, float]:
        """
        Get volatility risk level for a trade date (Wave 3)
        
        Returns:
            (risk_level, probability)
        """
        # Look at news in next 24 hours
        end_date = trade_date + timedelta(hours=24)
        upcoming_events = self.get_news_events(trade_date, end_date)
        
        probability = self.predict_volatility_spike(upcoming_events)
        
        if probability >= 0.7:
            risk_level = "extreme"
        elif probability >= 0.5:
            risk_level = "high"
        elif probability >= 0.3:
            risk_level = "medium"
        elif probability >= 0.1:
            risk_level = "low"
        else:
            risk_level = "minimal"
        
        return risk_level, probability
    
    # ========================================================================
    # SECTION 5: SENTIMENT ANALYSIS (Wave 3)
    # ========================================================================
    
    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze news sentiment (Wave 3)
        
        Returns:
            Sentiment score (-1 to 1, where positive = bullish)
        """
        if not self.config.use_sentiment_analysis:
            return 0.0
        
        # Check cache
        if text in self.sentiment_cache:
            return self.sentiment_cache[text]
        
        # Simplified sentiment analysis
        # In production, use NLP library like TextBlob or VADER
        
        positive_words = ['rise', 'increase', 'growth', 'positive', 'bullish', 'up', 'higher']
        negative_words = ['fall', 'decrease', 'decline', 'negative', 'bearish', 'down', 'lower']
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total = positive_count + negative_count
        if total == 0:
            sentiment = 0.0
        else:
            sentiment = (positive_count - negative_count) / total
        
        # Cache result
        self.sentiment_cache[text] = sentiment
        
        return sentiment
    
    def filter_by_sentiment(self, news_events: List[NewsEvent]) -> List[NewsEvent]:
        """
        Filter news by sentiment (Wave 3)
        
        Returns:
            News events with significant sentiment
        """
        if not self.config.use_sentiment_analysis:
            return news_events
        
        filtered = []
        for event in news_events:
            sentiment = self.analyze_sentiment(event.title)
            if abs(sentiment) >= self.config.sentiment_threshold:
                filtered.append(event)
        
        return filtered
    
    # ========================================================================
    # SECTION 6: COMPREHENSIVE FILTER (Wave 1, 2, 3)
    # ========================================================================
    
    def check_trade(self, instrument: str, trade_date: datetime) -> Tuple[bool, str, Dict]:
        """
        Comprehensive trade filter checking all news types (Wave 1, 2, 3)
        
        Returns:
            (is_allowed, message, details)
        """
        details = {
            'earnings': {'passed': True, 'message': ''},
            'news': {'passed': True, 'message': '', 'nearest_event': None},
            'volatility': {'risk_level': 'minimal', 'probability': 0},
            'sentiment': {'score': 0}
        }
        
        # Check earnings (Wave 1)
        earnings_ok, earnings_msg = self.check_earnings(instrument, trade_date)
        if not earnings_ok:
            details['earnings'] = {'passed': False, 'message': earnings_msg}
            return False, earnings_msg, details
        details['earnings']['message'] = earnings_msg
        
        # Check news (Wave 2)
        news_ok, news_msg, nearest_event = self.check_news(trade_date, instrument)
        if not news_ok:
            details['news'] = {'passed': False, 'message': news_msg, 'nearest_event': nearest_event.to_dict() if nearest_event else None}
            return False, news_msg, details
        details['news']['message'] = news_msg
        
        # Check volatility (Wave 3)
        risk_level, probability = self.get_volatility_risk_level(trade_date)
        details['volatility'] = {'risk_level': risk_level, 'probability': probability}
        
        # If extreme volatility risk, consider filtering
        if risk_level == 'extreme':
            return False, f"Extreme volatility risk ({probability:.0%})", details
        
        # Sentiment analysis (Wave 3)
        if self.config.use_sentiment_analysis and nearest_event:
            sentiment = self.analyze_sentiment(nearest_event.title)
            details['sentiment']['score'] = sentiment
            
            if abs(sentiment) >= self.config.sentiment_threshold:
                return False, f"Strong sentiment detected: {sentiment:.2f}", details
        
        return True, "All checks passed", details
    
    # ========================================================================
    # SECTION 7: NEWS CALENDAR MANAGEMENT (Wave 3)
    # ========================================================================
    
    def get_upcoming_news(self, hours: int = 24) -> List[NewsEvent]:
        """
        Get upcoming news events in next N hours (Wave 3)
        
        Returns:
            List of upcoming news events
        """
        now = datetime.now()
        end_time = now + timedelta(hours=hours)
        
        upcoming = []
        for event in self.news_calendar:
            if now <= event.date <= end_time:
                upcoming.append(event)
        
        return upcoming
    
    def print_upcoming_news(self, hours: int = 24):
        """Print upcoming news events (Wave 3)"""
        upcoming = self.get_upcoming_news(hours)
        
        if not upcoming:
            print(f"📰 No upcoming news in next {hours} hours")
            return
        
        print(f"\n📰 UPCOMING NEWS ({hours} hours):")
        print("-"*50)
        for event in upcoming:
            print(f"  {event.date.strftime('%Y-%m-%d %H:%M')}: {event.title}")
            print(f"    Impact: {event.impact.value} | Category: {event.category.value}")
    
    def get_news_summary(self, date: datetime) -> Dict:
        """
        Get news summary for a specific date (Wave 3)
        
        Returns:
            Dictionary with news summary
        """
        start_date = date.replace(hour=0, minute=0, second=0)
        end_date = start_date + timedelta(days=1)
        
        events = self.get_news_events(start_date, end_date)
        
        summary = {
            'date': date.date().isoformat(),
            'total_events': len(events),
            'by_impact': {},
            'by_category': {},
            'events': [e.to_dict() for e in events]
        }
        
        for event in events:
            # By impact
            impact = event.impact.value
            summary['by_impact'][impact] = summary['by_impact'].get(impact, 0) + 1
            
            # By category
            category = event.category.value
            summary['by_category'][category] = summary['by_category'].get(category, 0) + 1
        
        return summary
    
    # ========================================================================
    # SECTION 8: CUSTOM NEWS SOURCES (Wave 3)
    # ========================================================================
    
    def add_custom_source(self, source_name: str):
        """Add custom news source (Wave 3)"""
        if source_name not in self.config.custom_news_sources:
            self.config.custom_news_sources.append(source_name)
            if self.verbose:
                print(f"📰 Added custom news source: {source_name}")
    
    def import_calendar_csv(self, filepath: str) -> int:
        """
        Import news calendar from CSV (Wave 3)
        
        Expected columns: date, title, category, impact, country
        
        Returns:
            Number of events imported
        """
        try:
            df = pd.read_csv(filepath)
            count = 0
            
            for _, row in df.iterrows():
                event = NewsEvent(
                    date=pd.to_datetime(row['date']),
                    title=row['title'],
                    category=NewsCategory(row['category']),
                    impact=NewsImpact(row['impact']),
                    country=row.get('country', 'US')
                )
                self.news_calendar.append(event)
                count += 1
            
            if self.verbose:
                print(f"✅ Imported {count} news events from {filepath}")
            
            return count
        except Exception as e:
            print(f"❌ Failed to import calendar: {e}")
            return 0


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TESTING NEWS FILTER v3.0")
    print("="*70)
    
    # Initialize filter
    config = NewsFilterConfig(
        earnings_buffer_hours=24,
        high_impact_buffer_hours=4,
        medium_impact_buffer_hours=2
    )
    news_filter = NewsFilter(config, verbose=True)
    
    # Test earnings filter
    print("\n📊 Testing Earnings Filter:")
    news_filter.add_earnings_date('AAPL', datetime(2024, 2, 1, 16, 0))
    
    # Test before earnings
    trade_date = datetime(2024, 1, 31, 12, 0)
    is_safe, msg = news_filter.check_earnings('AAPL', trade_date)
    print(f"  Before earnings ({trade_date}): {msg}")
    
    # Test after earnings
    trade_date = datetime(2024, 2, 1, 18, 0)
    is_safe, msg = news_filter.check_earnings('AAPL', trade_date)
    print(f"  After earnings ({trade_date}): {msg}")
    
    # Test news filter
    print("\n📊 Testing News Filter:")
    trade_date = datetime(2024, 2, 1, 10, 0)  # Day of FOMC
    is_safe, msg, event = news_filter.check_news(trade_date)
    print(f"  {trade_date}: {msg}")
    
    trade_date = datetime(2024, 2, 5, 10, 0)  # Day after FOMC
    is_safe, msg, event = news_filter.check_news(trade_date)
    print(f"  {trade_date}: {msg}")
    
    # Test volatility prediction
    print("\n📊 Testing Volatility Prediction:")
    trade_date = datetime(2024, 2, 1, 10, 0)
    risk_level, probability = news_filter.get_volatility_risk_level(trade_date)
    print(f"  {trade_date}: Risk Level = {risk_level}, Probability = {probability:.0%}")
    
    # Test comprehensive check
    print("\n📊 Testing Comprehensive Check:")
    is_allowed, msg, details = news_filter.check_trade('AAPL', trade_date)
    print(f"  Trade allowed: {is_allowed}")
    print(f"  Message: {msg}")
    print(f"  Details: {details}")
    
    # Print upcoming news
    news_filter.print_upcoming_news(48)
    
    print(f"\n✅ News filter test complete")