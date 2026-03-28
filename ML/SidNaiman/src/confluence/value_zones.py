"""
Institutional Value Zone Detection
Simon Pullen: Areas where institutions accumulate/distribute
These zones act as strong support/resistance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValueZone:
    """Represents an institutional value zone"""
    top: float
    bottom: float
    midpoint: float
    strength: float  # 0-1, based on how many times it's been respected
    first_detected: pd.Timestamp
    last_tested: Optional[pd.Timestamp] = None
    test_count: int = 0


class InstitutionalValueZoneDetector:
    """
    Detects institutional value zones
    These are areas where institutions accumulate or distribute
    
    Simon: "These zones have been in my chart for years. They're still being used."
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_touches = config.get('zone_min_touches', 3)
        self.zone_width_pct = config.get('zone_width_pct', 0.005)  # 0.5% width
        self.lookback_years = config.get('zone_lookback_years', 2)
        
    def detect_zones(self, df: pd.DataFrame) -> List[ValueZone]:
        """
        Detect value zones in historical data
        Zones are areas where price has reversed multiple times
        """
        # Find swing highs and lows
        swing_highs = self._find_swing_points(df, 'high')
        swing_lows = self._find_swing_points(df, 'low')
        
        # Cluster swing points into zones
        zones = []
        
        # Cluster swing highs (resistance zones)
        high_clusters = self._cluster_points(swing_highs)
        for cluster in high_clusters:
            if len(cluster) >= self.min_touches:
                zone = self._create_zone(cluster, 'resistance', df)
                if zone:
                    zones.append(zone)
        
        # Cluster swing lows (support zones)
        low_clusters = self._cluster_points(swing_lows)
        for cluster in low_clusters:
            if len(cluster) >= self.min_touches:
                zone = self._create_zone(cluster, 'support', df)
                if zone:
                    zones.append(zone)
        
        return zones
    
    def _find_swing_points(self, df: pd.DataFrame, price_type: str, window: int = 10) -> List[Dict]:
        """
        Find swing highs or lows
        """
        points = []
        
        for i in range(window, len(df) - window):
            if price_type == 'high':
                current = df.iloc[i]['high']
                left_max = df.iloc[i-window:i]['high'].max()
                right_max = df.iloc[i+1:i+window+1]['high'].max()
                
                if current > left_max and current > right_max:
                    points.append({
                        'idx': i,
                        'price': current,
                        'timestamp': df.index[i]
                    })
            else:
                current = df.iloc[i]['low']
                left_min = df.iloc[i-window:i]['low'].min()
                right_min = df.iloc[i+1:i+window+1]['low'].min()
                
                if current < left_min and current < right_min:
                    points.append({
                        'idx': i,
                        'price': current,
                        'timestamp': df.index[i]
                    })
        
        return points
    
    def _cluster_points(self, points: List[Dict]) -> List[List[Dict]]:
        """
        Cluster points that are close in price
        """
        if not points:
            return []
            
        # Sort by price
        sorted_points = sorted(points, key=lambda x: x['price'])
        clusters = []
        current_cluster = [sorted_points[0]]
        
        for point in sorted_points[1:]:
            # Check if price is within zone width
            price_diff_pct = abs(point['price'] - current_cluster[-1]['price']) / current_cluster[-1]['price']
            
            if price_diff_pct <= self.zone_width_pct:
                current_cluster.append(point)
            else:
                if len(current_cluster) >= self.min_touches:
                    clusters.append(current_cluster)
                current_cluster = [point]
        
        # Add last cluster
        if len(current_cluster) >= self.min_touches:
            clusters.append(current_cluster)
        
        return clusters
    
    def _create_zone(self, cluster: List[Dict], zone_type: str, df: pd.DataFrame) -> Optional[ValueZone]:
        """
        Create a value zone from a cluster of points
        """
        prices = [p['price'] for p in cluster]
        timestamps = [p['timestamp'] for p in cluster]
        
        top = max(prices)
        bottom = min(prices)
        midpoint = (top + bottom) / 2
        
        # Calculate strength based on number of touches and recency
        touch_count = len(cluster)
        recency_factor = min((df.index[-1] - min(timestamps)).days / 365, 1.0)
        strength = min(touch_count / 10 * (1 - recency_factor * 0.3), 1.0)
        
        return ValueZone(
            top=top,
            bottom=bottom,
            midpoint=midpoint,
            strength=strength,
            first_detected=min(timestamps),
            last_tested=max(timestamps),
            test_count=touch_count
        )
    
    def is_in_zone(self, price: float, zones: List[ValueZone]) -> Tuple[bool, Optional[ValueZone]]:
        """
        Check if price is in any value zone
        """
        for zone in zones:
            if zone.bottom <= price <= zone.top:
                return True, zone
        return False, None
    
    def get_nearest_zone(self, price: float, zones: List[ValueZone], direction: str) -> Optional[ValueZone]:
        """
        Get nearest value zone in given direction
        """
        if direction == 'above':
            above = [z for z in zones if z.bottom > price]
            if not above:
                return None
            return min(above, key=lambda z: z.bottom - price)
        else:
            below = [z for z in zones if z.top < price]
            if not below:
                return None
            return max(below, key=lambda z: price - z.top)
    
    def would_act_as_resistance(self, price: float, zone: ValueZone, direction: str) -> bool:
        """
        Check if zone would act as resistance/support
        """
        if direction == 'long':
            # For long trades, zones above act as resistance
            return zone.bottom > price
        else:
            # For short trades, zones below act as support
            return zone.top < price
