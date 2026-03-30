"""
Standalone OANDA client for Simon Pullen ML Co-Pilot
Fixed version - Separates public and account-specific endpoints
Following Academy Support Session rules with proper error handling
"""

import requests
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Callable, Union
import time
import math

logger = logging.getLogger(__name__)

# OANDA Configuration - YOUR WORKING CREDENTIALS
OANDA_API_KEY = "fceae94e861642af6c9d9de1bf6a319d-52074b19e787811450bc44152cb71e78"
OANDA_ENVIRONMENT = "practice"  # or "live"

# Base URLs
if OANDA_ENVIRONMENT == "practice":
    OANDA_BASE_URL = "https://api-fxpractice.oanda.com/v3"
else:
    OANDA_BASE_URL = "https://api-fxtrade.oanda.com/v3"

# Account ID (from your test)
ACCOUNT_ID = "101-004-35778624-001"

# Headers for all requests
OANDA_HEADERS = {
    "Authorization": f"Bearer {OANDA_API_KEY}",
    "Content-Type": "application/json",
}

# ============================================================================
# VERIFIED WORKING OANDA PAIRS (Tested and confirmed)
# ============================================================================
VALID_OANDA_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_AUD", "EUR_CHF", "AUD_NZD",
    "NZD_JPY", "GBP_AUD", "GBP_CAD", "GBP_CHF", "GBP_NZD", "CAD_JPY", "CHF_JPY",
    "EUR_CAD", "AUD_CAD", "NZD_CAD", "EUR_NZD", "USD_NOK", "USD_SEK", "USD_TRY",
    "EUR_NOK", "EUR_SEK", "EUR_TRY", "GBP_NOK", "GBP_SEK", "GBP_TRY"
]

# ============================================================================
# GRANULARITY MAPPING
# ============================================================================
# UI-friendly names to OANDA granularity
UI_TO_OANDA_GRANULARITY = {
    '5s': 'S5', '10s': 'S10', '15s': 'S15', '30s': 'S30',
    '1m': 'M1', '2m': 'M2', '4m': 'M4', '5m': 'M5',
    '10m': 'M10', '15m': 'M15', '30m': 'M30',
    '1h': 'H1', '2h': 'H2', '3h': 'H3', '4h': 'H4',
    '6h': 'H6', '8h': 'H8', '12h': 'H12',
    '1d': 'D', '1w': 'W', '1M': 'M'
}

# Comprehensive granularity mapping for all formats
GRANULARITY_MAP = {
    # UI common formats
    '1m': 'M1', '5m': 'M5', '15m': 'M15', '30m': 'M30',
    '1h': 'H1', '2h': 'H2', '3h': 'H3', '4h': 'H4',
    '6h': 'H6', '8h': 'H8', '12h': 'H12',
    '1d': 'D', '1w': 'W', '1M': 'M',
    
    # Alternative formats
    '1H': 'H1', '2H': 'H2', '3H': 'H3', '4H': 'H4',
    '6H': 'H6', '8H': 'H8', '12H': 'H12',
    '1D': 'D', '1W': 'W', '1M': 'M',
    '5M': 'M5', '15M': 'M15', '30M': 'M30',
    
    # Direct OANDA formats (pass through)
    'M1': 'M1', 'M5': 'M5', 'M15': 'M15', 'M30': 'M30',
    'H1': 'H1', 'H2': 'H2', 'H3': 'H3', 'H4': 'H4',
    'H6': 'H6', 'H8': 'H8', 'H12': 'H12',
    'D': 'D', 'W': 'W', 'M': 'M',
    'S5': 'S5', 'S10': 'S10', 'S15': 'S15', 'S30': 'S30',
}

# Valid OANDA granularities
VALID_GRANULARITIES = ['S5', 'S10', 'S15', 'S30', 'M1', 'M2', 'M4', 'M5', 
                       'M10', 'M15', 'M30', 'H1', 'H2', 'H3', 'H4', 'H6', 
                       'H8', 'H12', 'D', 'W', 'M']

# Pip values for different pair types
PIP_VALUES = {
    'JPY': 0.01,      # Yen pairs: 2nd decimal
    'DEFAULT': 0.0001  # Most pairs: 4th decimal
}

# Stop loss rules from sessions
STOP_PIPS = {
    'JPY': 10,     # Yen pairs: 10 pips behind zone
    'DEFAULT': 5   # Others: 5 pips behind zone
}


class OANDAClient:
    """Fixed OANDA API client with proper endpoint separation"""
    
    def __init__(self):
        """Initialize the OANDA client"""
        self.api_key = OANDA_API_KEY
        self.base_url = OANDA_BASE_URL
        self.account_id = ACCOUNT_ID
        self.headers = OANDA_HEADERS.copy()
        self.valid_pairs = set(VALID_OANDA_PAIRS)
        
        # Create session for connection pooling
        self._session = requests.Session()
        self._session.headers.update(self.headers)
        
        # Verify connection on init
        self._verify_connection()
        
        logger.info(f"✅ OANDAClient initialized for {OANDA_ENVIRONMENT}")
        logger.info(f"   Account ID: {self.account_id}")
        logger.info(f"   Valid pairs: {len(self.valid_pairs)}")
    
    def _verify_connection(self):
        """Verify API connection is working"""
        try:
            # Test a simple candle request
            test_df = self.fetch_candles("EUR_USD", "H1", count=5)
            if not test_df.empty:
                logger.info("✅ OANDA API connection verified")
            else:
                logger.warning("⚠️ OANDA API connection test returned no data")
        except Exception as e:
            logger.error(f"❌ OANDA API connection failed: {e}")
    
    # ========================================================================
    # PAIR VALIDATION
    # ========================================================================
    
    def is_pair_supported(self, pair: str) -> bool:
        """Check if pair is in valid list"""
        return pair in self.valid_pairs
    
    def get_valid_pairs(self) -> List[str]:
        """Return list of valid pairs"""
        return sorted(list(self.valid_pairs))
    
    # ========================================================================
    # PIP CALCULATIONS (From Academy sessions)
    # ========================================================================
    
    def get_pip_value(self, instrument: str) -> float:
        """
        Get pip value for an instrument
        From sessions: 
        - Most pairs: 4th decimal (0.0001)
        - Yen pairs: 2nd decimal (0.01)
        """
        if instrument.endswith('_JPY') or '_JPY' in instrument:
            return PIP_VALUES['JPY']
        return PIP_VALUES['DEFAULT']
    
    def get_stop_pips(self, instrument: str) -> int:
        """
        Get number of pips for stop loss placement
        From sessions:
        - Yen pairs: 10 pips behind zone
        - Others: 5 pips behind zone
        """
        if instrument.endswith('_JPY') or '_JPY' in instrument:
            return STOP_PIPS['JPY']
        return STOP_PIPS['DEFAULT']
    
    def calculate_pips(self, instrument: str, price1: float, price2: float) -> float:
        """
        Calculate number of pips between two prices
        Used for precise stop loss placement
        """
        pip_value = self.get_pip_value(instrument)
        return abs(price1 - price2) / pip_value
    
    def add_pips(self, instrument: str, price: float, pips: float, direction: str = 'up') -> float:
        """
        Add/subtract pips from a price
        For stop loss placement
        """
        pip_value = self.get_pip_value(instrument)
        adjustment = pips * pip_value
        
        if direction == 'up':
            return round(price + adjustment, 5)
        else:
            return round(price - adjustment, 5)
    
    def calculate_stop_from_zone(self, instrument: str, zone_price: float, 
                                   trade_direction: str = 'long') -> float:
        """
        Calculate proper stop loss distance from zone
        From sessions: Yen pairs 10 pips, others 5 pips
        """
        pips = self.get_stop_pips(instrument)
        pip_value = self.get_pip_value(instrument)
        
        if trade_direction == 'long':
            # Stop below zone
            return round(zone_price - (pips * pip_value), 5)
        else:
            # Stop above zone
            return round(zone_price + (pips * pip_value), 5)
    
    # ========================================================================
    # TIME FORMATTING
    # ========================================================================
    
    def format_time(self, dt: datetime) -> str:
        """Format datetime for OANDA API (RFC3339)"""
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    
    def parse_time(self, time_str: str) -> datetime:
        """Parse OANDA time string to datetime"""
        time_str = time_str.rstrip("Z")
        if "." in time_str:
            return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%f")
        else:
            return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S")
    
    # ========================================================================
    # PUBLIC ENDPOINTS (No account ID required)
    # ========================================================================
    
    def fetch_candles(self, instrument: str, granularity: str = 'H1',
                       from_time: Optional[Union[datetime, str]] = None,
                       to_time: Optional[Union[datetime, str]] = None,
                       count: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch candle data from OANDA - PUBLIC ENDPOINT
        This endpoint does NOT require an account ID
        """
        # Validate pair
        if not self.is_pair_supported(instrument):
            logger.warning(f"Instrument {instrument} not in valid pairs list")
            return pd.DataFrame()
        
        # Convert granularity using comprehensive map
        oanda_gran = GRANULARITY_MAP.get(granularity, granularity)
        
        # Validate granularity format
        if oanda_gran not in VALID_GRANULARITIES:
            logger.error(f"❌ Invalid granularity: '{granularity}' -> '{oanda_gran}'")
            logger.error(f"Valid options: {VALID_GRANULARITIES}")
            return pd.DataFrame()
        
        # Build URL - PUBLIC endpoint (no account ID needed)
        url = f"{self.base_url}/instruments/{instrument}/candles"
        
        # Build parameters
        params = {
            'granularity': oanda_gran,
            'price': 'MBA',  # Mid, Bid, Ask
        }
        
        # Handle datetime objects
        if from_time:
            if isinstance(from_time, datetime):
                params['from'] = self.format_time(from_time)
            else:
                params['from'] = from_time
        if to_time:
            if isinstance(to_time, datetime):
                params['to'] = self.format_time(to_time)
            else:
                params['to'] = to_time
        if count:
            params['count'] = min(count, 5000)
        
        try:
            logger.debug(f"Fetching candles for {instrument} with granularity {oanda_gran}")
            response = self._session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                candles = data.get('candles', [])
                
                if not candles:
                    logger.info(f"No candles returned for {instrument}")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                df_list = []
                for c in candles:
                    # Only include complete candles
                    if c.get('complete', True):
                        df_list.append({
                            'time': c['time'],
                            'open': float(c['mid']['o']),
                            'high': float(c['mid']['h']),
                            'low': float(c['mid']['l']),
                            'close': float(c['mid']['c']),
                            'volume': c['volume']
                        })
                
                if not df_list:
                    return pd.DataFrame()
                
                df = pd.DataFrame(df_list)
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                
                # Ensure timezone is UTC
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                
                logger.info(f"✅ Fetched {len(df)} candles for {instrument}")
                return df
                
            elif response.status_code == 400:
                logger.error(f"❌ Bad request for {instrument}: {response.text}")
                return pd.DataFrame()
            else:
                logger.error(f"❌ OANDA API error for {instrument}: {response.status_code}")
                return pd.DataFrame()
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Request error for {instrument}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"❌ Unexpected error for {instrument}: {e}")
            return pd.DataFrame()
    
    def get_instrument_details(self, instrument: str) -> Dict:
        """
        Get detailed information about an instrument
        PUBLIC ENDPOINT - no account ID needed
        """
        if not self.is_pair_supported(instrument):
            return {}
        
        url = f"{self.base_url}/instruments/{instrument}"
        
        try:
            response = self._session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                inst = data.get('instrument', {})
                return {
                    'name': inst.get('name'),
                    'type': inst.get('type'),
                    'display_name': inst.get('displayName'),
                    'pip_location': inst.get('pipLocation'),
                    'minimum_trade_size': inst.get('minimumTradeSize'),
                    'maximum_trade_size': inst.get('maximumTradeSize'),
                    'margin_rate': inst.get('marginRate')
                }
            else:
                logger.error(f"Failed to get instrument details: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Exception getting instrument details: {e}")
            return {}
    
    # ========================================================================
    # ACCOUNT-SPECIFIC ENDPOINTS (Require account ID)
    # ========================================================================
    
    def get_account_summary(self) -> Dict:
        """Get account details and balance"""
        if not self.account_id:
            logger.error("No account ID available")
            return {}
        
        url = f"{self.base_url}/accounts/{self.account_id}"
        try:
            response = self._session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data
            else:
                logger.error(f"Failed to get account details: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Exception getting account details: {e}")
            return {}
    
    def get_current_price(self, instrument: str) -> Dict:
        """
        Get current bid/ask for an instrument
        ACCOUNT-SPECIFIC ENDPOINT
        """
        if not self.account_id:
            logger.error("No account ID available")
            return {}
        
        if not self.is_pair_supported(instrument):
            logger.warning(f"Instrument {instrument} not supported")
            return {}
        
        url = f"{self.base_url}/accounts/{self.account_id}/pricing"
        params = {"instruments": instrument}
        
        try:
            response = self._session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                prices = data.get('prices', [])
                if prices:
                    price = prices[0]
                    bid = float(price['bids'][0]['price'])
                    ask = float(price['asks'][0]['price'])
                    mid = (bid + ask) / 2
                    
                    return {
                        'instrument': price['instrument'],
                        'bid': bid,
                        'ask': ask,
                        'mid': mid,
                        'spread': ask - bid,
                        'spread_pips': self.calculate_pips(instrument, ask, bid),
                        'time': price['time']
                    }
            else:
                logger.error(f"Failed to get price: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Exception getting price: {e}")
            return {}
    
    def get_account_instruments(self) -> List[str]:
        """Get list of tradable instruments for this account"""
        if not self.account_id:
            logger.error("No account ID available")
            return []
        
        url = f"{self.base_url}/accounts/{self.account_id}/instruments"
        try:
            response = self._session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                instruments = data.get('instruments', [])
                return [inst['name'] for inst in instruments]
            else:
                logger.error(f"Failed to get instruments: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Exception getting instruments: {e}")
            return []
    
    # ========================================================================
    # CONVENIENCE METHODS
    # ========================================================================
    
    def get_recent_data(self, instrument: str, granularity: str = '1h',
                         hours: int = 24) -> pd.DataFrame:
        """Get recent data for an instrument"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        return self.fetch_candles(
            instrument=instrument,
            granularity=granularity,
            from_time=start_time,
            to_time=end_time
        )
    
    def get_historical_data(self, instrument: str, granularity: str = 'H1',
                              days: int = 30) -> pd.DataFrame:
        """
        Get historical data for backtesting
        Handles OANDA's 5000 candle limit by making multiple requests
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        all_dfs = []
        current_end = end_time
        
        while current_end > start_time:
            batch_days = min(7, (current_end - start_time).days + 1)
            current_start = max(start_time, current_end - timedelta(days=batch_days))
            
            df = self.fetch_candles(
                instrument=instrument,
                granularity=granularity,
                from_time=current_start,
                to_time=current_end
            )
            
            if not df.empty:
                all_dfs.append(df)
            
            current_end = current_start - timedelta(minutes=1)
            time.sleep(0.5)  # Rate limiting
        
        if all_dfs:
            combined = pd.concat(all_dfs)
            combined = combined[~combined.index.duplicated(keep='first')]
            combined.sort_index(inplace=True)
            return combined
        
        return pd.DataFrame()
    
    # ========================================================================
    # INDICATOR CALCULATIONS
    # ========================================================================
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI from price data"""
        if df.empty or len(df) < period + 1:
            return pd.Series(index=df.index)
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Avoid division by zero
        rs = gain / loss.replace(0, float('nan'))
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD from price data"""
        if df.empty or len(df) < slow + signal:
            return pd.DataFrame(index=df.index)
        
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': macd_line - signal_line
        })
    
    def get_weekly_data(self, instrument: str, weeks: int = 52) -> pd.DataFrame:
        """Get weekly data specifically for weekly RSI check"""
        return self.get_historical_data(
            instrument=instrument,
            granularity='W',
            days=weeks * 7
        )
    
    # ========================================================================
    # ZONE QUALITY ASSESSMENT (From Academy sessions)
    # ========================================================================
    
    def assess_zone_quality(self, df: pd.DataFrame, zone_start_idx: int, 
                              zone_end_idx: int) -> Dict:
        """
        Assess quality of a supply/demand zone
        From Academy sessions:
        - Consolidation tightness
        - Breakout violence
        """
        zone_data = df.iloc[zone_start_idx:zone_end_idx+1]
        
        # 1. Consolidation tightness (small bodies, small wicks)
        body_sizes = abs(zone_data['close'] - zone_data['open'])
        wick_sizes = zone_data['high'] - zone_data['low']
        
        avg_body = body_sizes.mean()
        avg_wick = wick_sizes.mean()
        
        tightness_score = 100 - min(100, (avg_wick / avg_body * 100) if avg_body > 0 else 100)
        
        # 2. Breakout violence (candle size, momentum)
        if zone_end_idx + 1 < len(df):
            breakout_candle = df.iloc[zone_end_idx + 1]
            prev_candle = df.iloc[zone_end_idx]
            
            breakout_size = abs(breakout_candle['close'] - breakout_candle['open'])
            prev_size = abs(prev_candle['close'] - prev_candle['open'])
            
            violence_score = min(100, (breakout_size / prev_size * 100) if prev_size > 0 else 50)
        else:
            violence_score = 50
        
        overall_quality = (tightness_score * 0.6 + violence_score * 0.4)
        
        quality_rating = 'Excellent' if overall_quality >= 80 else \
                        'Good' if overall_quality >= 60 else \
                        'Fair' if overall_quality >= 40 else 'Poor'
        
        return {
            'tightness_score': tightness_score,
            'violence_score': violence_score,
            'overall_quality': overall_quality,
            'quality_rating': quality_rating,
            'consolidation_candles': len(zone_data)
        }
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def set_environment(self, environment: str):
        """Switch between practice and live environments"""
        if environment not in ['practice', 'live']:
            raise ValueError("Environment must be 'practice' or 'live'")
        
        self.environment = environment
        if environment == "practice":
            self.base_url = "https://api-fxpractice.oanda.com/v3"
        else:
            self.base_url = "https://api-fxtrade.oanda.com/v3"
        
        logger.info(f"Switched to {environment} environment")
    
    def get_connection_status(self) -> Dict:
        """Get current connection status"""
        return {
            'environment': OANDA_ENVIRONMENT,
            'account_id': self.account_id,
            'valid_pairs_count': len(self.valid_pairs),
            'base_url': self.base_url,
            'connected': self.account_id is not None
        }


# ============================================================================
# STANDALONE TEST FUNCTION
# ============================================================================
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("TESTING OANDA CLIENT")
    print("="*60)
    
    # Create client
    client = OANDAClient()
    
    # Test 1: Get account summary
    print("\n📊 Account Summary:")
    summary = client.get_account_summary()
    if summary:
        account = summary.get('account', {})
        print(f"  Balance: {account.get('balance')} {account.get('currency')}")
        print(f"  NAV: {account.get('NAV')}")
        print(f"  Open Trades: {account.get('openTradeCount')}")
    
    # Test 2: Fetch candles for major pairs
    print("\n📈 Testing candle fetch:")
    test_pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]
    for pair in test_pairs:
        df = client.fetch_candles(pair, "H1", count=10)
        if not df.empty:
            print(f"  ✅ {pair}: {len(df)} candles")
            print(f"     Range: {df.index[0]} to {df.index[-1]}")
        else:
            print(f"  ❌ {pair}: Failed")
    
    # Test 3: Get current price
    print("\n💰 Current Prices:")
    for pair in test_pairs:
        price = client.get_current_price(pair)
        if price:
            print(f"  {pair}: Bid={price['bid']:.5f} Ask={price['ask']:.5f}")
    
    # Test 4: Pip calculations
    print("\n📏 Pip Calculations:")
    for pair in test_pairs:
        pip_value = client.get_pip_value(pair)
        stop_pips = client.get_stop_pips(pair)
        print(f"  {pair}: pip={pip_value}, stop={stop_pips} pips")
    
    print("\n" + "="*60)
    print("✅ Test complete")
    print("="*60)