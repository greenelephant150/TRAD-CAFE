"""
OANDA Trading Module for executing live trades
Supports both practice (demo) and live (real) environments
"""

import requests
import json
import logging
from typing import Dict, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Your OANDA credentials
PRACTICE_API_KEY = "fceae94e861642af6c9d9de1bf6a319d-52074b19e787811450bc44152cb71e78"
LIVE_API_KEY = "c4f1b1d0739e88bcc604b26115db4787-1abf7348cce8c429b76a9fcf97b0b97a"

# Account IDs
PRACTICE_ACCOUNT_ID = "101-004-35778624-001"  # £100k demo account
LIVE_ACCOUNT_ID = "001-004-17934933-001"      # £10 real account

# API endpoints
PRACTICE_URL = "https://api-fxpractice.oanda.com/v3"
LIVE_URL = "https://api-fxtrade.oanda.com/v3"


class OANDATrader:
    """Execute trades on OANDA (supports both practice and live)"""
    
    def __init__(self, environment: str = "practice"):
        """
        Initialize OANDA trader
        
        Args:
            environment: 'practice' (demo) or 'live' (real money)
        """
        self.environment = environment
        
        if environment == "live":
            self.api_key = LIVE_API_KEY
            self.account_id = LIVE_ACCOUNT_ID
            self.base_url = LIVE_URL
            self.account_type = "LIVE (real money)"
            logger.warning("⚠️  INITIALIZING LIVE TRADING - REAL MONEY ACCOUNT")
        else:
            self.api_key = PRACTICE_API_KEY
            self.account_id = PRACTICE_ACCOUNT_ID
            self.base_url = PRACTICE_URL
            self.account_type = "PRACTICE (demo)"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        logger.info(f"Initialized OANDA trader for {self.account_type} account")
    
    def get_account_summary(self) -> Dict:
        """Get account details and balance"""
        url = f"{self.base_url}/accounts/{self.account_id}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get account details: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            logger.error(f"Exception getting account details: {e}")
            return {}
    
    def get_instruments(self) -> List[str]:
        """Get list of tradable instruments"""
        url = f"{self.base_url}/accounts/{self.account_id}/instruments"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            
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
    
    def get_current_price(self, instrument: str) -> Dict:
        """Get current bid/ask for an instrument"""
        url = f"{self.base_url}/accounts/{self.account_id}/pricing"
        params = {"instruments": instrument}
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                prices = data.get('prices', [])
                if prices:
                    price = prices[0]
                    return {
                        'instrument': price['instrument'],
                        'bid': float(price['bids'][0]['price']),
                        'ask': float(price['asks'][0]['price']),
                        'mid': (float(price['bids'][0]['price']) + float(price['asks'][0]['price'])) / 2,
                        'time': price['time']
                    }
            else:
                logger.error(f"Failed to get price: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Exception getting price: {e}")
            return {}
    
    def place_order(self, 
                    instrument: str,
                    units: float,
                    stop_loss: Optional[float] = None,
                    take_profit: Optional[float] = None,
                    order_type: str = "MARKET") -> Dict:
        """
        Place a trade order
        """
        url = f"{self.base_url}/accounts/{self.account_id}/orders"
        
        # Build order
        order = {
            "order": {
                "type": order_type,
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK" if order_type == "MARKET" else "GTC"
            }
        }
        
        # Add stop loss if specified
        if stop_loss:
            order["order"]["stopLossOnFill"] = {
                "price": str(round(stop_loss, 5))
            }
        
        # Add take profit if specified
        if take_profit:
            order["order"]["takeProfitOnFill"] = {
                "price": str(round(take_profit, 5))
            }
        
        try:
            response = requests.post(url, headers=self.headers, json=order, timeout=10)
            
            if response.status_code == 201:
                result = response.json()
                logger.info(f"Order placed: {instrument} {units} units")
                return result
            else:
                logger.error(f"Order failed: {response.status_code} - {response.text}")
                return {"error": response.text, "status_code": response.status_code}
        except Exception as e:
            logger.error(f"Exception placing order: {e}")
            return {"error": str(e)}
    
    def place_trade_from_pattern(self, pattern_data: Dict, account_balance: float) -> Dict:
        """
        Execute a trade based on a detected pattern
        """
        instrument = pattern_data['trading_pair']
        
        # Determine direction
        if pattern_data['pattern']['type'] in ['W_bottom', 'inv_HS']:
            units = 1000  # Base units for BUY
        else:
            units = -1000  # Base units for SELL
        
        # Scale position size based on account balance and risk
        risk_amount = account_balance * 0.01  # 1% risk
        stop_distance = abs(pattern_data['pattern']['entry'] - pattern_data['pattern']['stop'])
        
        if stop_distance > 0:
            # Calculate units to risk exactly 1%
            position_units = int(risk_amount / stop_distance * 10000)
            if position_units < 1:
                position_units = 1
            units = position_units if units > 0 else -position_units
        
        # For live account, log a prominent warning
        if self.environment == "live":
            print("\n" + "!" * 60)
            print("!!! LIVE TRADE EXECUTION - REAL MONEY !!!")
            print("!" * 60)
            logger.warning(f"⚠️  Placing REAL MONEY trade on {instrument}")
        
        return self.place_order(
            instrument=instrument,
            units=units,
            stop_loss=pattern_data['pattern']['stop'],
            take_profit=pattern_data['pattern']['target']
        )
    
    def get_open_trades(self) -> List[Dict]:
        """Get all open trades"""
        url = f"{self.base_url}/accounts/{self.account_id}/openTrades"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('trades', [])
            else:
                logger.error(f"Failed to get open trades: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Exception getting open trades: {e}")
            return []
    
    def close_trade(self, trade_id: str) -> Dict:
        """Close a specific trade"""
        url = f"{self.base_url}/accounts/{self.account_id}/trades/{trade_id}/close"
        try:
            response = requests.put(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Trade {trade_id} closed")
                return response.json()
            else:
                logger.error(f"Failed to close trade: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Exception closing trade: {e}")
            return {}
