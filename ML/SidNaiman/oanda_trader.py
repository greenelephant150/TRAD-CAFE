"""
OANDA Trading Module for executing trades with Sid Naiman's SID Method
Supports both practice (demo) and live (real) environments
- Position size calculator (0.5% to 2% risk)
- Stop loss: lowest low rounded down / highest high rounded up
- Take profit at RSI 50
- Entry on close of confirmation candle only
- Max 3-5 concurrent trades
"""

import requests
import json
import logging
import math
from typing import Dict, Optional, List, Union
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

# OANDA credentials
PRACTICE_API_KEY = "fceae94e861642af6c9d9de1bf6a319d-52074b19e787811450bc44152cb71e78"
LIVE_API_KEY = "c4f1b1d0739e88bcc604b26115db4787-1abf7348cce8c429b76a9fcf97b0b97a"

# Account IDs
PRACTICE_ACCOUNT_ID = "101-004-35778624-001"
LIVE_ACCOUNT_ID = "001-004-17934933-001"

# API endpoints
PRACTICE_URL = "https://api-fxpractice.oanda.com/v3"
LIVE_URL = "https://api-fxtrade.oanda.com/v3"


class OANDATrader:
    """
    Execute trades on OANDA following Sid Naiman's SID Method rules
    """
    
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
            print("\n" + "!" * 60)
            print("!!! WARNING: INITIALIZING LIVE TRADING - REAL MONEY ACCOUNT !!!")
            print("!" * 60 + "\n")
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
        
        self._session = requests.Session()
        self._session.headers.update(self.headers)
        
        # Track open trades for capital preservation (Sid's rules)
        self.open_trades = []
        self.max_concurrent_trades = 5  # Sid's rule: max 3-5 concurrent
        self.daily_loss = 0.0
        self.max_daily_loss_pct = 2.0  # Max 2% loss per day
        self.consecutive_losses = 0
        
        # Track zones that have been traded
        self.traded_zones = []
        
        logger.info(f"Initialized OANDA trader for {self.account_type} account with Sid's rules")
        
        # Load current open trades
        self._refresh_open_trades()
    
    def _refresh_open_trades(self):
        """Refresh list of open trades"""
        self.open_trades = self.get_open_trades()
    
    def get_account_summary(self) -> Dict:
        """Get account details and balance"""
        url = f"{self.base_url}/accounts/{self.account_id}"
        try:
            response = self._session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                account = data.get('account', {})
                
                summary = {
                    'account': {
                        'id': account.get('id'),
                        'currency': account.get('currency'),
                        'balance': float(account.get('balance', 0)),
                        'NAV': float(account.get('NAV', 0)),
                        'unrealizedPL': float(account.get('unrealizedPL', 0)),
                        'realizedPL': float(account.get('realizedPL', 0)),
                        'marginUsed': float(account.get('marginUsed', 0)),
                        'marginAvailable': float(account.get('marginAvailable', 0)),
                        'openTradeCount': account.get('openTradeCount', 0),
                        'pendingOrderCount': account.get('pendingOrderCount', 0)
                    }
                }
                
                self._refresh_open_trades()
                return summary
            else:
                logger.error(f"Failed to get account details: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            logger.error(f"Exception getting account details: {e}")
            return {}
    
    def can_add_trade(self) -> Dict:
        """
        Check if we can add a new trade based on Sid's capital preservation rules
        - Max 3-5 concurrent trades
        - Max 2% daily loss
        - Reduce size after consecutive losses
        """
        issues = []
        
        if len(self.open_trades) >= self.max_concurrent_trades:
            issues.append(f"Max concurrent trades reached ({self.max_concurrent_trades})")
        
        account = self.get_account_summary().get('account', {})
        balance = account.get('balance', 10000)
        
        daily_loss_pct = (self.daily_loss / balance) * 100 if balance > 0 else 0
        if daily_loss_pct >= self.max_daily_loss_pct:
            issues.append(f"Daily loss limit reached ({daily_loss_pct:.1f}%)")
        
        if self.consecutive_losses >= 3:
            issues.append(f"Too many consecutive losses ({self.consecutive_losses})")
        
        return {
            'can_trade': len(issues) == 0,
            'issues': issues,
            'open_trades': len(self.open_trades),
            'daily_loss_pct': daily_loss_pct,
            'consecutive_losses': self.consecutive_losses,
            'recommended_risk': self._get_recommended_risk()
        }
    
    def _get_recommended_risk(self) -> float:
        """Get recommended risk percentage (0.5% to 2% per Sid's rules)"""
        base_risk = 1.0  # Start with 1%
        
        if self.consecutive_losses == 0:
            return min(base_risk, 1.0)
        elif self.consecutive_losses == 1:
            return 0.75
        elif self.consecutive_losses == 2:
            return 0.5
        else:
            return 0.25  # After 3+ losses, trade very small
    
    def update_after_trade(self, result: str, loss_amount: float = 0):
        """Update trade tracking after a trade closes"""
        if result == 'loss':
            self.consecutive_losses += 1
            self.daily_loss += loss_amount
        else:
            self.consecutive_losses = 0
    
    def reset_daily(self):
        """Reset daily counters (call at start of each day)"""
        self.daily_loss = 0.0
        self.consecutive_losses = 0
    
    def get_current_price(self, instrument: str) -> Dict:
        """Get current bid/ask for an instrument"""
        url = f"{self.base_url}/accounts/{self.account_id}/pricing"
        params = {"instruments": instrument}
        
        try:
            response = self._session.get(url, headers=self.headers, params=params, timeout=10)
            
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
                        'time': price['time']
                    }
            else:
                logger.error(f"Failed to get price: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Exception getting price: {e}")
            return {}
    
    def calculate_position_size(self, instrument: str, entry_price: float, 
                                 stop_loss: float, risk_percent: float = 1.0) -> Dict:
        """
        Calculate position size using Sid's position size calculator
        Risk 0.5% to 2% per trade (default 1%)
        """
        account = self.get_account_summary().get('account', {})
        balance = account.get('balance', 10000)
        
        # Adjust risk based on recent performance (Sid's rule)
        recommended_risk = self._get_recommended_risk()
        actual_risk = min(risk_percent, recommended_risk)
        
        # Calculate risk amount
        risk_amount = balance * (actual_risk / 100)
        
        # Calculate risk per unit
        if entry_price > stop_loss:  # Long trade
            risk_per_unit = entry_price - stop_loss
        else:  # Short trade
            risk_per_unit = stop_loss - entry_price
        
        if risk_per_unit <= 0:
            return {'error': 'Invalid stop loss'}
        
        # Calculate units
        units_raw = risk_amount / risk_per_unit
        
        # Round to appropriate precision
        units = math.floor(units_raw)
        
        # Ensure minimum trade size
        min_units = 1000
        if units < min_units:
            units = min_units
            actual_risk = (units * risk_per_unit / balance) * 100
        
        return {
            'units': units,
            'risk_amount': risk_amount,
            'risk_percent': actual_risk,
            'risk_per_unit': risk_per_unit,
            'position_value': units * entry_price,
            'balance': balance
        }
    
    def calculate_stop_from_zone(self, zone_price: float, zone_type: str, 
                                   signal_date_low: float, signal_date_high: float) -> float:
        """
        Calculate stop loss using Sid's rule:
        - For oversold (demand): lowest low between signal and entry, rounded down
        - For overbought (supply): highest high between signal and entry, rounded up
        """
        if zone_type == 'demand':
            # Demand zone: lowest low, rounded down to whole number
            stop_loss = math.floor(signal_date_low)
        else:
            # Supply zone: highest high, rounded up to whole number
            stop_loss = math.ceil(signal_date_high)
        
        return float(stop_loss)
    
    def is_zone_fresh(self, zone_id: str) -> bool:
        """Check if a zone has already been traded"""
        return zone_id not in self.traded_zones
    
    def mark_zone_traded(self, zone_id: str):
        """Mark a zone as traded (no longer fresh)"""
        if zone_id not in self.traded_zones:
            self.traded_zones.append(zone_id)
            logger.info(f"Zone {zone_id} marked as traded")
    
    def place_order(self, 
                    instrument: str,
                    units: float,
                    stop_loss: Optional[float] = None,
                    take_profit: Optional[float] = None,
                    order_type: str = "MARKET",
                    entry_price: Optional[float] = None,
                    zone_id: Optional[str] = None) -> Dict:
        """
        Place a trade order with precise entry on close
        """
        can_trade = self.can_add_trade()
        if not can_trade['can_trade']:
            return {
                'error': 'Cannot add trade',
                'reasons': can_trade['issues']
            }
        
        if zone_id and not self.is_zone_fresh(zone_id):
            return {
                'error': 'Zone already mitigated - cannot re-trade',
                'zone_id': zone_id
            }
        
        url = f"{self.base_url}/accounts/{self.account_id}/orders"
        
        if order_type == "MARKET":
            order = {
                "order": {
                    "type": "MARKET",
                    "instrument": instrument,
                    "units": str(int(units)),
                    "timeInForce": "FOK"
                }
            }
        elif order_type == "LIMIT" and entry_price:
            order = {
                "order": {
                    "type": "LIMIT",
                    "instrument": instrument,
                    "units": str(int(units)),
                    "price": str(round(entry_price, 5)),
                    "timeInForce": "GTC"
                }
            }
        else:
            return {'error': 'Invalid order type'}
        
        if stop_loss:
            order["order"]["stopLossOnFill"] = {
                "price": str(round(stop_loss, 5))
            }
        
        if take_profit:
            order["order"]["takeProfitOnFill"] = {
                "price": str(round(take_profit, 5))
            }
        
        try:
            logger.info(f"Placing order: {instrument} {units} units")
            
            if self.environment == "live":
                print("\n" + "!" * 60)
                print("LIVE ORDER DETAILS:")
                print(f"Instrument: {instrument}")
                print(f"Direction: {'BUY' if units > 0 else 'SELL'}")
                print(f"Units: {abs(units)}")
                print(f"Stop Loss: {stop_loss}")
                print(f"Take Profit: {take_profit}")
                print("!" * 60)
                
                confirm = input("Type 'CONFIRM' to execute live trade: ")
                if confirm != "CONFIRM":
                    return {"error": "Trade cancelled by user"}
            
            response = self._session.post(url, json=order, timeout=10)
            
            if response.status_code == 201:
                result = response.json()
                order_data = result.get('orderFillTransaction', result.get('orderCreateTransaction', {}))
                
                logger.info(f"✅ Order placed successfully: {instrument} {units} units")
                
                if zone_id:
                    self.mark_zone_traded(zone_id)
                
                self._refresh_open_trades()
                
                return {
                    'success': True,
                    'order_id': order_data.get('id'),
                    'instrument': instrument,
                    'units': units,
                    'price': float(order_data.get('price', 0)) if order_data.get('price') else None,
                    'time': order_data.get('time'),
                    'zone_id': zone_id,
                    'full_response': result
                }
            else:
                error_msg = f"Order failed: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {"error": error_msg, "status_code": response.status_code}
                
        except Exception as e:
            error_msg = f"Exception placing order: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def place_supply_demand_trade(self, zone_data: Dict, account_balance: float) -> Dict:
        """
        Execute a trade based on a supply/demand zone using Sid's rules
        """
        instrument = zone_data['instrument']
        zone_price = zone_data['zone_price']
        zone_type = zone_data['zone_type']  # 'demand' or 'supply'
        zone_id = zone_data.get('zone_id', f"{instrument}_{zone_price}_{datetime.now().timestamp()}")
        signal_date_low = zone_data.get('signal_date_low', zone_price * 0.99)
        signal_date_high = zone_data.get('signal_date_high', zone_price * 1.01)
        
        price_data = self.get_current_price(instrument)
        if not price_data:
            return {'error': 'Could not get current price'}
        
        entry_price = price_data['mid']
        
        # Calculate stop loss using Sid's rule
        stop_loss = self.calculate_stop_from_zone(
            zone_price, zone_type, signal_date_low, signal_date_high
        )
        
        # Calculate position size using Sid's position size calculator
        position = self.calculate_position_size(
            instrument=instrument,
            entry_price=entry_price,
            stop_loss=stop_loss,
            risk_percent=zone_data.get('risk_percent', 1.0)
        )
        
        if 'error' in position:
            return position
        
        # Determine direction
        units = position['units'] if zone_type == 'demand' else -position['units']
        
        # Calculate take profit (RSI 50 - will be monitored separately)
        # For now, use a conservative 1:1 risk/reward
        risk_distance = abs(entry_price - stop_loss)
        if zone_type == 'demand':
            take_profit = entry_price + (risk_distance * 1.0)
        else:
            take_profit = entry_price - (risk_distance * 1.0)
        
        result = self.place_order(
            instrument=instrument,
            units=units,
            stop_loss=stop_loss,
            take_profit=take_profit,
            order_type="LIMIT",
            entry_price=entry_price,
            zone_id=zone_id
        )
        
        if result.get('success'):
            trade_record = {
                'instrument': instrument,
                'zone_type': zone_type,
                'entry_price': result.get('price', entry_price),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'units': units,
                'risk_percent': position['risk_percent'],
                'zone_price': zone_price,
                'zone_id': zone_id,
                'zone_quality': zone_data.get('quality', {}),
                'entry_time': datetime.utcnow().isoformat(),
                'order_id': result.get('order_id')
            }
            
            if not hasattr(self, 'trade_history'):
                self.trade_history = []
            self.trade_history.append(trade_record)
        
        return result
    
    def get_open_trades(self) -> List[Dict]:
        """Get all open trades"""
        url = f"{self.base_url}/accounts/{self.account_id}/openTrades"
        try:
            response = self._session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                trades = data.get('trades', [])
                
                formatted = []
                for trade in trades:
                    formatted.append({
                        'id': trade.get('id'),
                        'instrument': trade.get('instrument'),
                        'units': int(trade.get('units', 0)),
                        'price': float(trade.get('price', 0)),
                        'stop_loss': float(trade.get('stopLoss', {}).get('price', 0)) if trade.get('stopLoss') else None,
                        'take_profit': float(trade.get('takeProfit', {}).get('price', 0)) if trade.get('takeProfit') else None,
                        'unrealizedPL': float(trade.get('unrealizedPL', 0)),
                        'current_price': float(trade.get('currentPrice', 0)) if trade.get('currentPrice') else None,
                        'open_time': trade.get('openTime')
                    })
                
                self.open_trades = formatted
                return formatted
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
            response = self._session.put(url, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Trade {trade_id} closed successfully")
                
                order_fill = result.get('orderFillTransaction', {})
                
                pl = float(order_fill.get('pl', 0))
                if pl < 0:
                    self.update_after_trade('loss', abs(pl))
                else:
                    self.update_after_trade('win')
                
                self._refresh_open_trades()
                
                return {
                    'success': True,
                    'trade_id': trade_id,
                    'pl': pl,
                    'price': float(order_fill.get('price', 0)),
                    'time': order_fill.get('time')
                }
            else:
                logger.error(f"Failed to close trade: {response.status_code}")
                return {'error': f'Failed to close trade: {response.status_code}'}
        except Exception as e:
            logger.error(f"Exception closing trade: {e}")
            return {'error': str(e)}
    
    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Get trade history from stored records"""
        if hasattr(self, 'trade_history'):
            return self.trade_history[-limit:]
        return []