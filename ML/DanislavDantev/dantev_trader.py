#!/usr/bin/env python3
"""
Danislav Dantev Trading System
Implements institutional order flow trading rules
- Never trade against order flow
- Wait for liquidity sweep before entry
- Enter on retest of order block
- Minimum 2:1 risk/reward
- Scale into positions at OTE levels
"""

import requests
import json
import logging
import math
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime, timedelta
import time

from dantev_config import (
    PRACTICE_API_KEY, LIVE_API_KEY, PRACTICE_ACCOUNT_ID, LIVE_ACCOUNT_ID,
    PRACTICE_URL, LIVE_URL, VALID_OANDA_PAIRS, RISK_CONFIG
)

logger = logging.getLogger(__name__)


class DantevTrader:
    """
    Danislav Dantev's Institutional Trading System
    Executes trades based on order flow analysis
    """
    
    def __init__(self, environment: str = "practice"):
        self.environment = environment
        
        if environment == "live":
            self.api_key = LIVE_API_KEY
            self.account_id = LIVE_ACCOUNT_ID
            self.base_url = LIVE_URL
            self.account_type = "LIVE"
            self.is_live = True
        else:
            self.api_key = PRACTICE_API_KEY
            self.account_id = PRACTICE_ACCOUNT_ID
            self.base_url = PRACTICE_URL
            self.account_type = "PRACTICE"
            self.is_live = False
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        self._session = requests.Session()
        self._session.headers.update(self.headers)
        
        # Risk tracking
        self.open_positions = []
        self.trade_history = []
        self.daily_pnl = 0.0
        self.daily_start_balance = 0.0
        self.consecutive_losses = 0
        
        # Risk limits
        self.max_daily_loss_pct = RISK_CONFIG.get('max_daily_risk', 2.0)
        self.max_consecutive_losses = RISK_CONFIG.get('max_consecutive_losses', 3)
        self.default_risk_pct = RISK_CONFIG.get('default_risk_percent', 0.5)
        self.max_risk_pct = RISK_CONFIG.get('max_risk_percent', 1.0)
        self.min_rr = RISK_CONFIG.get('risk_reward_min', 2.0)
        
        # Track today's date for daily reset
        self.today = datetime.now().date()
        
        logger.info(f"DantevTrader initialized for {self.account_type} account")
        logger.info(f"  Default risk: {self.default_risk_pct}%")
        logger.info(f"  Max daily loss: {self.max_daily_loss_pct}%")
        logger.info(f"  Min R:R: {self.min_rr}:1")
    
    def get_pip_value(self, instrument: str) -> float:
        """Get pip value for instrument"""
        if 'JPY' in instrument:
            return 0.01
        elif 'XAU' in instrument or 'XAG' in instrument:
            return 0.01
        return 0.0001
    
    def calculate_position_size(self, instrument: str, entry_price: float,
                                 stop_loss: float, risk_percent: float = None) -> Dict:
        """
        Calculate position size based on risk
        Danislav: 0.5% default, 1% max with high confluence
        """
        if risk_percent is None:
            risk_percent = self.default_risk_pct
        
        # Reduce risk after losses
        if self.consecutive_losses >= 2:
            risk_percent *= 0.5
        
        # Cap at max risk
        risk_percent = min(risk_percent, self.max_risk_pct)
        
        account = self.get_account_summary()
        balance = float(account.get('balance', 10000))
        
        # Update daily tracking
        if self.daily_start_balance == 0:
            self.daily_start_balance = balance
        
        # Check daily loss limit
        daily_loss_pct = (self.daily_pnl / self.daily_start_balance) * 100 if self.daily_start_balance > 0 else 0
        if daily_loss_pct <= -self.max_daily_loss_pct:
            return {'error': f'Daily loss limit reached: {abs(daily_loss_pct):.1f}%'}
        
        risk_amount = balance * (risk_percent / 100)
        
        if entry_price > stop_loss:  # Long
            risk_per_unit = entry_price - stop_loss
        else:  # Short
            risk_per_unit = stop_loss - entry_price
        
        if risk_per_unit <= 0:
            return {'error': 'Invalid stop loss'}
        
        units = int(risk_amount / risk_per_unit)
        
        # Round to appropriate lot size
        min_units = 1000
        if units < min_units:
            units = min_units
            risk_percent = (units * risk_per_unit / balance) * 100
        
        return {
            'units': units,
            'risk_amount': risk_amount,
            'risk_percent': risk_percent,
            'balance': balance,
            'daily_loss_pct': daily_loss_pct
        }
    
    def get_account_summary(self) -> Dict:
        """Get account details"""
        url = f"{self.base_url}/accounts/{self.account_id}"
        try:
            response = self._session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('account', {})
        except Exception as e:
            logger.error(f"Error getting account: {e}")
        return {}
    
    def get_current_price(self, instrument: str) -> Dict:
        """Get current price"""
        url = f"{self.base_url}/accounts/{self.account_id}/pricing"
        params = {"instruments": instrument}
        
        try:
            response = self._session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                prices = data.get('prices', [])
                if prices:
                    price = prices[0]
                    return {
                        'bid': float(price['bids'][0]['price']),
                        'ask': float(price['asks'][0]['price']),
                        'mid': (float(price['bids'][0]['price']) + float(price['asks'][0]['price'])) / 2,
                        'spread': float(price['asks'][0]['price']) - float(price['bids'][0]['price']),
                        'time': price['time']
                    }
        except Exception as e:
            logger.error(f"Error getting price: {e}")
        return {}
    
    def get_open_trades(self) -> List[Dict]:
        """Get all open trades"""
        url = f"{self.base_url}/accounts/{self.account_id}/openTrades"
        try:
            response = self._session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                trades = data.get('trades', [])
                self.open_positions = trades
                return trades
        except Exception as e:
            logger.error(f"Error getting open trades: {e}")
        return []
    
    def place_order(self, instrument: str, units: int, 
                    stop_loss: float, take_profit: float,
                    entry_price: Optional[float] = None,
                    comment: str = "") -> Dict:
        """
        Place institutional order
        Danislav: Entry on retest of order block
        """
        url = f"{self.base_url}/accounts/{self.account_id}/orders"
        
        order = {
            "order": {
                "type": "LIMIT" if entry_price else "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "GTC",
                "stopLossOnFill": {
                    "price": str(round(stop_loss, 5)),
                    "timeInForce": "GTC"
                },
                "takeProfitOnFill": {
                    "price": str(round(take_profit, 5)),
                    "timeInForce": "GTC"
                }
            }
        }
        
        if entry_price:
            order["order"]["price"] = str(round(entry_price, 5))
        
        # Live confirmation
        if self.is_live:
            print("\n" + "!" * 70)
            print("⚠️  LIVE TRADE EXECUTION - REAL MONEY ⚠️")
            print("!" * 70)
            print(f"Instrument: {instrument}")
            print(f"Direction: {'BUY' if units > 0 else 'SELL'}")
            print(f"Units: {abs(units)}")
            print(f"Entry: {entry_price or 'MARKET'}")
            print(f"Stop Loss: {stop_loss}")
            print(f"Take Profit: {take_profit}")
            print(f"Risk/Reward: {abs((take_profit - entry_price) / (entry_price - stop_loss)):.2f}:1" if entry_price else "N/A")
            print(f"Comment: {comment}")
            print("!" * 70)
            
            confirm = input("Type 'CONFIRM' to execute live trade: ")
            if confirm != "CONFIRM":
                return {"error": "Trade cancelled by user"}
        
        try:
            response = self._session.post(url, json=order, timeout=10)
            
            if response.status_code in [200, 201]:
                result = response.json()
                order_fill = result.get('orderFillTransaction', result.get('orderCreateTransaction', {}))
                
                logger.info(f"✅ Order placed: {instrument} {'BUY' if units > 0 else 'SELL'} {abs(units)} units")
                
                return {
                    'success': True,
                    'order_id': order_fill.get('id'),
                    'instrument': instrument,
                    'units': units,
                    'entry_price': float(order_fill.get('price', entry_price or 0)),
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'time': order_fill.get('time'),
                    'comment': comment
                }
            else:
                error_msg = f"Order failed: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {"error": error_msg}
                
        except Exception as e:
            error_msg = f"Exception placing order: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def execute_institutional_trade(self, analysis: Dict) -> Dict:
        """
        Execute trade based on institutional order flow analysis
        """
        if not analysis.get('trade_direction'):
            return {'error': 'No valid trade setup', 'setup_valid': False}
        
        trade_direction = analysis['trade_direction']
        entry_price = analysis['entry_price']
        stop_loss = analysis['stop_loss']
        take_profit = analysis['take_profit']
        risk_reward = analysis['risk_reward']
        confluence_score = analysis['confluence_score']
        signal_strength = analysis['signal_strength']
        
        # Check risk/reward minimum
        if risk_reward < self.min_rr:
            return {
                'error': f'Risk/Reward too low: {risk_reward:.1f}:1 < {self.min_rr}:1',
                'setup_valid': False
            }
        
        # Check confluence (minimum 40 for weak signal, 60 for strong)
        min_confluence = 40 if signal_strength == "WEAK" else 60
        if confluence_score < min_confluence:
            return {
                'error': f'Confluence too low: {confluence_score} < {min_confluence}',
                'setup_valid': False
            }
        
        # Check daily loss limit
        account = self.get_account_summary()
        balance = float(account.get('balance', 10000))
        
        # Reset daily if new day
        current_date = datetime.now().date()
        if current_date != self.today:
            self.daily_pnl = 0.0
            self.daily_start_balance = balance
            self.today = current_date
            logger.info(f"Daily reset - New day: {current_date}")
        
        daily_loss_pct = (self.daily_pnl / self.daily_start_balance) * 100 if self.daily_start_balance > 0 else 0
        if daily_loss_pct <= -self.max_daily_loss_pct:
            return {
                'error': f'Daily loss limit reached: {abs(daily_loss_pct):.1f}%',
                'setup_valid': False
            }
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return {
                'error': f'Max consecutive losses reached: {self.consecutive_losses}',
                'setup_valid': False
            }
        
        # Determine risk percentage based on confluence
        if confluence_score >= 80 and signal_strength == "STRONG":
            risk_percent = 1.0
        elif confluence_score >= 65:
            risk_percent = 0.75
        else:
            risk_percent = 0.5
        
        # Calculate position size
        instrument = analysis.get('instrument', 'EUR_USD')
        position = self.calculate_position_size(
            instrument, entry_price, stop_loss, risk_percent
        )
        
        if 'error' in position:
            return {'error': position['error'], 'setup_valid': False}
        
        units = position['units'] if trade_direction == 'long' else -position['units']
        
        # Create comment with setup details
        comment = f"Dantev:{signal_strength}:CS{confluence_score}:RR{risk_reward:.1f}"
        
        # Place order
        result = self.place_order(
            instrument=instrument,
            units=units,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_price=entry_price,
            comment=comment
        )
        
        if result.get('success'):
            # Record trade
            trade_record = {
                'instrument': instrument,
                'direction': trade_direction,
                'entry_price': result.get('entry_price', entry_price),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'units': units,
                'risk_percent': position['risk_percent'],
                'confluence_score': confluence_score,
                'signal_strength': signal_strength,
                'risk_reward': risk_reward,
                'entry_time': datetime.now().isoformat(),
                'order_id': result.get('order_id'),
                'comment': comment
            }
            self.trade_history.append(trade_record)
        
        return result
    
    def close_trade(self, trade_id: str) -> Dict:
        """Close a specific trade"""
        url = f"{self.base_url}/accounts/{self.account_id}/trades/{trade_id}/close"
        try:
            response = self._session.put(url, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                order_fill = result.get('orderFillTransaction', {})
                pl = float(order_fill.get('pl', 0))
                
                # Update loss tracking
                if pl < 0:
                    self.consecutive_losses += 1
                    self.daily_pnl += pl
                else:
                    self.consecutive_losses = 0
                    self.daily_pnl += pl
                
                logger.info(f"Trade {trade_id} closed. P&L: ${pl:.2f}")
                
                return {
                    'success': True,
                    'trade_id': trade_id,
                    'pl': pl,
                    'price': float(order_fill.get('price', 0)),
                    'time': order_fill.get('time')
                }
            else:
                return {'error': f'Failed to close trade: {response.status_code}'}
        except Exception as e:
            return {'error': str(e)}
    
    def modify_stop_loss(self, trade_id: str, new_stop: float) -> Dict:
        """Modify stop loss on an existing trade"""
        url = f"{self.base_url}/accounts/{self.account_id}/trades/{trade_id}/orders"
        
        modifications = {
            'stopLoss': {
                'price': str(round(new_stop, 5)),
                'timeInForce': 'GTC'
            }
        }
        
        try:
            response = self._session.put(url, json=modifications, timeout=10)
            if response.status_code in [200, 201]:
                return {'success': True}
            else:
                return {'error': f'Failed to modify stop: {response.status_code}'}
        except Exception as e:
            return {'error': str(e)}
    
    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Get trade history"""
        return self.trade_history[-limit:]
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        if not self.trade_history:
            return {'total_trades': 0}
        
        trades_df = []
        for t in self.trade_history:
            trades_df.append({
                'direction': t['direction'],
                'risk_percent': t['risk_percent'],
                'confluence_score': t['confluence_score'],
                'signal_strength': t['signal_strength'],
                'risk_reward': t['risk_reward']
            })
        
        import pandas as pd
        df = pd.DataFrame(trades_df)
        
        return {
            'total_trades': len(df),
            'avg_risk_percent': df['risk_percent'].mean(),
            'avg_confluence_score': df['confluence_score'].mean(),
            'signal_strength_distribution': df['signal_strength'].value_counts().to_dict(),
            'avg_risk_reward': df['risk_reward'].mean()
        }