"""
Trade logging for Simon Pullen backtesting
Matches his detailed trade log methodology
"""

import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class TradeRecord:
    """Complete trade record following Simon's format"""
    entry_time: Any  # Can be datetime or int
    instrument: str
    pattern_type: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    stop_loss_type: str
    position_size_units: float
    risk_percent: float
    exit_time: Optional[Any] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_pips: Optional[float] = None
    pnl_percent: Optional[float] = None
    bars_held: Optional[int] = None
    max_favorable: Optional[float] = None
    max_adverse: Optional[float] = None
    confluence_factors: List[str] = None
    notes: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary, handling datetime conversion"""
        d = asdict(self)
        # Convert datetime to string if it's a datetime object
        if hasattr(self.entry_time, 'isoformat'):
            d['entry_time'] = self.entry_time.isoformat()
        if hasattr(self.exit_time, 'isoformat') and self.exit_time is not None:
            d['exit_time'] = self.exit_time.isoformat()
        return d


class TradeLogger:
    """
    Detailed trade logger matching Simon's methodology
    Used for post-trade analysis and strategy refinement
    """
    
    def __init__(self):
        self.trades: List[TradeRecord] = []
        self.current_trade = None
        
    def log_entry(self, position: Dict, balance: float):
        """Log trade entry"""
        signal = position['entry_signal']
        
        record = TradeRecord(
            entry_time=position['entry_time'],
            instrument=signal.instrument,
            pattern_type=signal.pattern_type,
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            stop_loss_type=signal.stop_loss_type,
            position_size_units=position['position_size'].units,
            risk_percent=position['position_size'].risk_percent,
            confluence_factors=self._get_confluence_factors(signal)
        )
        
        self.current_trade = record
        
    def log_exit(self, position: Dict, exit_status: Dict, balance: float):
        """Log trade exit"""
        if self.current_trade is None:
            return
            
        self.current_trade.exit_time = position.get('exit_time', datetime.now())
        self.current_trade.exit_price = exit_status['exit_price']
        self.current_trade.exit_reason = exit_status['exit_reason']
        self.current_trade.pnl_pips = exit_status['pnl_pips']
        self.current_trade.pnl_percent = exit_status['pnl_pct']
        self.current_trade.bars_held = exit_status['bars_held']
        self.current_trade.max_favorable = exit_status.get('max_favorable')
        self.current_trade.max_adverse = exit_status.get('max_adverse')
        
        self.trades.append(self.current_trade)
        self.current_trade = None
        
    def _get_confluence_factors(self, signal) -> List[str]:
        """Extract confluence factors from signal"""
        factors = []
        
        if hasattr(signal, 'confluence_score') and signal.confluence_score > 0.1:
            # This would be populated by confluence scorer
            pass
            
        # Add pattern-specific factors
        if signal.pattern_type in ['M', 'W']:
            factors.append('impulsive_move')
        else:
            factors.append('break_close_retest')
            
        return factors
    
    def get_all_trades(self) -> List[Dict]:
        """Get all trades as dictionaries"""
        return [t.to_dict() for t in self.trades]
    
    def export_to_csv(self, filename: str):
        """Export trades to CSV"""
        df = pd.DataFrame([t.to_dict() for t in self.trades])
        df.to_csv(filename, index=False)
        
    def export_to_json(self, filename: str):
        """Export trades to JSON"""
        with open(filename, 'w') as f:
            json.dump([t.to_dict() for t in self.trades], f, indent=2)
    
    def analyze_by_pattern(self) -> Dict:
        """Analyze performance by pattern type"""
        if not self.trades:
            return {}
            
        result = {}
        for t in self.trades:
            ptype = t.pattern_type
            if ptype not in result:
                result[ptype] = {
                    'total': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pnl': 0,
                    'avg_pnl': 0,
                    'avg_bars': 0
                }
            
            result[ptype]['total'] += 1
            if t.pnl_percent and t.pnl_percent > 0:
                result[ptype]['wins'] += 1
            else:
                result[ptype]['losses'] += 1
                
            result[ptype]['total_pnl'] += t.pnl_percent or 0
            result[ptype]['avg_bars'] += t.bars_held or 0
            
        # Calculate averages
        for ptype in result:
            if result[ptype]['total'] > 0:
                result[ptype]['avg_pnl'] = result[ptype]['total_pnl'] / result[ptype]['total']
                result[ptype]['win_rate'] = result[ptype]['wins'] / result[ptype]['total'] * 100
                result[ptype]['avg_bars'] /= result[ptype]['total']
                
        return result
    
    def analyze_by_stop_type(self) -> Dict:
        """Analyze performance by stop loss type"""
        if not self.trades:
            return {}
            
        result = {}
        for t in self.trades:
            stype = t.stop_loss_type
            if stype not in result:
                result[stype] = {
                    'total': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pnl': 0,
                    'avg_risk_reward': 0
                }
            
            result[stype]['total'] += 1
            if t.pnl_percent and t.pnl_percent > 0:
                result[stype]['wins'] += 1
            else:
                result[stype]['losses'] += 1
                
            result[stype]['total_pnl'] += t.pnl_percent or 0
            
            # Calculate risk/reward
            if t.stop_loss and t.take_profit:
                if t.direction == 'long':
                    risk = t.entry_price - t.stop_loss
                    reward = t.take_profit - t.entry_price
                else:
                    risk = t.stop_loss - t.entry_price
                    reward = t.entry_price - t.take_profit
                    
                if risk > 0:
                    result[stype]['avg_risk_reward'] += reward / risk
        
        # Calculate averages
        for stype in result:
            if result[stype]['total'] > 0:
                result[stype]['avg_pnl'] = result[stype]['total_pnl'] / result[stype]['total']
                result[stype]['win_rate'] = result[stype]['wins'] / result[stype]['total'] * 100
                result[stype]['avg_risk_reward'] /= result[stype]['total']
                
        return result
    
    def analyze_by_day(self) -> Dict:
        """Analyze performance by day of week (Tue/Wed/Thu best)"""
        if not self.trades:
            return {}
            
        result = {}
        for t in self.trades:
            if t.entry_time and hasattr(t.entry_time, 'strftime'):
                day = t.entry_time.strftime('%A')
                if day not in result:
                    result[day] = {
                        'total': 0,
                        'wins': 0,
                        'losses': 0,
                        'total_pnl': 0
                    }
                
                result[day]['total'] += 1
                if t.pnl_percent and t.pnl_percent > 0:
                    result[day]['wins'] += 1
                else:
                    result[day]['losses'] += 1
                    
                result[day]['total_pnl'] += t.pnl_percent or 0
        
        # Calculate win rates
        for day in result:
            if result[day]['total'] > 0:
                result[day]['win_rate'] = result[day]['wins'] / result[day]['total'] * 100
                
        return result
