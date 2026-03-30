"""
Colorful logging utility for Simon Pullen trading system
Adds emojis and colors to terminal output
"""

import logging
import sys
from datetime import datetime
from typing import Optional

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
    # Background colors
    BG_BLUE = '\033[44m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_RED = '\033[41m'
    BG_PURPLE = '\033[45m'
    BG_CYAN = '\033[46m'

# Emoji mappings
EMOJIS = {
    'rocket': '🚀',
    'chart': '📊',
    'money': '💰',
    'fire': '🔥',
    'warning': '⚠️',
    'check': '✅',
    'cross': '❌',
    'bull': '🐂',
    'bear': '🐻',
    'pattern': '🔍',
    'trade': '📈',
    'profit': '💵',
    'loss': '💸',
    'stop': '🛑',
    'entry': '🎯',
    'exit': '🚪',
    'time': '⏰',
    'calendar': '📅',
    'cpu': '💻',
    'gpu': '🎮',
    'brain': '🧠',
    'magic': '✨',
    'star': '⭐',
    'trophy': '🏆',
    'info': 'ℹ️',
    'debug': '🔧',
    'error': '❌',
    'warning_emoji': '⚠️',
    'data': '💾',
    'save': '💾',
    'list': '📋',
    'settings': '⚙️'
}


class ColorLogger:
    """Colorful logger with emojis and formatting"""
    
    def __init__(self, name: str, verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time = datetime.now()
        
    def _format(self, level: str, message: str, emoji: str = '') -> str:
        """Format message with colors and emoji"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        if level == 'INFO':
            color = Colors.GREEN
        elif level == 'DEBUG':
            color = Colors.CYAN
        elif level == 'WARNING':
            color = Colors.YELLOW
        elif level == 'ERROR':
            color = Colors.RED
        elif level == 'HEADER':
            color = Colors.HEADER
        elif level == 'BOLD':
            color = Colors.BOLD
        else:
            color = Colors.END
            
        emoji_str = f"{EMOJIS.get(emoji, '')} " if emoji else ''
        return f"{Colors.BLUE}[{timestamp}]{Colors.END} {color}{emoji_str}{message}{Colors.END}"
    
    def header(self, message: str, emoji: str = 'star'):
        """Print a header"""
        line = "=" * 60
        print(f"\n{Colors.BOLD}{Colors.BG_PURPLE}{line}{Colors.END}")
        print(self._format('HEADER', f" {message}", emoji))
        print(f"{Colors.BOLD}{Colors.BG_PURPLE}{line}{Colors.END}\n")
    
    def section(self, message: str, emoji: str = 'info'):
        """Print a section header"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}▶ {message}{Colors.END}")
    
    def info(self, message: str, emoji: str = 'info'):
        """Print info message"""
        if self.verbose:
            print(self._format('INFO', message, emoji))
    
    def debug(self, message: str, emoji: str = 'debug'):
        """Print debug message"""
        if self.verbose:
            print(self._format('DEBUG', message, emoji))
    
    def warning(self, message: str, emoji: str = 'warning'):
        """Print warning message"""
        print(self._format('WARNING', message, emoji))
    
    def error(self, message: str, emoji: str = 'error'):
        """Print error message"""
        print(self._format('ERROR', message, emoji))
    
    def success(self, message: str, emoji: str = 'check'):
        """Print success message"""
        print(self._format('INFO', f"✅ {message}", ''))
    
    def trade_entry(self, instrument: str, pattern: str, price: float, stop: float, tp: float):
        """Print trade entry with formatting"""
        print(f"\n{Colors.BOLD}{Colors.BG_GREEN}🚀 NEW TRADE ENTRY 🚀{Colors.END}")
        print(f"{Colors.GREEN}├─ Instrument: {Colors.BOLD}{instrument}{Colors.END}")
        print(f"{Colors.GREEN}├─ Pattern: {Colors.BOLD}{pattern}{Colors.END}")
        print(f"{Colors.GREEN}├─ Entry: {Colors.BOLD}{price:.5f}{Colors.END}")
        print(f"{Colors.GREEN}├─ Stop Loss: {Colors.BOLD}{stop:.5f}{Colors.END}")
        print(f"{Colors.GREEN}├─ Take Profit: {Colors.BOLD}{tp:.5f}{Colors.END}")
        risk_reward = abs((tp-price)/(stop-price)) if stop != price else 0
        print(f"{Colors.GREEN}└─ Risk/Reward: {Colors.BOLD}{risk_reward:.2f}{Colors.END}\n")
    
    def trade_exit(self, instrument: str, pnl: float, reason: str, bars_held: int):
        """Print trade exit with formatting"""
        color = Colors.GREEN if pnl > 0 else Colors.RED
        emoji = 'profit' if pnl > 0 else 'loss'
        
        print(f"\n{color}{'📈 EXIT TRADE ' if pnl>0 else '📉 EXIT TRADE '}{Colors.END}")
        print(f"{color}├─ Instrument: {Colors.BOLD}{instrument}{Colors.END}")
        print(f"{color}├─ P&L: {Colors.BOLD}{pnl:+.2f}%{Colors.END}")
        print(f"{color}├─ Reason: {Colors.BOLD}{reason}{Colors.END}")
        print(f"{color}└─ Bars Held: {Colors.BOLD}{bars_held}{Colors.END}\n")
    
    def pattern_found(self, pattern_type: str, candle_count: int, confidence: float):
        """Print pattern detection"""
        print(f"{Colors.CYAN}  {EMOJIS['pattern']} Found {pattern_type}: {candle_count} candles, confidence: {confidence:.1%}{Colors.END}")
    
    def device_info(self, device: str, name: str, device_type: str):
        """Print device information"""
        emoji = 'gpu' if device_type == 'gpu' else 'cpu'
        print(f"{Colors.YELLOW}{EMOJIS[emoji]} Device: {Colors.BOLD}{name}{Colors.END}")
    
    def progress(self, current: int, total: int, message: str = ""):
        """Print progress bar"""
        percent = current / total * 100
        bar_length = 30
        filled = int(bar_length * current // total)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\r{Colors.CYAN}{bar} {percent:.1f}% {message}{Colors.END}", end='')
        if current == total:
            print()
    
    def summary(self, results: dict):
        """Print summary with formatting"""
        print(f"\n{Colors.BOLD}{Colors.BG_CYAN}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BG_CYAN}📊 BACKTEST SUMMARY 📊{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BG_CYAN}{'='*60}{Colors.END}\n")
        
        # Main stats
        print(f"{Colors.BOLD}Total Trades:{Colors.END} {results.get('total_trades', 0)}")
        
        win_rate = results.get('win_rate', 0)
        if win_rate >= 70:
            wr_color = Colors.GREEN
        elif win_rate >= 50:
            wr_color = Colors.YELLOW
        else:
            wr_color = Colors.RED
            
        print(f"{Colors.BOLD}Winners:{Colors.END} {results.get('winners', 0)} {EMOJIS['profit']}")
        print(f"{Colors.BOLD}Losers:{Colors.END} {results.get('losers', 0)} {EMOJIS['loss']}")
        print(f"{Colors.BOLD}Win Rate:{Colors.END} {wr_color}{win_rate:.1f}%{Colors.END}")
        
        total_return = results.get('return_pct', 0)
        if total_return > 20:
            ret_color = Colors.GREEN
        elif total_return > 0:
            ret_color = Colors.YELLOW
        else:
            ret_color = Colors.RED
            
        print(f"{Colors.BOLD}Total Return:{Colors.END} {ret_color}{total_return:+.2f}%{Colors.END}")
        print(f"{Colors.BOLD}Final Balance:{Colors.END} ${results.get('final_balance', 0):,.2f}")
        
        # Pattern breakdown
        if results.get('by_pattern'):
            print(f"\n{Colors.BOLD}{EMOJIS['pattern']} Pattern Breakdown:{Colors.END}")
            for ptype, stats in results['by_pattern'].items():
                pattern_emoji = 'bull' if 'W' in ptype or 'inverted' in ptype else 'bear'
                pnl_color = Colors.GREEN if stats['pnl'] > 0 else Colors.RED
                total = stats['wins'] + stats['losses']
                win_pct = (stats['wins'] / total * 100) if total > 0 else 0
                print(f"  {EMOJIS[pattern_emoji]} {ptype}: {stats['wins']}W/{stats['losses']}L "
                      f"({win_pct:.0f}%) "
                      f"{pnl_color}P&L: {stats['pnl']:+.2f}%{Colors.END}")
        
        # Performance rating
        print(f"\n{Colors.BOLD}⭐ Performance Rating:{Colors.END} ", end='')
        if win_rate >= 70 and total_return > 20:
            print(f"{Colors.BOLD}{Colors.GREEN}EXCELLENT {EMOJIS['trophy']}{Colors.END}")
        elif win_rate >= 60 and total_return > 10:
            print(f"{Colors.BOLD}{Colors.CYAN}GOOD {EMOJIS['chart']}{Colors.END}")
        elif win_rate >= 50 and total_return > 0:
            print(f"{Colors.BOLD}{Colors.YELLOW}FAIR {EMOJIS['warning']}{Colors.END}")
        else:
            print(f"{Colors.BOLD}{Colors.RED}POOR {EMOJIS['cross']}{Colors.END}")
        
        print(f"\n{Colors.BLUE}{'='*60}{Colors.END}\n")


# Global logger instance
_logger = None


def get_logger(verbose: bool = True) -> ColorLogger:
    """Get or create the global logger instance"""
    global _logger
    if _logger is None:
        _logger = ColorLogger("SimonSystem", verbose)
    return _logger
