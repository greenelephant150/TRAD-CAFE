"""
Sid Naiman's SID Method - Strategy Rules
Based on meticulous review of 50+ hours of trading sessions

Core Rules:
1. RSI Signal (oversold <30, overbought >70)
2. MACD Alignment (both pointing same direction)
3. Earnings Check (no trades within 14 calendar days before earnings)
4. Entry at current price when aligned
5. Stop Loss: 
   - For oversold: lowest low between signal and entry, rounded DOWN
   - For overbought: highest high between signal and entry, rounded UP
6. Take Profit: RSI 50 (set alert)
7. Exit after 2 consecutive reversal days
8. Position Size Calculator: 0.5% to 2% risk per trade
9. Daily charts only
"""

# Sid's RSI thresholds
RSI_RULES = {
    'oversold': 30,
    'overbought': 70,
    'target': 50,
    'rsi_period': 14
}

# Sid's MACD settings (standard)
MACD_RULES = {
    'fast_period': 12,
    'slow_period': 26,
    'signal_period': 9
}

# Sid's entry rules
ENTRY_RULES = {
    'requires_alignment': True,  # RSI and MACD must point same direction
    'prefer_cross': True,        # MACD cross gives stronger confirmation
    'require_earnings_buffer': True,
    'earnings_days_buffer': 14,  # No trades within 14 days before earnings
    'daily_chart_only': True,    # Only trade on daily charts
    'no_trading_earnings_day': True
}

# Sid's exit rules
EXIT_RULES = {
    'primary_exit': 'rsi_50',     # Exit when RSI reaches 50
    'secondary_exit': 'stop_loss', # Stop loss hit
    'reversal_exit': 2,            # Exit after 2 consecutive reversal days
    'stop_loss_type': {
        'oversold': 'lowest_low_rounded_down',
        'overbought': 'highest_high_rounded_up'
    }
}

# Sid's risk management rules
RISK_RULES = {
    'default_risk_percent': 1.0,      # 1% per trade standard
    'min_risk_percent': 0.5,          # 0.5% minimum
    'max_risk_percent': 2.0,           # 2% maximum
    'max_concurrent_trades': 5,        # Max 3-5 trades at once
    'max_daily_loss_percent': 2.0,     # Stop trading after 2% loss in a day
    'reduce_risk_after_losses': True,  # Reduce position size after consecutive losses
    'risk_reduction_factors': {
        1: 0.75,   # After 1 loss, trade at 0.75%
        2: 0.5,    # After 2 losses, trade at 0.5%
        3: 0.25    # After 3+ losses, trade at 0.25%
    },
    'position_sizing_method': 'fixed_percentage'
}

# Sid's alternative take profit (if you can't watch RSI 50)
ALTERNATIVE_TP_RULES = {
    'stocks_under_200': 4,    # Add 4 points for stocks under $200
    'stocks_over_200': 8,      # Add 8 points for stocks over $200
    'use_moving_average': True  # Can also use 50/200 day MA as secondary TP
}

# Timeframes Sid uses
TIMEFRAMES = {
    'trading': ['1d'],        # Only daily charts for trading
    'analysis': ['1d'],       # Only daily charts for analysis
    'preferred': '1d'         # Preferred timeframe
}

# Trading days (Sid doesn't have strong day preferences like Simon)
TRADING_DAYS = {
    'can_trade': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
    'avoid': []  # No specific days to avoid
}

# Confluence factors (Sid uses price patterns for confirmation only)
PATTERN_WEIGHTS = {
    'double_bottom': 0.1,     # Adds 10% confidence for oversold trades
    'double_top': 0.1,        # Adds 10% confidence for overbought trades
    'inverted_head_shoulders': 0.15,
    'head_shoulders': 0.15
}