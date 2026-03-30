"""
Simon Pullen's strict pattern rules
Based on analysis of 80+ hours of trading sessions
"""

# M-Top and W-Bottom Rules
MW_RULES = {
    'min_candles': 7,
    'max_candles': 30,
    'peak_similarity_tolerance': 0.02,  # 2% price difference maximum
    'neckline_based_on': 'bodies',  # Never use wicks
    'ignore_rogue_wicks': True,  # Can ignore single outlier wicks
    'entry_requires_retest': False,  # Break only, no retest needed
    'default_risk_reward': 1.0,  # 1:1 minimum
    'average_risk_reward': 1.2,  # 1.2:1 average
    'max_sideways_bars': 15,  # Close after 15 bars sideways
    'success_rate': 0.72,  # 72% historical success rate
}

# Head and Shoulders Rules
HS_RULES = {
    'min_candles': 30,
    'max_candles': 120,
    'neckline_based_on': 'bodies',
    'requires_break': True,
    'requires_close': True,
    'requires_retest': True,
    'requires_entry_candle': True,
    'valid_entry_candles': ['pin', 'engulfing', 'tweezer'],
    'stop_loss_options': {
        'conservative': 'behind_head',  # Higher win rate, lower R:R
        'moderate': 'behind_shoulder',
        'aggressive': 'behind_entry'  # Lower win rate, much higher R:R
    },
    'average_rr_conservative': 1.2,
    'average_rr_moderate': 2.1,
    'average_rr_aggressive': 3.1,
    'success_rate_conservative': 0.75,
    'success_rate_aggressive': 0.64,
    'max_sideways_multiplier': 1.0,  # Close after pattern-width sideways
}

# Risk Management Rules
RISK_RULES = {
    'default_risk_percent': 1.0,  # 1% per trade
    'max_risk_percent': 2.0,  # Only with high confluence
    'max_correlation_risk': 2.0,  # Max 2% across correlated pairs
    'correlation_groups': {
        'jpy_pairs': ['USD/JPY', 'EUR/JPY', 'GBP/JPY', 'AUD/JPY', 'CAD/JPY', 'NZD/JPY', 'CHF/JPY'],
        'eur_pairs': ['EUR/USD', 'EUR/GBP', 'EUR/JPY', 'EUR/CHF', 'EUR/AUD', 'EUR/CAD', 'EUR/NZD'],
        'gbp_pairs': ['GBP/USD', 'GBP/JPY', 'GBP/CHF', 'GBP/AUD', 'GBP/CAD', 'GBP/NZD'],
        'aud_pairs': ['AUD/USD', 'AUD/JPY', 'AUD/CHF', 'AUD/CAD', 'AUD/NZD'],
        'usd_pairs': ['USD/JPY', 'USD/CHF', 'USD/CAD', 'EUR/USD', 'GBP/USD', 'AUD/USD', 'NZD/USD']
    },
    'trading_days': {
        'preferred': ['Tuesday', 'Wednesday', 'Thursday'],
        'avoid': ['Monday', 'Friday'],
        'friday_afternoon_avoid': True  # UK time afternoons
    },
    'news_avoid_hours': 2,  # Avoid trading 2 hours before major news
}

# Confluence Factors - How much they increase probability
CONFLUENCE_WEIGHTS = {
    'divergence': 0.09,  # 71% -> 80% (9% increase)
    'inefficient_candle': 0.05,  # Acts as magnet
    'institutional_zone': 0.07,
    'weekly_trendline_alignment': 0.10,  # Never trade against
    'adr_level': {
        0.50: 0.00,  # 99% hit rate - baseline
        0.75: 0.05,  # 79% hit rate - slight edge
        1.00: 0.10,  # 41% hit rate - reversal edge
        1.25: 0.15,  # 14% hit rate - strong reversal edge
    }
}

# ADR (Average Daily Range) probabilities
ADR_PROBABILITIES = {
    0.50: 0.99,  # 50% of ADR: 99% chance of being hit
    0.75: 0.79,  # 75% of ADR: 79% chance
    1.00: 0.41,  # 100% of ADR: 41% chance
    1.25: 0.14,  # 125% of ADR: 14% chance (strong reversal signal)
}

# Timeframes Simon uses
TIMEFRAMES = {
    'mw_patterns': ['1h', '4h', '1d'],  # Never lower than 1h
    'head_shoulders': ['15m', '1h', '4h', '1d'],  # 15m is most active
    'preferred_mw': '1h',
    'preferred_hs': '15m',
    'analysis_order': ['1h', '15m', '4h']  # Morning analysis order
}
