SIMON PULLEN TRADING SYSTEM - QUICK START
==========================================

This implementation follows Simon Pullen's complete methodology from 80+ hours of training sessions.

CORE PRINCIPLES:
----------------
1. M-Tops & W-Bottoms (Quick Win Strategy)
   - 7-30 candles in pattern
   - Entry on break of neckline (no retest)
   - 1:1 risk/reward minimum
   - 72% historical success rate

2. Head & Shoulders (Primary Strategy)
   - 30-120 candles in pattern
   - Must have: Break + Close + Retest + Entry Candle
   - Entry candles: pin, engulfing, tweezer
   - 3 stop loss options with different R:R profiles
   - Conservative: 75% win rate, 1.2:1 R:R
   - Aggressive: 64% win rate, 3.1:1 R:R

3. Confluence Factors (Increase Probability)
   - Divergence: 71% → 80% win rate (40% profit increase)
   - Inefficient candles: act as magnets
   - Institutional Value Zones
   - ADR levels: 50% (99% hit), 75% (79%), 100% (41%), 125% (14%)

4. Risk Management
   - 1% per trade standard, 2% max with high confluence
   - Max 2% across correlated pairs (JPY, EUR, GBP groups)
   - Best days: Tue, Wed, Thu (avoid Mon/Fri)
   - Never hold through high-impact news

QUICK START:
------------
1. Install requirements:
   pip install -r requirements.txt

2. Run backtest:
   python scripts/backtest_patterns.py --instrument EUR/USD --timeframe 1h --days 365

3. Analyze trade log:
   python scripts/analyze_trade_log.py --input backtest_results.json --plot

4. Validate rules:
   python scripts/validate_rules.py

KEY FILES:
----------
src/core/mw_pattern.py           # M&W detection with all rules
src/core/head_shoulders.py       # H&S detection with all rules
src/confluence/divergence.py      # RSI divergence detection
src/risk/position_sizer.py        # 1% rule with correlation
src/execution/entry_rules.py      # Pattern-specific entry rules
src/backtesting/bar_replay.py     # Bar replay backtesting

SIMON'S WIN RATES:
------------------
- M&W (unfiltered): 68-72%
- M&W (filtered): 75-82%
- H&S conservative: 75%
- H&S aggressive: 64%
- With divergence: +9% to win rate

For complete documentation, see docs/ folder.
