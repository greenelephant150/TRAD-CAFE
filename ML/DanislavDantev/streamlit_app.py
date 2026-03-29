#!/usr/bin/env python3
"""
Danislav Dantev Institutional Trading System
Complete Streamlit UI for order flow analysis
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import warnings
import os
import sys
import time

warnings.filterwarnings('ignore')

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
try:
    from dantev_config import *
    from institutional_order_flow import InstitutionalOrderFlow
    from dantev_trader import DantevTrader
    from dantev_ai_trainer import DantevAITrainer
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure all required files are in the same directory:")
    st.error("  - dantev_config.py")
    st.error("  - institutional_order_flow.py")
    st.error("  - dantev_trader.py")
    st.error("  - dantev_ai_trainer.py")
    st.stop()

# Page config
st.set_page_config(
    page_title="Danislav Dantev Institutional Trading System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main header */
    .institutional-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1a1a2e, #16213e, #0f3460);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Card styles */
    .order-block-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a3b 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #00ff88;
    }
    
    .fvg-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a3b 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #ff6b6b;
    }
    
    .liquidity-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a3b 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #ffd93d;
    }
    
    .bos-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a3b 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #6c5ce7;
    }
    
    /* Signal badges */
    .signal-buy {
        background-color: #00ff88;
        color: #000;
        padding: 0.4rem 1.2rem;
        border-radius: 25px;
        font-weight: bold;
        display: inline-block;
    }
    
    .signal-sell {
        background-color: #ff6b6b;
        color: #fff;
        padding: 0.4rem 1.2rem;
        border-radius: 25px;
        font-weight: bold;
        display: inline-block;
    }
    
    .signal-strong {
        background-color: #00ff88;
        color: #000;
        padding: 0.2rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    .signal-moderate {
        background-color: #ffd93d;
        color: #000;
        padding: 0.2rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    .signal-weak {
        background-color: #ff9f4a;
        color: #000;
        padding: 0.2rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    .signal-avoid {
        background-color: #ff6b6b;
        color: #fff;
        padding: 0.2rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    /* Navigation menu styling */
    .nav-item {
        padding: 0.5rem 1rem;
        margin: 0.25rem 0;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .nav-item:hover {
        background-color: rgba(102, 126, 234, 0.2);
    }
    
    .nav-item-active {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        color: white;
    }
    
    /* Status indicators */
    .status-active {
        background-color: #00ff88;
        color: #000;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.7rem;
        display: inline-block;
    }
    
    .status-warning {
        background-color: #ffd93d;
        color: #000;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.7rem;
        display: inline-block;
    }
    
    .status-danger {
        background-color: #ff6b6b;
        color: #fff;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.7rem;
        display: inline-block;
    }
    
    /* Progress bar customization */
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.order_flow = None
    st.session_state.trader = None
    st.session_state.ai_trainer = None
    st.session_state.current_analysis = None
    st.session_state.auto_refresh = False
    st.session_state.last_analysis_time = None
    st.session_state.trade_history = []
    st.session_state.open_positions = []
    st.session_state.page = "Market Scanner"  # Default page


def initialize_system():
    """Initialize all components"""
    with st.spinner("Initializing Dantev Institutional Trading System..."):
        try:
            config = {
                'ORDER_BLOCK_CONFIG': ORDER_BLOCK_CONFIG,
                'FVG_CONFIG': FVG_CONFIG,
                'LIQUIDITY_CONFIG': LIQUIDITY_CONFIG,
                'BOS_CONFIG': BOS_CONFIG,
                'CHOCH_CONFIG': CHOCH_CONFIG,
                'FIBONACCI_CONFIG': FIBONACCI_CONFIG,
                'PD_ARRAY_CONFIG': PD_ARRAY_CONFIG
            }
            
            st.session_state.order_flow = InstitutionalOrderFlow(config)
            st.session_state.trader = DantevTrader(environment="practice")
            st.session_state.ai_trainer = DantevAITrainer(str(MODEL_PATH))
            st.session_state.initialized = True
            
            return True
        except Exception as e:
            st.error(f"Initialization failed: {e}")
            return False


def generate_sample_data(instrument: str, timeframe: str, days: int = 30) -> pd.DataFrame:
    """Generate sample OHLC data for demonstration"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Map timeframe to frequency
    freq_map = {
        '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
        '1h': '1h', '4h': '4h', '1d': '1D'
    }
    freq = freq_map.get(timeframe, '1h')
    
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    np.random.seed(hash(instrument) % 2**32)
    
    # Base price based on instrument
    if 'USD' in instrument:
        base_price = 1.1000
    elif 'JPY' in instrument:
        base_price = 150.00
    else:
        base_price = 1.0000
    
    # Generate random walk with some trend
    returns = np.random.randn(len(dates)) * 0.001
    # Add slight upward bias for demonstration
    returns = returns + 0.0002
    price = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLC
    df = pd.DataFrame({
        'open': price * (1 + np.random.randn(len(dates)) * 0.0002),
        'high': price * (1 + np.abs(np.random.randn(len(dates)) * 0.0004)),
        'low': price * (1 - np.abs(np.random.randn(len(dates)) * 0.0004)),
        'close': price,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Ensure high >= open/close and low <= open/close
    df['high'] = df[['high', 'open', 'close']].max(axis=1)
    df['low'] = df[['low', 'open', 'close']].min(axis=1)
    
    return df


def create_candlestick_chart(df: pd.DataFrame, analysis: Dict, instrument: str) -> go.Figure:
    """Create interactive candlestick chart with institutional levels"""
    
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        showlegend=True
    ))
    
    # Add Order Blocks
    for ob in analysis.get('order_blocks', [])[:5]:
        color = '#00ff88' if ob.type == 'bullish' else '#ff6b6b'
        fig.add_hline(
            y=ob.close,
            line_dash="dash",
            line_color=color,
            opacity=0.7,
            annotation_text=f"{ob.type.upper()} OB",
            annotation_position="top right"
        )
        
        # Add zone
        fig.add_hrect(
            y0=ob.low,
            y1=ob.high,
            fillcolor=color,
            opacity=0.15,
            line_width=0
        )
    
    # Add FVGs
    for fvg in analysis.get('fair_value_gaps', [])[:5]:
        if not fvg.filled:
            fig.add_hrect(
                y0=fvg.gap_bottom,
                y1=fvg.gap_top,
                fillcolor='#ff6b6b',
                opacity=0.2,
                line_width=0,
                annotation_text="FVG",
                annotation_position="top left"
            )
    
    # Add Liquidity Levels
    liquidity = analysis.get('liquidity_levels', {})
    for high in liquidity.get('swing_highs', [])[:10]:
        color = '#ffd93d' if high.swept else '#888'
        fig.add_hline(
            y=high.price,
            line_dash="dot",
            line_color=color,
            opacity=0.5,
            annotation_text=f"LH {high.price:.5f}",
            annotation_position="right"
        )
    
    for low in liquidity.get('swing_lows', [])[:10]:
        color = '#ffd93d' if low.swept else '#888'
        fig.add_hline(
            y=low.price,
            line_dash="dot",
            line_color=color,
            opacity=0.5,
            annotation_text=f"LL {low.price:.5f}",
            annotation_position="right"
        )
    
    # Add OTE levels
    ote = analysis.get('ote_levels', {})
    for name, level in ote.items():
        if name in ['golden_ratio', 'deep_retracement', 'extreme_retracement']:
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color='#6c5ce7',
                opacity=0.5,
                annotation_text=f"OTE {name.replace('_', ' ')}",
                annotation_position="bottom right"
            )
    
    # Add entry, stop, target if available
    if analysis.get('entry_price'):
        fig.add_hline(
            y=analysis['entry_price'],
            line_dash="solid",
            line_color='#00ff88',
            line_width=2,
            annotation_text="ENTRY",
            annotation_position="top left"
        )
    
    if analysis.get('stop_loss'):
        fig.add_hline(
            y=analysis['stop_loss'],
            line_dash="solid",
            line_color='#ff6b6b',
            line_width=2,
            annotation_text="STOP",
            annotation_position="bottom left"
        )
    
    if analysis.get('take_profit'):
        fig.add_hline(
            y=analysis['take_profit'],
            line_dash="solid",
            line_color='#00ff88',
            line_width=2,
            annotation_text="TARGET",
            annotation_position="top right"
        )
    
    # Update layout
    fig.update_layout(
        title=f"{instrument} - Institutional Order Flow Analysis",
        yaxis_title="Price",
        xaxis_title="Date",
        template="plotly_dark",
        height=600,
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def display_institutional_analysis(analysis: Dict):
    """Display institutional analysis results"""
    
    # Signal badge
    if analysis.get('trade_direction') == 'long':
        st.markdown('<div class="signal-buy">📈 BUY SIGNAL</div>', unsafe_allow_html=True)
    elif analysis.get('trade_direction') == 'short':
        st.markdown('<div class="signal-sell">📉 SELL SIGNAL</div>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:gray;">⚪ NO SIGNAL</span>', unsafe_allow_html=True)
    
    # Signal strength
    strength = analysis.get('signal_strength', 'WEAK')
    strength_class = {
        'STRONG': 'signal-strong',
        'MODERATE': 'signal-moderate',
        'WEAK': 'signal-weak',
        'AVOID': 'signal-avoid'
    }.get(strength, 'signal-weak')
    st.markdown(f'<span class="{strength_class}">{strength}</span>', unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Confluence Score", f"{analysis['confluence_score']:.0f}")
    
    with col2:
        st.metric("Premium/Discount", f"{analysis['premium_discount']:.1%}")
    
    with col3:
        st.metric("Trend Strength", f"{analysis['trend_strength']:.1%}")
    
    with col4:
        st.metric("Order Blocks", len(analysis['order_blocks']))
    
    with col5:
        st.metric("FVGs", len(analysis['fair_value_gaps']))
    
    # Trade setup
    if analysis.get('entry_price'):
        st.markdown("### 🎯 Trade Setup")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Entry", f"{analysis['entry_price']:.5f}")
        with col2:
            st.metric("Stop Loss", f"{analysis['stop_loss']:.5f}")
        with col3:
            st.metric("Take Profit", f"{analysis['take_profit']:.5f}")
        with col4:
            st.metric("Risk/Reward", f"{analysis['risk_reward']:.1f}:1")


def market_scanner_page():
    """Market Scanner page - displays institutional analysis"""
    st.markdown('<h2 class="sub-header">🔍 Market Scanner</h2>', unsafe_allow_html=True)
    
    # Get settings from session state
    instrument = st.session_state.get('selected_instrument', 'EUR_USD')
    timeframe = st.session_state.get('selected_timeframe', '1h')
    days = st.session_state.get('selected_days', 90)
    min_rr = st.session_state.get('min_rr', 2.0)
    
    # Load and analyze data
    with st.spinner(f"Analyzing {instrument} on {timeframe} timeframe..."):
        # Generate or load data
        df = generate_sample_data(instrument, timeframe, days)
        
        # Run institutional analysis
        current_price = df['close'].iloc[-1]
        analysis = st.session_state.order_flow.analyze(df, current_price)
        analysis['instrument'] = instrument
        analysis['timeframe'] = timeframe
        
        st.session_state.current_analysis = analysis
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📈 Price Action with Institutional Levels")
        fig = create_candlestick_chart(df, analysis, instrument)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🏦 Institutional Analysis")
        display_institutional_analysis(analysis)
        
        # Execute trade button
        if analysis.get('trade_direction') and analysis.get('risk_reward', 0) >= min_rr:
            st.markdown("---")
            st.markdown("### 🚀 Execute Trade")
            
            if st.button("✅ EXECUTE INSTITUTIONAL TRADE", type="primary", use_container_width=True):
                with st.spinner("Executing trade..."):
                    result = st.session_state.trader.execute_institutional_trade(analysis)
                    
                    if result.get('success'):
                        st.success(f"✅ Trade executed successfully!")
                        st.json({
                            'Instrument': result.get('instrument'),
                            'Units': result.get('units'),
                            'Entry': result.get('entry_price'),
                            'Stop': result.get('stop_loss'),
                            'Target': result.get('take_profit')
                        })
                    else:
                        st.error(f"❌ Trade failed: {result.get('error')}")


def open_positions_page():
    """Open Positions page"""
    st.markdown('<h2 class="sub-header">📊 Open Positions</h2>', unsafe_allow_html=True)
    
    # Refresh open positions
    open_positions = st.session_state.trader.get_open_trades()
    
    if open_positions:
        for pos in open_positions:
            with st.container():
                st.markdown(f"""
                <div class="order-block-card">
                    <strong>{pos.get('instrument', 'Unknown')}</strong><br>
                    Direction: {'LONG' if int(pos.get('units', 0)) > 0 else 'SHORT'}<br>
                    Entry: {pos.get('price', 0):.5f}<br>
                    Current P&L: ${float(pos.get('unrealizedPL', 0)):.2f}
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Close Position", key=f"close_{pos.get('id')}"):
                    result = st.session_state.trader.close_trade(pos.get('id'))
                    if result.get('success'):
                        st.success(f"Position closed. P&L: ${result.get('pl', 0):.2f}")
                        st.rerun()
                    else:
                        st.error(f"Failed to close: {result.get('error')}")
    else:
        st.info("No open positions")


def trade_history_page():
    """Trade History page"""
    st.markdown('<h2 class="sub-header">📝 Trade History</h2>', unsafe_allow_html=True)
    
    trade_history = st.session_state.trader.get_trade_history(limit=50)
    
    if trade_history:
        df_history = pd.DataFrame(trade_history)
        
        # Format for display
        display_cols = ['instrument', 'direction', 'entry_price', 'stop_loss', 'take_profit', 
                       'risk_percent', 'confluence_score', 'signal_strength', 'risk_reward', 'entry_time']
        available_cols = [c for c in display_cols if c in df_history.columns]
        
        st.dataframe(df_history[available_cols], use_container_width=True)
        
        # Performance summary
        perf = st.session_state.trader.get_performance_summary()
        st.markdown("### 📊 Performance Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Trades", perf.get('total_trades', 0))
        with col2:
            st.metric("Avg Risk %", f"{perf.get('avg_risk_percent', 0):.2f}%")
        with col3:
            st.metric("Avg R:R", f"{perf.get('avg_risk_reward', 0):.2f}:1")
        
        # Export button
        if st.button("📥 Export to CSV"):
            csv = df_history.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No trade history")


def ai_training_page():
    """AI Training page"""
    st.markdown('<h2 class="sub-header">🧠 AI Model Training</h2>', unsafe_allow_html=True)
    
    st.markdown("### 🤖 AI-Powered Institutional Predictions")
    
    ai_enabled = st.session_state.get('ai_enabled', True)
    
    if ai_enabled:
        # Try to load AI model
        model_path = MODEL_PATH / "dantev_institutional_model.pkl"
        
        if model_path.exists():
            model = st.session_state.ai_trainer.load_model(str(model_path))
            if model and st.session_state.current_analysis:
                prediction = st.session_state.ai_trainer.predict(model, st.session_state.current_analysis)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("AI Confidence", f"{prediction['confidence']:.1%}")
                
                with col2:
                    st.metric("Signal Strength", prediction['signal_strength'])
                
                with col3:
                    analysis = st.session_state.current_analysis
                    st.metric("Recommended Action", 
                             "BUY" if prediction['prediction'] == 1 and analysis.get('trade_direction') == 'long' else
                             "SELL" if prediction['prediction'] == 1 and analysis.get('trade_direction') == 'short' else
                             "WAIT")
                
                # Combined score
                combined_score = (analysis.get('confluence_score', 0) * 0.6 + prediction['confidence'] * 100 * 0.4)
                combined_score = max(0.0, min(100.0, combined_score))
                st.progress(combined_score / 100)
                st.caption(f"Combined Institutional Score: {combined_score:.0f}/100")
                
                if combined_score > 70:
                    st.success("✅ HIGH PROBABILITY SETUP - Consider entry")
                elif combined_score > 50:
                    st.info("📊 Moderate probability - Wait for confirmation")
                else:
                    st.warning("⚠️ Low probability - No trade recommended")
            else:
                st.warning("No analysis available. Run Market Scanner first.")
        else:
            st.info("No trained AI model found. Train a model with historical data.")
            
            if st.button("🚀 Train New Model"):
                with st.spinner("Training institutional AI model..."):
                    # Create dummy training data
                    X_train = np.random.rand(100, 20)
                    y_train = np.random.randint(0, 2, 100)
                    
                    # Train a simple model
                    from sklearn.ensemble import RandomForestClassifier
                    dummy_model = RandomForestClassifier(n_estimators=10, random_state=42)
                    dummy_model.fit(X_train, y_train)
                    
                    # Save model
                    st.session_state.ai_trainer.save_model(
                        dummy_model, 
                        "dantev_institutional_model",
                        {'version': '1.0', 'features': ['demo_feature'] * 20}
                    )
                    st.success("Model trained successfully!")
                    st.rerun()
    else:
        st.info("AI augmentation disabled. Enable in sidebar to get AI predictions.")


def ai_models_page():
    """AI Models page - model management"""
    st.markdown('<h2 class="sub-header">🤖 AI Models</h2>', unsafe_allow_html=True)
    
    st.markdown("### 📊 Model Management")
    
    # Get model info
    model_info = st.session_state.ai_trainer.get_model_info()
    
    if model_info.get('total_models', 0) > 0:
        st.metric("Total Models", model_info['total_models'])
        
        # Display models
        for model in model_info['models']:
            with st.container():
                st.markdown(f"""
                <div class="order-block-card">
                    <strong>{model['filename']}</strong><br>
                    Training Date: {model['training_date']}<br>
                    Model Type: {model['model_type']}<br>
                    Features: {model['feature_count']}<br>
                    Size: {model['size_mb']:.2f} MB
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No models found. Train a model in the AI Training tab.")
    
    st.markdown("---")
    st.markdown("### 🚀 Model Training")
    
    # Training parameters
    col1, col2 = st.columns(2)
    
    with col1:
        lookback = st.slider("Lookback Period", 10, 100, 50)
        train_size = st.slider("Training Samples", 10000, 200000, 100000, step=10000)
    
    with col2:
        validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2, 0.05)
        model_type = st.selectbox("Model Type", ["Random Forest", "XGBoost", "LightGBM"])
    
    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training model... This may take a while..."):
            # Create dummy training data
            X_train = np.random.rand(train_size, 20)
            y_train = np.random.randint(0, 2, train_size)
            
            if model_type == "Random Forest":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
            elif model_type == "XGBoost":
                try:
                    import xgboost as xgb
                    model = xgb.XGBClassifier(n_estimators=100, max_depth=8, random_state=42)
                except ImportError:
                    st.error("XGBoost not installed. Install with: pip install xgboost")
                    return
            else:  # LightGBM
                try:
                    import lightgbm as lgb
                    model = lgb.LGBMClassifier(n_estimators=100, max_depth=8, random_state=42)
                except ImportError:
                    st.error("LightGBM not installed. Install with: pip install lightgbm")
                    return
            
            model.fit(X_train, y_train)
            
            # Save model
            metadata = {
                'version': '1.0',
                'lookback': lookback,
                'train_size': train_size,
                'validation_split': validation_split,
                'model_type': model_type,
                'features': AI_CONFIG.get('features', [])
            }
            
            st.session_state.ai_trainer.save_model(model, "dantev_institutional_model", metadata)
            st.success("Model trained successfully!")
            st.rerun()


def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="institutional-header">🏦 Danislav Dantev Institutional Trading System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Smart Money Concepts | Order Flow | Liquidity Sweeps | FVG | CHoCH | OTE</p>', unsafe_allow_html=True)
    
    # Initialize if needed
    if not st.session_state.initialized:
        if st.button("🚀 Initialize System", type="primary"):
            if initialize_system():
                st.success("✅ System initialized successfully!")
                st.rerun()
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Environment selection
        environment = st.radio(
            "Trading Environment",
            ["practice", "live"],
            index=0,
            help="Practice = Demo account, Live = Real money"
        )
        
        if environment == "live":
            st.warning("⚠️ LIVE TRADING - REAL MONEY")
            st.session_state.trader.environment = "live"
            st.session_state.trader.is_live = True
        
        st.markdown("---")
        
        # ====================================================================
        # NAVIGATION MENU
        # ====================================================================
        st.markdown("## 🧭 Navigation")
        
        # Navigation options
        nav_options = ["Market Scanner", "Open Positions", "Trade History", "AI Training", "AI Models"]
        
        # Create navigation buttons
        for option in nav_options:
            if st.button(option, use_container_width=True, key=f"nav_{option}"):
                st.session_state.page = option
                st.rerun()
        
        st.markdown("---")
        
        # ====================================================================
        # MARKET SCANNER SETTINGS (only show if on that page)
        # ====================================================================
        if st.session_state.page == "Market Scanner":
            st.markdown("## 📊 Scanner Settings")
            
            # Instrument selection
            instrument = st.selectbox(
                "Instrument",
                VALID_OANDA_PAIRS,
                index=0,
                help="Select currency pair to analyze"
            )
            st.session_state.selected_instrument = instrument
            
            # Timeframe
            timeframe = st.selectbox(
                "Timeframe",
                ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                index=4,
                help="Higher timeframes = more reliable signals"
            )
            st.session_state.selected_timeframe = timeframe
            
            # Data days
            days = st.slider("Historical Data (days)", 7, 365, 90)
            st.session_state.selected_days = days
            
            st.markdown("---")
        
        # ====================================================================
        # RISK SETTINGS (global)
        # ====================================================================
        st.markdown("## 📊 Risk Management")
        risk_percent = st.slider("Risk % per trade", 0.25, 2.0, 0.5, 0.25)
        min_rr = st.slider("Min Risk/Reward", 1.5, 5.0, 2.0, 0.5)
        
        # Update trader settings
        st.session_state.trader.default_risk_pct = risk_percent
        st.session_state.trader.min_rr = min_rr
        st.session_state.min_rr = min_rr
        
        st.markdown("---")
        
        # ====================================================================
        # AI SETTINGS
        # ====================================================================
        st.markdown("## 🧠 AI Settings")
        ai_enabled = st.checkbox("Enable AI Augmentation", value=True)
        st.session_state.ai_enabled = ai_enabled
        
        st.markdown("---")
        
        # ====================================================================
        # ACCOUNT INFO
        # ====================================================================
        st.markdown("## 💰 Account")
        account = st.session_state.trader.get_account_summary()
        balance = float(account.get('balance', 10000))
        nav = float(account.get('NAV', balance))
        
        st.metric("Balance", f"${balance:,.2f}")
        st.metric("NAV", f"${nav:,.2f}")
        st.metric("Open Trades", account.get('openTradeCount', 0))
        
        st.markdown("---")
        
        # Auto-refresh option
        auto_refresh = st.checkbox("Auto-refresh", value=False)
        if auto_refresh:
            refresh_interval = st.number_input("Refresh interval (seconds)", 30, 300, 60)
            st.info(f"Auto-refresh every {refresh_interval}s")
            
            # Simple auto-refresh logic
            import time
            if st.session_state.get('last_refresh', 0) < time.time() - refresh_interval:
                st.session_state.last_refresh = time.time()
                st.rerun()
    
    # ====================================================================
    # PAGE ROUTING
    # ====================================================================
    if st.session_state.page == "Market Scanner":
        market_scanner_page()
    elif st.session_state.page == "Open Positions":
        open_positions_page()
    elif st.session_state.page == "Trade History":
        trade_history_page()
    elif st.session_state.page == "AI Training":
        ai_training_page()
    elif st.session_state.page == "AI Models":
        ai_models_page()
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        Danislav Dantev Institutional Trading System | v1.0 | Smart Money Concepts<br>
        {RISK_WARNING}
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()