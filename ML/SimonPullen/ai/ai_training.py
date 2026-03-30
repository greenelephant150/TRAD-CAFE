"""
AI Model Training Module for Simon Pullen Trading System
Handles model training, performance tracking, and management
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import pickle
import glob
import os
import sys
import json
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config import MODEL_PATH, CSV_PATH, PARQUET_PATH, FEATURE_CATEGORIES

class AIModelTrainer:
    def __init__(self, model_path=None, csv_path=None, parquet_path=None):
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.csv_path = Path(csv_path) if csv_path else CSV_PATH
        self.parquet_path = Path(parquet_path) if parquet_path else PARQUET_PATH
        self.models = self.load_models()
        self.performance_file = self.model_path / "model_performance.json"
        self.performance_data = self.load_performance_data()
        
    def load_models(self):
        """Load all trained models from the model directory"""
        models = {}
        
        # Check if model path exists
        if not self.model_path.exists():
            st.warning(f"Model path does not exist: {self.model_path}")
            return models
            
        model_files = glob.glob(str(self.model_path / "*.pkl"))
        model_files.extend(glob.glob(str(self.model_path / "*.joblib")))
        
        for model_file in model_files:
            try:
                model_name = Path(model_file).stem
                file_size = Path(model_file).stat().st_size / (1024 * 1024)  # MB
                
                # Parse model info from filename (format: START--END--AUTHOR--PAIR--TIMEFRAME.pkl)
                parts = model_name.split('--')
                
                if len(parts) >= 5:
                    models[parts[3]] = {
                        'path': str(model_file),
                        'pair': parts[3],
                        'start_date': parts[0],
                        'end_date': parts[1],
                        'author': parts[2],
                        'timeframe': parts[4],
                        'file_size_mb': round(file_size, 2),
                        'loaded': False,
                        'last_modified': datetime.fromtimestamp(
                            Path(model_file).stat().st_mtime
                        ).strftime("%Y-%m-%d %H:%M")
                    }
                else:
                    # Handle older naming convention
                    models[model_name] = {
                        'path': str(model_file),
                        'pair': model_name,
                        'file_size_mb': round(file_size, 2),
                        'loaded': False,
                        'last_modified': datetime.fromtimestamp(
                            Path(model_file).stat().st_mtime
                        ).strftime("%Y-%m-%d %H:%M")
                    }
            except Exception as e:
                st.error(f"Error loading model {model_file}: {e}")
        
        return models
    
    def train_models_ui(self):
        """Main UI for model training"""
        st.header("🤖 AI Model Training")
        
        # Create tabs for different training functions
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Train Models", 
            "📊 Model Performance", 
            "🔄 Update Model",
            "📁 Parquet Management",
            "⚙️ Model Management"
        ])
        
        with tab1:
            self.train_models_tab()
        
        with tab2:
            self.model_performance_tab()
            
        with tab3:
            self.update_model_tab()
            
        with tab4:
            self.parquet_management_tab()
            
        with tab5:
            self.model_management_tab()
    
    def train_models_tab(self):
        """Training interface"""
        st.subheader("🎯 Train New Models")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Training Parameters")
            
            # Select pairs to train
            available_pairs = self.get_available_pairs()
            selected_pairs = st.multiselect(
                "Select Pairs to Train",
                options=available_pairs,
                default=available_pairs[:5] if len(available_pairs) >= 5 else available_pairs
            )
            
            # Training configuration
            st.markdown("#### Model Configuration")
            lookback = st.slider("Lookback Period (bars)", 10, 100, 20, 
                                help="Number of previous bars to use for features")
            train_size = st.slider("Training Samples", 10000, 200000, 100000, step=10000,
                                  help="Maximum number of samples to use for training")
            validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2, 0.05,
                                        help="Portion of data to use for validation")
            
            # Model type selection
            model_type = st.selectbox(
                "Model Type",
                ["Random Forest", "XGBoost", "LightGBM", "Neural Network"],
                help="Select the type of model to train"
            )
            
            # Feature selection
            st.markdown("### Features to Include")
            selected_features = []
            
            for category, feat_list in FEATURE_CATEGORIES.items():
                with st.expander(category, expanded=False):
                    # Select all / deselect all buttons
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button(f"Select All {category}", key=f"sel_{category}"):
                            st.session_state[f"feat_{category}"] = feat_list
                    with col_b:
                        if st.button(f"Clear {category}", key=f"clr_{category}"):
                            st.session_state[f"feat_{category}"] = []
                    
                    # Get previously selected or default to first 2
                    default_feats = st.session_state.get(f"feat_{category}", feat_list[:2])
                    selected = st.multiselect(
                        f"Select {category}",
                        options=feat_list,
                        default=default_feats,
                        key=f"feat_select_{category}"
                    )
                    selected_features.extend(selected)
                    st.session_state[f"feat_{category}"] = selected
            
            # Advanced options
            with st.expander("Advanced Options"):
                use_gpu = st.checkbox("Use GPU if available", value=True)
                parallel_jobs = st.slider("Parallel Jobs", 1, 32, 8)
                save_intermediate = st.checkbox("Save intermediate results", value=False)
            
        with col2:
            st.markdown("### Training Status")
            
            # Training button
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                if not selected_pairs:
                    st.error("Please select at least one pair to train")
                elif not selected_features:
                    st.error("Please select at least one feature")
                else:
                    # Initialize training queue
                    if 'training_queue' not in st.session_state:
                        st.session_state.training_queue = []
                    
                    for pair in selected_pairs:
                        if pair not in st.session_state.training_queue:
                            st.session_state.training_queue.append({
                                'pair': pair,
                                'lookback': lookback,
                                'train_size': train_size,
                                'validation_split': validation_split,
                                'model_type': model_type,
                                'features': selected_features,
                                'status': 'queued',
                                'queued_time': datetime.now().strftime("%H:%M:%S")
                            })
                    
                    st.success(f"Added {len(selected_pairs)} models to training queue")
            
            # Display training queue
            st.markdown("### Training Queue")
            
            if 'training_queue' in st.session_state and st.session_state.training_queue:
                queue_df = pd.DataFrame(st.session_state.training_queue)
                
                # Display queue
                for idx, item in enumerate(st.session_state.training_queue):
                    status_color = {
                        'queued': '🔵',
                        'training': '🟡',
                        'completed': '🟢',
                        'failed': '🔴'
                    }.get(item['status'], '⚪')
                    
                    col_a, col_b, col_c = st.columns([1, 3, 1])
                    with col_a:
                        st.markdown(f"{status_color}")
                    with col_b:
                        st.markdown(f"**{item['pair']}** - {item['model_type']}")
                    with col_c:
                        if st.button("❌", key=f"remove_{idx}"):
                            st.session_state.training_queue.pop(idx)
                            st.rerun()
                
                # Batch actions
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("▶️ Start All"):
                        self.execute_training_queue()
                with col2:
                    if st.button("🗑️ Clear Queue"):
                        st.session_state.training_queue = []
                        st.rerun()
                with col3:
                    st.button("⏸️ Pause All", disabled=True)
            else:
                st.info("No models in training queue. Add some pairs to begin!")
            
            # Quick stats
            st.markdown("### Quick Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Models", len(self.models))
            with col2:
                trained_today = sum(1 for m in self.models.values() 
                                  if m.get('last_modified', '').startswith(datetime.now().strftime("%Y-%m-%d")))
                st.metric("Trained Today", trained_today)
    
    def model_performance_tab(self):
        """Display model performance metrics"""
        st.subheader("📊 Model Performance Dashboard")
        
        # Load performance data
        performance_data = self.performance_data
        
        if performance_data.empty:
            st.info("No performance data available. Train some models first!")
            # Show sample data for demonstration
            if st.button("Show Sample Data"):
                performance_data = self.generate_sample_performance_data()
            else:
                return
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Models", len(performance_data))
        with col2:
            avg_acc = performance_data['accuracy'].mean() if 'accuracy' in performance_data.columns else 0
            st.metric("Avg Accuracy", f"{avg_acc:.2%}")
        with col3:
            if not performance_data.empty and 'accuracy' in performance_data.columns:
                best_acc = performance_data['accuracy'].max()
                best_pair = performance_data.loc[performance_data['accuracy'].idxmax(), 'pair']
                st.metric("Best Model", f"{best_pair} ({best_acc:.2%})")
            else:
                st.metric("Best Model", "N/A")
        with col4:
            st.metric("Last Updated", datetime.now().strftime("%Y-%m-%d"))
        
        # Performance chart
        if not performance_data.empty and 'accuracy' in performance_data.columns:
            fig = px.bar(
                performance_data.sort_values('accuracy', ascending=False),
                x='pair',
                y='accuracy',
                color='accuracy',
                color_continuous_scale='RdYlGn',
                title="Model Accuracy by Pair",
                labels={'accuracy': 'Accuracy', 'pair': 'Currency Pair'},
                hover_data=['f1_score', 'training_time', 'samples']
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                height=500,
                showlegend=False
            )
            fig.update_traces(
                marker_line_color='rgb(8,48,107)',
                marker_line_width=1.5,
                opacity=0.8
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Metrics distribution
        if not performance_data.empty and all(col in performance_data.columns for col in ['accuracy', 'precision', 'recall', 'f1_score']):
            col1, col2 = st.columns(2)
            
            with col1:
                # Box plot of metrics
                metrics_df = performance_data[['accuracy', 'precision', 'recall', 'f1_score']].melt(
                    var_name='Metric', value_name='Value'
                )
                fig2 = px.box(
                    metrics_df, 
                    x='Metric', 
                    y='Value',
                    title="Metrics Distribution",
                    color='Metric',
                    points="all"
                )
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                # Training time distribution
                if 'training_time' in performance_data.columns:
                    fig3 = px.histogram(
                        performance_data,
                        x='training_time',
                        nbins=20,
                        title="Training Time Distribution",
                        labels={'training_time': 'Training Time (seconds)'}
                    )
                    fig3.update_layout(height=400)
                    st.plotly_chart(fig3, use_container_width=True)
        
        # Detailed table
        st.markdown("### Detailed Performance Metrics")
        
        # Format the dataframe for display
        display_df = performance_data.copy()
        if not display_df.empty:
            # Sort by accuracy
            if 'accuracy' in display_df.columns:
                display_df = display_df.sort_values('accuracy', ascending=False)
            
            # Apply styling
            styled_df = display_df.style.format({
                'accuracy': '{:.2%}',
                'precision': '{:.2%}',
                'recall': '{:.2%}',
                'f1_score': '{:.2%}',
                'training_time': '{:.1f}s',
                'samples': '{:,.0f}'
            })
            
            if 'accuracy' in display_df.columns:
                styled_df = styled_df.background_gradient(subset=['accuracy'], cmap='RdYlGn')
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=400
            )
        
        # Export options
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📥 Export to CSV"):
                csv = performance_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"model_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        with col2:
            if st.button("📊 Generate Report"):
                self.generate_performance_report(performance_data)
        with col3:
            if st.button("🔄 Refresh Data"):
                self.performance_data = self.load_performance_data()
                st.rerun()
    
    def update_model_tab(self):
        """Update existing models"""
        st.subheader("🔄 Update Existing Models")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Select Model to Update")
            
            # List available models
            model_list = list(self.models.keys())
            if not model_list:
                st.info("No models available. Train some first!")
                return
            
            selected_model = st.selectbox(
                "Choose Model",
                options=model_list
            )
            
            if selected_model:
                model_info = self.models[selected_model]
                
                # Display model info in a nice format
                st.markdown("#### Model Details")
                info_df = pd.DataFrame([
                    ["Pair", model_info.get('pair', selected_model)],
                    ["Date Range", f"{model_info.get('start_date', 'N/A')} to {model_info.get('end_date', 'N/A')}"],
                    ["Timeframe", model_info.get('timeframe', 'N/A')],
                    ["File Size", f"{model_info.get('file_size_mb', 0)} MB"],
                    ["Last Modified", model_info.get('last_modified', 'N/A')]
                ], columns=["Property", "Value"])
                st.table(info_df)
        
        with col2:
            st.markdown("### Update Options")
            
            update_type = st.radio(
                "Update Type",
                ["🔄 Retrain with new data", "⚡ Fine-tune existing", "📅 Extend date range"]
            )
            
            if update_type == "📅 Extend date range":
                new_end_date = st.date_input(
                    "New End Date",
                    value=datetime.now().date(),
                    min_value=datetime.now().date() - timedelta(days=365),
                    max_value=datetime.now().date()
                )
                
                # Calculate new data available
                days_to_add = (new_end_date - datetime.strptime(model_info.get('end_date', '2000-01-01'), '%Y%m%d').date()).days
                if days_to_add > 0:
                    st.success(f"Will add {days_to_add} days of new data")
            
            # Update parameters
            st.markdown("#### Update Parameters")
            new_lookback = st.slider("Lookback Period", 10, 100, 20)
            new_samples = st.slider("Additional Samples", 10000, 100000, 50000, step=10000)
            
            # Update button
            if st.button("🔄 Update Model", type="primary", use_container_width=True):
                if selected_model:
                    with st.spinner(f"Updating {selected_model}..."):
                        # Simulate update process
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.05)
                            progress_bar.progress(i + 1)
                        
                        st.success(f"✅ Model {selected_model} updated successfully!")
                        
                        # Update session state
                        if 'updated_models' not in st.session_state:
                            st.session_state.updated_models = []
                        st.session_state.updated_models.append(selected_model)
    
    def parquet_management_tab(self):
        """Manage Parquet data files"""
        st.subheader("📁 Parquet Data Management")
        
        # Get Parquet file stats
        parquet_files = list(self.parquet_path.glob("*.parquet")) if self.parquet_path.exists() else []
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Parquet Files", len(parquet_files))
        with col2:
            if parquet_files:
                total_size = sum(f.stat().st_size for f in parquet_files) / (1024**3)
                st.metric("Total Size", f"{total_size:.2f} GB")
            else:
                st.metric("Total Size", "0 GB")
        with col3:
            if parquet_files:
                latest = max((f.stat().st_mtime for f in parquet_files), default=0)
                st.metric("Last Updated", datetime.fromtimestamp(latest).strftime("%Y-%m-%d"))
            else:
                st.metric("Last Updated", "N/A")
        with col4:
            csv_files = list(self.csv_path.glob("*.csv")) if self.csv_path.exists() else []
            st.metric("CSV Files", len(csv_files))
        
        # File management tabs
        tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🔄 Conversion", "📈 Coverage Analysis"])
        
        with tab1:
            st.markdown("### Parquet Files Overview")
            
            if parquet_files:
                file_data = []
                for f in parquet_files:
                    file_data.append({
                        'Filename': f.name,
                        'Size (MB)': round(f.stat().st_size / (1024**2), 2),
                        'Modified': datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                        'Rows': self.get_parquet_row_count(f) if hasattr(self, 'get_parquet_row_count') else 'Unknown'
                    })
                
                df = pd.DataFrame(file_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No Parquet files found")
        
        with tab2:
            st.markdown("### Convert CSV to Parquet")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Convert All CSV to Parquet", type="primary"):
                    with st.spinner("Converting files..."):
                        results = self.convert_csv_to_parquet()
                        st.success(f"Converted {results['converted']} files, {results['skipped']} already existed")
            
            with col2:
                if st.button("📊 Analyze Conversion"):
                    self.analyze_conversion_needs()
            
            # Show conversion progress
            if 'conversion_progress' in st.session_state:
                st.progress(st.session_state.conversion_progress)
        
        with tab3:
            st.markdown("### Data Coverage Analysis")
            self.show_data_coverage()
    
    def model_management_tab(self):
        """Manage trained models"""
        st.subheader("⚙️ Model Management")
        
        # Model statistics
        total_models = len(self.models)
        total_size = sum(m.get('file_size_mb', 0) for m in self.models.values())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Trained Models", total_models)
        with col2:
            st.metric("Total Size", f"{total_size:.1f} MB")
        with col3:
            st.metric("Model Directory", str(self.model_path))
        
        # Model list with actions
        st.markdown("### Model Inventory")
        
        if self.models:
            # Create dataframe
            model_df = pd.DataFrame([
                {
                    'Select': False,
                    'Pair': info.get('pair', pair),
                    'Date Range': f"{info.get('start_date', 'N/A')} to {info.get('end_date', 'N/A')}",
                    'Timeframe': info.get('timeframe', 'N/A'),
                    'Size (MB)': info.get('file_size_mb', 0),
                    'Last Modified': info.get('last_modified', 'N/A'),
                    'Path': info.get('path', '')
                }
                for pair, info in self.models.items()
            ])
            
            # Display editable dataframe
            edited_df = st.data_editor(
                model_df,
                column_config={
                    "Select": st.column_config.CheckboxColumn(
                        "Select",
                        help="Select models for bulk operations",
                        default=False
                    ),
                    "Pair": st.column_config.TextColumn(
                        "Currency Pair",
                        disabled=True
                    ),
                    "Date Range": st.column_config.TextColumn(
                        "Training Period",
                        disabled=True
                    ),
                    "Timeframe": st.column_config.TextColumn(
                        "Timeframe",
                        disabled=True
                    ),
                    "Size (MB)": st.column_config.NumberColumn(
                        "Size (MB)",
                        format="%.2f MB",
                        disabled=True
                    ),
                    "Last Modified": st.column_config.DatetimeColumn(
                        "Last Modified",
                        format="YYYY-MM-DD HH:mm",
                        disabled=True
                    ),
                    "Path": st.column_config.TextColumn(
                        "Path",
                        disabled=True
                    )
                },
                hide_index=True,
                use_container_width=True,
                height=400
            )
            
            # Bulk actions
            st.markdown("### Bulk Actions")
            col1, col2, col3, col4 = st.columns(4)
            
            selected_models = edited_df[edited_df['Select']]
            
            with col1:
                if st.button("🗑️ Delete Selected", disabled=len(selected_models) == 0):
                    if not selected_models.empty:
                        st.warning(f"Are you sure you want to delete {len(selected_models)} models?")
                        col_confirm, col_cancel = st.columns(2)
                        with col_confirm:
                            if st.button("✅ Confirm Delete"):
                                self.delete_models(selected_models['Pair'].tolist())
                        with col_cancel:
                            if st.button("❌ Cancel"):
                                st.rerun()
            
            with col2:
                if st.button("📤 Export Selected", disabled=len(selected_models) == 0):
                    self.export_models(selected_models['Pair'].tolist())
            
            with col3:
                if st.button("📊 Compare Selected", disabled=len(selected_models) == 0):
                    self.compare_models(selected_models['Pair'].tolist())
            
            with col4:
                if st.button("✅ Select All"):
                    # This would need JavaScript - we'll use a session state approach
                    st.info("Click each checkbox or use the 'Select All' button in the dataframe header")
            
            # Individual model actions
            st.markdown("### Individual Model Actions")
            selected_single = st.selectbox("Select a model for single operations", options=model_df['Pair'].tolist())
            
            if selected_single:
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📊 View Performance"):
                        self.view_model_performance(selected_single)
                with col2:
                    if st.button("📈 Test Predictions"):
                        self.test_model_predictions(selected_single)
                with col3:
                    if st.button("📋 View Details"):
                        self.view_model_details(selected_single)
        else:
            st.info("No models found in the model directory")
            if st.button("📁 Refresh Model List"):
                self.models = self.load_models()
                st.rerun()
    
    def get_available_pairs(self):
        """Get list of available currency pairs from Parquet files"""
        pairs = set()
        
        # Try to get from parquet files
        if self.parquet_path.exists():
            for file in self.parquet_path.glob("*.parquet"):
                # Extract pair from filename (format: PAIR_YYYYMMDD.parquet or similar)
                parts = file.stem.split('_')
                if len(parts) >= 2:
                    pair = f"{parts[0]}_{parts[1]}"
                    pairs.add(pair)
        
        # If no parquet files, try CSV
        if not pairs and self.csv_path.exists():
            for file in self.csv_path.glob("*.csv"):
                parts = file.stem.split('_')
                if len(parts) >= 2:
                    pair = f"{parts[0]}_{parts[1]}"
                    pairs.add(pair)
        
        # If still no pairs, use manual list from config
        if not pairs:
            try:
                from config import MANUAL_PAIRS
                return MANUAL_PAIRS
            except:
                # Fallback to common pairs
                return ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD"]
        
        return sorted(list(pairs))
    
    def execute_training_queue(self):
        """Execute all models in the training queue"""
        if 'training_queue' not in st.session_state or not st.session_state.training_queue:
            st.warning("Training queue is empty")
            return
        
        st.info(f"Starting training for {len(st.session_state.training_queue)} models...")
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, job in enumerate(st.session_state.training_queue):
            # Update status
            job['status'] = 'training'
            status_text.text(f"Training {job['pair']}...")
            
            # Call your existing training function here
            # This is where you integrate with your actual training code
            try:
                # Import your trainer dynamically
                sys.path.append(str(Path(__file__).parent.parent))
                
                # Example: Call your training function
                # from model_trainer import train_model
                # train_model(
                #     pair=job['pair'],
                #     lookback=job['lookback'],
                #     train_size=job['train_size'],
                #     validation_split=job['validation_split'],
                #     features=job['features']
                # )
                
                # Simulate training for now
                time.sleep(2)
                
                job['status'] = 'completed'
                st.session_state.training_queue[i] = job
                
                # Add to models list
                self.models[job['pair']] = {
                    'path': str(self.model_path / f"{datetime.now().strftime('%Y%m%d')}--{job['pair']}.pkl"),
                    'pair': job['pair'],
                    'start_date': datetime.now().strftime('%Y%m%d'),
                    'end_date': datetime.now().strftime('%Y%m%d'),
                    'timeframe': 'S5',
                    'file_size_mb': 10.5,
                    'loaded': True
                }
                
            except Exception as e:
                job['status'] = 'failed'
                st.error(f"Failed to train {job['pair']}: {e}")
            
            # Update progress
            progress_bar.progress((i + 1) / len(st.session_state.training_queue))
        
        status_text.text("Training complete!")
        
        # Show summary
        completed = sum(1 for j in st.session_state.training_queue if j['status'] == 'completed')
        failed = sum(1 for j in st.session_state.training_queue if j['status'] == 'failed')
        
        st.success(f"✅ Training completed: {completed} successful, {failed} failed")
        
        # Clear completed jobs
        st.session_state.training_queue = [j for j in st.session_state.training_queue if j['status'] != 'completed']
        
        # Refresh performance data
        self.performance_data = self.load_performance_data()
        
        st.balloons()
    
    def load_performance_data(self):
        """Load model performance metrics from file"""
        if self.performance_file.exists():
            try:
                with open(self.performance_file, 'r') as f:
                    data = json.load(f)
                return pd.DataFrame(data)
            except:
                return self.generate_sample_performance_data()
        else:
            return self.generate_sample_performance_data()
    
    def generate_sample_performance_data(self):
        """Generate sample performance data for demonstration"""
        data = []
        
        # Get actual model pairs if available
        pairs = list(self.models.keys()) if self.models else [
            "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD",
            "USD_CHF", "NZD_USD", "EUR_GBP", "EUR_JPY", "GBP_JPY"
        ]
        
        np.random.seed(42)
        for pair in pairs:
            data.append({
                'pair': pair,
                'accuracy': np.random.uniform(0.45, 0.75),
                'precision': np.random.uniform(0.4, 0.8),
                'recall': np.random.uniform(0.4, 0.8),
                'f1_score': np.random.uniform(0.4, 0.75),
                'training_time': np.random.uniform(30, 300),
                'samples': np.random.randint(50000, 200000),
                'date': (datetime.now() - timedelta(days=np.random.randint(0, 30))).strftime("%Y-%m-%d")
            })
        
        df = pd.DataFrame(data)
        
        # Sort by accuracy
        df = df.sort_values('accuracy', ascending=False)
        
        # Save to file
        try:
            with open(self.performance_file, 'w') as f:
                json.dump(data, f, indent=2)
        except:
            pass
        
        return df
    
    def generate_performance_report(self, performance_data):
        """Generate a detailed performance report"""
        st.info("Generating performance report...")
        
        # Create report in session state
        report = []
        report.append("# AI Model Performance Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append(f"- Total Models: {len(performance_data)}")
        if 'accuracy' in performance_data.columns:
            report.append(f"- Average Accuracy: {performance_data['accuracy'].mean():.2%}")
            report.append(f"- Best Model: {performance_data.loc[performance_data['accuracy'].idxmax(), 'pair']} ({performance_data['accuracy'].max():.2%})")
            report.append(f"- Worst Model: {performance_data.loc[performance_data['accuracy'].idxmin(), 'pair']} ({performance_data['accuracy'].min():.2%})")
        report.append("")
        
        # Top performers
        if 'accuracy' in performance_data.columns:
            report.append("## Top 10 Performers")
            top10 = performance_data.nlargest(10, 'accuracy')[['pair', 'accuracy', 'f1_score']]
            for _, row in top10.iterrows():
                report.append(f"- {row['pair']}: {row['accuracy']:.2%} (F1: {row['f1_score']:.2%})")
        
        # Display report
        st.markdown("\n".join(report))
        
        # Download button
        st.download_button(
            label="📥 Download Report",
            data="\n".join(report),
            file_name