import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import download_data, load_data
from features import prepare_data_for_ml
from model_trainer import TimeSeriesPredictor, PROPHET_AVAILABLE

st.set_page_config(page_title="Advanced Time Series Forecasting", layout="wide", page_icon="📈")

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Comparative Analysis of ML Models on Time Series Data")
st.markdown("### Real-World Financial Data Forecasting & Analysis")

# Sidebar
st.sidebar.header("🛠️ Data Configuration")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL", help="Enter a valid Yahoo Finance ticker (e.g., AAPL, MSFT, BTC-USD)")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))

# Define data directory relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

if st.sidebar.button("📥 Fetch Data"):
    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            data_path = os.path.join(DATA_DIR, f"{ticker}_data.csv")
            download_data(ticker, start_date, end_date, data_path)
            st.session_state['data_path'] = data_path
            st.session_state['ticker'] = ticker
            st.sidebar.success("Data loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error fetching data: {e}")

# Main App Logic
if 'data_path' in st.session_state:
    df = load_data(st.session_state['data_path'])
    ticker = st.session_state['ticker']
    
    if df is not None:
        # Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Data Analysis", "⚙️ Model Training", "🏆 Results Comparison"])
        
        with tab1:
            st.subheader(f"Exploratory Data Analysis: {ticker}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Observations", len(df))
            col1.metric("Start Date", str(df.index.min().date()))
            col1.metric("End Date", str(df.index.max().date()))
            
            col2.metric("Highest Price", f"${df['High'].max():.2f}")
            col2.metric("Lowest Price", f"${df['Low'].min():.2f}")
            
            col3.metric("Average Volume", f"{int(df['Volume'].mean()):,}")
            
            # Interactive Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
            fig.update_layout(title=f"{ticker} Closing Price History", xaxis_title="Date", yaxis_title="Price (USD)")
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("View Raw Data"):
                st.dataframe(df.head())
                st.write(df.describe())

        with tab2:
            st.subheader("Model Configuration")
            
            col_conf1, col_conf2 = st.columns(2)
            
            with col_conf1:
                test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
                scale_data = st.checkbox("Scale Data (MinMax)", value=True)
            
            with col_conf2:
                available_models = ["ARIMA", "XGBoost", "Random Forest", "SVR"]
                if PROPHET_AVAILABLE:
                    available_models.append("Prophet")
                
                selected_models = st.multiselect("Select Models to Train", available_models, default=["ARIMA", "XGBoost"])

            # Hyperparameters (Simplified)
            st.markdown("#### Hyperparameters")
            params = {}
            if "ARIMA" in selected_models:
                with st.expander("ARIMA Settings"):
                    p = st.number_input("p (AR)", 0, 10, 5)
                    d = st.number_input("d (I)", 0, 2, 1)
                    q = st.number_input("q (MA)", 0, 10, 0)
                    params['arima'] = {'order': (p, d, q)}
            
            if "XGBoost" in selected_models:
                with st.expander("XGBoost Settings"):
                    n_est = st.number_input("n_estimators (XGB)", 50, 500, 100)
                    lr = st.number_input("learning_rate (XGB)", 0.01, 0.5, 0.1)
                    params['xgboost'] = {'n_estimators': n_est, 'learning_rate': lr}

            if "Random Forest" in selected_models:
                with st.expander("Random Forest Settings"):
                    rf_est = st.number_input("n_estimators (RF)", 50, 500, 100)
                    params['random_forest'] = {'n_estimators': rf_est}

            if "SVR" in selected_models:
                with st.expander("SVR Settings"):
                    C = st.number_input("C (Regularization)", 0.1, 100.0, 1.0)
                    params['svr'] = {'C': C}

            if st.button("🚀 Train & Evaluate Models"):
                with st.spinner("Preprocessing data..."):
                    X_train, y_train, X_test, y_test, feature_cols, scaler = prepare_data_for_ml(
                        df, target_col='Close', test_size=test_size, scale=scale_data
                    )
                    
                    # Store in session state for results tab
                    st.session_state['X_train'] = X_train
                    st.session_state['y_train'] = y_train
                    st.session_state['X_test'] = X_test
                    st.session_state['y_test'] = y_test
                    st.session_state['scaler'] = scaler
                    st.session_state['results'] = {}
                    
                    # Train Loop
                    progress_bar = st.progress(0)
                    step = 1.0 / len(selected_models)
                    current_progress = 0.0
                    
                    for model_name in selected_models:
                        st.write(f"Training {model_name}...")
                        model_key = model_name.lower().replace(" ", "_")
                        model_params = params.get(model_key, {})
                        
                        predictor = TimeSeriesPredictor(model_type=model_key, params=model_params)
                        
                        try:
                            predictor.train(X_train, y_train)
                            preds = predictor.predict(X_test)
                            
                            # Align predictions
                            if isinstance(preds, pd.Series):
                                pred_series = preds
                            else:
                                pred_series = pd.Series(preds, index=y_test.index)
                            
                            metrics = predictor.evaluate(y_test, pred_series)
                            st.session_state['results'][model_name] = {'predictions': pred_series, 'metrics': metrics}
                            
                        except Exception as e:
                            st.error(f"Error training {model_name}: {e}")
                        
                        current_progress += step
                        progress_bar.progress(min(current_progress, 1.0))
                    
                    st.success("Training Complete! Go to 'Results Comparison' tab.")

        with tab3:
            if 'results' in st.session_state and st.session_state['results']:
                st.subheader("🏆 Model Performance Comparison")
                
                results = st.session_state['results']
                y_test = st.session_state['y_test']
                
                # Metrics Table
                metrics_list = []
                for name, res in results.items():
                    m = res['metrics']
                    m['Model'] = name
                    metrics_list.append(m)
                
                metrics_df = pd.DataFrame(metrics_list).set_index('Model')
                st.table(metrics_df.style.highlight_min(axis=0, color='lightgreen'))
                
                # Visualization
                st.subheader("Forecast vs Actual")
                fig_res = go.Figure()
                
                # Actual
                fig_res.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Actual', line=dict(color='black', width=2)))
                
                # Predictions
                colors = ['red', 'blue', 'green', 'orange', 'purple']
                for i, (name, res) in enumerate(results.items()):
                    color = colors[i % len(colors)]
                    fig_res.add_trace(go.Scatter(x=res['predictions'].index, y=res['predictions'], mode='lines', name=name, line=dict(color=color, dash='dash')))
                
                fig_res.update_layout(title="Model Predictions on Test Set", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig_res, use_container_width=True)
                
                # Best Model
                best_model = metrics_df['RMSE'].idxmin()
                st.info(f"🌟 Best Performing Model based on RMSE: **{best_model}**")
                
            else:
                st.info("Please train models in the 'Model Training' tab first.")

else:
    st.info("👈 Please enter a ticker and click 'Fetch Data' to begin.")
