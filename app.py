# ==============================================================================
# UNIFIED AI FORECASTER v6.0 (Monolithic Build)
# All logic is contained in this single file for robust, atomic deployment.
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import time
import firebase_admin
from firebase_admin import credentials, firestore
import plotly.graph_objects as go
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
import traceback

# ==============================================================================
# SECTION 1: DATA PROCESSING LOGIC
# (Formerly data_processing.py)
# ==============================================================================

def load_from_firestore(db_client, collection_name):
    """
    Loads and preprocesses data from a Firestore collection.
    Ensures data is sorted, de-duplicated, and correctly typed.
    """
    if db_client is None:
        return pd.DataFrame()
    
    try:
        docs = db_client.collection(collection_name).stream()
        records = [doc.to_dict() for doc in docs]
    except Exception as e:
        st.error(f"Error fetching from Firestore collection '{collection_name}': {e}")
        return pd.DataFrame()
        
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    if 'date' not in df.columns:
        return pd.DataFrame()

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    
    if pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = df['date'].dt.tz_localize(None).dt.normalize()
    
    df.sort_values(by='date', inplace=True)
    df.drop_duplicates(subset=['date'], keep='last', inplace=True)
    
    numeric_cols = ['sales', 'customers', 'add_on_sales']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df.reset_index(drop=True)

def create_features(df, events_df):
    """
    Engineers a rich feature set for a deep learning time series model.
    """
    df_copy = df.copy()

    base_sales = df_copy['sales'] - df_copy.get('add_on_sales', 0)
    customers_safe = df_copy['customers'].replace(0, np.nan)
    df_copy['atv'] = (base_sales / customers_safe).fillna(method='ffill').fillna(0)

    df_copy['month'] = df_copy['date'].dt.month
    df_copy['dayofyear'] = df_copy['date'].dt.dayofyear
    df_copy['weekofyear'] = df_copy['date'].dt.isocalendar().week.astype('int')
    df_copy['year'] = df_copy['date'].dt.year
    df_copy['dayofweek'] = df_copy['date'].dt.dayofweek

    df_copy['is_payday_period'] = df_copy['date'].apply(
        lambda x: 1 if x.day in [14, 15, 16, 29, 30, 31, 1, 2] else 0
    ).astype(int)
    
    if events_df is not None and not events_df.empty:
        events_df_unique = events_df.drop_duplicates(subset=['date'], keep='first').copy()
        events_df_unique['date'] = pd.to_datetime(events_df_unique['date']).dt.normalize()
        df_copy = pd.merge(df_copy, events_df_unique[['date', 'activity_name']], on='date', how='left')
        df_copy['is_event'] = df_copy['activity_name'].notna().astype(int)
        df_copy.drop(columns=['activity_name'], inplace=True)
    else:
        df_copy['is_event'] = 0

    if 'day_type' in df_copy.columns:
        df_copy['is_not_normal_day'] = (df_copy['day_type'] == 'Not Normal Day').astype(int)
    else:
        df_copy['is_not_normal_day'] = 0

    df_copy['is_weekend'] = (df_copy['dayofweek'] >= 5).astype(int)
    df_copy['payday_weekend_interaction'] = df_copy['is_payday_period'] * df_copy['is_weekend']
    
    df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month']/12)
    df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month']/12)
    df_copy['dayofweek_sin'] = np.sin(2 * np.pi * df_copy['dayofweek']/7)
    df_copy['dayofweek_cos'] = np.cos(2 * np.pi * df_copy['dayofweek']/7)

    targets_for_rolling = ['sales', 'customers', 'atv']
    windows = [7, 14]

    for target in targets_for_rolling:
        if target in df_copy.columns:
            for w in windows:
                shifted = df_copy[target].shift(1)
                df_copy[f'{target}_rolling_mean_{w}d'] = shifted.rolling(window=w).mean()
                df_copy[f'{target}_rolling_std_{w}d'] = shifted.rolling(window=w).std()

    df_copy.fillna(method='ffill', inplace=True)
    df_copy.fillna(method='bfill', inplace=True)
    df_copy.fillna(0, inplace=True)

    return df_copy

# ==============================================================================
# SECTION 2: FORECASTING LOGIC
# (Formerly forecasting.py)
# ==============================================================================

def generate_forecast(historical_df, events_df, periods=15):
    """
    Generates a multivariate, point forecast using a Darts NBEATS model.
    """
    try:
        future_date_range = pd.date_range(
            start=historical_df['date'].max() + pd.Timedelta(days=1),
            periods=periods
        )
        future_df_template = pd.DataFrame({'date': future_date_range})
        combined_df = pd.concat([historical_df, future_df_template], ignore_index=True)
        df_featured = create_features(combined_df, events_df)
        
        target_cols = ['customers', 'atv']
        feature_cols = [
            col for col in df_featured.columns if col not in 
            ['date', 'doc_id', 'sales', 'customers', 'atv', 'add_on_sales', 'day_type', 'day_type_notes']
        ]

        ts_target = TimeSeries.from_dataframe(
            historical_df, time_col='date', value_cols=target_cols, freq='D'
        )
        ts_features = TimeSeries.from_dataframe(
            df_featured, time_col='date', value_cols=feature_cols, freq='D'
        )

        scaler_target = Scaler()
        scaler_features = Scaler()
        ts_target_scaled = scaler_target.fit_transform(ts_target)
        ts_features_scaled = scaler_features.fit_transform(ts_features)

        input_chunk_length = max(2 * periods, 30)
        model = NBEATSModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=periods,
            n_epochs=50,
            num_stacks=2, num_blocks=3, num_layers=4, layer_widths=256,
            random_state=42,
            pl_trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": False}
        )

        st.info("Training model... This may take a minute.")
        model.fit(series=ts_target_scaled, past_covariates=ts_features_scaled)
        
        st.info("Generating forecast...")
        prediction_scaled = model.predict(
            n=periods, series=ts_target_scaled, past_covariates=ts_features_scaled
        )
        
        prediction = scaler_target.inverse_transform(prediction_scaled)
        forecast_df = prediction.pd_dataframe().reset_index()
        forecast_df.rename(columns={'time': 'ds', 'customers': 'forecast_customers', 'atv': 'forecast_atv'}, inplace=True)

        forecast_df['forecast_sales'] = forecast_df['forecast_customers'] * forecast_df['forecast_atv']
        forecast_df['forecast_sales'] = forecast_df['forecast_sales'].clip(lower=0)
        forecast_df['forecast_customers'] = forecast_df['forecast_customers'].clip(lower=0).round().astype(int)
        forecast_df['forecast_atv'] = forecast_df['forecast_atv'].clip(lower=0)

        return forecast_df.round(2)

    except Exception as e:
        st.error("An error occurred during forecast generation.")
        st.error(f"Details: {e}")
        traceback.print_exc()
        return pd.DataFrame()

# ==============================================================================
# SECTION 3: STREAMLIT APPLICATION UI & MAIN LOGIC
# (Formerly app.py)
# ==============================================================================

def main():
    st.set_page_config(
        page_title="Unified AI Forecaster v6.0",
        page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="st-"] { font-family: 'Poppins', sans-serif; }
        .main > div { background-color: #1a1a1a; color: #e0e0e0; }
        .block-container { padding: 2rem 2rem !important; }
        [data-testid="stSidebar"] { background-color: #252525; border-right: 1px solid #444; }
        .stButton > button { border-radius: 8px; font-weight: 600; transition: all 0.2s ease-in-out; border: none; padding: 10px 16px; }
        .stButton:has(button:contains("Generate")) > button { background: linear-gradient(45deg, #c8102e, #e01a37); color: #FFFFFF; }
        .stButton:has(button:contains("Generate")):hover > button { transform: translateY(-2px); box-shadow: 0 4px 15px 0 rgba(200, 16, 46, 0.4); }
        .stTabs [data-baseweb="tab"] { border-radius: 8px; background-color: transparent; color: #d3d3d3; padding: 8px 14px; font-weight: 600; font-size: 0.9rem; }
        .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #c8102e; color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)

    @st.cache_resource
    def init_firestore():
        try:
            if not firebase_admin._apps:
                creds_dict = {
                    "type": st.secrets.firebase_credentials.type,
                    "project_id": st.secrets.firebase_credentials.project_id,
                    "private_key_id": st.secrets.firebase_credentials.private_key_id,
                    "private_key": st.secrets.firebase_credentials.private_key.replace('\\n', '\n'),
                    "client_email": st.secrets.firebase_credentials.client_email,
                    "client_id": st.secrets.firebase_credentials.client_id,
                    "auth_uri": st.secrets.firebase_credentials.auth_uri,
                    "token_uri": st.secrets.firebase_credentials.token_uri,
                    "auth_provider_x509_cert_url": st.secrets.firebase_credentials.auth_provider_x509_cert_url,
                    "client_x509_cert_url": st.secrets.firebase_credentials.client_x509_cert_url
                }
                cred = credentials.Certificate(creds_dict)
                firebase_admin.initialize_app(cred)
            return firestore.client()
        except Exception as e:
            st.error(f"Firestore Connection Error: {e}")
            return None

    @st.cache_data(ttl=600)
    def get_data(_db_client):
        if _db_client is None: return pd.DataFrame(), pd.DataFrame()
        historical_df = load_from_firestore(_db_client, 'historical_data')
        events_df = load_from_firestore(_db_client, 'future_activities')
        return historical_df, events_df

    db = init_firestore()

    if db:
        if 'forecast_df' not in st.session_state: st.session_state.forecast_df = pd.DataFrame()

        with st.sidebar:
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png")
            st.title("Unified AI Forecaster")
            st.info("Atomic Build v6.0")
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.rerun()
            if st.button("📈 Generate Forecast", type="primary", use_container_width=True):
                historical_df, events_df = get_data(db)
                if len(historical_df) < 30:
                    st.error("Need at least 30 days of data for the model to train.")
                else:
                    st.session_state.forecast_df = pd.DataFrame()
                    forecast_df = generate_forecast(historical_df, events_df, periods=15)
                    if not forecast_df.empty:
                        st.session_state.forecast_df = forecast_df
                        st.success("Forecast Generated Successfully!")
                    else:
                        st.error("Forecast generation failed.")

        tab_list = ["📈 Forecast Dashboard", "🔢 Forecast Data", "✍️ Edit Historical Data"]
        tabs = st.tabs(tab_list)

        with tabs[0]:
            st.header("📈 Forecast Dashboard")
            if not st.session_state.forecast_df.empty:
                historical_df, _ = get_data(db)
                for target in ['sales', 'customers', 'atv']:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=historical_df['date'], y=historical_df[target], mode='lines', name='Historical', line=dict(color='#ffffff')))
                    fig.add_trace(go.Scatter(x=st.session_state.forecast_df['ds'], y=st.session_state.forecast_df[f'forecast_{target}'], mode='lines', name='Forecast', line=dict(color='#c8102e', dash='dash')))
                    title_map = {'sales': 'Sales (PHP)', 'customers': 'Customer Count', 'atv': 'Avg. Transaction (PHP)'}
                    fig.update_layout(title=f'Historical vs. Forecasted {title_map[target]}', template='plotly_dark', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Click 'Generate Forecast' to begin.")
        
        with tabs[1]:
            st.header("🔢 Forecast Data")
            if not st.session_state.forecast_df.empty:
                df_to_show = st.session_state.forecast_df.rename(columns={'ds': 'Date', 'forecast_customers': 'Pred. Customers', 'forecast_atv': 'Pred. Avg. Sale (₱)', 'forecast_sales': 'Pred. Sales (₱)'}).set_index('Date')
                st.dataframe(df_to_show, use_container_width=True)
            else:
                st.info("Generate a forecast to see the raw data output.")

        with tabs[2]:
            st.header("✍️ Edit Historical Data")
            historical_df, _ = get_data(db)
            if not historical_df.empty:
                recent_df = historical_df.sort_values(by="date", ascending=False).head(30)
                for _, row in recent_df.iterrows():
                    doc_id = row.get('doc_id', str(row['date']))
                    with st.expander(f"{row['date'].strftime('%B %d, %Y')} - Sales: ₱{row.get('sales', 0):,.2f}"):
                        with st.form(key=f"edit_form_{doc_id}", border=False):
                            day_type = st.selectbox("Day Type", ["Normal Day", "Not Normal Day"], index=["Normal Day", "Not Normal Day"].index(row.get('day_type', 'Normal Day')), key=f"dtype_{doc_id}")
                            if st.form_submit_button("💾 Update"):
                                db.collection('historical_data').document(row['doc_id']).update({'day_type': day_type})
                                st.success(f"Updated record for {row['date'].strftime('%Y-%m-%d')}")
                                st.cache_data.clear()
                                time.sleep(1); st.rerun()
            else:
                st.info("No historical data found.")
    else:
        st.error("Fatal Error: Could not connect to Firestore.")

if __name__ == "__main__":
    main()

