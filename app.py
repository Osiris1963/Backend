import streamlit as st
import pandas as pd
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import plotly.graph_objs as go
import time
from datetime import timedelta, date
import firebase_admin
from firebase_admin import credentials, firestore
import logging
import bcrypt

# --- Suppress Prophet's informational messages ---
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

# --- Page Configuration ---
st.set_page_config(
    page_title="McDonald's San Carlos 688",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Professional UI ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    /* Hide the default sidebar and hamburger menu */
    [data-testid="stSidebar"], [data-testid="main-menu-button"] {
        display: none;
    }
    /* Main Font & Colors */
    html, body, [class*="st-"], .stButton>button, .stTextInput>div>div>input, .stSelectbox>div>div>div {
        font-family: 'Poppins', sans-serif;
    }
    .main > div {
        background-color: #F7F7F7;
    }
    /* Custom Navigation Bar */
    .nav-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 2rem;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        position: sticky;
        top: 0;
        z-index: 1000;
    }
    .nav-title {
        font-size: 1.5rem;
        font-weight: 700;
        display: flex;
        align-items: center;
    }
    .nav-title img {
        height: 30px;
        margin-right: 10px;
    }
    .nav-links {
        display: flex;
        gap: 5px;
    }
    .nav-button {
        padding: 8px 16px;
        border-radius: 8px;
        border: none;
        background-color: transparent;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .nav-button:hover {
        background-color: #f0f0f0;
    }
    .nav-button.active {
        background-color: #DA291C;
        color: white;
    }
    .logout-button {
        background-color: #4B5563;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Firestore Initialization ---
@st.cache_resource
def init_firestore():
    try:
        if not firebase_admin._apps:
            creds_dict = st.secrets.firebase_credentials
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"Firestore Connection Error: {e}")
        return None

# --- User Management Functions ---
def get_user(db_client, username):
    users_ref = db_client.collection('users')
    query = users_ref.where('username', '==', username).limit(1)
    results = list(query.stream())
    return results[0] if results else None

def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

# --- Data Loading ---
@st.cache_data(ttl="10m")
def load_all_data(_db_client):
    collections = ["historical_data", "future_activities", "future_events", "users", "forecast_log", "forecast_insights"]
    data = {}
    for collection in collections:
        docs = _db_client.collection(collection).stream()
        records = []
        for doc in docs:
            record = doc.to_dict()
            record['doc_id'] = doc.id
            records.append(record)
        
        df = pd.DataFrame(records)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            df = df.sort_values(by='date', ascending=False).reset_index(drop=True)
        elif 'ds' in df.columns:
             df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
             df = df.sort_values(by='ds', ascending=False).reset_index(drop=True)
        data[collection] = df
    return data

# --- SUPER GENIUS FORECASTING ENGINE ---
def create_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    return df

def create_multi_output_target(df, target_col, horizon):
    df = df.sort_values('date').reset_index(drop=True)
    for i in range(1, horizon + 1):
        df[f'y_t+{i}'] = df[target_col].shift(-i)
    df.dropna(inplace=True)
    features = ['dayofweek', 'month', 'year', 'dayofyear', 'weekofyear']
    X = df[features]
    y = df[[f'y_t+{i}' for i in range(1, horizon + 1)]]
    return X, y

def train_xgboost_direct(df, target_col, horizon):
    df_featured = create_features(df.copy())
    X_train, y_train = create_multi_output_target(df_featured, target_col, horizon)
    if X_train.empty: return None
    base_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
    multi_output_model = MultiOutputRegressor(base_model)
    multi_output_model.fit(X_train, y_train)
    return multi_output_model

def train_and_forecast_stacked(historical_df, target_col, horizon):
    split_date = historical_df['date'].max() - timedelta(days=horizon*2)
    train_base, train_meta = historical_df[historical_df['date'] <= split_date], historical_df[historical_df['date'] > split_date]

    df_prophet = train_base.rename(columns={'date': 'ds', target_col: 'y'})
    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.add_country_holidays(country_name='PH')
    prophet_model.fit(df_prophet)
    
    xgb_model = train_xgboost_direct(train_base, target_col, horizon)
    if not xgb_model: return None, None

    future_meta_prophet = prophet_model.make_future_dataframe(periods=len(train_meta))
    prophet_meta_preds = prophet_model.predict(future_meta_prophet)['yhat'].tail(len(train_meta)).values
    
    xgb_meta_features = create_features(train_base.tail(1))
    xgb_meta_preds = xgb_model.predict(xgb_meta_features[xgb_model.estimator_.feature_names_in_])[0]

    meta_X = pd.DataFrame({'prophet_pred': prophet_meta_preds[:len(xgb_meta_preds)], 'xgb_pred': xgb_meta_preds})
    meta_y = train_meta[target_col].values[:len(meta_X)]

    meta_model = Ridge()
    meta_model.fit(meta_X, meta_y)

    prophet_model.fit(historical_df.rename(columns={'date': 'ds', target_col: 'y'}))
    xgb_model = train_xgboost_direct(historical_df, target_col, horizon)

    future_prophet = prophet_model.make_future_dataframe(periods=horizon)
    prophet_future_preds = prophet_model.predict(future_prophet)['yhat'].tail(horizon).values
    
    xgb_future_features = create_features(historical_df.tail(1))
    xgb_future_preds = xgb_model.predict(xgb_future_features[xgb_model.estimator_.feature_names_in_])[0]

    final_meta_X = pd.DataFrame({'prophet_pred': prophet_future_preds, 'xgb_pred': xgb_future_preds})
    final_forecast = meta_model.predict(final_meta_X)

    future_dates = pd.date_range(start=historical_df['date'].max() + timedelta(days=1), periods=horizon)
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': final_forecast})
    
    return forecast_df, prophet_model

# --- UI Rendering Functions ---
def render_dashboard(data):
    # UI Code for Dashboard
    pass
def render_insights(data):
    # UI Code for Insights
    pass
def render_evaluator(data):
    # UI Code for Evaluator
    pass
def render_add_data(db):
    # UI Code for Add Data
    pass
def render_activities(db, data):
    # UI Code for Activities
    pass
def render_historical(db, data):
    # UI Code for Historical
    pass
def render_admin(db, data):
    # UI Code for Admin
    pass

# --- Main Application ---
db = init_firestore()

if 'authentication_status' not in st.session_state:
    st.session_state.authentication_status = False
    st.session_state.username = None
    st.session_state.access_level = 3
    st.session_state.current_view = "dashboard"

if not st.session_state.authentication_status:
    st.title("McDonald's San Carlos 688")
    username = st.text_input("Username (Email)")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user_record = get_user(db, username)
        if user_record and verify_password(password, user_record.to_dict()['password']):
            st.session_state.authentication_status = True
            st.session_state.username = username
            st.session_state.access_level = user_record.to_dict()['access_level']
            st.rerun()
        else:
            st.error("Incorrect username or password")
else:
    # --- Custom Navigation Bar ---
    st.markdown("""
        <div class="nav-container">
            <div class="nav-title">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/McDonald%27s_Golden_Arches.svg/1200px-McDonald%27s_Golden_Arches.svg.png" alt="Logo">
                <span>McDonald's San Carlos 688</span>
            </div>
            <div id="nav-links" class="nav-links">
            </div>
            <div id="user-info"></div>
        </div>
    """, unsafe_allow_html=True)

    # --- Load Data ---
    data = load_all_data(db)

    # --- Render Navigation and Views ---
    tabs = ["Dashboard", "Insights", "Evaluator", "Add Data", "Activities", "History"]
    if st.session_state.access_level == 1:
        tabs.append("Admin")
    
    cols = st.columns(len(tabs) + 2)
    for i, tab in enumerate(tabs):
        if cols[i].button(tab, key=f"nav_{tab}"):
            st.session_state.current_view = tab.lower().replace(" ", "")
    
    if cols[-1].button("Logout", key="logout"):
        st.session_state.authentication_status = False
        st.rerun()

    # --- Display Current View ---
    if st.session_state.current_view == "dashboard":
        render_dashboard(data)
    elif st.session_state.current_view == "insights":
        render_insights(data)
    elif st.session_state.current_view == "evaluator":
        render_evaluator(data)
    elif st.session_state.current_view == "adddata":
        render_add_data(db)
    elif st.session_state.current_view == "activities":
        render_activities(db, data)
    elif st.session_state.current_view == "history":
        render_historical(db, data)
    elif st.session_state.current_view == "admin":
        render_admin(db, data)
