import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import re
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# Streamlit page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hybrid Financial Analysis & Forecasting",
    page_icon="🌊",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Application header and description
# -----------------------------------------------------------------------------
st.title("🌊 Hybrid Analysis Dashboard: Elliott Wave & Machine Learning")
st.markdown("An integrated platform for projecting price movement using classical technical analysis (Elliott Wave) and modern statistical models (Machine Learning).")


# -----------------------------------------------------------------------------
# SECTION 1: ELLIOTT WAVE LOGIC (TRANSLATED FROM PINE SCRIPT)
# -----------------------------------------------------------------------------

# Data structures used to store pivot points and detected patterns
@dataclass
class Point:
    price: float
    bar: int
    time: any # pd.Timestamp
    rsiVal: float = 0.0
    volumeVal: float = 0.0

@dataclass
class WavePattern:
    p0_idx: int = 0
    p1_idx: int = 0
    p2_idx: int = 0
    p3_idx: int = 0
    p4_idx: int = 0
    p5_idx: int = 0
    isBull: bool = True
    isDiag: bool = False
    confidenceScore: float = -1.0
    confirmationState: str = 'NONE' # WAITING, CONFIRMED, INVALIDATED

# -----------------------------------------------------------------------------
# Utility indicators
# -----------------------------------------------------------------------------

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index (RSI) with division-safe logic."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    loss_safe = avg_loss.replace(0, np.nan)
    rs = avg_gain / loss_safe
    rsi = 100 - (100 / (1 + rs))

    # Resolve boundary conditions explicitly
    no_loss_mask = (avg_loss == 0) & (avg_gain > 0)
    no_gain_mask = (avg_gain == 0) & (avg_loss > 0)

    rsi = rsi.mask(no_loss_mask, 100)
    rsi = rsi.mask(no_gain_mask, 0)
    rsi = rsi.fillna(50).clip(lower=0, upper=100)
    return rsi

def find_pivots(data, left, right):
    """Identify high and low pivot points within the price series."""
    pivots = []

    high_series = data['High']
    low_series = data['Low']

    for i in range(left, len(data) - right):
        window_slice = slice(i - left, i + right + 1)

        current_high = high_series.iloc[i]
        current_low = low_series.iloc[i]
        local_high = high_series.iloc[window_slice].max()
        local_low = low_series.iloc[window_slice].min()

        is_pivot_high = current_high == local_high
        is_pivot_low = current_low == local_low

        if is_pivot_high:
            pivots.append(Point(price=current_high, bar=i, time=data.index[i], rsiVal=data['RSI'].iloc[i]))
        elif is_pivot_low:
            pivots.append(Point(price=current_low, bar=i, time=data.index[i], rsiVal=data['RSI'].iloc[i]))
            
    pivots.sort(key=lambda p: p.time)
    
    unique_pivots = []
    if pivots:
        unique_pivots.append(pivots[0])
        for i in range(1, len(pivots)):
            is_prev_high = pivots[i-1].price >= unique_pivots[-1].price
            is_curr_high = pivots[i].price >= unique_pivots[-1].price
            if (is_prev_high and not is_curr_high) or (not is_prev_high and is_curr_high):
                 unique_pivots.append(pivots[i])
                 
    return unique_pivots

def calculate_ew_score(p0, p1, p2, p3, p4, p5, isBull, isDiag, useNeowaveLogic, useRsiDivergence):
    """Calculate a confidence score for the detected Elliott Wave pattern."""
    rule_w2_retraces_w1 = (p2.price < p1.price) if isBull else (p2.price > p1.price)
    rule_w2_not_exceed_w0 = (p2.price > p0.price) if isBull else (p2.price < p0.price)
    rule_w4_overlap = (p4.price < p1.price) if isBull else (p4.price > p1.price)
    rule_w3_progress = (p3.price > p1.price) if isBull else (p3.price < p1.price)
    rule_w5_progress = (p5.price > p3.price) if isBull else (p5.price < p3.price)
    
    len1 = abs(p1.price - p0.price)
    len3 = abs(p3.price - p2.price)
    len5 = abs(p5.price - p4.price)
    rule_w3_not_shortest = (len3 > len1 and len3 > len5) if len1 > 0 and len5 > 0 else True
    
    impulse_rules_ok = rule_w2_retraces_w1 and rule_w2_not_exceed_w0 and not rule_w4_overlap and rule_w3_progress and rule_w5_progress and rule_w3_not_shortest
    diag_rules_ok = rule_w2_retraces_w1 and rule_w2_not_exceed_w0 and rule_w4_overlap and rule_w3_progress and rule_w5_progress and rule_w3_not_shortest

    basic_structure_ok = diag_rules_ok if isDiag else impulse_rules_ok
    if not basic_structure_ok:
        return -1.0

    current_score = 5.0
    len2 = abs(p2.price - p1.price)
    if len1 > 0:
        w2_retr = len2 / len1
        if 0.5 < w2_retr < 0.67: current_score += 2.0
        if abs(len3 / len1 - 1.618) < 0.4: current_score += 1.5
        if abs(len5 / len1 - 1.0) < 0.4 or abs(len5 / len3 - 0.618) < 0.3: current_score += 1.0

    if useRsiDivergence:
        if (isBull and p5.price > p3.price and p5.rsiVal < p3.rsiVal) or \
           (not isBull and p5.price < p3.price and p5.rsiVal > p3.rsiVal):
            current_score += 2.0
            
    if useNeowaveLogic and len1 > 0:
        deep_w2 = (len2 / len1) > 0.618
        extended_w3 = (len3 / len1) > 1.618
        if deep_w2 and extended_w3:
            current_score += 2.5

    return current_score


def run_ew_analysis(data, params):
    """Execute the full Elliott Wave analysis pipeline."""
    pivot_params = {
        'Scalping': (8, 5), 'Day Trading': (21, 13), 'Swing Trading': (55, 34)
    }
    pivotLeft, pivotRight = pivot_params.get(params['trading_style'], (21, 13))

    data['RSI'] = calculate_rsi(data['Gold'])

    pivots = find_pivots(data, pivotLeft, pivotRight)
    
    if len(pivots) < 6:
        return None, "Not enough pivots were detected to build a pattern."

    currentBestPattern = WavePattern(confidenceScore=-1.0)

    start_scan_index = max(0, len(pivots) - params['scanDepth'])
    for i in range(start_scan_index, len(pivots) - 5):
        p = pivots[i:i+6]
        
        isBullSequence = p[0].price < p[1].price > p[2].price < p[3].price > p[4].price < p[5].price
        isBearSequence = p[0].price > p[1].price < p[2].price > p[3].price < p[4].price > p[5].price

        for isBull in [True, False]:
            if (isBull and not isBullSequence) or (not isBull and not isBearSequence):
                continue
            
            for isDiag in [False, True]:
                if isDiag and not params['useEndingDiagonals']:
                    continue
                
                score = calculate_ew_score(p[0], p[1], p[2], p[3], p[4], p[5], isBull, isDiag, params['useNeowaveLogic'], params['useRsiDivergence'])
                if score > currentBestPattern.confidenceScore:
                    currentBestPattern = WavePattern(
                        p0_idx=i, p1_idx=i+1, p2_idx=i+2, p3_idx=i+3, p4_idx=i+4, p5_idx=i+5,
                        isBull=isBull, isDiag=isDiag, confidenceScore=score, confirmationState='WAITING'
                    )

    if currentBestPattern.confidenceScore > 0:
        return currentBestPattern, pivots
    
    return None, "No high-quality Elliott Wave pattern was found."

# -----------------------------------------------------------------------------
# SECTION 2: MACHINE LEARNING FORECASTING LOGIC (FROM ORIGINAL SCRIPT)
# -----------------------------------------------------------------------------

@st.cache_data
def load_ml_data(start_date, end_date):
    """Load and merge historical data for gold and exogenous variables."""
    symbols = {'Gold': 'GC=F', 'DXY': 'DX-Y.NYB', 'VIX': '^VIX', 'TNX': '^TNX'}
    data_frames = {key: yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)[['Close']].rename(columns={'Close': key}) for key, ticker in symbols.items()}
    full_df = pd.concat(data_frames.values(), axis=1).interpolate(method='linear').dropna()
    return full_df

@st.cache_data
def create_features(_data):
    """Perform feature engineering for the ML model."""
    df = _data.copy()
    df['Gold_Lag_1'] = df['Gold'].shift(1)
    df['Gold_SMA_20'] = df['Gold'].rolling(window=20).mean()
    df['Gold_SMA_50'] = df['Gold'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Gold'])
    df['Volatility_20D'] = df['Gold'].pct_change().rolling(window=20).std() * np.sqrt(20)
    return df.dropna()


def run_ml_prediction(featured_data, prediction_days):
    """Train the LightGBM models and produce quantile forecasts."""
    df_model = featured_data.copy()
    df_model['Gold_Target'] = df_model['Gold'].shift(-prediction_days)
    df_model.dropna(inplace=True)
    
    features = [col for col in df_model.columns if col not in ['Gold_Target']]
    X = df_model[features]
    y = df_model['Gold_Target']
    X.columns = [re.sub(r'[^A-Za-z0-9_]+', '', str(col)) for col in X.columns]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    models = {}
    quantiles = {'lower': 0.1, 'median': 0.5, 'upper': 0.9}
    
    for name, q in quantiles.items():
        model = lgb.LGBMRegressor(
            objective='quantile', alpha=q, metric='quantile',
            n_estimators=1000, learning_rate=0.05,
            num_leaves=31, verbose=-1, n_jobs=-1, seed=42
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='quantile', callbacks=[lgb.early_stopping(100, verbose=False)])
        models[name] = model
        
    last_known_features = X.tail(1)
    predictions = {name: model.predict(last_known_features)[0] for name, model in models.items()}
    return predictions, models, X_test, y_test


# -----------------------------------------------------------------------------
# USER INTERFACE (UI) WITH TABS
# -----------------------------------------------------------------------------

# --- Sidebar for global inputs ---
with st.sidebar:
    st.header("⚙️ Core Analysis Parameters")
    selected_ticker = st.text_input("Ticker Symbol", "GC=F", help="Examples: GC=F (Gold), CL=F (Crude Oil), BTC-USD (Bitcoin), AAPL (Apple)")
    start_date = st.date_input("Start Date", pd.to_datetime('2022-01-01'))
    end_date = st.date_input("End Date", pd.to_datetime('today'))

# --- Tab definitions ---
tab_ew, tab_ml, tab_method = st.tabs(["**🌊 Elliott Wave Forecast**", "**🤖 Machine Learning Forecast**", "**📜 Methodology**"])

# --- TAB 1: ELLIOTT WAVE ---
with tab_ew:
    st.header("Algorithmic Elliott Wave Analysis")

    with st.sidebar:
        with st.expander("🔬 Elliott Wave Parameters", expanded=True):
            ew_params = {
                'trading_style': st.selectbox('Analysis Style', ['Day Trading', 'Scalping', 'Swing Trading']),
                'scanDepth': st.slider('Pattern Scan Depth', 50, 500, 250),
                'useEndingDiagonals': st.checkbox('Detect Ending Diagonals', True),
                'useRsiDivergence': st.checkbox('Use RSI Divergence', True),
                'useNeowaveLogic': st.checkbox('Apply NEoWave Logic', True),
                'showWaveLines': st.checkbox('Show Wave Lines', True),
                'showWaveLabels': st.checkbox('Show Wave Labels', True),
                'showChannel': st.checkbox('Show Trend Channel', True),
                'reward_ratio_input': st.slider('Risk/Reward Ratio for TP', 1.0, 5.0, 2.0, 0.5),
            }
            ew_params['capital_input'] = st.number_input('Starting Capital ($)', 1000.0, 1000000.0, 10000.0, 100.0)
            ew_params['risk_percent_input'] = st.number_input('Risk per Trade (%)', 0.5, 5.0, 1.0, 0.5)

    if st.button("Run Elliott Wave Analysis", type="primary"):
        with st.spinner(f"Loading data for {selected_ticker} and running EW analysis..."):
            try:
                ew_data = yf.download(selected_ticker, start=start_date, end=end_date, progress=False)
                if ew_data.empty:
                    st.error("Failed to load data for the selected ticker. Please verify the symbol or date range.")
                else:
                    ew_data = ew_data.rename(columns={'Close': 'Gold'}) # Generalize column name
                    pattern, pivots = run_ew_analysis(ew_data, ew_params)
                    st.session_state.ew_pattern = pattern
                    st.session_state.ew_pivots = pivots
                    st.session_state.ew_data = ew_data
            except Exception as e:
                st.error(f"An error occurred while running the Elliott Wave analysis: {e}")

    if 'ew_pattern' in st.session_state and st.session_state.ew_pattern:
        pattern = st.session_state.ew_pattern
        pivots = st.session_state.ew_pivots
        ew_data = st.session_state.ew_data
        
        col1, col2 = st.columns([3, 1])
        with col2:
            st.subheader("Pattern Summary")
            pattern_text = f"{'Bullish' if pattern.isBull else 'Bearish'} {'Ending Diagonal' if pattern.isDiag else 'Impulse'}"
            st.metric("Active Pattern", pattern_text)
            st.metric("Confidence Score", f"{pattern.confidenceScore:.1f}")

            with st.expander("Trade Planning Panel"):
                 p5 = pivots[pattern.p5_idx]
                 p4 = pivots[pattern.p4_idx]
                 entry = p5.price
                 sl = p4.price
                 risk_unit = abs(entry-sl)
                 tp = entry + risk_unit * ew_params['reward_ratio_input'] if pattern.isBull else entry - risk_unit * ew_params['reward_ratio_input']
                 risk_amount = ew_params['capital_input'] * (ew_params['risk_percent_input']/100)
                 pos_size = risk_amount / risk_unit if risk_unit > 0 else 0

                 st.metric(f"Potential {'Long' if pattern.isBull else 'Short'} Entry", f"{entry:,.2f}")
                 st.metric("Stop Loss", f"{sl:,.2f}", delta=f"{-risk_unit:,.2f}", delta_color="inverse")
                 st.metric("Take Profit (RR 1:{})".format(ew_params['reward_ratio_input']), f"{tp:,.2f}")
                 st.write(f"Position Size: **{pos_size:,.4f} units**")

        with col1:
            fig_ew = go.Figure(data=[go.Candlestick(x=ew_data.index, open=ew_data['Open'], high=ew_data['High'], low=ew_data['Low'], close=ew_data['Gold'], name="Price")])
            
            if ew_params['showWaveLines'] and pattern:
                wave_points = [pivots[i] for i in [pattern.p0_idx, pattern.p1_idx, pattern.p2_idx, pattern.p3_idx, pattern.p4_idx, pattern.p5_idx]]
                fig_ew.add_trace(go.Scatter(
                    x=[p.time for p in wave_points],
                    y=[p.price for p in wave_points],
                    mode='lines+markers', name='Wave Path', line=dict(color='yellow', width=2)
                ))
            if ew_params['showWaveLabels'] and pattern:
                labels = ['0', '1', '2', '3', '4', '5(D)' if pattern.isDiag else '5']
                for i, p_idx in enumerate([pattern.p0_idx, pattern.p1_idx, pattern.p2_idx, pattern.p3_idx, pattern.p4_idx, pattern.p5_idx]):
                    fig_ew.add_annotation(x=pivots[p_idx].time, y=pivots[p_idx].price, text=labels[i], showarrow=True, arrowhead=2, ax=0, ay=-30 if pattern.isBull and i % 2 != 0 else 30)
            
            fig_ew.update_layout(title=f"Elliott Wave Analysis for {selected_ticker}", template='plotly_dark', xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_ew, use_container_width=True)

    elif 'ew_pattern' in st.session_state and st.session_state.ew_pattern is None:
         st.info(st.session_state.ew_pivots)

# --- TAB 2: MACHINE LEARNING ---
with tab_ml:
    st.header("Machine Learning Forecast (Quantile Regression)")
    with st.sidebar:
        with st.expander("🤖 Machine Learning Parameters"):
            prediction_days = st.slider("Prediction Horizon (Days)", 1, 90, 14, help="Length of the forward forecast window.")

    with st.spinner("Loading macroeconomic data and training the model..."):
        ml_raw_data = load_ml_data(start_date, end_date)

    if not ml_raw_data.empty:
        featured_data = create_features(ml_raw_data)
        predictions, models, X_test, y_test = run_ml_prediction(featured_data, prediction_days)

        # Use .item() to extract a scalar value for Streamlit metrics
        last_close_price = featured_data['Gold'].iloc[-1].item()

        future_dates = pd.to_datetime(pd.date_range(start=featured_data.index[-1], periods=prediction_days + 1, freq='B'))
        plot_data = featured_data.tail(90).copy()
        fig_ml = go.Figure()

        fig_ml.add_trace(go.Scatter(x=[plot_data.index[-1], future_dates[-1]], y=[last_close_price, predictions['upper']], fill=None, mode='lines', line_color='rgba(211,211,211,0.5)', name='Upper Bound'))
        fig_ml.add_trace(go.Scatter(x=[plot_data.index[-1], future_dates[-1]], y=[last_close_price, predictions['lower']], fill='tonexty', mode='lines', line_color='rgba(211,211,211,0.5)', name='Lower Bound'))
        fig_ml.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Gold'], mode='lines', name='Historical Price', line=dict(color='gold', width=3)))
        fig_ml.add_trace(go.Scatter(x=[plot_data.index[-1], future_dates[-1]], y=[last_close_price, predictions['median']], mode='lines+markers', name='Median Forecast', line=dict(color='red', dash='dot', width=2), marker=dict(size=8, symbol='x')))

        fig_ml.update_layout(title=f'Gold Price Projection for the Next {prediction_days} Trading Days', template='plotly_dark')

        st.subheader("Forecast Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Median Forecast", f"${predictions['median']:,.2f}", f"{predictions['median'] - last_close_price:,.2f}")
        col2.metric("Last Close", f"${last_close_price:,.2f}")
        col3.metric("Forecast Horizon", f"{prediction_days} Days")

        st.plotly_chart(fig_ml, use_container_width=True)

        with st.expander("Model Performance & Drivers"):
             y_pred = models['median'].predict(X_test)
             rmse = np.sqrt(mean_squared_error(y_test, y_pred))
             mae = mean_absolute_error(y_test, y_pred)
             r2 = r2_score(y_test, y_pred)
             c1, c2, c3 = st.columns(3)
             c1.metric("RMSE", f"${rmse:,.2f}")
             c2.metric("MAE", f"${mae:,.2f}")
             c3.metric("R² Score", f"{r2:.2%}")

             feature_importance = pd.DataFrame({'feature': X_test.columns, 'importance': models['median'].feature_importances_}).sort_values('importance', ascending=False)
             fig_imp = go.Figure(go.Bar(x=feature_importance['importance'], y=feature_importance['feature'], orientation='h'))
             fig_imp.update_layout(title='Feature Importance Ranking', template='plotly_dark', yaxis={'categoryorder':'total ascending'})
             st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.error("Failed to load data for the machine learning analysis.")

# --- TAB 3: METHODOLOGY ---
with tab_method:
    st.header("Analytical Methodology")
    st.subheader("🌊 Elliott Wave Theory")
    st.markdown("""
    This analysis is rooted in **Elliott Wave Principles**, a form of technical analysis that identifies repeating fractal patterns in price movement. The theory suggests that markets follow a natural psychological rhythm that can be anticipated.
    - **Impulse Pattern (5 Waves):** Waves 1, 3, and 5 move with the dominant trend, while waves 2 and 4 are corrective.
    - **Corrective Pattern (3 Waves):** Typically labeled A, B, C, and move against the dominant trend.
    - **Algorithmic Implementation:** The script automatically detects pivot points (swing highs and lows), then matches them to Elliott Wave rules and guidelines. Confidence scores are awarded based on alignment with classic Fibonacci ratios and additional **NEoWave** criteria.
    """)

    st.subheader("🤖 Machine Learning (Quantile Regression)")
    st.markdown("""
    This approach uses the **Light Gradient Boosting Machine (LightGBM)** algorithm to project future prices. Instead of producing a single-value forecast, the method employs **Quantile Regression** to estimate a probabilistic price range (for example, an 80% confidence interval between the upper and lower bounds).
    - **Features (Input Variables):** The model considers not only historical price data, but also key macroeconomic drivers such as the US Dollar Index (DXY), Volatility Index (VIX), and US Treasury Yield (TNX).
    - **Objective:** Capture complex non-linear relationships between multiple market drivers and the asset price.
    """)

    st.warning("""
    **IMPORTANT DISCLAIMER:** This application is an analytical and educational tool, **NOT FINANCIAL ADVICE**. Past performance does not guarantee future results. Always conduct your own research and manage risk carefully.
    """, icon="⚠️")
