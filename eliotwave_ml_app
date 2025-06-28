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
# Konfigurasi Halaman Utama Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Analisis & Prediksi Finansial Hibrida",
    page_icon="�",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Judul dan Deskripsi Aplikasi
# -----------------------------------------------------------------------------
st.title("🌊 Dasbor Analisis Hibrida: Elliott Wave & Machine Learning")
st.markdown("Sebuah platform terintegrasi untuk memproyeksikan pergerakan harga menggunakan analisis teknikal klasik (Elliott Wave) dan model statistik modern (Machine Learning).")


# -----------------------------------------------------------------------------
# BAGIAN 1: LOGIKA ELLIOTT WAVE (DITERJEMAHKAN DARI PINE SCRIPT)
# -----------------------------------------------------------------------------

# Struktur data untuk menyimpan informasi titik pivot dan pola
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

def find_pivots(data, left, right):
    """Menemukan titik pivot high dan low dalam data."""
    pivots = []
    # Menggunakan shift untuk perbandingan yang lebih efisien
    highs = (data['High'] > data['High'].shift(1)) & (data['High'] > data['High'].shift(-1))
    lows = (data['Low'] < data['Low'].shift(1)) & (data['Low'] < data['Low'].shift(-1))
    
    for i in range(left, len(data) - right):
        is_pivot_high = data['High'][i] == data['High'][i-left:i+right+1].max()
        is_pivot_low = data['Low'][i] == data['Low'][i-left:i+right+1].min()

        if is_pivot_high:
            pivots.append(Point(price=data['High'][i], bar=i, time=data.index[i], rsiVal=data['RSI'][i]))
        elif is_pivot_low:
            pivots.append(Point(price=data['Low'][i], bar=i, time=data.index[i], rsiVal=data['RSI'][i]))
            
    # Sortir pivot berdasarkan waktu untuk memastikan urutan yang benar
    pivots.sort(key=lambda p: p.time)
    
    # Hapus pivot berturut-turut pada jenis yang sama (misalnya, dua high berturut-turut)
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
    """Menghitung skor kepercayaan untuk pola Elliott Wave yang diberikan."""
    # Aturan Wajib Elliott Wave
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

    # Penilaian berdasarkan Pedoman Fibonacci & lainnya
    current_score = 5.0
    len2 = abs(p2.price - p1.price)
    if len1 > 0:
        w2_retr = len2 / len1
        if 0.5 < w2_retr < 0.67: current_score += 2.0
        if abs(len3 / len1 - 1.618) < 0.4: current_score += 1.5
        if abs(len5 / len1 - 1.0) < 0.4 or abs(len5 / len3 - 0.618) < 0.3: current_score += 1.0

    # Divergensi RSI
    if useRsiDivergence:
        if (isBull and p5.price > p3.price and p5.rsiVal < p3.rsiVal) or \
           (not isBull and p5.price < p3.price and p5.rsiVal > p3.rsiVal):
            current_score += 2.0
            
    # Logika NEoWave
    if useNeowaveLogic and len1 > 0:
        deep_w2 = (len2 / len1) > 0.618
        extended_w3 = (len3 / len1) > 1.618
        if deep_w2 and extended_w3:
            current_score += 2.5

    return current_score


def run_ew_analysis(data, params):
    """Menjalankan seluruh proses analisis Elliott Wave."""
    pivot_params = {
        'Scalping': (8, 5), 'Day Trading': (21, 13), 'Swing Trading': (55, 34)
    }
    pivotLeft, pivotRight = pivot_params.get(params['trading_style'], (21, 13))

    data['RSI'] = 100 - (100 / (1 + data['Gold'].diff().where(data['Gold'].diff() > 0, 0).rolling(14).mean() / -data['Gold'].diff().where(data['Gold'].diff() < 0, 0).rolling(14).mean()))

    pivots = find_pivots(data, pivotLeft, pivotRight)
    
    if len(pivots) < 6:
        return None, "Tidak cukup pivot yang terdeteksi untuk membentuk pola."

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
    
    return None, "Tidak ada pola Elliott Wave berkualitas tinggi yang ditemukan."

# -----------------------------------------------------------------------------
# BAGIAN 2: LOGIKA PREDIKSI MACHINE LEARNING (DARI SKRIP ASLI)
# -----------------------------------------------------------------------------

@st.cache_data
def load_ml_data(start_date, end_date):
    """Memuat dan menggabungkan data historis untuk Emas dan variabel eksogen."""
    symbols = {'Gold': 'GC=F', 'DXY': 'DX-Y.NYB', 'VIX': '^VIX', 'TNX': '^TNX'}
    data_frames = {key: yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)[['Close']].rename(columns={'Close': key}) for key, ticker in symbols.items()}
    full_df = pd.concat(data_frames.values(), axis=1).interpolate(method='linear').dropna()
    return full_df

@st.cache_data
def create_features(_data):
    """Melakukan rekayasa fitur (feature engineering)."""
    df = _data.copy()
    df['Gold_Lag_1'] = df['Gold'].shift(1)
    df['Gold_SMA_20'] = df['Gold'].rolling(window=20).mean()
    df['Gold_SMA_50'] = df['Gold'].rolling(window=50).mean()
    delta = df['Gold'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Volatility_20D'] = df['Gold'].pct_change().rolling(window=20).std() * np.sqrt(20)
    return df.dropna()


def run_ml_prediction(featured_data, prediction_days):
    """Melatih model LightGBM dan menghasilkan prediksi."""
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
# ANTARMUKA PENGGUNA (UI) DENGAN TAB
# -----------------------------------------------------------------------------

# --- Sidebar untuk Input Umum ---
with st.sidebar:
    st.header("⚙️ Parameter Analisis Utama")
    selected_ticker = st.text_input("Simbol Ticker", "GC=F", help="Contoh: GC=F (Emas), CL=F (Minyak), BTC-USD (Bitcoin), AAPL (Apple)")
    start_date = st.date_input("Tanggal Mulai", pd.to_datetime('2022-01-01'))
    end_date = st.date_input("Tanggal Selesai", pd.to_datetime('today'))

# --- Definisi Tab ---
tab_ew, tab_ml, tab_method = st.tabs(["**🌊 Prediksi Elliott Wave**", "**🤖 Prediksi Machine Learning**", "**📜 Metodologi**"])

# --- TAB 1: ELLIOTT WAVE ---
with tab_ew:
    st.header("Analisis Algoritmik Elliott Wave")
    
    # --- Sidebar khusus EW ---
    with st.sidebar:
        with st.expander("🔬 Parameter Elliott Wave", expanded=True):
            ew_params = {
                'trading_style': st.selectbox('Gaya Analisis', ['Day Trading', 'Scalping', 'Swing Trading']),
                'scanDepth': st.slider('Kedalaman Pindai Pola', 50, 500, 250),
                'useEndingDiagonals': st.checkbox('Deteksi Ending Diagonals', True),
                'useRsiDivergence': st.checkbox('Gunakan Divergensi RSI', True),
                'useNeowaveLogic': st.checkbox('Terapkan Logika NEoWave', True),
                'showWaveLines': st.checkbox('Tampilkan Garis Gelombang', True),
                'showWaveLabels': st.checkbox('Tampilkan Label Gelombang', True),
                'showChannel': st.checkbox('Tampilkan Kanal Tren', True),
                'reward_ratio_input': st.slider('Rasio Risk/Reward untuk TP', 1.0, 5.0, 2.0, 0.5),
            }
            ew_params['capital_input'] = st.number_input('Modal Awal ($)', 1000.0, 1000000.0, 10000.0, 100.0)
            ew_params['risk_percent_input'] = st.number_input('Risiko per Trade (%)', 0.5, 5.0, 1.0, 0.5)

    if st.button("Jalankan Analisis Elliott Wave", type="primary"):
        with st.spinner(f"Memuat data untuk {selected_ticker} dan menjalankan analisis EW..."):
            try:
                ew_data = yf.download(selected_ticker, start=start_date, end=end_date, progress=False)
                if ew_data.empty:
                    st.error("Gagal memuat data untuk ticker yang dipilih. Periksa kembali simbol atau rentang tanggal.")
                else:
                    ew_data = ew_data.rename(columns={'Close': 'Gold'}) # Generalisasi kolom
                    pattern, pivots = run_ew_analysis(ew_data, ew_params)
                    st.session_state.ew_pattern = pattern
                    st.session_state.ew_pivots = pivots
                    st.session_state.ew_data = ew_data
            except Exception as e:
                st.error(f"Terjadi kesalahan saat analisis Elliott Wave: {e}")

    # --- Tampilan Hasil EW ---
    if 'ew_pattern' in st.session_state and st.session_state.ew_pattern:
        pattern = st.session_state.ew_pattern
        pivots = st.session_state.ew_pivots
        ew_data = st.session_state.ew_data
        
        # Dashboard
        col1, col2 = st.columns([3, 1])
        with col2:
            st.subheader("Ringkasan Pola")
            pattern_text = f"{'Bullish' if pattern.isBull else 'Bearish'} {'Ending Diagonal' if pattern.isDiag else 'Impulse'}"
            st.metric("Pola Aktif", pattern_text)
            st.metric("Skor Kepercayaan", f"{pattern.confidenceScore:.1f}")
            
            with st.expander("Panel Perencanaan Trade"):
                 p5 = pivots[pattern.p5_idx]
                 p4 = pivots[pattern.p4_idx]
                 entry = p5.price
                 sl = p4.price
                 risk_unit = abs(entry-sl)
                 tp = entry + risk_unit * ew_params['reward_ratio_input'] if pattern.isBull else entry - risk_unit * ew_params['reward_ratio_input']
                 risk_amount = ew_params['capital_input'] * (ew_params['risk_percent_input']/100)
                 pos_size = risk_amount / risk_unit if risk_unit > 0 else 0

                 st.metric(f"Potensi Entri {'Long' if pattern.isBull else 'Short'}", f"{entry:,.2f}")
                 st.metric("Stop Loss", f"{sl:,.2f}", delta=f"{-risk_unit:,.2f}", delta_color="inverse")
                 st.metric("Take Profit (RR 1:{})".format(ew_params['reward_ratio_input']), f"{tp:,.2f}")
                 st.write(f"Ukuran Posisi: **{pos_size:,.4f} unit**")
        
        # Grafik
        with col1:
            fig_ew = go.Figure(data=[go.Candlestick(x=ew_data.index, open=ew_data['Open'], high=ew_data['High'], low=ew_data['Low'], close=ew_data['Gold'], name="Harga")])
            
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
            
            fig_ew.update_layout(title=f"Analisis Elliott Wave untuk {selected_ticker}", template='plotly_dark', xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_ew, use_container_width=True)

    elif 'ew_pattern' in st.session_state and st.session_state.ew_pattern is None:
         st.info(st.session_state.ew_pivots) # Menampilkan pesan error dari fungsi analisis

# --- TAB 2: MACHINE LEARNING ---
with tab_ml:
    st.header("Prediksi Berbasis Machine Learning (Quantile Regression)")
    with st.sidebar:
        with st.expander("🤖 Parameter Machine Learning"):
            prediction_days = st.slider("Horizon Prediksi (Hari)", 1, 90, 14, help="Jangka waktu prediksi ke depan.")

    with st.spinner("Memuat data makroekonomi dan melatih model..."):
        ml_raw_data = load_ml_data(start_date, end_date)

    if not ml_raw_data.empty:
        featured_data = create_features(ml_raw_data)
        predictions, models, X_test, y_test = run_ml_prediction(featured_data, prediction_days)
        
        last_close_price = featured_data['Gold'].iloc[-1]
        
        future_dates = pd.to_datetime(pd.date_range(start=featured_data.index[-1], periods=prediction_days + 1, freq='B'))
        plot_data = featured_data.tail(90).copy()
        fig_ml = go.Figure()

        # Area prediksi
        fig_ml.add_trace(go.Scatter(x=[plot_data.index[-1], future_dates[-1]], y=[last_close_price, predictions['upper']], fill=None, mode='lines', line_color='rgba(211,211,211,0.5)', name='Batas Atas'))
        fig_ml.add_trace(go.Scatter(x=[plot_data.index[-1], future_dates[-1]], y=[last_close_price, predictions['lower']], fill='tonexty', mode='lines', line_color='rgba(211,211,211,0.5)', name='Batas Bawah'))
        # Garis harga historis dan prediksi
        fig_ml.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Gold'], mode='lines', name='Harga Historis', line=dict(color='gold', width=3)))
        fig_ml.add_trace(go.Scatter(x=[plot_data.index[-1], future_dates[-1]], y=[last_close_price, predictions['median']], mode='lines+markers', name='Prediksi Median', line=dict(color='red', dash='dot', width=2), marker=dict(size=8, symbol='x')))
        
        fig_ml.update_layout(title=f'Proyeksi Harga Emas untuk {prediction_days} Hari ke Depan', template='plotly_dark')
        
        # Tampilan hasil ML
        st.subheader("Ringkasan Prediksi")
        col1, col2, col3 = st.columns(3)
        col1.metric("Proyeksi Harga (Median)", f"${predictions['median']:,.2f}", f"${predictions['median'] - last_close_price:,.2f}")
        col2.metric("Harga Penutupan Terakhir", f"${last_close_price:,.2f}")
        col3.metric("Horizon Waktu", f"{prediction_days} Hari")
        
        st.plotly_chart(fig_ml, use_container_width=True)

        with st.expander("Detail Kinerja Model & Faktor Penggerak"):
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
             fig_imp.update_layout(title='Peringkat Kepentingan Fitur', template='plotly_dark', yaxis={'categoryorder':'total ascending'})
             st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.error("Gagal memuat data untuk analisis machine learning.")

# --- TAB 3: METODOLOGI ---
with tab_method:
    st.header("Metodologi Analisis")
    st.subheader("🌊 Teori Gelombang Elliott")
    st.markdown("""
    Analisis ini didasarkan pada **Prinsip Gelombang Elliott**, sebuah bentuk analisis teknikal yang mengidentifikasi pola fraktal berulang dalam pergerakan harga. Teori ini menyatakan bahwa pergerakan pasar mengikuti urutan psikologis alami yang dapat diprediksi.
    - **Pola Impuls (5 Gelombang):** Gelombang 1, 3, 5 bergerak searah dengan tren utama, sementara gelombang 2 dan 4 adalah koreksi.
    - **Pola Koreksi (3 Gelombang):** Biasanya diberi label A, B, C, dan bergerak melawan tren utama.
    - **Implementasi Algoritmik:** Skrip ini secara otomatis mendeteksi titik-titik pivot (puncak dan lembah), kemudian mencoba mencocokkannya dengan aturan dan pedoman Elliott Wave. Skor kepercayaan diberikan berdasarkan seberapa baik pola tersebut sesuai dengan rasio Fibonacci klasik dan kriteria tambahan dari **NEoWave**.
    """)
    
    st.subheader("🤖 Machine Learning (Quantile Regression)")
    st.markdown("""
    Pendekatan ini menggunakan model statistik **Light Gradient Boosting Machine (LightGBM)** untuk memprediksi harga di masa depan. Berbeda dengan prediksi nilai tunggal, metode ini menggunakan **Quantile Regression** untuk memproyeksikan rentang probabilitas harga (misalnya, ada keyakinan 80% harga akan berada di antara batas atas dan bawah).
    - **Fitur (Variabel Input):** Model tidak hanya melihat harga historis, tetapi juga mempertimbangkan faktor makroekonomi penting seperti Indeks Dolar AS (DXY), Indeks Volatilitas (VIX), dan Suku Bunga AS (TNX).
    - **Tujuan:** Untuk menangkap hubungan non-linear yang kompleks antara berbagai pendorong pasar dan harga aset.
    """)

    st.warning("""
    **DISCLAIMER PENTING:** Aplikasi ini adalah alat bantu analitis dan edukatif, **BUKAN SARAN KEUANGAN**. Kinerja masa lalu tidak menjamin hasil di masa depan. Selalu lakukan riset Anda sendiri dan kelola risiko dengan cermat.
    """, icon="⚠️")
�
