import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- 1. KONFIGURASI HALAMAN (LAYOUT WIDE) ---
st.set_page_config(page_title="Telco Churn Analytics Dashboard", layout="wide", page_icon="📊")

# --- 2. STYLE CSS UNTUK TAMPILAN MODERN ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e1e4e8;
    }
    .stPlotlyChart {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. DATA CLEANING & CACHING ---
@st.cache_data
def load_data():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    return df

df = load_data()

# --- 4. DATA MINING (TRAINING MODEL) ---
@st.cache_resource
def train_model(data):
    df_m = data.copy().drop('customerID', axis=1)
    
    encoders = {}
    for col in df_m.columns:
        if df_m[col].dtype == 'object':
            le = LabelEncoder()
            df_m[col] = le.fit_transform(df_m[col])
            encoders[col] = le
            
    X = df_m.drop('Churn', axis=1)
    y = df_m['Churn']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return model, encoders, X.columns, importance

model, encoders, feature_cols, feat_importances = train_model(df)

# --- 5. SIDEBAR ---
st.sidebar.title("📡 Panel Kontrol")
st.sidebar.markdown("Gunakan menu ini untuk berpindah halaman.")
menu = st.sidebar.radio("Pilih Tampilan:", ["Dashboard Executive", "Alat Prediksi AI"])
st.sidebar.divider()
st.sidebar.info("KDD Process: Data Cleaning -> Transformation -> Mining -> Knowledge Presentation")

# --- 6. HALAMAN 1: DASHBOARD RINGKASAN ---
if menu == "Dashboard Executive":
    st.title("📊 Telco Business Insights Dashboard")
    st.markdown("Ringkasan data pelanggan dan pola Churn.")

    # --- BARIS 1: KARTU KPI ---
    col1, col2, col3, col4 = st.columns(4)
    churn_rate = (df['Churn'] == 'Yes').mean() * 100
    
    col1.metric("Total Pelanggan", f"{len(df):,}")
    col2.metric("Rasio Churn", f"{churn_rate:.1f}%")
    col3.metric("Rata-rata Tagihan", f"${df['MonthlyCharges'].mean():.2f}")
    col4.metric("Total Pendapatan", f"${df['TotalCharges'].sum()/1e6:.2f}M")

    st.divider()

    # --- BARIS 2: GRAFIK SEJAJAR (DASHBOARD LOOK) ---
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Distribusi Churn per Kontrak")
        fig1 = px.histogram(df, x="Contract", color="Churn", barmode="group",
                            color_discrete_map={'Yes':'#FF4B4B', 'No':'#1C83E1'},
                            template="plotly_white")
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.subheader("Analisis Tenure vs Churn")
        fig2 = px.box(df, x="Churn", y="tenure", color="Churn",
                      color_discrete_map={'Yes':'#FF4B4B', 'No':'#1C83E1'})
        st.plotly_chart(fig2, use_container_width=True)

    # --- BARIS 3: FITUR PALING BERPENGARUH ---
    st.subheader("Faktor Penentu Pelanggan Berhenti (Top 10)")
    fig3 = px.bar(feat_importances.head(10), orientation='h', 
                  color_discrete_sequence=['#1C83E1'])
    st.plotly_chart(fig3, use_container_width=True)

# --- 7. HALAMAN 2: SISTEM PREDIKSI ---
else:
    st.title("🔮 AI Churn Predictor System")
    st.write("Masukkan profil pelanggan untuk menghitung probabilitas risiko secara real-time.")

    # --- INPUT FORM DALAM KOLOM ---
    with st.form("form_prediksi"):
        st.markdown("### 📝 Input Data Pelanggan")
        f1, f2, f3 = st.columns(3)
        
        with f1:
            tenure = st.slider("Tenure (Bulan)", 0, 72, 12)
            contract = st.selectbox("Tipe Kontrak", df['Contract'].unique())
            monthly = st.number_input("Monthly Charges ($)", value=65.0)
        
        with f2:
            internet = st.selectbox("Layanan Internet", df['InternetService'].unique())
            payment = st.selectbox("Metode Bayar", df['PaymentMethod'].unique())
            gender = st.selectbox("Gender", df['gender'].unique())

        with f3:
            tech = st.selectbox("Tech Support", df['TechSupport'].unique())
            paperless = st.selectbox("Paperless Billing", df['PaperlessBilling'].unique())
            multi = st.selectbox("Multiple Lines", df['MultipleLines'].unique())

        submitted = st.form_submit_button("ANALISIS RISIKO SEKARANG")

    if submitted:
        # Menyiapkan data untuk dikirim ke model
        input_dict = {col: 0 for col in feature_cols}
        input_dict.update({
            'gender': gender, 'tenure': tenure, 'MonthlyCharges': monthly,
            'Contract': contract, 'InternetService': internet,
            'TechSupport': tech, 'PaymentMethod': payment,
            'PaperlessBilling': paperless, 'MultipleLines': multi,
            'TotalCharges': tenure * monthly
        })
        
        input_df = pd.DataFrame([input_dict])

        # Encoding input otomatis
        for col, le in encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = le.transform([input_df[col][0]])
                except:
                    input_df[col] = 0

        # Hasil Prediksi
        prob = model.predict_proba(input_df[feature_cols])[0][1]
        
        st.markdown("---")
        res1, res2 = st.columns([1, 2])
        
        with res1:
            # GAUGE METER PROFESIONAL
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                title = {'text': "Risk Score %"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#FF4B4B" if prob > 0.5 else "#1C83E1"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "#ffcccb" if prob > 0.5 else "lightgray"}
                    ]
                }))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res2:
            st.write("### Rekomendasi Bisnis:")
            if prob > 0.5:
                st.error(f"**STATUS: RISIKO TINGGI ({(prob*100):.1f}%)**")
                st.write("- Hubungi pelanggan dalam waktu 1x24 jam.")
                st.write("- Tawarkan peralihan ke kontrak 1 tahun dengan diskon 15%.")
            else:
                st.success(f"**STATUS: RISIKO RENDAH ({(prob*100):.1f}%)**")
                st.write("- Pelanggan dalam kondisi aman.")
                st.write("- Cocok untuk ditawarkan upgrade layanan internet.")