import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- 1. SETTING HALAMAN ---
st.set_page_config(page_title="Telco Churn Analytics", layout="wide")

# --- 2. LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        # Pastikan TotalCharges bersih dari spasi kosong
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"File CSV tidak ditemukan atau rusak: {e}")
        return None

df = load_data()

# --- 3. TRAINING MODEL (DILAKUKAN SEKALI) ---
@st.cache_resource
def train_model(data):
    df_m = data.copy()
    df_m.drop('customerID', axis=1, inplace=True)
    
    # Label Encoding untuk kolom kategori
    encoders = {}
    for col in df_m.columns:
        if df_m[col].dtype == 'object':
            le = LabelEncoder()
            df_m[col] = le.fit_transform(df_m[col])
            encoders[col] = le
            
    X = df_m.drop('Churn', axis=1)
    y = df_m['Churn']
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Feature Importance
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    return model, encoders, X.columns, importances

if df is not None:
    model, encoders, feature_cols, feat_importances = train_model(df)

    # --- 4. SIDEBAR ---
    st.sidebar.title("Menu Navigasi")
    page = st.sidebar.selectbox("Pilih Tampilan:", ["Dashboard Insights", "Sistem Prediksi"])

    # --- 5. HALAMAN 1: DASHBOARD ---
    if page == "Dashboard Insights":
        st.title("📊 Customer Business Dashboard")
        
        # Row 1: KPI Metrics
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Pelanggan", f"{len(df):,}")
        with c2:
            churn_rate = (df['Churn'] == 'Yes').mean() * 100
            st.metric("Tingkat Churn", f"{churn_rate:.1f}%")
        with c3:
            st.metric("Rata-rata Tagihan", f"${df['MonthlyCharges'].mean():.2f}")

        st.markdown("---")

        # Row 2: Charts
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Churn Berdasarkan Tipe Kontrak")
            fig1 = px.histogram(df, x="Contract", color="Churn", barmode="group",
                                color_discrete_sequence=['#2E7BCF', '#FF4B4B'])
            st.plotly_chart(fig1, use_container_width=True)
        
        with col_b:
            st.subheader("Distribusi Masa Berlangganan (Tenure)")
            fig2 = px.box(df, x="Churn", y="tenure", color="Churn")
            st.plotly_chart(fig2, use_container_width=True)

        # Row 3: Importance
        st.subheader("Faktor Utama Penyebab Churn")
        fig3 = px.bar(feat_importances.head(10), orientation='h', 
                      labels={'value':'Skor Penting', 'index':'Fitur'})
        st.plotly_chart(fig3, use_container_width=True)

    # --- 6. HALAMAN 2: PREDIKSI ---
    else:
        st.title("🔮 Prediksi Churn Pelanggan")
        st.write("Masukkan data pelanggan untuk melihat probabilitas mereka berhenti.")

        with st.form("input_pelanggan"):
            col1, col2 = st.columns(2)
            with col1:
                tenure = st.slider("Tenure (Bulan)", 0, 72, 12)
                contract = st.selectbox("Kontrak", df['Contract'].unique())
                internet = st.selectbox("Layanan Internet", df['InternetService'].unique())
            with col2:
                monthly = st.number_input("Biaya Bulanan ($)", value=60.0)
                payment = st.selectbox("Metode Bayar", df['PaymentMethod'].unique())
                tech_support = st.selectbox("Tech Support", df['TechSupport'].unique())
            
            submit = st.form_submit_button("ANALISIS RISIKO")

        if submit:
            # Buat dummy data dengan semua kolom yang dibutuhkan model
            input_raw = {col: 0 for col in feature_cols} # Inisialisasi awal
            
            # Isi data dari input form
            input_raw['tenure'] = tenure
            input_raw['Contract'] = contract
            input_raw['MonthlyCharges'] = monthly
            input_raw['InternetService'] = internet
            input_raw['PaymentMethod'] = payment
            input_raw['TechSupport'] = tech_support
            input_raw['TotalCharges'] = tenure * monthly
            
            input_df = pd.DataFrame([input_raw])

            # Encoding categorical data
            for col, le in encoders.items():
                if col in input_df.columns:
                    try:
                        input_df[col] = le.transform([input_df[col][0]])
                    except:
                        input_df[col] = 0

            # Prediksi
            prob = model.predict_proba(input_df[feature_cols])[0][1]
            
            st.markdown("---")
            res_c1, res_c2 = st.columns([1, 2])
            with res_c1:
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100,
                    title = {'text': "Skor Risiko %"},
                    gauge = {'bar': {'color': "red" if prob > 0.5 else "blue"}}))
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with res_c2:
                if prob > 0.5:
                    st.error("### HASIL: BERISIKO TINGGI")
                    st.write("Pelanggan ini kemungkinan besar akan Churn. Disarankan memberikan promo retensi.")
                else:
                    st.success("### HASIL: RISIKO RENDAH")
                    st.write("Pelanggan cenderung loyal. Pertahankan kualitas layanan.")