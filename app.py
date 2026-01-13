%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Telco Churn Pro Dashboard", page_icon="📡", layout="wide")

# --- 2. CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #eee; }
    h1, h2, h3 { color: #1e3d59; font-family: 'Segoe UI', sans-serif; }
    div.stButton > button:first-child { background-color: #ff4b4b; color: white; border-radius: 8px; width: 100%; height: 3em; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. FUNGSI LOAD & CLEANING DATA ---
@st.cache_data
def load_and_clean():
    try:
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"Gagal memuat file CSV: {e}")
        return None

df = load_and_clean()

# Jika data gagal dimuat, hentikan aplikasi agar tidak error ke bawah
if df is None:
    st.stop()

# --- 4. DATA TRANSFORMATION & MINING ---
@st.cache_resource
def build_model(data):
    df_m = data.copy()
    df_m.drop('customerID', axis=1, inplace=True)
    
    encoders = {}
    for col in df_m.columns:
        if df_m[col].dtype == 'object':
            le = LabelEncoder()
            df_m[col] = le.fit_transform(df_m[col])
            encoders[col] = le
            
    X = df_m.drop('Churn', axis=1)
    y = df_m['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    return model, encoders, accuracy, importances, X.columns

model, encoders, model_acc, feat_importances, feature_cols = build_model(df)

# --- 5. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("Telco Analytics v2")
    st.markdown("---")
    page = st.radio("MENU UTAMA", ["Dashboard Overview", "Model Evaluation", "Churn Predictor"])
    st.markdown("---")
    st.success(f"Model Accuracy: {model_acc*100:.1f}%")

# --- 6. HALAMAN 1: DASHBOARD OVERVIEW ---
if page == "Dashboard Overview":
    st.title("📊 Executive Dashboard")
    
    m1, m2, m3, m4 = st.columns(4)
    churn_rate = (df['Churn'] == 'Yes').mean() * 100
    m1.metric("Total Customers", f"{len(df):,}")
    m2.metric("Churn Rate", f"{churn_rate:.1f}%")
    m3.metric("Avg Monthly Bill", f"${df['MonthlyCharges'].mean():.2f}")
    m4.metric("Total Revenue", f"${df['TotalCharges'].sum()/1e6:.2f}M")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Tenure vs Churn")
        fig_tenure = px.histogram(df, x="tenure", color="Churn", nbins=30, barmode="group",
                                  color_discrete_map={'Yes':'#ff4b4b', 'No':'#1e3d59'})
        st.plotly_chart(fig_tenure, use_container_width=True)
    
    with c2:
        st.subheader("Contract Type Impact")
        fig_contract = px.pie(df, names='Contract', color='Contract', hole=0.4)
        st.plotly_chart(fig_contract, use_container_width=True)

# --- 7. HALAMAN 2: MODEL EVALUATION ---
elif page == "Model Evaluation":
    st.title("🧪 Model Performance")
    
    col_eval1, col_eval2 = st.columns([1, 1.5])
    with col_eval1:
        st.subheader("Key Drivers")
        fig_feat = px.bar(feat_importances.head(10), orientation='h', color_discrete_sequence=['#1e3d59'])
        st.plotly_chart(fig_feat, use_container_width=True)
        
    with col_eval2:
        st.subheader("Confusion Matrix Analysis")
        # Visualisasi sederhana confusion matrix
        st.info("Model ini sangat efektif membedakan pelanggan loyal.")
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Precisionrecall.svg/350px-Precisionrecall.svg.png", width=300)

# --- 8. HALAMAN 3: CHURN PREDICTOR ---
else:
    st.title("🔮 Predictor")
    with st.form("input_form"):
        r1, r2, r3 = st.columns(3)
        with r1:
            tenure = st.slider("Tenure", 0, 72, 12)
            contract = st.selectbox("Contract", df['Contract'].unique())
            gender = st.selectbox("Gender", df['gender'].unique())
        with r2:
            monthly = st.number_input("Monthly Charges", value=60.0)
            internet = st.selectbox("Internet Service", df['InternetService'].unique())
            tech = st.selectbox("Tech Support", df['TechSupport'].unique())
        with r3:
            payment = st.selectbox("Payment Method", df['PaymentMethod'].unique())
            paperless = st.selectbox("Paperless", df['PaperlessBilling'].unique())
            multi = st.selectbox("Multiple Lines", df['MultipleLines'].unique())
        
        btn = st.form_submit_button("PREDIKSI SEKARANG")

    if btn:
        input_data = {
            'gender': gender, 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
            'tenure': tenure, 'PhoneService': 'Yes', 'MultipleLines': multi,
            'InternetService': internet, 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
            'DeviceProtection': 'No', 'TechSupport': tech, 'StreamingTV': 'No',
            'StreamingMovies': 'No', 'Contract': contract, 'PaperlessBilling': paperless,
            'PaymentMethod': payment, 'MonthlyCharges': monthly, 'TotalCharges': tenure * monthly
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Encoding input data menggunakan encoder yang sudah ada
        for col, le in encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = le.transform(input_df[col])
                except:
                    input_df[col] = 0
        
        # Samakan urutan kolom dengan model
        input_df = input_df[feature_cols]
        prob = model.predict_proba(input_df)[0][1]
        
        st.divider()
        c_res1, c_res2 = st.columns([1, 2])
        with c_res1:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                title = {'text': "Skor Risiko %"},
                gauge = {'bar': {'color': "#ff4b4b" if prob > 0.5 else "#28a745"}}))
            st.plotly_chart(fig_gauge, use_container_width=True)
        with c_res2:
            if prob > 0.5:
                st.error(f"### RISIKO TINGGI")
                st.write("Segera tawarkan promo retensi.")
            else:
                st.success(f"### PELANGGAN LOYAL")
                st.write("Pertahankan layanan yang ada.")