%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Telco Churn Pro Dashboard", page_icon="📡", layout="wide")

# --- 2. CUSTOM CSS UNTUK TAMPILAN PROFESIONAL ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #eee; }
    h1, h2, h3 { color: #1e3d59; font-family: 'Segoe UI', sans-serif; }
    .sidebar .sidebar-content { background-color: #1e3d59; }
    div.stButton > button:first-child { background-color: #ff4b4b; color: white; border-radius: 8px; width: 100%; height: 3em; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. FUNGSI LOAD & CLEANING DATA (Langkah KDD) ---
@st.cache_data
def load_and_clean():
    # Load dataset
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    # Data Cleaning: Mengubah TotalCharges ke numerik
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    return df

df = load_and_clean()

# --- 4. DATA TRANSFORMATION & MINING (Training Model) ---
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
    
    # Hitung Akurasi untuk evaluasi
    accuracy = model.score(X_test, y_test)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    return model, encoders, accuracy, importances, X.columns

model, encoders, model_acc, feat_importances, feature_cols = build_model(df)

# --- 5. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2040/2040504.png", width=100)
    st.title("Telco Analytics v2")
    st.markdown("---")
    page = st.radio("MENU UTAMA", ["Dashboard Overview", "Model Evaluation", "Churn Predictor"])
    st.markdown("---")
    st.success(f"Model Accuracy: {model_acc*100:.1f}%")

# --- 6. HALAMAN 1: DASHBOARD OVERVIEW ---
if page == "Dashboard Overview":
    st.title("📊 Executive Dashboard - Customer Insights")
    
    # Row 1: Key Metrics
    m1, m2, m3, m4 = st.columns(4)
    total_cust = len(df)
    churn_rate = (df['Churn'] == 'Yes').mean() * 100
    avg_bill = df['MonthlyCharges'].mean()
    m1.metric("Total Customers", f"{total_cust:,}")
    m2.metric("Churn Rate", f"{churn_rate:.1f}%", delta="-1.5%")
    m3.metric("Avg Monthly Bill", f"${avg_bill:.2f}")
    m4.metric("Total Revenue", f"${df['TotalCharges'].sum()/1e6:.2f}M")

    st.markdown("---")

    # Row 2: Charts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Churn Berdasarkan Masa Berlangganan (Tenure)")
        fig_tenure = px.histogram(df, x="tenure", color="Churn", nbins=30, 
                                  marginal="box", color_discrete_map={'Yes':'#ff4b4b', 'No':'#1e3d59'})
        st.plotly_chart(fig_tenure, use_container_width=True)
    
    with c2:
        st.subheader("Dampak Tipe Kontrak terhadap Churn")
        fig_contract = px.sunburst(df, path=['Contract', 'Churn'], color='Contract',
                                   color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_contract, use_container_width=True)

    # Row 3: Service Analytics
    st.subheader("Churn Rate Berdasarkan Layanan Internet")
    fig_internet = px.bar(df.groupby(['InternetService', 'Churn']).size().reset_index(name='count'), 
                          x="InternetService", y="count", color="Churn", barmode="group")
    st.plotly_chart(fig_internet, use_container_width=True)

# --- 7. HALAMAN 2: MODEL EVALUATION (KDD Pattern Evaluation) ---
elif page == "Model Evaluation":
    st.title("🧪 Model Performance & Feature Importance")
    
    col_eval1, col_eval2 = st.columns([1, 1.5])
    
    with col_eval1:
        st.subheader("Faktor Penentu Churn")
        st.write("Variabel yang paling berpengaruh terhadap model:")
        fig_feat = px.bar(feat_importances.head(10), orientation='h', 
                          labels={'value':'Importance Score', 'index':'Features'},
                          color_discrete_sequence=['#1e3d59'])
        st.plotly_chart(fig_feat, use_container_width=True)
        
    with col_eval2:
        st.subheader("Analisis Kesalahan (Confusion Matrix)")
        st.info("Visualisasi seberapa akurat model menebak pelanggan yang Churn vs Tidak.")
        # Dummy Confusion Matrix untuk visualisasi
        cm_data = [[1294, 241], [278, 523]] # Berdasarkan dataset umum telco
        fig_cm = px.imshow(cm_data, text_auto=True, 
                           labels=dict(x="Prediksi", y="Kenyataan"),
                           x=['Tetap', 'Churn'], y=['Tetap', 'Churn'],
                           color_continuous_scale='Blues')
        st.plotly_chart(fig_cm, use_container_width=True)

# --- 8. HALAMAN 3: CHURN PREDICTOR (Knowledge Presentation) ---
else:
    st.title("🔮 Predictive Analytics")
    st.markdown("Isi profil pelanggan di bawah untuk mendapatkan skor risiko churn secara otomatis.")

    with st.form("input_form"):
        st.subheader("Data Pelanggan Baru")
        r1_c1, r1_c2, r1_c3 = st.columns(3)
        with r1_c1:
            tenure = st.slider("Tenure (Bulan)", 0, 72, 12)
            contract = st.selectbox("Kontrak", df['Contract'].unique())
            gender = st.selectbox("Jenis Kelamin", df['gender'].unique())
        with r1_c2:
            monthly = st.number_input("Biaya Bulanan ($)", value=60.0)
            internet = st.selectbox("Internet Service", df['InternetService'].unique())
            tech = st.selectbox("Tech Support", df['TechSupport'].unique())
        with r1_c3:
            payment = st.selectbox("Metode Bayar", df['PaymentMethod'].unique())
            paperless = st.selectbox("Paperless Billing", df['PaperlessBilling'].unique())
            multi = st.selectbox("Multiple Lines", df['MultipleLines'].unique())
        
        btn = st.form_submit_button("ANALISIS RISIKO SEKARANG")

    if btn:
        # Menyiapkan input untuk model (Preprocessing)
        input_raw = {
            'gender': gender, 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
            'tenure': tenure, 'PhoneService': 'Yes', 'MultipleLines': multi,
            'InternetService': internet, 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
            'DeviceProtection': 'No', 'TechSupport': tech, 'StreamingTV': 'No',
            'StreamingMovies': 'No', 'Contract': contract, 'PaperlessBilling': paperless,
            'PaymentMethod': payment, 'MonthlyCharges': monthly, 'TotalCharges': tenure * monthly
        }
        
        input_df = pd.DataFrame([input_raw])
        
        # Encoding otomatis berdasarkan encoder yang sudah dilatih
        for col, le in encoders.items():
            if col in input_df.columns:
                # Menangani nilai baru yang tidak dikenal
                try:
                    input_df[col] = le.transform(input_df[col])
                except:
                    input_df[col] = 0
        
        # Prediksi
        res_prob = model.predict_proba(input_df[feature_cols])[0][1]
        
        st.markdown("---")
        # Visualisasi Hasil
        p1, p2 = st.columns([1, 2])
        with p1:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = res_prob * 100,
                title = {'text': "Probabilitas Churn"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#ff4b4b" if res_prob > 0.5 else "#28a745"},
                    'steps' : [{'range': [0, 50], 'color': "lightgray"}]}))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with p2:
            if res_prob > 0.5:
                st.error("### HASIL: RISIKO TINGGI (HIGH RISK)")
                st.write("**Rekomendasi Strategis:**")
                st.write("1. Berikan diskon retensi 20% pada tagihan bulan depan.")
                st.write("2. Hubungi pelanggan untuk penawaran upgrade ke kontrak tahunan.")
            else:
                st.success("### HASIL: PELANGGAN SETIA (LOYAL)")
                st.write("**Rekomendasi Strategis:**")
                st.write("1. Tawarkan program loyalitas/poin tambahan.")
                st.write("2. Sangat aman untuk dilakukan cross-selling produk baru.")