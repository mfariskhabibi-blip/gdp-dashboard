%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- 1. CONFIG HALAMAN ---
st.set_page_config(page_title="Telco Churn Insight Pro", page_icon="📡", layout="wide")

# --- 2. STYLE CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    div[data-testid="stExpander"] { background-color: #ffffff; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD & PREPROCESS DATA ---
@st.cache_data
def load_and_process_data():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    return df

df = load_and_process_data()

# --- 4. MODEL TRAINING (DILAKUKAN SEKALI SAAT APP DIBUKA) ---
@st.cache_resource
def train_model(data):
    df_model = data.copy()
    df_model.drop('customerID', axis=1, inplace=True)
    
    # Encoding categorical data
    le_dict = {}
    for col in df_model.columns:
        if df_model[col].dtype == 'object':
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col])
            le_dict[col] = le
            
    X = df_model.drop('Churn', axis=1)
    y = df_model['Churn']
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return rf, le_dict, X.columns

model, encoders, feature_names = train_model(df)

# --- 5. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2040/2040504.png", width=80)
    st.title("Telco Dashboard")
    st.markdown("---")
    menu = st.radio("Pilih Menu:", ["Business Overview", "Predictive Analytics"])
    st.markdown("---")
    st.write("**Metode:** KDD Process")
    st.write("**Model:** Random Forest Classifier")

# --- 6. HALAMAN 1: BUSINESS OVERVIEW ---
if menu == "Business Overview":
    st.title("📊 Executive Business Dashboard")
    
    # Row 1: Key Metrics
    m1, m2, m3, m4 = st.columns(4)
    churn_val = (df['Churn'] == 'Yes').sum()
    churn_rate = (churn_val / len(df)) * 100
    
    m1.metric("Total Customers", f"{len(df):,}")
    m2.metric("Churn Rate", f"{churn_rate:.1f}%", delta="-1.2%")
    m3.metric("Avg Monthly Bill", f"${df['MonthlyCharges'].mean():.2f}")
    m4.metric("Total Revenue", f"${df['TotalCharges'].sum()/1e6:.2f}M")

    st.markdown("---")

    # Row 2: Charts
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Churn Distribution by Contract")
        fig_cont = px.histogram(df, x="Contract", color="Churn", barmode="group", 
                                color_discrete_map={'Yes':'#EF553B', 'No':'#636EFA'})
        st.plotly_chart(fig_cont, use_container_width=True)
        
    with c2:
        st.subheader("Monthly Charges vs Churn")
        fig_box = px.box(df, x="Churn", y="MonthlyCharges", color="Churn",
                         color_discrete_map={'Yes':'#EF553B', 'No':'#636EFA'})
        st.plotly_chart(fig_box, use_container_width=True)

# --- 7. HALAMAN 2: PREDICTIVE ANALYTICS ---
else:
    st.title("🔮 Customer Churn Prediction")
    st.info("Input data pelanggan di bawah ini untuk melihat probabilitas Churn.")

    with st.form("predict_form"):
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.subheader("Demografi")
            gender = st.selectbox("Gender", df['gender'].unique())
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", df['Partner'].unique())
            tenure = st.slider("Tenure (Bulan)", 0, 72, 12)

        with col_b:
            st.subheader("Layanan")
            internet = st.selectbox("Internet Service", df['InternetService'].unique())
            tech = st.selectbox("Tech Support", df['TechSupport'].unique())
            contract = st.selectbox("Contract", df['Contract'].unique())
            method = st.selectbox("Payment Method", df['PaymentMethod'].unique())

        with col_c:
            st.subheader("Finansial")
            monthly = st.number_input("Monthly Charges ($)", value=50.0)
            total = st.number_input("Total Charges ($)", value=float(tenure * 50))
            paperless = st.selectbox("Paperless Billing", df['PaperlessBilling'].unique())
            multi = st.selectbox("Multiple Lines", df['MultipleLines'].unique())

        submitted = st.form_submit_button("ANALISIS RISIKO PELANGGAN")

    if submitted:
        # Menyiapkan data untuk prediksi
        input_raw = {
            'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 
            'Dependents': 'No', 'tenure': tenure, 'PhoneService': 'Yes', 
            'MultipleLines': multi, 'InternetService': internet, 'OnlineSecurity': 'No',
            'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': tech,
            'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': contract,
            'PaperlessBilling': paperless, 'PaymentMethod': method, 
            'MonthlyCharges': monthly, 'TotalCharges': total
        }
        
        input_df = pd.DataFrame([input_raw])
        
        # Encoding input sesuai dengan training
        for col, le in encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])
        
        # Prediksi
        prob = model.predict_proba(input_df[feature_names])[0][1]
        
        # Tampilan Hasil
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Skor Risiko %", 'font': {'size': 24}},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#EF553B" if prob > 0.5 else "#636EFA"},
                    'steps' : [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}]}))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            if prob > 0.5:
                st.error(f"### PELANGGAN BERISIKO TINGGI")
                st.write("**Analisis:** Pelanggan memiliki kecenderungan kuat untuk berhenti berlangganan.")
                st.write("**Rekomendasi:** Berikan penawaran khusus pada kontrak " + ("Two Year" if contract != "Two year" else "Loyalty Bonus") + ".")
            else:
                st.success(f"### PELANGGAN AMAN (LOYAL)")
                st.write("**Analisis:** Pelanggan kemungkinan besar akan tetap berlangganan.")
                st.write("**Rekomendasi:** Tawarkan upselling produk internet atau layanan streaming.")