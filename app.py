import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Configure the page
st.set_page_config(
    page_title="AI Fraud Sentinel",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS for glassmorphism and modern look
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0d0e15 0%, #1a1c2c 100%);
        color: white;
    }
    h1 {
        color: #00ffff !important;
        text-align: center;
        text-shadow: 0 0 10px rgba(0,255,255,0.5);
    }
    .glass-panel {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .fraud-alert {
        background: rgba(255, 0, 85, 0.2);
        border: 1px solid #ff0055;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        color: #ff0055;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .safe-alert {
        background: rgba(0, 255, 136, 0.2);
        border: 1px solid #00ff88;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        color: #00ff88;
        font-weight: bold;
        font-size: 1.5rem;
    }
    /* Make standard Streamlit widgets blend in */
    .stNumberInput > div > div > input {
        color: white;
        background-color: rgba(255, 255, 255, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ AI Fraud Sentinel")
st.markdown("<p style='text-align: center; color: #a0a5b5;'>Local Machine Learning Fraud Detection</p>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if os.path.exists('fraud_model.joblib'):
        return joblib.load('fraud_model.joblib')
    return None

model = load_model()

if model is None:
    st.error("⚠️ Model not found! Please close this window, run the training bat file, and try again.")
    st.stop()

expected_features = model.feature_names_in_

# Layout with two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Test a Single Transaction")
    
    amount = st.number_input("Amount ($)", value=15.99, step=0.01)
    time = st.number_input("Time (Seconds since first transaction)", value=0.0)
    v1 = st.number_input("V1 (PCA Component 1)", value=0.0)
    v2 = st.number_input("V2 (PCA Component 2)", value=0.0)
    
    st.markdown("<small style='color: #a0a5b5;'>Note: V3 through V28 will be auto-filled with 0.0 for this manual test.</small>", unsafe_allow_html=True)
    
    if st.button("Predict Fraud", use_container_width=True, type="primary"):
        row = {}
        for f in expected_features:
            if f.lower() == 'amount': row[f] = amount
            elif f.lower() == 'time': row[f] = time
            elif f.lower() == 'v1': row[f] = v1
            elif f.lower() == 'v2': row[f] = v2
            else: row[f] = 0.0
            
        df_test = pd.DataFrame([row])
        pred = model.predict(df_test)[0]
        prob = model.predict_proba(df_test)[0]
        
        st.markdown("### Result")
        if pred == 1:
            st.markdown(f"<div class='fraud-alert'>🚨 FRAUD DETECTED<br><span style='font-size:1rem'>Confidence: {prob[1]*100:.2f}%</span></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='safe-alert'>✅ NORMAL<br><span style='font-size:1rem'>Confidence: {prob[0]*100:.2f}%</span></div>", unsafe_allow_html=True)
            

with col2:
    st.subheader("Batch Testing (Upload CSV)")
    
    uploaded_file = st.file_uploader("Upload a CSV with transactions", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} transactions.")
            
            # Ensure all features exist
            for f in expected_features:
                if f not in df.columns:
                    df[f] = 0.0
            
            # Reorder columns
            X = df[expected_features]
            
            if st.button("Run Batch Prediction", type="primary", use_container_width=True):
                preds = model.predict(X)
                df['Fraud_Prediction'] = preds
                
                fraud_count = sum(preds == 1)
                st.write(f"**Found {fraud_count} fraudulent transactions!**")
                
                # Show fraudulent ones
                if fraud_count > 0:
                    st.dataframe(df[df['Fraud_Prediction'] == 1].head(10))
                else:
                    st.success("No fraud found in this batch!")
                    
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
