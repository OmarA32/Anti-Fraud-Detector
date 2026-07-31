import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import pairwise_distances_argmin_min

# Set page config
st.set_page_config(page_title="AI Fraud Sentinel", layout="wide", page_icon="🛡️")

# Custom CSS for glassmorphism
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: white;
    }
    .stApp {
        background-image: radial-gradient(circle at 50% -20%, #4c1d95, transparent 50%);
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-text {
        color: #10b981;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .danger-text {
        color: #ef4444;
        font-weight: bold;
        font-size: 1.2rem;
    }
    h1, h2, h3 {
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.05);
        color: white;
    }
    .stNumberInput > div > div > input {
        color: white;
        background-color: rgba(255, 255, 255, 0.05);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    if not os.path.exists('rf_model.joblib'):
        return None, None, None, None, None
    
    preprocessor = joblib.load('preprocessor.joblib')
    rf_model = joblib.load('rf_model.joblib')
    autoencoder = joblib.load('autoencoder.joblib')
    with open('ae_threshold.txt', 'r') as f:
        ae_threshold = float(f.read().strip())
    iso_forest = joblib.load('iso_forest_model.joblib')
    
    return preprocessor, rf_model, autoencoder, ae_threshold, iso_forest

preprocessor, rf_model, autoencoder, ae_threshold, iso_forest = load_models()

st.title("🛡️ AI Fraud Sentinel (Multi-Model)")
st.markdown("Test your financial transactions against **3 entirely different AI philosophies**.")

if preprocessor is None:
    st.error("Models not found! Please wait for the training script to finish.")
    st.stop()

# --- MODEL SELECTION ---
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("🧠 Select Artificial Intelligence Engine")
model_choice = st.selectbox("Choose the algorithm to process the transaction:", [
    "🌲 Random Forest (Supervised Classification)",
    "🤖 Autoencoder (Deep Learning Reconstruction Anomaly)",
    "🌲 Isolation Forest (Tree-Based Anomaly Detection)"
])
st.markdown('</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

# --- MANUAL TRANSACTION TESTING ---
with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("💳 Single Transaction Test")
    st.markdown("Enter transaction details using human-readable features (PaySim).")
    
    txn_type = st.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
    amount = st.number_input("Amount ($)", min_value=0.0, value=500.0)
    
    st.markdown("##### Sender Details")
    oldbalanceOrg = st.number_input("Sender Initial Balance", min_value=0.0, value=1000.0)
    newbalanceOrig = st.number_input("Sender New Balance", min_value=0.0, value=500.0)
    
    st.markdown("##### Recipient Details")
    oldbalanceDest = st.number_input("Recipient Initial Balance", min_value=0.0, value=0.0)
    newbalanceDest = st.number_input("Recipient New Balance", min_value=0.0, value=500.0)

    if st.button("🔍 Scan Transaction", type="primary", use_container_width=True):
        # Prepare data
        input_data = pd.DataFrame([{
            'type': txn_type,
            'amount': amount,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrig': newbalanceOrig,
            'oldbalanceDest': oldbalanceDest,
            'newbalanceDest': newbalanceDest
        }])
        
        # Preprocess
        X_processed = preprocessor.transform(input_data)
        
        is_fraud = False
        message = ""
        
        # 1. RANDOM FOREST
        if "Random Forest" in model_choice:
            pred = rf_model.predict(X_processed)[0]
            prob = rf_model.predict_proba(X_processed)[0][1] * 100
            is_fraud = (pred == 1)
            message = f"Random Forest voted: {'Fraud' if is_fraud else 'Normal'} ({prob:.2f}% confidence)."
            
        # 2. AUTOENCODER
        elif "Autoencoder" in model_choice:
            reconstruction = autoencoder.predict(X_processed)
            mse = np.mean(np.power(X_processed - reconstruction, 2), axis=1)[0]
            is_fraud = (mse > ae_threshold)
            message = f"Autoencoder Reconstruction Error: {mse:.4f} (Threshold is {ae_threshold:.4f}). "
            message += "High error means the AI has never seen this pattern before!" if is_fraud else "Low error means this looks like perfectly normal behavior."
            
        # 3. ISOLATION FOREST
        elif "Isolation Forest" in model_choice:
            pred = iso_forest.predict(X_processed)[0]
            # Isolation Forest returns -1 for anomaly (fraud) and 1 for normal
            is_fraud = (pred == -1)
            message = "Isolation Forest Result: Anomaly! The math of this transaction looks extremely weird." if is_fraud else "Isolation Forest Result: Normal. This blends right in with standard transactions."

        # Display result
        st.markdown("---")
        if is_fraud:
            st.markdown(f'<p class="danger-text">🚨 THREAT DETECTED: SUSPECTED FRAUD</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="success-text">✅ VERIFIED: NORMAL TRANSACTION</p>', unsafe_allow_html=True)
            
        st.info(message)
        
    st.markdown('</div>', unsafe_allow_html=True)


# --- BATCH TRANSACTION TESTING ---
with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📁 CSV Batch Processing")
    st.markdown("Upload a CSV file with PaySim columns to scan thousands of transactions instantly.")
    
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(batch_df)} transactions.")
        
        required_cols = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        if not all(col in batch_df.columns for col in required_cols):
            st.error(f"CSV must contain the following columns: {required_cols}")
        else:
            if st.button("Scan Entire Batch", type="primary", use_container_width=True):
                X_batch_processed = preprocessor.transform(batch_df)
                
                results = []
                if "Random Forest" in model_choice:
                    preds = rf_model.predict(X_batch_processed)
                    results = preds == 1
                elif "Autoencoder" in model_choice:
                    recon = autoencoder.predict(X_batch_processed)
                    mses = np.mean(np.power(X_batch_processed - recon, 2), axis=1)
                    results = mses > ae_threshold
                elif "Isolation Forest" in model_choice:
                    preds = iso_forest.predict(X_batch_processed)
                    results = preds == -1
                
                fraud_count = sum(results)
                normal_count = len(results) - fraud_count
                
                st.markdown(f"### Results using {model_choice.split()[1]}")
                st.markdown(f"<p class='danger-text'>🚨 Fraudulent Transactions: {fraud_count}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='success-text'>✅ Normal Transactions: {normal_count}</p>", unsafe_allow_html=True)
                
    st.markdown('</div>', unsafe_allow_html=True)
