import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import time

# PyTorch
import torch
import torch.nn as nn

# Hardware Acceleration Setup
device = torch.device("cpu")
try:
    import intel_extension_for_pytorch as ipex
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
except ImportError:
    pass

# --- PyTorch Autoencoder Architecture ---
class PyTorchAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(PyTorchAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

st.set_page_config(page_title="Fraud Sentinel AI", layout="wide", page_icon="🛡️")

# Custom CSS for glassmorphism and premium aesthetics
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #020617 0%, #1e1b4b 100%);
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
    .success-text { color: #10b981; font-weight: bold; font-size: 1.3rem; }
    .danger-text { color: #ef4444; font-weight: bold; font-size: 1.3rem; }
    h1, h2, h3 { color: #f8fafc; font-family: 'Inter', sans-serif; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models_and_stats():
    if not os.path.exists('rf_model.joblib') or not os.path.exists('pytorch_ae.pth'):
        return None, None, None, None, None, None, None
        
    preprocessor = joblib.load('preprocessor.joblib')
    rf_model = joblib.load('rf_model.joblib')
    
    # Load PyTorch Model (input dim is 10 due to OneHotEncoding of 'type' column)
    autoencoder = PyTorchAutoencoder(input_dim=10).to(device)
    autoencoder.load_state_dict(torch.load('pytorch_ae.pth', weights_only=True, map_location=device))
    autoencoder.eval() # Set to evaluation mode
    
    iso_forest = joblib.load('iso_forest_model.joblib')
    
    with open('ae_threshold.txt', 'r') as f:
        ae_threshold = float(f.read().strip())
        
    try:
        with open('db_stats.json', 'r') as f:
            db_stats = json.load(f)
        with open('model_stats.json', 'r') as f:
            model_stats = json.load(f)
    except:
        db_stats, model_stats = {}, {}
        
    return preprocessor, rf_model, autoencoder, ae_threshold, iso_forest, db_stats, model_stats

preprocessor, rf_model, autoencoder, ae_threshold, iso_forest, db_stats, model_stats = load_models_and_stats()

if preprocessor is None:
    st.error("🚀 Training in progress! Please wait for the script to finish and refresh the page.")
    st.stop()

# --- SIDEBAR: Configuration & Stats ---
with st.sidebar:
    st.title("🛡️ Fraud Sentinel AI")
    st.markdown("---")
    
    st.subheader("🧠 Engine Selection")
    model_choice = st.radio("Active Defense Algorithm:", [
        "🌲 Random Forest (Supervised)",
        "🤖 PyTorch Autoencoder (Deep Learning)",
        "🌲 Isolation Forest (Anomaly)"
    ])
    
    st.markdown("---")
    st.subheader("📊 Database Intelligence")
    if db_stats:
        # Format the huge numbers so they don't get cut off!
        total_rec_formatted = f"{db_stats['total_transactions'] / 1000000:.2f}M"
        total_fraud_formatted = f"{db_stats['total_fraud']:,}"
        
        col1, col2 = st.columns(2)
        col1.metric("Total Records", total_rec_formatted)
        col2.metric("Total Fraud", total_fraud_formatted)
        st.metric("Dataset Fraud Rate", f"{db_stats['fraud_percentage']}%")
        
    st.markdown("---")
    st.subheader("⚙️ Active Engine Stats")
    if model_stats:
        if "Random Forest" in model_choice:
            stats = model_stats['rf']
        elif "Autoencoder" in model_choice:
            stats = model_stats['ae']
        else:
            stats = model_stats['iso']
            
        st.metric("Validation Accuracy", f"{stats['validation_accuracy']}%")
        st.metric("Fraud Detection Rate", f"{stats['fraud_detection_rate']}%")
        st.metric("Training Time", f"{stats['training_time_sec']}s")

# --- MAIN DASHBOARD ---
st.title("Network Scanning Terminal")
st.markdown("Execute deep-packet inspection of financial transactions using state-of-the-art AI architectures.")

tab1, tab2 = st.tabs(["💳 Manual Single Scan", "📁 CSV Batch Processing"])

with tab1:
    st.subheader("Run a deep scan on a single transaction")
    
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            txn_type = st.selectbox("Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
            amount = st.number_input("Amount ($)", min_value=0.0, value=1500.0)
        with col2:
            oldbalanceOrg = st.number_input("Sender Old Balance", min_value=0.0, value=3000.0)
            newbalanceOrig = st.number_input("Sender New Balance", min_value=0.0, value=1500.0)
        with col3:
            oldbalanceDest = st.number_input("Recipient Old Balance", min_value=0.0, value=0.0)
            newbalanceDest = st.number_input("Recipient New Balance", min_value=0.0, value=1500.0)

    if st.button("🔍 Execute Deep Scan", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            'type': txn_type, 'amount': amount,
            'oldbalanceOrg': oldbalanceOrg, 'newbalanceOrig': newbalanceOrig,
            'oldbalanceDest': oldbalanceDest, 'newbalanceDest': newbalanceDest
        }])
        
        with st.spinner(f"Routing transaction to {model_choice} engine for structural analysis..."):
            time.sleep(1.2) # Simulate deep scan
            X_processed = preprocessor.transform(input_data)
            
            is_fraud = False
            msg = ""
            
            if "Random Forest" in model_choice:
                pred = rf_model.predict(X_processed)[0]
                prob = rf_model.predict_proba(X_processed)[0][1] * 100
                is_fraud = (pred == 1)
                msg = f"Random Forest Pattern Match: {'FRAUD' if is_fraud else 'CLEAN'} ({prob:.1f}% Confidence)"
                
            elif "Autoencoder" in model_choice:
                # PyTorch Inference
                tensor_X = torch.FloatTensor(X_processed).to(device)
                with torch.no_grad():
                    recon = autoencoder(tensor_X)
                    mse = torch.mean(torch.pow(tensor_X - recon, 2), dim=1)[0].item()
                
                is_fraud = (mse > ae_threshold)
                msg = f"Reconstruction MSE: {mse:.4f} (Threshold: {ae_threshold:.4f}). "
                msg += "High anomaly score detected." if is_fraud else "Low anomaly score."
                
            elif "Isolation Forest" in model_choice:
                pred = iso_forest.predict(X_processed)[0]
                is_fraud = (pred == -1)
                msg = "Transaction successfully isolated deep in the forest structure (High Anomaly)." if is_fraud else "Transaction blended with normal density groups."

            st.markdown("---")
            if is_fraud:
                st.markdown(f'<p class="danger-text">🚨 THREAT DETECTED: CONNECTION SEVERED</p>', unsafe_allow_html=True)
                st.error(msg)
            else:
                st.markdown(f'<p class="success-text">✅ VERIFIED: TRANSACTION SECURED</p>', unsafe_allow_html=True)
                st.success(msg)

with tab2:
    st.subheader("Mass Transaction Inspection")
    with st.container(border=True):
        uploaded_file = st.file_uploader("Upload PaySim CSV dataset segment", type="csv")
        
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df):,} transactions for inspection.")
            
            if st.button("Initiate Grid Scan", type="primary"):
                with st.spinner("Analyzing mass batch for anomalies..."):
                    time.sleep(1.5)
                    X_batch_processed = preprocessor.transform(batch_df)
                    
                    if "Random Forest" in model_choice:
                        preds = rf_model.predict(X_batch_processed)
                        results = (preds == 1)
                    elif "Autoencoder" in model_choice:
                        tensor_X_batch = torch.FloatTensor(X_batch_processed).to(device)
                        with torch.no_grad():
                            recon = autoencoder(tensor_X_batch)
                            mses = torch.mean(torch.pow(tensor_X_batch - recon, 2), dim=1).cpu().numpy()
                        results = mses > ae_threshold
                    elif "Isolation Forest" in model_choice:
                        preds = iso_forest.predict(X_batch_processed)
                        results = (preds == -1)
                    
                    batch_df['AI_Flag'] = ['Fraud' if r else 'Normal' for r in results]
                    fraud_count = sum(results)
                    
                    c1, c2 = st.columns(2)
                    c1.metric("🚨 Total Threats Detected", f"{fraud_count:,}")
                    c2.metric("✅ Cleared Transactions", f"{len(batch_df) - fraud_count:,}")
                    
                    st.markdown("### Threat Topography Map")
                    st.markdown("Visualizing Transaction Amount vs. Sender Balance")
                    # Scatter chart coloring by our AI_Flag
                    st.scatter_chart(data=batch_df, x='oldbalanceOrg', y='amount', color='AI_Flag', height=400)
