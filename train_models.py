import kagglehub
import pandas as pd
import numpy as np
import os
import joblib
import json
import time
import sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, recall_score

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Hardware Acceleration Setup
device = torch.device("cpu")
try:
    import intel_extension_for_pytorch as ipex
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        print("[System] Hardware Acceleration ACTIVE: Intel XPU detected.")
    else:
        print("[System] Intel XPU not found. Defaulting to CPU.")
except ImportError:
    print("[System] intel_extension_for_pytorch not installed. Defaulting to CPU.")

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def download_dataset():
    print("[1/5] Downloading PaySim Dataset automatically from Kaggle...")
    try:
        path = kagglehub.dataset_download("sriharshaeedala/financial-fraud-detection-dataset")
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".csv"):
                    return os.path.join(root, file)
        raise FileNotFoundError("CSV file not found in downloaded dataset.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        sys.exit(1)

def preprocess_and_sample(csv_file):
    print(f"[2/5] Loading and Preprocessing MAX DATASET (6.3 Million Rows) from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    total_rows = len(df)
    total_fraud = int(df['isFraud'].sum())
    
    db_stats = {
        "total_transactions": total_rows,
        "total_fraud": total_fraud,
        "fraud_percentage": round((total_fraud / total_rows) * 100, 2)
    }
    with open('db_stats.json', 'w') as f:
        json.dump(db_stats, f)
        
    print(f"      -> Total Database Rows: {total_rows:,} | Fraud: {total_fraud:,}")
    
    # MAX POWER: Do not downsample! Use the full 6.3 Million rows.
    # We simply shuffle the dataframe.
    df_sampled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_df, val_df = train_test_split(df_sampled, test_size=0.2, random_state=42, stratify=df_sampled['isFraud'])
    
    features = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    X_train = train_df[features]
    y_train = train_df['isFraud']
    X_val = val_df[features]
    y_val = val_df['isFraud']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['type'])
        ])
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    joblib.dump(preprocessor, 'preprocessor.joblib')
    
    return X_train_processed, y_train, X_val_processed, y_val, df_sampled

def train_random_forest(X_train, y_train, X_val, y_val):
    print("[3/5] Training Aggressive Random Forest Classifier (Supervised) on CPU...")
    start_time = time.time()
    
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    preds = rf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    recall = recall_score(y_val, preds, zero_division=0)
    
    joblib.dump(rf, 'rf_model.joblib')
    
    return {
        "name": "Random Forest",
        "training_time_sec": round(time.time() - start_time, 2),
        "validation_accuracy": round(acc * 100, 2),
        "fraud_detection_rate": round(recall * 100, 2)
    }

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

def train_autoencoder_pytorch(X_train, y_train, X_val, y_val):
    print(f"[4/5] Training PyTorch Autoencoder (Deep Learning) on {device}...")
    start_time = time.time()
    
    X_normal_train = X_train[y_train == 0]
    
    tensor_X_train = torch.FloatTensor(X_normal_train)
    tensor_X_val = torch.FloatTensor(X_val).to(device)
    
    dataset = TensorDataset(tensor_X_train, tensor_X_train)
    # MASSIVE BATCH SIZE to fully utilize 16GB GPU VRAM
    dataloader = DataLoader(dataset, batch_size=8192, shuffle=True)
    
    input_dim = X_train.shape[1]
    model = PyTorchAutoencoder(input_dim).to(device)
    
    # Use ipex optimization if XPU is available
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    if device.type == "xpu":
        try:
            model, optimizer = ipex.optimize(model, optimizer=optimizer)
        except Exception as e:
            print(f"[Warning] Failed to apply IPEX optimization: {e}")
    
    # 30 Epochs for deep learning convergence on massive dataset
    epochs = 30
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"      -> Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss/len(dataloader):.6f}", flush=True)
            
    model.eval()
    with torch.no_grad():
        # Evaluate in chunks to prevent VRAM overflow during massive predictions
        train_recon = []
        for i in range(0, len(tensor_X_train), 8192):
            chunk = tensor_X_train[i:i+8192].to(device)
            train_recon.append(model(chunk).cpu())
        train_recon = torch.cat(train_recon)
        mse_train = torch.mean(torch.pow(tensor_X_train - train_recon, 2), dim=1).numpy()
        
    threshold = float(np.percentile(mse_train, 99.0))
    
    with torch.no_grad():
        val_recon = []
        for i in range(0, len(tensor_X_val), 8192):
            chunk = tensor_X_val[i:i+8192].to(device)
            val_recon.append(model(chunk).cpu())
        val_recon = torch.cat(val_recon)
        val_mse = torch.mean(torch.pow(tensor_X_val.cpu() - val_recon, 2), dim=1).numpy()
        
    preds = (val_mse > threshold).astype(int)
    
    acc = accuracy_score(y_val, preds)
    recall = recall_score(y_val, preds, zero_division=0)
    
    torch.save(model.state_dict(), 'pytorch_ae.pth')
    with open('ae_threshold.txt', 'w') as f:
        f.write(str(threshold))
        
    return {
        "name": "PyTorch Autoencoder",
        "training_time_sec": round(time.time() - start_time, 2),
        "validation_accuracy": round(acc * 100, 2),
        "fraud_detection_rate": round(recall * 100, 2),
        "threshold": round(threshold, 4)
    }

def train_isolation_forest(X_train, y_train, X_val, y_val):
    print("[5/5] Training Aggressive Isolation Forest (Tree-Based Anomaly) on CPU...")
    start_time = time.time()
    
    iso = IsolationForest(n_estimators=150, contamination=0.15, random_state=42, n_jobs=-1)
    iso.fit(X_train)
    
    preds_val = iso.predict(X_val)
    preds = np.where(preds_val == -1, 1, 0)
    
    acc = accuracy_score(y_val, preds)
    recall = recall_score(y_val, preds, zero_division=0)
    
    joblib.dump(iso, 'iso_forest_model.joblib')
    
    return {
        "name": "Isolation Forest",
        "training_time_sec": round(time.time() - start_time, 2),
        "validation_accuracy": round(acc * 100, 2),
        "fraud_detection_rate": round(recall * 100, 2)
    }

def main():
    try:
        csv_file = download_dataset()
        X_train, y_train, X_val, y_val, df_sampled = preprocess_and_sample(csv_file)
        
        model_stats = {}
        model_stats['rf'] = train_random_forest(X_train, y_train, X_val, y_val)
        model_stats['ae'] = train_autoencoder_pytorch(X_train, y_train, X_val, y_val)
        model_stats['iso'] = train_isolation_forest(X_train, y_train, X_val, y_val)
        
        with open('model_stats.json', 'w') as f:
            json.dump(model_stats, f, indent=4)
            
        print("\nAll 3 models successfully trained and saved!")
        print("Training complete. You can now run the Streamlit interface.")
        
    except Exception as e:
        print(f"\nFATAL ERROR during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
