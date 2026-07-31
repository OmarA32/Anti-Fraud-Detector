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
    print(f"[2/5] Loading and Preprocessing Dataset from: {csv_file}")
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
    
    fraud_df = df[df['isFraud'] == 1]
    normal_df = df[df['isFraud'] == 0].sample(n=100000, random_state=42)
    df_sampled = pd.concat([fraud_df, normal_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    
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
    print("[3/5] Training Aggressive Random Forest Classifier (Supervised)...")
    start_time = time.time()
    
    # HYPERPARAMETER OPTIMIZATION: 
    # class_weight='balanced' drastically penalizes missing fraud cases
    # max_depth=15 allows the model to learn more complex patterns
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
        # Encoder (Compresses the data)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        # Decoder (Reconstructs the data)
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
    print("[4/5] Training PyTorch Autoencoder (Deep Learning)...")
    start_time = time.time()
    
    # Train only on normal data
    X_normal_train = X_train[y_train == 0]
    
    # Convert numpy arrays to PyTorch tensors
    tensor_X_train = torch.FloatTensor(X_normal_train)
    tensor_X_val = torch.FloatTensor(X_val)
    
    # Create DataLoader
    dataset = TensorDataset(tensor_X_train, tensor_X_train)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    input_dim = X_train.shape[1]
    model = PyTorchAutoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # Training Loop (15 Epochs)
    epochs = 15
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    # Calculate Reconstruction Error on Training Data
    model.eval()
    with torch.no_grad():
        train_recon = model(tensor_X_train)
        # Calculate MSE per row
        mse_train = torch.mean(torch.pow(tensor_X_train - train_recon, 2), dim=1).numpy()
        
    # HYPERPARAMETER OPTIMIZATION:
    # Lower threshold from 99.9 to 99.0 to catch way more anomalies!
    threshold = float(np.percentile(mse_train, 99.0))
    
    # Validation stats
    with torch.no_grad():
        val_recon = model(tensor_X_val)
        val_mse = torch.mean(torch.pow(tensor_X_val - val_recon, 2), dim=1).numpy()
        
    preds = (val_mse > threshold).astype(int)
    
    acc = accuracy_score(y_val, preds)
    recall = recall_score(y_val, preds, zero_division=0)
    
    # Save the PyTorch Model
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
    print("[5/5] Training Aggressive Isolation Forest (Tree-Based Anomaly)...")
    start_time = time.time()
    
    # HYPERPARAMETER OPTIMIZATION:
    # contamination=0.15 makes it flag significantly more transactions as outliers
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
