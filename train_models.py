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
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score

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
    
    # Save database stats
    db_stats = {
        "total_transactions": total_rows,
        "total_fraud": total_fraud,
        "fraud_percentage": round((total_fraud / total_rows) * 100, 2)
    }
    with open('db_stats.json', 'w') as f:
        json.dump(db_stats, f)
        
    print(f"      -> Total Database Rows: {total_rows:,} | Fraud: {total_fraud:,}")
    
    # Downsample for local training speed
    fraud_df = df[df['isFraud'] == 1]
    normal_df = df[df['isFraud'] == 0].sample(n=100000, random_state=42)
    df_sampled = pd.concat([fraud_df, normal_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Validation set for calculating stats
    # 80% train, 20% validation
    train_df, val_df = train_test_split(df_sampled, test_size=0.2, random_state=42, stratify=df_sampled['isFraud'])
    
    features = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    X_train = train_df[features]
    y_train = train_df['isFraud']
    X_val = val_df[features]
    y_val = val_df['isFraud']
    
    # Preprocessing pipeline
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
    print("[3/5] Training Random Forest Classifier (Supervised)...")
    start_time = time.time()
    
    rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10, n_jobs=-1)
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

def train_autoencoder(X_train, y_train, X_val, y_val):
    print("[4/5] Training Autoencoder (Deep Learning Anomaly Detection)...")
    start_time = time.time()
    
    # Train only on normal data
    X_normal_train = X_train[y_train == 0]
    
    ae = MLPRegressor(hidden_layer_sizes=(16, 8, 16), max_iter=50, random_state=42)
    ae.fit(X_normal_train, X_normal_train)
    
    # Calculate threshold on training data (99.9th percentile)
    reconstructions = ae.predict(X_normal_train)
    mse = np.mean(np.power(X_normal_train - reconstructions, 2), axis=1)
    threshold = float(np.percentile(mse, 99.9))
    
    # Validation stats
    val_recon = ae.predict(X_val)
    val_mse = np.mean(np.power(X_val - val_recon, 2), axis=1)
    preds = (val_mse > threshold).astype(int)
    
    acc = accuracy_score(y_val, preds)
    recall = recall_score(y_val, preds, zero_division=0)
    
    joblib.dump(ae, 'autoencoder.joblib')
    with open('ae_threshold.txt', 'w') as f:
        f.write(str(threshold))
        
    return {
        "name": "Autoencoder",
        "training_time_sec": round(time.time() - start_time, 2),
        "validation_accuracy": round(acc * 100, 2),
        "fraud_detection_rate": round(recall * 100, 2),
        "threshold": round(threshold, 4)
    }

def train_isolation_forest(X_train, y_train, X_val, y_val):
    print("[5/5] Training Isolation Forest (Tree-Based Anomaly Detection)...")
    start_time = time.time()
    
    # Isolation forest handles data mixed with anomalies
    iso = IsolationForest(n_estimators=100, contamination=0.07, random_state=42, n_jobs=-1)
    iso.fit(X_train)
    
    preds_val = iso.predict(X_val)
    # IsolationForest returns -1 for anomaly, 1 for normal. Convert to 1 for fraud, 0 for normal
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
        model_stats['ae'] = train_autoencoder(X_train, y_train, X_val, y_val)
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
