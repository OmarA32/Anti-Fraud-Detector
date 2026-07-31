import kagglehub
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import pairwise_distances_argmin_min

print("[1/5] Downloading PaySim Dataset automatically from Kaggle...")
# This downloads the dataset automatically without needing an API key!
path = kagglehub.dataset_download("sriharshaeedala/financial-fraud-detection-dataset")

# Find the csv file in the downloaded folder
csv_file = None
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".csv"):
            csv_file = os.path.join(root, file)
            break

print(f"[2/5] Loading and Preprocessing Dataset from: {csv_file}")
# Load the dataset
df = pd.read_csv(csv_file)

# We want a sample so it trains in a reasonable time on local laptops.
# Keep all fraud, and a random sample of non-fraud.
fraud_df = df[df['isFraud'] == 1]
normal_df = df[df['isFraud'] == 0].sample(n=100000, random_state=42)

df_sampled = pd.concat([fraud_df, normal_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# Features and target
X = df_sampled[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
y = df_sampled['isFraud']

# Preprocessing pipeline
numeric_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
categorical_features = ['type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# Fit preprocessor
print("[3/5] Training Random Forest Classifier (Supervised)...")
X_processed = preprocessor.fit_transform(X)

# Save preprocessor
joblib.dump(preprocessor, 'preprocessor.joblib')

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10, n_jobs=-1)
rf_model.fit(X_processed, y)
joblib.dump(rf_model, 'rf_model.joblib')

print("[4/5] Training Autoencoder (Deep Learning Anomaly Detection)...")
# Train Autoencoder ONLY on normal data
X_normal = X[y == 0]
X_normal_processed = preprocessor.transform(X_normal)

# Simple neural network that tries to predict its own input
autoencoder = MLPRegressor(hidden_layer_sizes=(16, 8, 16), max_iter=50, random_state=42)
autoencoder.fit(X_normal_processed, X_normal_processed)

# Calculate reconstruction errors on normal data to find the 99.9th percentile threshold
reconstructions = autoencoder.predict(X_normal_processed)
mse = np.mean(np.power(X_normal_processed - reconstructions, 2), axis=1)
threshold = np.percentile(mse, 99.9)

joblib.dump(autoencoder, 'autoencoder.joblib')
with open('ae_threshold.txt', 'w') as f:
    f.write(str(threshold))

from sklearn.ensemble import RandomForestClassifier, IsolationForest

# ... skipping down to the 3rd model ...

print("[5/5] Training Isolation Forest (Anomaly Detection)...")
# Isolation Forest handles large datasets extremely well, so we can train it on the full sampled dataset
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1)
iso_forest.fit(X_processed)

# Save Isolation Forest model
joblib.dump(iso_forest, 'iso_forest_model.joblib')

print("\nAll 3 models successfully trained and saved!")
