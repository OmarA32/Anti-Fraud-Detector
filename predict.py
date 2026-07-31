import joblib
import pandas as pd
import numpy as np

def main():
    model_filename = 'fraud_model.joblib'
    try:
        clf = joblib.load(model_filename)
        print(f"Successfully loaded model from {model_filename}")
    except FileNotFoundError:
        print(f"Error: Could not find {model_filename}. Please run train.py first.")
        return

    print("\n--- Testing with Sample Data ---")
    
    # Credit Card Fraud dataset features typically look like:
    # V1 to V28, Amount. (Total 29 features for openml dataset 1597, but wait, usually Time is included or excluded)
    # The exact features depend on fetch_openml. 
    # To be robust, let's load the model and see what features it expects from its internal state.
    expected_features = clf.feature_names_in_
    
    # Let's create dummy samples based on expected features.
    sample_normal = []
    sample_anomaly = []
    
    for f in expected_features:
        if f.lower() == 'amount':
            sample_normal.append(15.99)
            sample_anomaly.append(9999.99)
        elif f.lower() == 'time':
            sample_normal.append(100.0)
            sample_anomaly.append(1000.0)
        else:
            # PCA components V1..V28 usually mean ~0 for normal
            sample_normal.append(0.0)
            # Add some extreme values for anomaly
            sample_anomaly.append(10.0 if np.random.rand() > 0.5 else -10.0)

    df_test = pd.DataFrame([sample_normal, sample_anomaly], columns=expected_features)
    
    print("\nSample Transactions:")
    print(df_test.head())
    
    predictions = clf.predict(df_test)
    probabilities = clf.predict_proba(df_test)
    
    print("\nPredictions:")
    for i, pred in enumerate(predictions):
        label = "Fraud" if pred == 1 else "Normal"
        prob = probabilities[i][1] if pred == 1 else probabilities[i][0]
        print(f"Transaction {i+1}: {label} (Confidence: {prob:.2%})")

if __name__ == "__main__":
    main()
