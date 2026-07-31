import sys
import joblib
import pandas as pd

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_new.py <path_to_csv>")
        return

    csv_path = sys.argv[1]
    
    # Remove quotes if user dragged and dropped
    if csv_path.startswith('"') and csv_path.endswith('"'):
        csv_path = csv_path[1:-1]
        
    model_filename = 'fraud_model.joblib'
    
    try:
        clf = joblib.load(model_filename)
        print(f"Successfully loaded model from {model_filename}")
    except FileNotFoundError:
        print(f"Error: Could not find {model_filename}. Please run train.py first.")
        return

    try:
        df_new = pd.read_csv(csv_path)
        print(f"Successfully loaded data from {csv_path} ({len(df_new)} rows)")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    expected_features = clf.feature_names_in_
    
    # Filter only expected features, fill missing with 0
    missing_cols = [col for col in expected_features if col not in df_new.columns]
    for col in missing_cols:
        df_new[col] = 0.0
        
    X_new = df_new[expected_features]
    
    predictions = clf.predict(X_new)
    probabilities = clf.predict_proba(X_new)
    
    print("\n--- Predictions ---")
    df_results = df_new.copy()
    df_results['Prediction'] = ["Fraud" if p == 1 else "Normal" for p in predictions]
    df_results['Confidence'] = [probabilities[i][1] if p == 1 else probabilities[i][0] for i, p in enumerate(predictions)]
    
    print(df_results[['Prediction', 'Confidence']])
    
    output_csv = "predictions_output.csv"
    df_results.to_csv(output_csv, index=False)
    print(f"\nSaved full results to {output_csv}")

if __name__ == "__main__":
    main()
