import pandas as pd
from sklearn.datasets import fetch_openml

def main():
    print("Loading cached OpenML data...")
    data = fetch_openml(data_id=1597, as_frame=True, parser='auto')
    
    y = data.target
    if y.dtype == 'category' or y.dtype == 'object':
        counts = y.value_counts()
        minority_label = counts.idxmin()
        fraud_mask = (y == minority_label)
    else:
        fraud_mask = (y == 1)
        
    X_fraud = data.data[fraud_mask].copy()
    
    # Take 5 known fraud transactions
    sample_fraud = X_fraud.head(5)
    
    sample_fraud.to_csv("true_fraud_data.csv", index=False)
    print("Saved 5 true fraud samples to true_fraud_data.csv")

if __name__ == "__main__":
    main()
