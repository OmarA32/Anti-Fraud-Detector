import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main():
    print("Fetching Credit Card Fraud dataset from OpenML (this may take a minute)...")
    # Fetch dataset (ID 1597 is the credit card fraud dataset)
    # Using as_frame=True to get a pandas DataFrame
    data = fetch_openml(data_id=1597, as_frame=True, parser='auto')
    
    df = data.frame
    X = data.data
    
    # OpenML target is often represented as strings ('1', '2' or '0', '1'). Let's convert to int.
    # For dataset 1597, Class is the target, '0' is normal, '1' is fraud.
    # Handling potential categorical string values like '1' and '2'.
    y = data.target
    if y.dtype == 'category' or y.dtype == 'object':
        # Sometimes fraud is '1' vs '0', or '2' vs '1' in openml. 
        # Let's map appropriately. The minority class is fraud.
        counts = y.value_counts()
        minority_label = counts.idxmin()
        y = (y == minority_label).astype(int)
    else:
        y = y.astype(int)

    print(f"Dataset loaded. Total samples: {len(df)}")
    print(f"Number of fraud cases: {y.sum()} ({y.sum()/len(y)*100:.2f}%)")

    # Split the dataset
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train the model
    # We use a RandomForest. Limit max_depth and n_estimators to keep training fast for this simple project.
    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Evaluate
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model
    model_filename = 'fraud_model.joblib'
    joblib.dump(clf, model_filename)
    print(f"Model saved to {model_filename}")

if __name__ == "__main__":
    main()
