# Anti-Fraud-detector

A simple machine learning project to detect fraudulent credit card transactions. 
It fetches a real-world dataset from OpenML, trains a Random Forest model, and provides a script to make predictions.

## Setup

1. Make sure you have Python installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Train the Model:**
   Run the training script to fetch the dataset, train the model, and save it as `fraud_model.joblib`.
   ```bash
   python train.py
   ```

2. **Make Predictions:**
   Run the prediction script to see the model in action on some sample synthetic transactions.
   ```bash
   python predict.py
   ```
