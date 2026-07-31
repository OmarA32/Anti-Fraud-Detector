# 🛡️ AI Fraud Sentinel

A complete, production-ready Multi-Model Cybersecurity Dashboard designed to detect anomalous financial transactions in massive datasets using a combination of supervised and unsupervised machine learning algorithms.

![Interface Screenshot](assets/screenshot.png)

## 🚀 Features

- **Automated Data Pipeline**: Silently pulls the massive 500MB+ [Kaggle PaySim Database](https://www.kaggle.com/datasets/sriharshaeedala/financial-fraud-detection-dataset) without requiring API keys or manual downloads.
- **Deep Packet Inspection UI**: A custom, dark-mode, glassmorphism UI built with Streamlit featuring live AI engine metrics, deep-scan simulations, and threat topography scatter plots.
- **One-Click Hacker Boot**: The entire system is modularized and launched securely via `start_interface.bat`.

## 🧠 3-Model AI Architecture

This project employs three distinct AI philosophies to battle financial fraud, which you can hot-swap in real-time from the dashboard:

1. 🌲 **Random Forest (Supervised Classification)**: Learns from known fraudulent transaction rules and splits data perfectly. Highly accurate. We utilize `class_weight='balanced'` to massively penalize missing fraud.
2. 🤖 **PyTorch Autoencoder (Deep Learning)**: A custom PyTorch `nn.Module` neural network trained exclusively on "safe" transactions. It attempts to mathematically reconstruct transactions and flags high "reconstruction errors" as anomalies.
3. 🌲 **Isolation Forest (Tree-Based Anomaly)**: Scalable unsupervised anomaly detection that builds random decision trees to isolate out-of-bounds data quickly.

## 📊 Live System Intelligence

The models are trained locally on a 6.3 Million row financial dataset. Below are the training and evaluation statistics:

### Database Stats (PaySim)
* **Total Records Analyzed**: 6,362,620
* **Total Fraudulent Transactions**: 8,213
* **Dataset Fraud Rate**: 0.13%

### AI Performance Metrics
| AI Engine | Validation Accuracy | Fraud Detection Rate (Recall) | Training Time |
| :--- | :--- | :--- | :--- |
| **Random Forest** | 99.43% | **99.45%** | 0.72s |
| **PyTorch Autoencoder** | 94.16% | 35.85% | 15.74s |
| **Isolation Forest** | 82.54% | 34.27% | 0.75s |

*Note: Since fraud is incredibly rare (0.13%), Unsupervised Anomaly Detection models (Autoencoder, Isolation Forest) naturally have lower recall without aggressive threshold tuning, while the Supervised model (Random Forest) excels by learning exactly what fraud looks like.*

## 💻 How to Run Locally

1. Clone this repository.
2. Double-click the **`start_interface.bat`** file.
3. The system will automatically construct a virtual environment, download the 500MB Kaggle dataset, rapidly train all 3 AI models, and launch the web dashboard in your browser.
