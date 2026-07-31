@echo off
title AI Fraud Sentinel
color 0A
cls

echo ===================================================
echo     AI FRAUD SENTINEL - DEEP PACKET INSPECTION
echo ===================================================
echo.
echo [System] Initializing neural pathways...
ping 127.0.0.1 -n 2 > nul

echo [System] Checking virtual environment integrity...
if not exist "venv" (
    echo [!] VENV missing. Constructing isolated environment...
    python -m venv venv
)
call venv\Scripts\activate
echo [OK] Virtual environment secured.

echo [System] Synchronizing dependencies...
pip install -r requirements.txt -q
python install_env.py
echo [OK] Dependencies synchronized.

echo [System] Checking ML models...
if not exist "weights\rf_model.joblib" (
    echo.
    echo [!] WARNING: AI models missing!
    echo [!] Initiating emergency deep-learning sequence...
    python train_models.py
) else (
    echo [OK] 3-Model Architecture loaded from weights directory.
)

echo.
echo ===================================================
echo     SYSTEM READY. LAUNCHING DUAL SERVERS.
echo ===================================================
echo [1/2] Booting MLflow Tracking Dashboard (Background)...
start /B mlflow ui --port 5000
echo [2/2] Booting Streamlit Interface...
echo.
ping 127.0.0.1 -n 2 > nul

streamlit run app.py
