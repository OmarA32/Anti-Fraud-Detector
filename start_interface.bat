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
echo [OK] Dependencies synchronized.

echo [System] Checking ML models...
if not exist rf_model.joblib (
    echo.
    echo [!] WARNING: AI models missing!
    echo [!] Initiating emergency deep-learning sequence...
    python train_models.py
) else (
    echo [OK] 3-Model Architecture loaded and ready.
)

echo.
echo ===================================================
echo     SYSTEM READY. LAUNCHING SECURE INTERFACE.
echo ===================================================
echo.
ping 127.0.0.1 -n 2 > nul

streamlit run app.py
