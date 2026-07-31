@echo off
setlocal
title AI Fraud Sentinel

echo ===================================================
echo     AI FRAUD SENTINEL - DEEP PACKET INSPECTION
echo ===================================================

echo [System] Initializing neural pathways...
echo [System] Checking virtual environment integrity...

:: Check if Conda is installed
call conda --version >nul 2>&1
if %errorlevel% neq 0 goto :use_venv

echo [System] Conda MLOps environment manager detected!
call conda info --envs | findstr /C:"fraud_env" >nul
if %errorlevel% neq 0 (
    echo [!] Constructing isolated Conda environment 'fraud_env'...
    call conda create -n fraud_env python=3.11 -y
)
call conda activate fraud_env
echo [OK] Conda environment secured.
goto :deps

:use_venv
echo [System] Conda not found. Falling back to standard Python venv...
if not exist "venv" (
    echo [!] VENV missing. Constructing isolated environment...
    python -m venv venv
)
call venv\Scripts\activate
echo [OK] Virtual environment secured.

:deps

echo [System] Synchronizing dependencies...
pip install -r requirements.txt
echo [OK] Dependencies synchronized.

echo ===================================================
echo     BOOTING MLOps TRACKING SERVER
echo ===================================================
echo [System] Launching MLflow Dashboard in the background...
start /B mlflow ui --port 5000 > nul 2>&1
ping 127.0.0.1 -n 3 > nul
echo [System] Opening MLflow Dashboard in your browser...
start http://localhost:5000

echo.
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
echo     SYSTEM READY. LAUNCHING AI INTERFACE.
echo ===================================================
echo [System] Streamlit will now open in your browser...

streamlit run app.py
