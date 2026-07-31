@echo off
title AI Fraud Sentinel Launcher
color 0B

echo =========================================
echo       AI Fraud Sentinel Launcher
echo =========================================
echo.

:: Check if virtual environment exists
if not exist venv (
    echo [1/3] Creating virtual environment...
    python -m venv venv
) else (
    echo [1/3] Virtual environment found.
)

:: Activate the virtual environment
call venv\Scripts\activate

echo.
echo [2/3] Checking and installing requirements...
:: Install requirements quietly
pip install -r requirements.txt -q

echo.
:: Check if the models exist. If not, train them.
if not exist rf_model.joblib (
    echo [!] AI models not found. Training all 3 models now, this may take a minute...
    python train_models.py
)

echo.
echo [3/3] Launching Local AI Web Interface...
echo Please wait while the interface opens in your browser.
echo Do not close this black window while using the app!
echo.
streamlit run app.py
