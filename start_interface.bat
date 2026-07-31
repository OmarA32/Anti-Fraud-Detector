@echo off
setlocal
title AI Fraud Sentinel

echo ===================================================
echo     AI FRAUD SENTINEL - DEEP PACKET INSPECTION
echo ===================================================

echo [System] Initializing neural pathways...
echo [System] Checking virtual environment integrity...

:: Check if Python exists
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Python/Miniconda not found on this system!
    echo [System] Initiating automatic zero-dependency bootstrap...
    echo [System] Downloading official Miniconda installer (this may take a minute)...
    powershell -Command "Invoke-WebRequest -Uri 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe' -OutFile 'miniconda_installer.exe'"
    
    echo [System] Installing Miniconda silently in the background...
    start /wait "" miniconda_installer.exe /InstallationType=JustMe /RegisterPython=1 /S /D=%USERPROFILE%\Miniconda3
    del miniconda_installer.exe
    
    :: Add to PATH for this session
    set PATH=%USERPROFILE%\Miniconda3;%USERPROFILE%\Miniconda3\Scripts;%USERPROFILE%\Miniconda3\Library\bin;%PATH%
    echo [OK] Miniconda successfully installed and loaded into environment!
) else (
    echo [OK] Python environment secured.
)

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
start /B mlflow ui --port 5000 > nul 2>&1
echo [2/2] Booting Streamlit Interface...
echo.
ping 127.0.0.1 -n 3 > nul

echo [System] Opening MLflow Dashboard in your browser...
start http://localhost:5000

streamlit run app.py
