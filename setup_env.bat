@echo off
echo Setting up Virtual Environment...
python -m venv venv
call venv\Scripts\activate
echo Installing dependencies...
pip install -r requirements.txt
echo Setup complete.
pause
