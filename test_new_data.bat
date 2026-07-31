@echo off
call venv\Scripts\activate
set /p filepath="Enter path to CSV file with new data (or drag and drop it here): "
python predict_new.py "%filepath%"
pause
