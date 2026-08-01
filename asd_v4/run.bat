@echo off
cd /d "%~dp0"
set PYTHON_EXE=C:\Users\love9\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe
"%PYTHON_EXE%" run.py --check
if errorlevel 1 (
    echo.
    echo 執行檢查未通過，請先安裝依賴後再試。
    pause
    exit /b 1
)
"%PYTHON_EXE%" run.py
pause
