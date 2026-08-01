@echo off
setlocal

cd /d "%~dp0"

where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Conda was not found.
    echo Install Miniconda or Anaconda, then run this file from Anaconda Prompt.
    exit /b 1
)

echo Creating or updating the asd-homecoach environment...
call conda env update --file "%~dp0environment-windows.yml" --prune
if errorlevel 1 (
    echo [ERROR] Environment installation failed.
    exit /b 1
)

echo Checking Python packages and NVIDIA CUDA access...
call conda run --no-capture-output -n asd-homecoach python -c "import flask, numpy, cv2, sklearn, xgboost, tensorflow, faster_whisper, ctranslate2; assert ctranslate2.get_cuda_device_count() ^> 0, 'CTranslate2 cannot detect an NVIDIA CUDA GPU'; print('Environment check passed. CUDA devices:', ctranslate2.get_cuda_device_count())"
if errorlevel 1 (
    echo [ERROR] A package could not be imported.
    exit /b 1
)

echo.
echo Environment is ready.
echo Activate it with: conda activate asd-homecoach
echo Start the app with: cd HomeCoach ^&^& python run.py

endlocal
