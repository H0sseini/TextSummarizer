@echo off

:: Check if Python is available
where python >nul 2>nul || (
    echo Python is not installed or not in PATH.
    pause
    exit /b 1
)



:: Creating environment and activating
echo Creating environment and activating
python -m venv venv
venv\Scripts\activate
:: conda create --name venv python=3.10
:: call "C:\Users\user\Anaconda3\Scripts\activate.bat" venv
pip install -r requirements.txt



pause
