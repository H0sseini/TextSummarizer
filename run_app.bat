@echo off

:: Check if Python is available
where python >nul 2>nul || (
    echo Python is not installed or not in PATH.
    pause
    exit /b 1
)

:: Check if uvicorn is available
where  python uvicorn >nul 2>nul || (
    echo Uvicorn is not installed. Please run: pip install -r requirements.txt
    pause
    exit /b 1
)

:: Starting the project
echo Starting TextSummarizer...

:: Launch browser
start http://127.0.0.1:8000

:: Start FastAPI server
python -m uvicorn frontend.app:app --reload

pause
