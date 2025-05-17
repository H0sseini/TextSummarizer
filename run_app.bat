@echo off
cd /d "%~dp0Frontend"
echo Starting TextSummarizer...
start http://127.0.0.1:8000
python -m uvicorn app:app --reload
pause
