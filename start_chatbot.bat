@echo off
echo ========================================
echo   UCSI University Chatbot - Quick Start
echo ========================================
echo.

echo [1/3] Checking Python...
python --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found! Please install Python 3.10+
    pause
    exit /b 1
)

echo.
echo [2/3] Checking Ollama...
ollama list
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo WARNING: Ollama not running!
    echo Please start Ollama from Start Menu or run: ollama serve
    echo.
    pause
)

echo.
echo [2.5/3] Checking Dependencies...
python -m pip install -r requirements.txt

echo.
echo [3/3] Starting server...
echo.
echo Server will start at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python main.py
pause
