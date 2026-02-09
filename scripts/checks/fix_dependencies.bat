@echo off
setlocal
echo ==========================================
echo Fixing Python dependencies for chatbot
echo ==========================================
echo.

echo Checking Python Environment...
where py
if %ERRORLEVEL% NEQ 0 (
    echo 'py' launcher not found. Using 'python' command.
    set PYTHON_CMD=python
) else (
    echo 'py' launcher found.
    set PYTHON_CMD=py
)

echo.
echo Listing installed Python versions (if using py launcher):
if "%PYTHON_CMD%"=="py" (
    py --list
)

echo.
echo Current active Python version:
%PYTHON_CMD% --version

echo.
echo ==========================================
echo Step 1: Uninstalling broken packages
echo ==========================================
%PYTHON_CMD% -m pip uninstall -y numpy pandas

echo.
echo ==========================================
echo Step 2: Reinstalling NumPy and pandas
echo ==========================================
echo Installing compatible versions...
:: Force reinstall without cache to avoid broken wheels
%PYTHON_CMD% -m pip install --upgrade --force-reinstall --no-cache-dir numpy pandas

echo.
echo ==========================================
echo Step 3: Verification
echo ==========================================
%PYTHON_CMD% -c "import numpy; print('NumPy version:', numpy.__version__, 'Path:', numpy.__file__)"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] NumPy import failed! 
    echo Please ensure you are NOT using the free-threaded (3.13t) version of Python.
    echo If you are, please install the standard Python 3.13 or 3.12.
) else (
    echo NumPy verification SUCCESS.
)

%PYTHON_CMD% -c "import pandas; print('Pandas version:', pandas.__version__)"

echo.
echo Done. You can now try 'python main.py' or scripts\run\start_standard.bat.
pause
