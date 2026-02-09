@echo off
setlocal
set "ROOT_DIR=%~dp0..\.."
pushd "%ROOT_DIR%"
echo Attempting to run chatbot with standard Python 3.13...
echo.
py -3.13 main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Execution failed.
    echo If 'py -3.13' is not available, install standard Python 3.13.
    echo You can also run scripts\checks\fix_dependencies.bat.
    popd
    pause
    exit /b 1
)
popd
endlocal
