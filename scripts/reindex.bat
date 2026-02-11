@echo off
setlocal

:: Set project root
pushd "%~dp0.."

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    echo [INFO] Activated .venv
) else if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo [INFO] Activated venv
) else (
    echo [WARN] No venv found. Using system Python.
)

:: Rebuild Index
python scripts/reindex_rag.py

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Reindexing failed with code %ERRORLEVEL%
    popd
    exit /b 1
)

echo.
echo [SUCCESS] Reindexing complete. All student data should be searchable.
popd
endlocal
