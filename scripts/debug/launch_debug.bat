@echo off
setlocal
cd /d "%~dp0\..\.."
echo Starting Main Server...
python main.py > data\reports\debug_server.log 2>&1
