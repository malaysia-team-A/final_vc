@echo off
setlocal
cd /d "%~dp0\..\.."
python scripts\checks\check_system.py > data\reports\check_debug.txt 2>&1
