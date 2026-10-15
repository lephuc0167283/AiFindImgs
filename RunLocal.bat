@echo off

echo =========================
echo ACTIVATE VENV
echo =========================

call .venv\Scripts\activate

echo.
echo Python dang dung:
where python
echo.

echo =========================
echo BUILD INDEX...
echo =========================

python build_index.py

echo.
echo =========================
echo LAN IP:
echo =========================

for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /C:"IPv4 Address"') do (
    set IP=%%a
    goto :found
)

:found
set IP=%IP: =%

echo Truy cap tu may khac:
echo http://%IP%:8000
echo.

echo =========================
echo START SERVER...
echo =========================

python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

pause