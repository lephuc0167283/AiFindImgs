@echo off
setlocal

cd /d "%~dp0"
set "PYTHON=%~dp0.venv\Scripts\python.exe"

echo =========================
echo CHECK PROJECT VENV
echo =========================
if not exist "%PYTHON%" (
    echo Khong tim thay Python trong .venv cua du an.
    echo Chay cac lenh sau trong thu muc du an:
    echo py -3.10 -m venv .venv
    echo .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo Python dang dung:
"%PYTHON%" --version

echo.
echo =========================
echo BUILD INDEX...
echo =========================
"%PYTHON%" build_index.py
if errorlevel 1 (
    echo.
    echo Build index loi. Dung chuong trinh.
    pause
    exit /b 1
)

echo.
echo =========================
echo START SERVER...
echo =========================
start "AI Find Server" /D "%~dp0" "%PYTHON%" -m uvicorn app:app --host 0.0.0.0 --port 8000

echo.
echo Doi server khoi dong...
timeout /t 5 /nobreak >nul

echo.
echo =========================
echo NGROK TUNNEL...
echo =========================
echo Dia chi co dinh:
echo https://pound-astronaut-chapter.ngrok-free.dev
echo.
ngrok http --domain=pound-astronaut-chapter.ngrok-free.dev 8000

endlocal