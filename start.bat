@echo off
setlocal

cd /d C:\Users\lephu\Documents\GitHub\TyncabinetFindImg

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
start "Tyncabinet Server" cmd /k python -m uvicorn app:app --host 0.0.0.0 --port 8000

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