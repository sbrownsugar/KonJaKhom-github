@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo   ก่อนจะค่อม : กระจกอนาคต  (prototype)
echo   กำลังเปิดแอป... เปิดเบราว์เซอร์ไปที่  http://localhost:8501
echo   ปิดแอป: กด Ctrl+C ในหน้าต่างนี้
echo ============================================================
if exist ".venv\Scripts\streamlit.exe" (
  ".venv\Scripts\streamlit.exe" run "project\app.py"
) else (
  echo [หมายเหตุ] ไม่พบ .venv - ใช้ Python ของระบบ
  echo ถ้ายังไม่ได้ติดตั้ง ให้รันก่อน:  pip install -r requirements.txt
  streamlit run "project\app.py"
)
pause
