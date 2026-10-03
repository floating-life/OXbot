@echo off
REM ===== run before EVERY upload: tests + official-judge legality sweep + pack =====
REM Put the official judge source at judge\judge_official.py first.
cd /d %~dp0\..
call .venv\Scripts\activate.bat
set W=%1
if "%W%"=="" set W=ckpts\run1\latest.npz
python tests\test_all.py || goto :fail
python tests\test_judge_compat.py || goto :fail
python tools\judge_runner.py --games 200 --weights %W% || goto :fail
python tools\judge_runner.py --games 40 --weights %W% --positional || goto :fail
python botzone\pack_bot.py --weights %W% --out dist\oxbot_fable.zip || goto :fail
echo.
echo ALL CHECKS PASSED.
echo   dist\oxbot_fable.zip        -^> Botzone bot source, compiler Python 3.6.5
echo   dist\fabledan_w_XXXXXXXX.npz -^> user storage (keep the exact file name)
goto :eof
:fail
echo CHECK FAILED - do not upload.
exit /b 1
