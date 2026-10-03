@echo off
setlocal EnableExtensions
REM ===== run before EVERY upload: official judge + packed process gate =====
REM The judge is intentionally mandatory. Do not silently substitute judge_fixed.py.
set "ROOT=%~dp0.."
set "PUSHED=0"
pushd "%ROOT%" >nul 2>&1
if errorlevel 1 goto :fail
set "PUSHED=1"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo CHECK FAILED: missing %PY%; run scripts\setup_windows.bat first.
  goto :fail
)
if not exist "judge\judge_official.py" (
  echo CHECK FAILED: put the verified official judge at judge\judge_official.py
  goto :fail
)
set "W=%~1"
if "%W%"=="" set "W=ckpts\real-v2\best.npz"
if not exist "%W%" (
  echo CHECK FAILED: weights not found: %W%
  goto :fail
)
%PY% tests\test_all.py || goto :fail
%PY% tests\test_judge_compat.py judge\judge_official.py || goto :fail
%PY% tools\judge_runner.py --judge judge\judge_official.py --games 200 --weights "%W%" --require-model --report reports\real_v2_official_model200.json || goto :fail
%PY% tools\judge_runner.py --judge judge\judge_official.py --games 40 --weights "%W%" --require-model --positional --report reports\real_v2_official_positional40.json || goto :fail
%PY% botzone\pack_bot.py --weights "%W%" --out dist\OXbot-real-v2.zip || goto :fail
%PY% tools\check_submission.py --zip dist\OXbot-real-v2.zip --weights "%W%" --judge judge\judge_official.py --games 1 --report reports\real_v2_packed_submission.json || goto :fail
echo.
echo ALL CHECKS PASSED: official judge and traditional/keep-running packed process.
echo   dist\OXbot-real-v2.zip      -^> Botzone bot source, compiler Python 3.6.5
echo   dist\fabledan_w_XXXXXXXX.npz -^> user storage (keep the exact file name)
if "%PUSHED%"=="1" popd >nul 2>&1
endlocal
exit /b 0
:fail
echo CHECK FAILED - do not upload.
if "%PUSHED%"=="1" popd >nul 2>&1
endlocal
exit /b 1
