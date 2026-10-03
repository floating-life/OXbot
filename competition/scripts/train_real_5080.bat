@echo off
setlocal EnableExtensions
REM Finite real-replay training on the 9800X3D + RTX 5080.
REM Prepare all source records first with: .venv\Scripts\python train_real.py prepare --include-test
REM The trainer reads train + validation only; final held-out evaluation is separate.
REM Examples:
REM   scripts\train_real_5080.bat
REM   scripts\train_real_5080.bat --epochs 12
REM OXBOT_REAL_DATA and OXBOT_REAL_SOURCE override trainer default paths.
REM Re-running resumes OUT\latest.pt; use a new output for a fresh experiment.

set "ROOT=%~dp0.."
set "PUSHED=0"
pushd "%ROOT%" >nul 2>&1
if errorlevel 1 goto :fail
set "PUSHED=1"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo Missing %PY%. Run scripts\setup_windows.bat first.
    goto :fail
)
set "OUT=ckpts\real-v2"
if defined OXBOT_REAL_OUT set "OUT=%OXBOT_REAL_OUT%"
set "EPOCHS=8"
if defined OXBOT_REAL_EPOCHS set "EPOCHS=%OXBOT_REAL_EPOCHS%"
set "DEVICE=cuda:0"
if defined OXBOT_REAL_DEVICE set "DEVICE=%OXBOT_REAL_DEVICE%"
set "BATCH=128"
if defined OXBOT_REAL_BATCH set "BATCH=%OXBOT_REAL_BATCH%"
set "CANDIDATE_BUDGET=8192"
if defined OXBOT_REAL_CANDIDATE_BUDGET set "CANDIDATE_BUDGET=%OXBOT_REAL_CANDIDATE_BUDGET%"
set "CANDIDATE_CHUNK=2048"
if defined OXBOT_REAL_CANDIDATE_CHUNK set "CANDIDATE_CHUNK=%OXBOT_REAL_CANDIDATE_CHUNK%"
set "THREADS=2"
if defined OXBOT_REAL_THREADS set "THREADS=%OXBOT_REAL_THREADS%"
set "DATA_ARG="
if defined OXBOT_REAL_DATA set DATA_ARG=--data "%OXBOT_REAL_DATA%"
set "SOURCE_ARG="
if defined OXBOT_REAL_SOURCE set SOURCE_ARG=--source "%OXBOT_REAL_SOURCE%"
set "RESUME_ARG="
if exist "%OUT%\latest.pt" set RESUME_ARG=--resume "%OUT%\latest.pt"

"%PY%" train_real.py train --out "%OUT%" --epochs "%EPOCHS%" ^
    --device "%DEVICE%" --batch "%BATCH%" ^
    --candidate-budget "%CANDIDATE_BUDGET%" --candidate-chunk "%CANDIDATE_CHUNK%" ^
    --threads "%THREADS%" %DATA_ARG% %SOURCE_ARG% %RESUME_ARG% %*
set "RC=%ERRORLEVEL%"
if "%PUSHED%"=="1" popd >nul 2>&1
endlocal & exit /b %RC%

:fail
echo REAL TRAINING LAUNCH FAILED - see the message above.
if "%PUSHED%"=="1" popd >nul 2>&1
endlocal
exit /b 1
