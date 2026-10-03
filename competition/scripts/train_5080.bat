@echo off
setlocal EnableExtensions
REM ===== OXbot/FableDan self-play on 9800X3D + one RTX 5080 =====
REM Defaults are conservative for an 8-core 9800X3D. Override with OXBOT_*
REM environment variables, or append normal train_fast options:
REM   scripts\train_5080.bat --cycles 1
REM   set OXBOT_OUT=ckpts\smoke & scripts\train_5080.bat --cycles 2
REM Re-running automatically resumes OUT\latest.pt unless OXBOT_NO_RESUME=1.

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

set "OUT=ckpts\run1"
if defined OXBOT_OUT set "OUT=%OXBOT_OUT%"
set "ACTORS=8"
if defined OXBOT_ACTORS set "ACTORS=%OXBOT_ACTORS%"
set "RING=32"
if defined OXBOT_RING set "RING=%OXBOT_RING%"
set "LADDER=0.5"
if defined OXBOT_LADDER_FRAC set "LADDER=%OXBOT_LADDER_FRAC%"
set "EVAL_GAMES=100"
if defined OXBOT_EVAL_GAMES set "EVAL_GAMES=%OXBOT_EVAL_GAMES%"
set "EVAL_CYCLES=25"
if defined OXBOT_EVAL_CYCLES set "EVAL_CYCLES=%OXBOT_EVAL_CYCLES%"
set "EXPORT_CYCLES=50"
if defined OXBOT_EXPORT_CYCLES set "EXPORT_CYCLES=%OXBOT_EXPORT_CYCLES%"
set "BATCH=4096"
if defined OXBOT_BATCH set "BATCH=%OXBOT_BATCH%"
set "MICRO_BATCH=256"
if defined OXBOT_MICRO_BATCH set "MICRO_BATCH=%OXBOT_MICRO_BATCH%"
set "DEVICE=cuda:0"
if defined OXBOT_DEVICE set "DEVICE=%OXBOT_DEVICE%"
set "INFER_DEVICE=cuda:0"
if defined OXBOT_INFER_DEVICE set "INFER_DEVICE=%OXBOT_INFER_DEVICE%"
set "CYCLE_ARG="
if defined OXBOT_CYCLES set "CYCLE_ARG=--cycles %OXBOT_CYCLES%"
set "HOURS_ARG="
if defined OXBOT_MAX_HOURS set "HOURS_ARG=--max-hours %OXBOT_MAX_HOURS%"

if exist "%OUT%\latest.pt" if /I not "%OXBOT_NO_RESUME%"=="1" goto :resume

echo Starting fresh training in "%OUT%". Extra options: %*
"%PY%" -m fabledan.train_fast --out "%OUT%" ^
    --actors "%ACTORS%" --ring "%RING%" ^
    --ladder-frac "%LADDER%" ^
    --batch "%BATCH%" --micro-batch "%MICRO_BATCH%" ^
    --eval-games "%EVAL_GAMES%" --eval-cycles "%EVAL_CYCLES%" ^
    --export-cycles "%EXPORT_CYCLES%" ^
    --device "%DEVICE%" --infer-device "%INFER_DEVICE%" ^
    %CYCLE_ARG% %HOURS_ARG% %*
goto :done

:resume
echo Resuming "%OUT%\latest.pt". Extra options: %*
"%PY%" -m fabledan.train_fast --out "%OUT%" --resume "%OUT%\latest.pt" ^
    --actors "%ACTORS%" --ring "%RING%" ^
    --ladder-frac "%LADDER%" ^
    --batch "%BATCH%" --micro-batch "%MICRO_BATCH%" ^
    --eval-games "%EVAL_GAMES%" --eval-cycles "%EVAL_CYCLES%" ^
    --export-cycles "%EXPORT_CYCLES%" ^
    --device "%DEVICE%" --infer-device "%INFER_DEVICE%" ^
    %CYCLE_ARG% %HOURS_ARG% %*

:done
set "RC=%ERRORLEVEL%"
if "%PUSHED%"=="1" popd >nul 2>&1
endlocal & exit /b %RC%

:fail
echo TRAINING LAUNCH FAILED - see the message above.
if "%PUSHED%"=="1" popd >nul 2>&1
endlocal
exit /b 1
