@echo off
setlocal EnableExtensions
REM ===== duplicate head-to-head of two checkpoints (seat-swapped pairs) =====
REM usage: scripts\duel.bat new.npz old.npz [games]
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
if "%~1"=="" (
    echo Usage: scripts\duel.bat new.npz old.npz [games]
    goto :fail
)
if "%~2"=="" (
    echo Usage: scripts\duel.bat new.npz old.npz [games]
    goto :fail
)
set "G=%~3"
if not defined G set "G=400"
"%PY%" -m fabledan.evaluate --a "%~1" --b "%~2" --games "%G%" --ladder-frac 1.0 --log-every 100
set "RC=%ERRORLEVEL%"
if "%PUSHED%"=="1" popd >nul 2>&1
endlocal & exit /b %RC%

:fail
echo DUEL LAUNCH FAILED - see the message above.
if "%PUSHED%"=="1" popd >nul 2>&1
endlocal
exit /b 1
