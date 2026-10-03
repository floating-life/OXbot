@echo off
setlocal EnableExtensions
REM ===== OXbot/FableDan one-time setup on Windows (RTX 5080) =====
REM The competition bot is Python 3.6-compatible, but training uses the
REM existing Python 3.12 + CUDA environment when available.

set "ROOT=%~dp0.."
set "PUSHED=0"
pushd "%ROOT%" >nul 2>&1
if errorlevel 1 goto :fail
set "PUSHED=1"

REM Select Python 3.12 explicitly. Falling back to an arbitrary Python can
REM select 3.14, for which a matching CUDA wheel may not exist yet.
set "BOOTSTRAP="
where py >nul 2>&1
if not errorlevel 1 (
    py -3.12 -c "import sys" >nul 2>&1
    if not errorlevel 1 set "BOOTSTRAP=py -3.12"
)
if not defined BOOTSTRAP (
    where python >nul 2>&1
    if not errorlevel 1 (
        python -c "import sys; assert sys.version_info[:2] == (3, 12)" >nul 2>&1
        if not errorlevel 1 set "BOOTSTRAP=python"
    )
)
if not defined BOOTSTRAP (
    echo Python 3.12 was not found. Install Python 3.12 or set up WSL first.
    goto :fail
)

if not exist ".venv\Scripts\python.exe" (
    REM Reuse an already provisioned Python 3.12 CUDA stack when present;
    REM pip can still shadow an absent or incompatible package in this venv.
    %BOOTSTRAP% -m venv --system-site-packages ".venv"
    if errorlevel 1 goto :fail
)
set "PY=.venv\Scripts\python.exe"
"%PY%" -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version" >nul 2>&1
if errorlevel 1 (
    echo Existing .venv is not Python 3.12. Remove it and rerun setup_windows.bat.
    goto :fail
)

REM Keep setup repeatable: do not reinstall a working CUDA wheel on every run.
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail
"%PY%" -c "import numpy" >nul 2>&1
if errorlevel 1 (
    "%PY%" -m pip install "numpy>=1.24"
    if errorlevel 1 goto :fail
)
"%PY%" -c "import torch; assert torch.cuda.is_available()" >nul 2>&1
if errorlevel 1 (
    echo Installing the CUDA 12.8 PyTorch build...
    "%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 goto :fail
)

"%PY%" -c "import torch,sys; assert torch.cuda.is_available(); print('python',sys.version.split()[0]); print('torch',torch.__version__,'cuda',torch.version.cuda); print('gpu',torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0)); x=torch.randn(2048,2048,device='cuda',dtype=torch.bfloat16); print('bf16 matmul ok',bool((x@x).abs().sum()>0))"
if errorlevel 1 goto :fail
"%PY%" tools\check_environment.py --micro-batch 256
if errorlevel 1 goto :fail

"%PY%" tests\test_all.py
if errorlevel 1 goto :fail
"%PY%" tests\test_judge_compat.py
if errorlevel 1 goto :fail

echo.
echo SETUP OK. Next: scripts\train_5080.bat
if "%PUSHED%"=="1" popd
endlocal
exit /b 0

:fail
echo SETUP FAILED - see the message above.
if "%PUSHED%"=="1" popd >nul 2>&1
endlocal
exit /b 1
