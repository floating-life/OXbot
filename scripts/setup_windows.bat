@echo off
REM ===== OXbot/FableDan one-time setup on Windows (RTX 5080) =====
REM RTX 50-series (Blackwell, sm_120) needs a PyTorch build with CUDA 12.8+.
cd /d %~dp0\..
where py >nul 2>nul && (py -3.12 -m venv .venv || py -3 -m venv .venv) || python -m venv .venv
call .venv\Scripts\activate.bat || goto :fail
python -m pip install --upgrade pip
pip install numpy || goto :fail
pip install torch --index-url https://download.pytorch.org/whl/cu128 || goto :fail
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda);print('gpu',torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0));x=torch.randn(2048,2048,device='cuda',dtype=torch.bfloat16);print('bf16 matmul ok',bool((x@x).abs().sum()>0))" || goto :fail
python tests\test_all.py || goto :fail
python tests\test_judge_compat.py || goto :fail
echo.
echo SETUP OK.  Next: scripts\train_5080.bat
goto :eof
:fail
echo SETUP FAILED - see the message above.
exit /b 1
