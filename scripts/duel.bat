@echo off
REM ===== duplicate head-to-head of two checkpoints (seat-swapped pairs) =====
REM usage: scripts\duel.bat new.npz old.npz [games]
cd /d %~dp0\..
call .venv\Scripts\activate.bat
set G=%3
if "%G%"=="" set G=400
python -m fabledan.evaluate --a %1 --b %2 --games %G% --ladder-frac 1.0 --log-every 100
