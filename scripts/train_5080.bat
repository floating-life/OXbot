@echo off
REM ===== OXbot/FableDan self-play training on 9800X3D + RTX 5080 =====
REM Re-run this file after any interruption: it resumes from latest.pt.
REM Tuning: watch "samples/s" in the log. If GPU (Task Manager > Performance >
REM GPU > Cuda) sits near 100%%, lower --actors; if CPU is at 100%% and GPU low,
REM the actors are the bottleneck (try --ring 48).
cd /d %~dp0\..
call .venv\Scripts\activate.bat
set OUT=ckpts\run1
set RESUME=
if exist %OUT%\latest.pt set RESUME=--resume %OUT%\latest.pt
REM --ladder-frac: share of games at level 2 / no tribute (Botzone default).
REM Start 0.5; raise to 0.85 once the ladder logs confirm level 2 / no tribute.
python -m fabledan.train_fast --out %OUT% %RESUME% ^
    --actors 12 --ring 32 ^
    --ladder-frac 0.5 ^
    --eval-games 100 --eval-cycles 25 ^
    --export-cycles 50
