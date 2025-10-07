@echo off
REM Simple Windows batch script to run experiments in background
REM Usage: run_experiment_background.bat <experiment_number>
REM Example: run_experiment_background.bat 7

if "%1"=="" (
    echo Usage: run_experiment_background.bat ^<experiment_number^>
    echo Example: run_experiment_background.bat 7
    echo.
    echo Available experiments:
    echo   1 - RQ1.1 Fairness Weights
    echo   2 - Comprehensive Parameter Sweep
    echo   3 - Custom Parameter Sweep
    echo   4 - Focused Parameter Sweep
    echo   5 - Bottleneck Analysis
    echo   6 - Comparative Parameter Sweep
    echo   7 - EWMA Gamma Sensitivity
    pause
    exit /b 1
)

echo ========================================
echo Running Experiment %1 in background...
echo ========================================
echo Started at: %date% %time%
echo.

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

REM Run in background and save output to log file
start /B cmd /c "python run_experiment.py %1 > logs/experiment_%1_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log 2>&1"

echo Experiment %1 started in background!
echo Check logs/ directory for progress
echo.
echo Estimated completion time:
echo   Experiment 7: ~2-3 hours
echo   Other experiments: ~30-60 minutes
echo.
pause
