@echo off
echo Setting up daily scheduled task for data pipeline...

REM Get the full path to the launch script
set SCRIPT_DIR=%~dp0
set LAUNCH_SCRIPT=%SCRIPT_DIR%launch_script.bat

REM Create a scheduled task to run the launch script daily at 7:30 AM
schtasks /create /tn "DataPipelineDaily" /tr "%LAUNCH_SCRIPT%" /sc DAILY /st 07:30:00 /ru "%USERNAME%" /f

if %ERRORLEVEL% equ 0 (
    echo.
    echo Success! The data pipeline has been scheduled to run daily at 7:30 AM.
    echo Task name: DataPipelineDaily
    echo.
    echo You can manage this task in the Windows Task Scheduler.
) else (
    echo.
    echo Error creating scheduled task. Error code: %ERRORLEVEL%
    echo Please run this script as administrator and try again.
)

pause