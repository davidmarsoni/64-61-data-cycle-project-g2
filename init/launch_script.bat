@echo off
echo Starting data pipeline at %date% %time%

echo Step 0: Ensuring dependencies are installed...
cd /d "%~dp0"
python setup_dependencies.py
set DEP_ERROR=%ERRORLEVEL%

if %DEP_ERROR% neq 0 (
    echo ERROR: Failed to install required dependencies, code: %DEP_ERROR%
    exit /b %DEP_ERROR%
) else (
    echo Dependencies installed successfully.
)
echo.

echo Step 1: Running data collector...
cd /d "%~dp0.."
python data_collector.py
set COLLECTOR_ERROR=%ERRORLEVEL%

if %COLLECTOR_ERROR% neq 0 (
    echo Warning: Data collector script returned error code: %COLLECTOR_ERROR%
) else (
    echo Data collection completed successfully.
)
echo.

echo Step 2: Running data cleaner...
python data_cleaner.py
set CLEANER_ERROR=%ERRORLEVEL%

if %CLEANER_ERROR% neq 0 (
    echo Warning: Data cleaner script returned error code: %CLEANER_ERROR%
) else (
    echo Data cleaning completed successfully.
)
echo.

echo Step 3: Running ETL process...
python data_ETL.py
set ETL_ERROR=%ERRORLEVEL%

if %ETL_ERROR% neq 0 (
    echo Warning: ETL script returned error code: %ETL_ERROR%
) else (
    echo ETL process completed successfully.
)
echo.

echo Data pipeline completed at %date% %time%

REM Return overall status
if %DEP_ERROR% neq 0 (
    exit /b %DEP_ERROR%
) else if %COLLECTOR_ERROR% neq 0 (
    exit /b %COLLECTOR_ERROR%
) else if %CLEANER_ERROR% neq 0 (
    exit /b %CLEANER_ERROR%
) else if %ETL_ERROR% neq 0 (
    exit /b %ETL_ERROR%
) else (
    exit /b 0
)