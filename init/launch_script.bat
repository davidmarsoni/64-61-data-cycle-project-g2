@echo off
echo Starting data pipeline at %date% %time%
echo.

echo Step 1: Running data collector...
cd /d "%~dp0.."
python data_collector.py
if %ERRORLEVEL% neq 0 (
    echo Error running data collector script.
    exit /b 1
)
echo Data collection completed successfully.
echo.

echo Step 2: Running data cleaner...
python data_cleaner.py
if %ERRORLEVEL% neq 0 (
    echo Error running data cleaner script.
    exit /b 1
)
echo Data cleaning completed successfully.
echo.

echo Step 3: Running ETL process...
python data_ETL.py
if %ERRORLEVEL% neq 0 (
    echo Error running ETL script.
    exit /b 1
)
echo ETL process completed successfully.
echo.

echo Data pipeline completed at %date% %time%
exit /b 0