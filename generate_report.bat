@echo off
echo === Generating PDF Report from existing results ===

REM Merge all CSV results
python src\aggregate_results.py

REM Generate PDF report
python src\generate_report.py

echo.
echo ✅ Report generated: summary_report.pdf
pause
