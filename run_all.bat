@echo off
echo === Preprocessing datasets ===
python src\02_preprocessing.py

echo.
echo === Cleaning old models & results ===
del /Q models\*.keras 2>nul
del /Q results\*.csv 2>nul

REM =======================
REM Run all models & datasets
REM =======================
for %%M in (lstm gru transformer) do (
    for %%D in (FD001 FD002 FD003 FD004) do (
        echo.
        echo === Training %%M on %%D ===
        python src\train.py --model %%M --dataset %%D --epochs 100 --batch_size 64
        python src\evaluate.py --model %%M --dataset %%D
    )
)

echo.
echo === Aggregating all results ===
python src\aggregate_results.py --pdf

echo.
echo === All tasks finished ===
pause
