@echo off
REM Windows helper to run training
set PYTHON=python
%PYTHON% -m pip install -r requirements.txt
%PYTHON% src\models\train_model.py --input data\raw\heart_disease_uci.csv --output models
