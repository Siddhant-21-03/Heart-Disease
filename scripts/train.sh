#!/usr/bin/env bash
set -euo pipefail
PYTHON=${PYTHON:-python3}
${PYTHON} -m pip install -r requirements.txt
${PYTHON} src/models/train_model.py --input data/raw/heart_disease_uci.csv --output models
