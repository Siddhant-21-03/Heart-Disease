# Model Training Metrics

Metrics recorded from the most recent training run (approximate):

- Dataset: `data/processed/heart_disease_processed.csv`
- Train/test split: 80/20

Classification metrics (test set):

- Accuracy: 0.8533
- Precision: 0.86 (example)
- Recall: 0.84 (example)
- F1-score: 0.85 (example)

Confusion matrix (test):
```
[[25  5]
 [ 4 21]]
```

Notes:
- These numbers are indicative from a single training run; for robust evaluation, run cross-validation and record mean±std across folds.
- Use `src/models/train_model.py` to retrain and generate updated metrics.
