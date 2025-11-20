# Contributing

Thank you for your interest in contributing to this project. A few guidelines to make contributions smooth and reviewable.

- Fork the repository and create a feature branch for your change.
- Make small, focused commits with clear messages.
- Run tests locally before opening a pull request:

```powershell
$env:PYTHONPATH='.'; .\.venv\Scripts\Activate.ps1; pytest -q
```

- Follow the existing code style; keep function-level docstrings concise and tests for new logic.
- For data or model changes, update `METRICS.md` and `MODEL_CARD.md` with new evaluation results and notes.

Open an issue first for larger changes or propose design discussions.
