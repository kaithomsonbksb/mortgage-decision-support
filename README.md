# Mortgage Credit-Risk Decision Support

A decision-support prototype for DN6MD001 Assignment 2, using synthetic mortgage applicant data to estimate repayment risk.
It combines a model score with advisory rules to suggest a review level for a human reviewer. It never approves or declines applications.

## Requirements

Use Python 3.13 and create a fresh virtual environment. Dependency versions are pinned in `requirements.txt`.
On macOS, install XGBoost's OpenMP dependency first: `brew install libomp`.

## Setup and run

Run these commands in order from the project folder.

### Windows (PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.data
.\.venv\Scripts\python.exe -m src.evaluate
.\.venv\Scripts\python.exe -m src.improve
.\.venv\Scripts\python.exe -m src.main
.\.venv\Scripts\python.exe -m src.applicant_example
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### macOS / Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m src.data
.venv/bin/python -m src.evaluate
.venv/bin/python -m src.improve
.venv/bin/python -m src.main
.venv/bin/python -m src.applicant_example
.venv/bin/python -m unittest discover -s tests -v
```

Run `src.main` before `src.applicant_example`. This saves the model used by the examples.

## What the commands produce

- `src.data`: data audit.
- `src.evaluate`: validation results, a threshold sweep, and charts.
- `src.improve`: parameter grid, cross-validation comparison, and improved-model sweep.
- `src.main`: model test results, a confusion matrix, and the advisory-system summary.
- `src.applicant_example`: example scores and advisory outcomes.
- `unittest`: 21 tests.

## Project structure

- `src/data.py` - loads the dataset and prints its core audit.
- `src/audit.py` - reports summary statistics and missing-data patterns.
- `src/model.py` - builds, trains, and evaluates the model pipelines.
- `src/evaluate.py` - evaluates a validation set and plots a threshold sweep.
- `src/improve.py` - compares XGBoost settings and evaluates an improved model.
- `src/decisions.py` - validates applications and applies advisory review rules.
- `src/main.py` - runs the model comparisons and advisory-system evaluation.
- `src/applicant_example.py` - scores sample applications and prints advisory outcomes.
- `tests/` - 21 tests covering decision rules, validation, and boundaries.
- `requirements.txt` - pinned dependency versions.

## Reproducibility

Package versions are pinned. The regularised model uses no random row or column sampling (`subsample` and `colsample_bytree` are set to 1.0) and a fixed seed of 42. Results are identical across the tested Windows, macOS, and Linux setups.

## Data note

The synthetic dataset is supplied in the DN6MD001 assignment pack and is included in this repository at `data/A2 Mortgage Credit Risk Dataset.csv`.

## Module-supplied code

The module supplied the starter Logistic Regression benchmark, data loading, `validate_application_fields`, `validate_score`, and four original tests in `test_validation.py`. The remaining implementation and tests are my work.

## AI use

GitHub Copilot helped with code suggestions, autocomplete and debugging. I reviewed and tested the code. Claude assisted with planning, review and checking results.
GitHub Copilot was also used to aid documentation and notes for code readability.  
