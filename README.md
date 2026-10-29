# Student Score Prediction

A Python machine learning project that cleans messy student records and predicts exam scores using study hours, attendance, previous scores, and sleep hours.

The project demonstrates data cleaning, exploratory visualization, train/test splitting, missing-value handling, regression, and evaluation with scikit-learn. The current implementation is a runnable script using a **Random Forest regressor**.

## Results at a glance

| Item | Verified result |
| --- | --- |
| Original dataset | 168 rows, 7 columns |
| Cleaned dataset | 148 rows, 7 columns |
| Training / test records | 118 / 30 |
| Model | `RandomForestRegressor(random_state=42)` |
| Test mean absolute error (MAE) | **5.9 score points** |
| Example prediction | **45.5 / 100** |

These results were reproduced from `main.py` using the included data and the dependency versions listed below. Results may vary across dependency versions.

**Interpreting MAE:** on this test split, predictions differ from the recorded exam scores by approximately 5.9 points on average. MAE is an error measure, not a classification accuracy percentage or a guaranteed error bound for individual predictions.

## Features

- Standardizes missing-value markers, student IDs, and student names.
- Converts numeric text containing units, percentage symbols, and decimal commas into numbers.
- Removes duplicate records and out-of-range values.
- Exports a cleaned CSV dataset.
- Displays a scatter plot of study hours against exam scores.
- Trains a Random Forest model using four numerical features.
- Evaluates predictions on a held-out test set using MAE.
- Predicts the exam score of an example student.

## Repository contents

| File | Purpose |
| --- | --- |
| [`main.py`](main.py) | Data cleaning, visualization, training, evaluation, and example prediction. |
| [`Student_performance_messy.csv`](Student_performance_messy.csv) | Input dataset with inconsistent formats, missing values, and duplicates. |
| [`Student_performance_cleaned.csv`](Student_performance_cleaned.csv) | Cleaned dataset; regenerated whenever the script runs. |
| [`README.md`](README.md) | Setup, methodology, results, and limitations. |

## Dataset

| Column | Meaning | Role |
| --- | --- | --- |
| `student_id` | Student identifier | Cleaning and deduplication only; excluded from model inputs. |
| `student_name` | Student name | Standardized during cleaning; excluded from model inputs. |
| `study_hours` | Recorded study hours | Input feature; 0–24. |
| `attendance_percent` | Attendance percentage | Input feature; 0–100. |
| `previous_score` | Previous academic score | Input feature; 0–100. |
| `sleep_hours` | Recorded sleep hours | Input feature; 0–24. |
| `exam_score` | Exam score to predict | Target; 0–100. |


### Cleaning workflow

1. Replace markers such as `N/A`, `null`, `none`, `unknown`, and `?` with missing values.
2. Trim and normalize student IDs and names.
3. Remove hour units and percent signs, normalize decimal commas, and convert numeric columns to numbers. Unparseable values become missing.
4. Remove fully duplicated rows and keep the first occurrence of each non-missing student ID. Rows without an ID may remain.
5. Remove rows with out-of-range feature values; retain rows with missing feature values for later imputation.
6. Remove rows with missing or out-of-range exam scores.
7. Reset the index and export the cleaned dataset.

The exported CSV can still contain missing feature values. Median imputation happens later for training and evaluation; those imputed values are not written back to the CSV.

## Getting started

The project was verified with **Python 3.12.14**.

### 1. Clone the repository

```bash
git clone https://github.com/DevAJZ/ML-Model-For-Student-Score-Prediction.git
cd ML-Model-For-Student-Score-Prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Use `python3` instead of `python` if that is your system's Python command.

### 3. Install dependencies

These versions were used to reproduce the reported results:

```bash
python -m pip install pandas==2.2.3 matplotlib==3.10.8 numpy==2.3.5 scikit-learn==1.8.0
```

### 4. Run the project

```bash
python main.py
```

Run from the repository root so the script can find the input CSV. Each run overwrites `Student_performance_cleaned.csv`.

The script prints data-quality summaries and opens a scatter plot. Depending on the plotting backend, **close the plot window to allow training and evaluation to continue**.

For a terminal-only run on macOS / Linux:

```bash
MPLBACKEND=Agg python main.py
```

For Windows PowerShell:

```powershell
$env:MPLBACKEND = "Agg"
python main.py
```

The `Agg` backend suppresses the interactive window; the current script does not save the plot as an image.

### Example output

Selected lines from a verified run:

```text
Original dataset shape: (168, 7)
Cleaned dataset shape: (148, 7)
MAE(Mean Absolute Error): 5.9
Predicted exam score for the new student: 45.5
```

## Modeling approach

1. Select the four numerical features and use `exam_score` as the target.
2. Split the cleaned data into 80% training and 20% testing using `random_state=42`.
3. Calculate feature medians from the **training set only**.
4. Fill missing values in both sets using those training-set medians.
5. Fit `RandomForestRegressor(random_state=42)` with its remaining default settings.
6. Predict the held-out test scores and calculate MAE.

Training-only medians prevent the test set from influencing the learned imputation values. The script fixes the random seed for both splitting and training.

## Predicting another student's score

Edit the `new_student` DataFrame near the end of `main.py`, then rerun the script:

```python
new_student = pd.DataFrame({
    "study_hours": [10],
    "attendance_percent": [80],
    "previous_score": [70],
    "sleep_hours": [7]
})
```

With the included data and verified environment, this example produces **45.5 / 100**.


Results are based on a small dataset and a single train/test split.”
This project demonstrates an ML workflow. Predictions describe learned associations, not evidence that changing one input will cause a particular exam result.

## Planned improvements

These are development goals, not implemented features:

- [ ] Document dataset provenance, units, and permitted use.
- [ ] Compare a mean baseline, Linear Regression, and Random Forest using cross-validation on the training set.
- [ ] Report MAE, RMSE, and R², reserving a final test set for evaluation.
- [ ] Package imputation and the estimator in a scikit-learn `Pipeline`.
- [ ] Separate cleaning, training, evaluation, and prediction into reusable functions or modules.
- [ ] Add a dependency file and tests for cleaning rules and prediction inputs.
- [ ] Save and reload the complete fitted pipeline for inference.
- [ ] Build a small prediction demo with input validation and clear result explanations.
- [ ] Add automated checks with GitHub Actions.
- [ ] Publish residual plots, model comparisons, and the reasoning behind the final model choice.

## Author
[DevAJZ](https://github.com/DevAJZ)