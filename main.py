import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import(train_test_split)
from sklearn.ensemble import(RandomForestRegressor)
from sklearn.metrics import mean_absolute_error
# 1. Load the messy dataset
df = pd.read_csv("Student_performance_messy.csv")

print("Original dataset shape:", df.shape)
#duplicated rows of original dataset
print("number of duplicate rows:", df.duplicated().sum())
#information about the original dataset
print(df.info())
#number of missing values of the original dataset
print("number of missing values:", df.isnull().sum().sum())

# Cleaning the messy dataset
# 2. Standardize missing-value representations
df = df.replace(
    r'(?i)^\s*(?:n/?a|null|none|unknown|\?)\s*$',
    pd.NA,
    regex=True
)

# 3. Fix student IDs
df["student_id"] = (
    df["student_id"]
    .astype("string")
    .str.strip()
    .str.upper()
    .str.replace(r"\s+", "", regex=True)
)

# 4. Fix student names
df["student_name"] = (
    df["student_name"]
    .astype("string")
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
    .str.title()
)

# 5. Fix numeric column formats
numeric_columns = [
    "study_hours",
    "attendance_percent",
    "previous_score",
    "sleep_hours",
    "exam_score"
]

for col in numeric_columns:
    values = df[col].astype("string").str.strip()

    # Remove units such as "hours" and "hrs"
    values = values.str.replace(
        r"(?i)\s*(?:hours?|hrs?)\s*$",
        "",
        regex=True
    )

    # Remove percentage symbols
    values = values.str.replace("%", "", regex=False)

    # Convert decimal commas to decimal points
    values = values.str.replace(",", ".", regex=False)

    # Convert to numeric; invalid text becomes NaN
    df[col] = pd.to_numeric(values, errors="coerce")

# 6. Remove completely duplicated rows
df = df.drop_duplicates()

# 7. Remove duplicate student IDs, keeping the first occurrence
df = df[
    df["student_id"].isna()
    | ~df["student_id"].duplicated(keep="first")
    ]

# 8. Remove invalid study hours
# Valid range: 0 to 24 hours
df = df[
    df["study_hours"].between(0, 24)
    | df["study_hours"].isna()
    ]

# 9. Remove invalid attendance percentages
# Valid range: 0% to 100%
df = df[
    df["attendance_percent"].between(0, 100)
    | df["attendance_percent"].isna()
    ]

# 10. Remove invalid previous scores
# Valid range: 0 to 100
df = df[
    df["previous_score"].between(0, 100)
    | df["previous_score"].isna()
    ]

# 11. Remove invalid sleep hours
# Valid range: 0 to 24 hours
df = df[
    df["sleep_hours"].between(0, 24)
    | df["sleep_hours"].isna()
    ]

# 12. Remove rows without an exam score
df = df.dropna(subset=["exam_score"])

# 13. Remove invalid exam scores
# Valid range: 0 to 100
df = df[df["exam_score"].between(0, 100)]

# 14. Reset the index
df = df.reset_index(drop=True)

# 15. Save the cleaned dataset
df.to_csv("Student_performance_cleaned.csv", index=False)

# 16. Display the results
print("\nCleaned dataset shape:", df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nColumn data types:")
print(df.dtypes)

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nRemaining duplicate rows:", df.duplicated().sum())

print("\nCleaned dataset saved as Student_performance_cleaned.csv")

#Visualize
plt.scatter(df.study_hours, df.exam_score)
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.show()

#Train
X = df[["study_hours", "attendance_percent", "previous_score", "sleep_hours"]]
y = df["exam_score"]
#80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#fill missing rows
medians = X_train.median()
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)
#Random Forest: collection of decision trees
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

#Evaluate: here we compare the predicted result against the actual result using MAE(Mean Absolute Error)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print("MAE(Mean Absolute Error):", round(mae, 1))

#Predict: predict the exam score of a student with 10 study hours, 80% attendance, 70 previous score, and 7 sleep hours
new_student = pd.DataFrame({
    "study_hours": [10],
    "attendance_percent": [80],
    "previous_score": [70],
    "sleep_hours": [7]
})
predicted_score = model.predict(new_student)[0]
print("Predicted exam score for the new student:", round(predicted_score, 1))