from pathlib import Path

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =========================
# 1. Load dataset
# =========================

wine = load_wine(as_frame=True)
df = wine.frame.copy()


# =========================
# 2. Basic information
# =========================

print("===== Dataset Shape =====")
print(df.shape)

print("\n===== First 5 Rows =====")
print(df.head())

print("\n===== Column Names =====")
print(df.columns.tolist())

print("\n===== Data Types =====")
print(df.dtypes)


# =========================
# 3. Save raw dataset
# =========================

raw_data_dir = Path("data/raw")
raw_data_dir.mkdir(parents=True, exist_ok=True)

raw_csv_path = raw_data_dir / "wine_raw.csv"
df.to_csv(raw_csv_path, index=False)

print("\n===== Raw CSV Saved =====")
print(raw_csv_path)


# =========================
# 4. Check missing values
# =========================

missing_values = df.isna().sum()
total_missing_values = missing_values.sum()

print("\n===== Missing Values =====")
print(missing_values)

print("\nTotal Missing Values:")
print(total_missing_values)


# =========================
# 5. Check duplicate rows
# =========================

duplicate_count = df.duplicated().sum()

print("\n===== Duplicate Rows =====")
print("Number of duplicate rows:", duplicate_count)


# =========================
# 6. Check data types
# =========================

print("\n===== Data Types Check =====")
print(df.dtypes)

print("\nNumber of columns by data type:")
print(df.dtypes.value_counts())


# =========================
# 7. Separate features and target
# =========================

X = df.drop(columns=["target"])
y = df["target"]

print("\n===== Features and Target =====")
print("X shape:", X.shape)
print("y shape:", y.shape)


# =========================
# 8. Train/test split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\n===== Train/Test Split =====")
print("Training samples:", len(X_train))
print("Test samples:", len(X_test))
print("Training ratio:", f"{len(X_train) / len(X):.2%}")
print("Test ratio:", f"{len(X_test) / len(X):.2%}")


# =========================
# 9. Preprocessing
# Fit ONLY on training data
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n===== Standardization =====")
print("Scaler fitted on training set only.")
print("Training set transformed.")
print("Test set transformed using training-set statistics.")


# =========================
# 10. Convert processed data
#     back to DataFrames
# =========================

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X.columns,
    index=X_train.index,
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X.columns,
    index=X_test.index,
)

train_processed = X_train_scaled.copy()
train_processed["target"] = y_train

test_processed = X_test_scaled.copy()
test_processed["target"] = y_test


# =========================
# 11. Save processed datasets
# =========================

processed_data_dir = Path("data/processed")
processed_data_dir.mkdir(parents=True, exist_ok=True)

train_csv_path = processed_data_dir / "wine_train.csv"
test_csv_path = processed_data_dir / "wine_test.csv"

train_processed.to_csv(train_csv_path, index=False)
test_processed.to_csv(test_csv_path, index=False)

print("\n===== Processed CSV Files Saved =====")
print(train_csv_path)
print(test_csv_path)


# =========================
# 12. Create data check report
# =========================

reports_dir = Path("reports")
reports_dir.mkdir(parents=True, exist_ok=True)

report_path = reports_dir / "wine_data_check.txt"

report = f"""Wine Dataset Data Check Report
================================

Dataset:
Wine dataset from scikit-learn

Original dataset shape:
{df.shape[0]} rows, {df.shape[1]} columns

Features:
{X.shape[1]}

Target:
target

Missing values:
{total_missing_values}

Duplicate rows:
{duplicate_count}

Data types:
{df.dtypes.value_counts().to_string()}

Cleaning result:
No missing values or duplicate rows were found.
All column data types are appropriate.
No additional cleaning operation was required.

Train/Test Split:
Training samples: {len(X_train)}
Test samples: {len(X_test)}
Training ratio: {len(X_train) / len(X):.2%}
Test ratio: {len(X_test) / len(X):.2%}
random_state: 42
stratify: target

Preprocessing:
StandardScaler was used for feature standardization.

Important:
The scaler was fitted ONLY on the training set.
The test set was transformed using statistics learned
from the training set to avoid data leakage.
"""

report_path.write_text(report, encoding="utf-8")

print("\n===== Data Check Report Saved =====")
print(report_path)


# =========================
# 13. Final summary
# =========================

print("\n===== Final Summary =====")
print("Raw dataset saved:", raw_csv_path)
print("Training dataset saved:", train_csv_path)
print("Test dataset saved:", test_csv_path)
print("Data check report saved:", report_path)
print("Data preparation completed successfully.")