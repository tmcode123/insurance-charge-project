"""
data_cleaning.py

Cleans the raw insurance dataset and loads it into a SQLite database.
Run this script first before opening the notebook, Power BI, Tableau, or SQL analysis.

Input   : data/insurance.csv
Outputs : data/insurance_clean.csv
          data/data_quality_report.csv
          data/insurance.db

Run: python data_cleaning.py
"""

import sqlite3
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")


# File paths
DATA_DIR     = Path("data")
RAW_PATH     = DATA_DIR / "insurance.csv"
CLEAN_PATH   = DATA_DIR / "insurance_clean.csv"
QUALITY_PATH = DATA_DIR / "data_quality_report.csv"
DB_PATH      = DATA_DIR / "insurance.db"

# BMI brackets used to group customers into health-risk categories.
BMI_BINS: list[float] = [0, 18.5, 25, 30, float("inf")]
BMI_LABELS: list[str] = [
    "Underweight",
    "Healthy",
    "Overweight",
    "Obese",
]

# Age brackets used for dashboard filters and summary analysis.
AGE_BINS: list[int] = [17, 25, 35, 45, 55, 64]
AGE_LABELS: list[str] = [
    "18-25",
    "26-35",
    "36-45",
    "46-55",
    "56-64",
]

# Required columns for the insurance charges analysis.
REQUIRED_COLS: list[str] = [
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region",
    "charges",
]


def log_step(msg: str) -> None:
    """Print a progress message so you can follow what the script is doing."""
    print(f"[ETL] {msg}")


def check_files_exist() -> None:
    """
    Check that the required input file exists before cleaning starts.
    Gives a clear error message if the file is missing.
    """
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Required input file not found: {RAW_PATH}\n"
            f"Make sure the data/ folder is in the same directory as this script."
        )


def quality_report(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Produces a data quality summary for a dataframe.

    For each column it reports dtype, null count, null percentage,
    and number of unique values.
    """
    return pd.DataFrame({
        "dataset": name,
        "column": df.columns,
        "dtype": df.dtypes.values,
        "null_count": df.isnull().sum().values,
        "null_pct": (df.isnull().mean() * 100).round(1).values,
        "unique": [df[c].apply(str).nunique() for c in df.columns],
    })


def validate_required_columns(df: pd.DataFrame) -> None:
    """Make sure the raw file contains every column needed for analysis."""
    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def standardize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text fields so categories are consistent for SQL and dashboards."""
    df["sex"] = df["sex"].str.strip().str.lower()
    df["smoker"] = df["smoker"].str.strip().str.lower()
    df["region"] = df["region"].str.strip().str.lower()

    return df


def validate_category_values(df: pd.DataFrame) -> None:
    """Check that category columns only contain expected values."""
    valid_values = {
        "sex": {"female", "male"},
        "smoker": {"yes", "no"},
        "region": {"northeast", "northwest", "southeast", "southwest"},
    }

    for column, allowed in valid_values.items():
        unexpected = sorted(set(df[column].dropna()) - allowed)

        if unexpected:
            raise ValueError(f"Unexpected values found in {column}: {unexpected}")


def validate_numeric_values(df: pd.DataFrame) -> None:
    """Check that numeric fields are within reasonable ranges."""
    rules = {
        "age": df["age"].between(18, 64),
        "bmi": df["bmi"].gt(0),
        "children": df["children"].between(0, 5),
        "charges": df["charges"].gt(0),
    }

    for column, rule in rules.items():
        invalid_count = (~rule | df[column].isna()).sum()

        if invalid_count > 0:
            raise ValueError(f"{column} has {invalid_count} invalid value(s).")


# Make sure everything is in place before we start
check_files_exist()
DATA_DIR.mkdir(exist_ok=True)


# Load the raw data
log_step("Loading raw data ...")

df = pd.read_csv(RAW_PATH)
log_step(f"  insurance_raw : {df.shape[0]:,} rows x {df.shape[1]} cols")


# Clean the data
log_step("Cleaning insurance data ...")

# Standardize column names for cleaner pandas, SQL, Power BI, and Tableau work
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

validate_required_columns(df)

# Keep only the columns needed for this project
df = df[REQUIRED_COLS].copy()

# Convert numeric fields explicitly
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
df["children"] = pd.to_numeric(df["children"], errors="coerce")
df["charges"] = pd.to_numeric(df["charges"], errors="coerce")

# Clean category fields
df = standardize_text_columns(df)

# Check for missing values before continuing
missing_total = int(df.isnull().sum().sum())

if missing_total > 0:
    print("\nMissing Values")
    print(df.isnull().sum())
    raise ValueError("Missing values found. Review the raw file before continuing.")

# Validate values before feature engineering
validate_category_values(df)
validate_numeric_values(df)

# Remove exact duplicate customer records
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
log_step(f"  Removed {before - len(df):,} duplicate rows")

# Convert integer-like fields after validation
df["age"] = df["age"].astype(int)
df["children"] = df["children"].astype(int)

# Derived columns used throughout the analysis
df["age_group"] = pd.cut(
    df["age"],
    bins=AGE_BINS,
    labels=AGE_LABELS,
)

df["bmi_category"] = pd.cut(
    df["bmi"],
    bins=BMI_BINS,
    labels=BMI_LABELS,
)

df["is_smoker"] = (df["smoker"] == "yes").astype(int)
df["is_obese"] = (df["bmi_category"] == "Obese").astype(int)

# Charge level uses quartiles from this dataset
q1 = df["charges"].quantile(0.25)
q3 = df["charges"].quantile(0.75)

df["charge_level"] = "Medium"
df.loc[df["charges"] <= q1, "charge_level"] = "Low"
df.loc[df["charges"] >= q3, "charge_level"] = "High"

log_step(f"  Clean shape: {df.shape}")
log_step(f"  Average charge: ${df['charges'].mean():,.2f}")
log_step(f"  Median charge:  ${df['charges'].median():,.2f}")
log_step(f"  Max charge:     ${df['charges'].max():,.2f}")


# Generate a data quality report
log_step("Generating data quality report ...")

qr = quality_report(df, "insurance")
print("\nData Quality Report")
print(qr.to_string(index=False))


# Save the cleaned data
log_step("Saving cleaned data ...")

df.to_csv(CLEAN_PATH, index=False)
qr.to_csv(QUALITY_PATH, index=False)

log_step(f"  -> {CLEAN_PATH}   ({len(df):,} rows)")
log_step(f"  -> {QUALITY_PATH}")


print("\nSummary Stats")
print(df[["age", "bmi", "children", "charges"]].describe().round(2))

print("\nAverage Charges by Smoker")
print(
    df.groupby("smoker")["charges"]
    .agg(["count", "mean", "median"])
    .round(2)
)

print("\nAverage Charges by BMI Category")
print(
    df.groupby("bmi_category")["charges"]
    .agg(["count", "mean", "median"])
    .round(2)
)


# Load into SQLite database
log_step("Creating SQLite database ...")

with sqlite3.connect(DB_PATH) as conn:
    df.to_sql("insurance", conn, if_exists="replace", index=False)

    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM insurance")
    row_count: int = cursor.fetchone()[0]

log_step(f"  -> {DB_PATH}   ({row_count:,} rows in 'insurance' table)")
log_step("Done")
