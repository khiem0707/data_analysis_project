import os
from pathlib import Path
from datetime import datetime

import pandas as pd


# =========================================================
# 1. Cấu hình đường dẫn
# =========================================================

RAW_DIR = Path("data/raw/B3_bike_sharing")
OUTPUT_TABLE_DIR = Path("outputs/tables/B3")
OUTPUT_LOG_DIR = Path("outputs/logs")

OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUTPUT_LOG_DIR / "B3_round1_data_audit_log.txt"


# =========================================================
# 2. Tìm file day.csv và hour.csv tự động
# =========================================================

def find_file(root_dir: Path, filename: str) -> Path:
    matches = list(root_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Không tìm thấy {filename} trong {root_dir}")
    return matches[0]


day_path = find_file(RAW_DIR, "day.csv")
hour_path = find_file(RAW_DIR, "hour.csv")


# =========================================================
# 3. Đọc dữ liệu
# =========================================================

day_df = pd.read_csv(day_path)
hour_df = pd.read_csv(hour_path)

# Chuyển dteday sang datetime nếu có
for df in [day_df, hour_df]:
    if "dteday" in df.columns:
        df["dteday"] = pd.to_datetime(df["dteday"], errors="coerce")


# =========================================================
# 4. Hàm kiểm kê chung
# =========================================================

def audit_dataframe(df: pd.DataFrame, dataset_name: str):
    # Shape
    shape_df = pd.DataFrame({
        "Item": ["Number of rows", "Number of columns"],
        "Value": [df.shape[0], df.shape[1]]
    })
    shape_df.to_csv(
        OUTPUT_TABLE_DIR / f"{dataset_name}_dataset_shape.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Dtypes
    dtypes_df = pd.DataFrame({
        "Column": df.columns,
        "Data_Type": [str(dtype) for dtype in df.dtypes]
    })
    dtypes_df.to_csv(
        OUTPUT_TABLE_DIR / f"{dataset_name}_dtypes.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Missing values
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing_Count": df.isnull().sum().values,
        "Missing_Rate": df.isnull().sum().values / len(df) * 100
    }).sort_values(by="Missing_Count", ascending=False)

    missing_df.to_csv(
        OUTPUT_TABLE_DIR / f"{dataset_name}_missing_values.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Unique values
    unique_df = pd.DataFrame({
        "Column": df.columns,
        "Unique_Count": df.nunique(dropna=False).values
    }).sort_values(by="Unique_Count", ascending=True)

    unique_df.to_csv(
        OUTPUT_TABLE_DIR / f"{dataset_name}_unique_values.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Numeric describe
    numeric_describe = df.describe().T.reset_index()
    numeric_describe = numeric_describe.rename(columns={"index": "Column"})

    numeric_describe.to_csv(
        OUTPUT_TABLE_DIR / f"{dataset_name}_numeric_describe_raw.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Duplicate count
    duplicate_count = int(df.duplicated().sum())
    duplicate_df = pd.DataFrame({
        "Item": ["Duplicated rows"],
        "Value": [duplicate_count]
    })

    duplicate_df.to_csv(
        OUTPUT_TABLE_DIR / f"{dataset_name}_duplicates.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Constant columns
    constant_columns = [
        col for col in df.columns
        if df[col].nunique(dropna=False) == 1
    ]

    constant_df = pd.DataFrame({
        "Constant_Column": constant_columns
    })

    constant_df.to_csv(
        OUTPUT_TABLE_DIR / f"{dataset_name}_constant_columns.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return {
        "shape": df.shape,
        "missing_df": missing_df,
        "duplicate_count": duplicate_count,
        "constant_columns": constant_columns
    }


# =========================================================
# 5. Kiểm kê day.csv và hour.csv
# =========================================================

day_audit = audit_dataframe(day_df, "B3_day")
hour_audit = audit_dataframe(hour_df, "B3_hour")


# =========================================================
# 6. Kiểm tra thời gian
# =========================================================

time_summary_rows = []

for dataset_name, df in [("B3_day", day_df), ("B3_hour", hour_df)]:
    if "dteday" in df.columns:
        min_date = df["dteday"].min()
        max_date = df["dteday"].max()
        num_days = df["dteday"].nunique()

        time_summary_rows.append({
            "Dataset": dataset_name,
            "Start_Date": min_date,
            "End_Date": max_date,
            "Number_Of_Unique_Days": num_days
        })

time_summary_df = pd.DataFrame(time_summary_rows)

time_summary_df.to_csv(
    OUTPUT_TABLE_DIR / "B3_time_range.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 7. Kiểm tra phân bố một số biến chính
# =========================================================

categorical_columns = [
    "season", "yr", "mnth", "hr", "holiday",
    "weekday", "workingday", "weathersit"
]

for col in categorical_columns:
    if col in hour_df.columns:
        freq_df = hour_df[col].value_counts(dropna=False).reset_index()
        freq_df.columns = [col, "Count"]
        freq_df["Rate_Percent"] = freq_df["Count"] / len(hour_df) * 100

        freq_df.to_csv(
            OUTPUT_TABLE_DIR / f"B3_hour_frequency_{col}.csv",
            index=False,
            encoding="utf-8-sig"
        )


# =========================================================
# 8. Ghi log
# =========================================================

with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("B3 Bike Sharing Dataset - Round 1 Data Audit\n")
    f.write("=" * 60 + "\n")
    f.write(f"Run time: {datetime.now()}\n")
    f.write(f"Raw dir: {RAW_DIR}\n")
    f.write(f"day.csv path: {day_path}\n")
    f.write(f"hour.csv path: {hour_path}\n\n")

    f.write("DAY DATASET\n")
    f.write("-" * 60 + "\n")
    f.write(f"Shape: {day_df.shape[0]} rows, {day_df.shape[1]} columns\n")
    f.write(f"Duplicate rows: {day_audit['duplicate_count']}\n")
    f.write(f"Constant columns: {day_audit['constant_columns']}\n")
    f.write("\nMissing values:\n")
    f.write(day_audit["missing_df"].to_string(index=False))
    f.write("\n\n")

    f.write("HOUR DATASET\n")
    f.write("-" * 60 + "\n")
    f.write(f"Shape: {hour_df.shape[0]} rows, {hour_df.shape[1]} columns\n")
    f.write(f"Duplicate rows: {hour_audit['duplicate_count']}\n")
    f.write(f"Constant columns: {hour_audit['constant_columns']}\n")
    f.write("\nMissing values:\n")
    f.write(hour_audit["missing_df"].to_string(index=False))
    f.write("\n\n")

    f.write("TIME RANGE\n")
    f.write("-" * 60 + "\n")
    f.write(time_summary_df.to_string(index=False))
    f.write("\n\n")

    if "cnt" in hour_df.columns:
        f.write("Target variable cnt summary from hour.csv\n")
        f.write("-" * 60 + "\n")
        f.write(hour_df["cnt"].describe().to_string())
        f.write("\n")


print("Hoàn thành kiểm kê dữ liệu B3 Bike Sharing.")
print(f"day.csv shape: {day_df.shape}")
print(f"hour.csv shape: {hour_df.shape}")
print(f"Output tables saved to: {OUTPUT_TABLE_DIR}")
print(f"Log saved to: {LOG_PATH}")