import os
import pandas as pd
import numpy as np
from datetime import datetime

# =========================================================
# 1. Cấu hình đường dẫn
# =========================================================

RAW_PATH = "data/raw/A3_customer_personality/marketing_campaign.csv"

OUTPUT_TABLE_DIR = "outputs/tables/A3"
OUTPUT_LOG_DIR = "outputs/logs"

os.makedirs(OUTPUT_TABLE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LOG_DIR, exist_ok=True)

LOG_PATH = os.path.join(OUTPUT_LOG_DIR, "A3_round1_data_audit_log.txt")

# =========================================================
# 2. Đọc dữ liệu
# Dataset này là CSV nhưng phân tách bằng tab
# =========================================================

df = pd.read_csv(RAW_PATH, sep="\t")

# =========================================================
# 3. Kiểm tra thông tin tổng quan
# =========================================================

shape_df = pd.DataFrame({
    "Item": ["Number of rows", "Number of columns"],
    "Value": [df.shape[0], df.shape[1]]
})

shape_df.to_csv(
    os.path.join(OUTPUT_TABLE_DIR, "A3_dataset_shape.csv"),
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# 4. Kiểm tra kiểu dữ liệu
# =========================================================

dtypes_df = pd.DataFrame({
    "Column": df.columns,
    "Data_Type": [str(dtype) for dtype in df.dtypes]
})

dtypes_df.to_csv(
    os.path.join(OUTPUT_TABLE_DIR, "A3_dtypes.csv"),
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# 5. Kiểm tra missing values
# =========================================================

missing_df = pd.DataFrame({
    "Column": df.columns,
    "Missing_Count": df.isnull().sum().values,
    "Missing_Rate": (df.isnull().sum().values / len(df)) * 100
})

missing_df = missing_df.sort_values(
    by="Missing_Count",
    ascending=False
)

missing_df.to_csv(
    os.path.join(OUTPUT_TABLE_DIR, "A3_missing_values.csv"),
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# 6. Kiểm tra số lượng giá trị duy nhất
# =========================================================

unique_df = pd.DataFrame({
    "Column": df.columns,
    "Unique_Count": df.nunique().values
})

unique_df = unique_df.sort_values(
    by="Unique_Count",
    ascending=True
)

unique_df.to_csv(
    os.path.join(OUTPUT_TABLE_DIR, "A3_unique_values.csv"),
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# 7. Kiểm tra phân bố target Response
# =========================================================

response_distribution = (
    df["Response"]
    .value_counts(dropna=False)
    .reset_index()
)

response_distribution.columns = ["Response", "Count"]
response_distribution["Rate_Percent"] = (
    response_distribution["Count"] / len(df) * 100
)

response_distribution.to_csv(
    os.path.join(OUTPUT_TABLE_DIR, "A3_response_distribution.csv"),
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# 8. Thống kê mô tả dữ liệu numeric thô
# =========================================================

numeric_describe = df.describe().T.reset_index()
numeric_describe = numeric_describe.rename(columns={"index": "Column"})

numeric_describe.to_csv(
    os.path.join(OUTPUT_TABLE_DIR, "A3_numeric_describe_raw.csv"),
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# 9. Kiểm tra duplicate
# =========================================================

duplicate_count = df.duplicated().sum()

duplicate_df = pd.DataFrame({
    "Item": ["Duplicated rows"],
    "Value": [duplicate_count]
})

duplicate_df.to_csv(
    os.path.join(OUTPUT_TABLE_DIR, "A3_duplicates.csv"),
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# 10. Kiểm tra cột hằng số
# =========================================================

constant_columns = [
    col for col in df.columns
    if df[col].nunique(dropna=False) == 1
]

constant_df = pd.DataFrame({
    "Constant_Column": constant_columns
})

constant_df.to_csv(
    os.path.join(OUTPUT_TABLE_DIR, "A3_constant_columns.csv"),
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# 11. Ghi log
# =========================================================

with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("A3 Customer Personality Analysis - Round 1 Data Audit\n")
    f.write("=" * 60 + "\n")
    f.write(f"Run time: {datetime.now()}\n")
    f.write(f"Raw path: {RAW_PATH}\n")
    f.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    f.write(f"Duplicate rows: {duplicate_count}\n")
    f.write("\nMissing values:\n")
    f.write(missing_df.to_string(index=False))
    f.write("\n\nResponse distribution:\n")
    f.write(response_distribution.to_string(index=False))
    f.write("\n\nConstant columns:\n")
    f.write(str(constant_columns))
    f.write("\n")

print("Hoàn thành kiểm kê dữ liệu A3.")
print(f"Shape: {df.shape}")
print(f"Duplicate rows: {duplicate_count}")
print(f"Output tables saved to: {OUTPUT_TABLE_DIR}")
print(f"Log saved to: {LOG_PATH}")