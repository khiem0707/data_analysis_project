import os
from pathlib import Path
from datetime import datetime

import pandas as pd


# =========================================================
# 1. Cấu hình đường dẫn
# =========================================================

RAW_DIR = Path("data/raw/C3_urbansound8k")
OUTPUT_TABLE_DIR = Path("outputs/tables/C3")
OUTPUT_LOG_DIR = Path("outputs/logs")

OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUTPUT_LOG_DIR / "C3_round1_data_audit_log.txt"


# =========================================================
# 2. Tìm file metadata UrbanSound8K.csv
# =========================================================

metadata_candidates = [
    RAW_DIR / "UrbanSound8K.csv",
    RAW_DIR / "metadata" / "UrbanSound8K.csv"
]

metadata_path = None
for candidate in metadata_candidates:
    if candidate.exists():
        metadata_path = candidate
        break

if metadata_path is None:
    raise FileNotFoundError(
        "Không tìm thấy UrbanSound8K.csv. "
        "Hãy kiểm tra file nằm ở data/raw/C3_urbansound8k/UrbanSound8K.csv "
        "hoặc data/raw/C3_urbansound8k/metadata/UrbanSound8K.csv"
    )


# =========================================================
# 3. Xác định thư mục chứa fold1 đến fold10
# Có thể là:
# - data/raw/C3_urbansound8k/fold1
# - data/raw/C3_urbansound8k/audio/fold1
# =========================================================

audio_base_candidates = [
    RAW_DIR,
    RAW_DIR / "audio"
]

audio_base_dir = None
for candidate in audio_base_candidates:
    if (candidate / "fold1").exists():
        audio_base_dir = candidate
        break

if audio_base_dir is None:
    raise FileNotFoundError(
        "Không tìm thấy thư mục fold1. "
        "Hãy kiểm tra cấu trúc data/raw/C3_urbansound8k/fold1...fold10 "
        "hoặc data/raw/C3_urbansound8k/audio/fold1...fold10"
    )


# =========================================================
# 4. Đọc metadata
# =========================================================

df = pd.read_csv(metadata_path)

required_columns = [
    "slice_file_name",
    "fsID",
    "start",
    "end",
    "salience",
    "fold",
    "classID",
    "class"
]

missing_required_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_required_columns:
    raise ValueError(
        f"Metadata thiếu các cột bắt buộc: {missing_required_columns}"
    )


# =========================================================
# 5. Tạo biến duration từ metadata
# =========================================================

df["duration_metadata"] = df["end"] - df["start"]


# =========================================================
# 6. Kiểm tra file audio có tồn tại không
# =========================================================

def build_expected_audio_path(row):
    return audio_base_dir / f"fold{int(row['fold'])}" / str(row["slice_file_name"])


df["expected_audio_path"] = df.apply(build_expected_audio_path, axis=1)
df["file_exists"] = df["expected_audio_path"].apply(lambda p: p.exists())

# Nếu có file không thấy theo đường dẫn chuẩn, thử tìm toàn bộ file .wav để đối chiếu
if not df["file_exists"].all():
    all_wav_files = list(audio_base_dir.rglob("*.wav"))
    filename_to_path = {p.name: p for p in all_wav_files}

    def find_audio_path(row):
        expected_path = row["expected_audio_path"]
        if expected_path.exists():
            return expected_path
        return filename_to_path.get(row["slice_file_name"], None)

    df["found_audio_path"] = df.apply(find_audio_path, axis=1)
    df["file_exists"] = df["found_audio_path"].notnull()
else:
    df["found_audio_path"] = df["expected_audio_path"]


# =========================================================
# 7. Kiểm kê tổng quan metadata
# =========================================================

shape_df = pd.DataFrame({
    "Item": ["Number of rows", "Number of columns"],
    "Value": [df.shape[0], df.shape[1]]
})

shape_df.to_csv(
    OUTPUT_TABLE_DIR / "C3_metadata_shape.csv",
    index=False,
    encoding="utf-8-sig"
)

dtypes_df = pd.DataFrame({
    "Column": df.columns,
    "Data_Type": [str(dtype) for dtype in df.dtypes]
})

dtypes_df.to_csv(
    OUTPUT_TABLE_DIR / "C3_metadata_dtypes.csv",
    index=False,
    encoding="utf-8-sig"
)

missing_df = pd.DataFrame({
    "Column": df.columns,
    "Missing_Count": df.isnull().sum().values,
    "Missing_Rate": df.isnull().sum().values / len(df) * 100
}).sort_values(by="Missing_Count", ascending=False)

missing_df.to_csv(
    OUTPUT_TABLE_DIR / "C3_metadata_missing_values.csv",
    index=False,
    encoding="utf-8-sig"
)

duplicate_count = int(df.duplicated(subset=["slice_file_name", "fold"]).sum())

duplicate_df = pd.DataFrame({
    "Item": ["Duplicated rows by slice_file_name and fold"],
    "Value": [duplicate_count]
})

duplicate_df.to_csv(
    OUTPUT_TABLE_DIR / "C3_metadata_duplicates.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 8. Phân bố class và fold
# =========================================================

class_distribution = (
    df["class"]
    .value_counts(dropna=False)
    .reset_index()
)

class_distribution.columns = ["Class", "Count"]
class_distribution["Rate_Percent"] = (
    class_distribution["Count"] / len(df) * 100
)

class_distribution.to_csv(
    OUTPUT_TABLE_DIR / "C3_class_distribution.csv",
    index=False,
    encoding="utf-8-sig"
)

fold_distribution = (
    df["fold"]
    .value_counts(dropna=False)
    .sort_index()
    .reset_index()
)

fold_distribution.columns = ["Fold", "Count"]
fold_distribution["Rate_Percent"] = (
    fold_distribution["Count"] / len(df) * 100
)

fold_distribution.to_csv(
    OUTPUT_TABLE_DIR / "C3_fold_distribution.csv",
    index=False,
    encoding="utf-8-sig"
)

class_fold_crosstab = pd.crosstab(
    df["class"],
    df["fold"],
    margins=True
)

class_fold_crosstab.to_csv(
    OUTPUT_TABLE_DIR / "C3_class_by_fold_crosstab.csv",
    encoding="utf-8-sig"
)


# =========================================================
# 9. Thống kê biến duration_metadata
# =========================================================

duration_stats = df["duration_metadata"].describe().reset_index()
duration_stats.columns = ["Statistic", "Value"]

duration_stats.to_csv(
    OUTPUT_TABLE_DIR / "C3_duration_metadata_describe.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 10. Kiểm tra audio file thực tế
# =========================================================

all_audio_files = list(audio_base_dir.rglob("*.wav"))

audio_file_summary = pd.DataFrame({
    "Item": [
        "Audio base directory",
        "Number of wav files found",
        "Number of metadata rows",
        "Number of metadata rows with existing file",
        "Number of metadata rows with missing file"
    ],
    "Value": [
        str(audio_base_dir),
        len(all_audio_files),
        len(df),
        int(df["file_exists"].sum()),
        int((~df["file_exists"]).sum())
    ]
})

audio_file_summary.to_csv(
    OUTPUT_TABLE_DIR / "C3_audio_file_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

missing_audio_files = df.loc[
    ~df["file_exists"],
    ["slice_file_name", "fold", "class", "expected_audio_path"]
]

missing_audio_files.to_csv(
    OUTPUT_TABLE_DIR / "C3_missing_audio_files.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 11. Phân loại bản chất biến metadata
# =========================================================

variable_classification_rows = [
    {
        "Variable": "slice_file_name",
        "Meaning": "Tên file âm thanh",
        "Python_Data_Type": "object",
        "Variable_Nature": "Qualitative",
        "Subtype": "Nominal",
        "Role": "Identifier"
    },
    {
        "Variable": "fsID",
        "Meaning": "ID nguồn âm thanh gốc",
        "Python_Data_Type": "int",
        "Variable_Nature": "Quantitative",
        "Subtype": "Discrete",
        "Role": "Identifier / group reference"
    },
    {
        "Variable": "start",
        "Meaning": "Thời điểm bắt đầu đoạn âm thanh trong file gốc",
        "Python_Data_Type": "float",
        "Variable_Nature": "Quantitative",
        "Subtype": "Continuous",
        "Role": "Metadata feature"
    },
    {
        "Variable": "end",
        "Meaning": "Thời điểm kết thúc đoạn âm thanh trong file gốc",
        "Python_Data_Type": "float",
        "Variable_Nature": "Quantitative",
        "Subtype": "Continuous",
        "Role": "Metadata feature"
    },
    {
        "Variable": "duration_metadata",
        "Meaning": "Thời lượng đoạn âm thanh, tính bằng end - start",
        "Python_Data_Type": "float",
        "Variable_Nature": "Quantitative",
        "Subtype": "Continuous",
        "Role": "Derived feature"
    },
    {
        "Variable": "salience",
        "Meaning": "Mức độ nổi bật của âm thanh foreground/background",
        "Python_Data_Type": "int",
        "Variable_Nature": "Qualitative",
        "Subtype": "Ordinal/Binary-like",
        "Role": "Metadata feature"
    },
    {
        "Variable": "fold",
        "Meaning": "Nhóm fold dùng cho chia dữ liệu/thực nghiệm",
        "Python_Data_Type": "int",
        "Variable_Nature": "Qualitative",
        "Subtype": "Nominal",
        "Role": "Split group"
    },
    {
        "Variable": "classID",
        "Meaning": "Mã số lớp âm thanh",
        "Python_Data_Type": "int",
        "Variable_Nature": "Qualitative",
        "Subtype": "Nominal encoded as integer",
        "Role": "Label code"
    },
    {
        "Variable": "class",
        "Meaning": "Tên lớp âm thanh",
        "Python_Data_Type": "object",
        "Variable_Nature": "Qualitative",
        "Subtype": "Nominal",
        "Role": "Target label"
    }
]

variable_classification_df = pd.DataFrame(variable_classification_rows)

variable_classification_df.to_csv(
    OUTPUT_TABLE_DIR / "C3_variable_classification_metadata.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 12. Ghi log
# =========================================================

with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("C3 UrbanSound8K - Round 1 Data Audit\n")
    f.write("=" * 60 + "\n")
    f.write(f"Run time: {datetime.now()}\n")
    f.write(f"Raw dir: {RAW_DIR}\n")
    f.write(f"Metadata path: {metadata_path}\n")
    f.write(f"Audio base dir: {audio_base_dir}\n\n")

    f.write("METADATA SUMMARY\n")
    f.write("-" * 60 + "\n")
    f.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns including derived audit columns\n")
    f.write(f"Original metadata columns: {required_columns}\n")
    f.write(f"Duplicate rows by slice_file_name and fold: {duplicate_count}\n")
    f.write("\nMissing values:\n")
    f.write(missing_df.to_string(index=False))
    f.write("\n\n")

    f.write("AUDIO FILE CHECK\n")
    f.write("-" * 60 + "\n")
    f.write(audio_file_summary.to_string(index=False))
    f.write("\n\n")

    f.write("CLASS DISTRIBUTION\n")
    f.write("-" * 60 + "\n")
    f.write(class_distribution.to_string(index=False))
    f.write("\n\n")

    f.write("FOLD DISTRIBUTION\n")
    f.write("-" * 60 + "\n")
    f.write(fold_distribution.to_string(index=False))
    f.write("\n\n")

    f.write("DURATION FROM METADATA SUMMARY\n")
    f.write("-" * 60 + "\n")
    f.write(duration_stats.to_string(index=False))
    f.write("\n")

print("Hoàn thành kiểm kê dữ liệu C3 UrbanSound8K.")
print(f"Metadata shape: {df.shape}")
print(f"Number of wav files found: {len(all_audio_files)}")
print(f"Rows with existing audio file: {int(df['file_exists'].sum())}/{len(df)}")
print(f"Output tables saved to: {OUTPUT_TABLE_DIR}")
print(f"Log saved to: {LOG_PATH}")