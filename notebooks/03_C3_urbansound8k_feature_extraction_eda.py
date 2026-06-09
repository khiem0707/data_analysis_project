# =========================================================
# VÒNG 4 - AUDIO FEATURE EXTRACTION VÀ EDA CHO C3: URBANSOUND8K
# =========================================================
# Mục tiêu:
# 1. Đọc metadata UrbanSound8K và file WAV trong fold1 đến fold10
# 2. Số hóa dữ liệu audio thô thành các đặc trưng định lượng
# 3. Tạo C3_audio_features.csv
# 4. Tính thống kê tổng thể và theo từng lớp
# 5. Vẽ Histogram, Boxplot, Scatter Plot, Heatmap, Waveform, Mel-spectrogram
# =========================================================

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import librosa
import librosa.display


# =========================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# =========================================================

RAW_DIR = Path("data/raw/C3_urbansound8k")

ROUND_DIR = Path("outputs/round_04_C3_audio_eda")
DATA_DIR = ROUND_DIR / "data"
LOG_DIR = ROUND_DIR / "logs"

TABLE_DATASET_DIR = ROUND_DIR / "tables" / "dataset_level"
TABLE_CLASS_DIR = ROUND_DIR / "tables" / "class_level"
TABLE_QUALITY_DIR = ROUND_DIR / "tables" / "quality_check"

FIG_DATASET_DIR = ROUND_DIR / "figures" / "dataset_level"
FIG_CLASS_DIR = ROUND_DIR / "figures" / "class_level"
FIG_WAVEFORM_DIR = ROUND_DIR / "figures" / "waveform_spectrogram"

for folder in [
    DATA_DIR, LOG_DIR,
    TABLE_DATASET_DIR, TABLE_CLASS_DIR, TABLE_QUALITY_DIR,
    FIG_DATASET_DIR, FIG_CLASS_DIR, FIG_WAVEFORM_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / "C3_round4_audio_eda_log.txt"
FEATURE_PATH = DATA_DIR / "C3_audio_features.csv"

# Nếu đã có C3_audio_features.csv và không muốn trích xuất lại, để False.
FORCE_REEXTRACT = False

# Nếu muốn test nhanh, đặt số nhỏ như 100. Chạy toàn bộ thì để None.
MAX_FILES_FOR_DEBUG = None

N_MFCC = 13


# =========================================================
# 2. HÀM TIỆN ÍCH
# =========================================================

def save_csv(df, path):
    """Lưu dataframe ra CSV với encoding phù hợp Excel tiếng Việt."""
    df.to_csv(path, index=False, encoding="utf-8-sig")


def safe_mode(series):
    """Lấy mode đầu tiên. Nếu không có mode thì trả về NaN."""
    mode_values = series.dropna().mode()
    if len(mode_values) == 0:
        return np.nan
    return mode_values.iloc[0]


def safe_cv(mean_value, std_value):
    """Tính hệ số biến thiên CV. Nếu mean bằng 0 thì trả về NaN."""
    if pd.isna(mean_value) or mean_value == 0:
        return np.nan
    return std_value / mean_value * 100


def iqr_bounds(series):
    """Tính Q1, Q3, IQR, lower bound và upper bound theo quy tắc IQR."""
    clean_series = series.dropna()
    if len(clean_series) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return q1, q3, iqr, lower, upper


def count_outliers(series):
    """Đếm số outliers theo quy tắc IQR."""
    q1, q3, iqr, lower, upper = iqr_bounds(series)
    if pd.isna(lower) or pd.isna(upper):
        return 0
    return int(((series < lower) | (series > upper)).sum())


def descriptive_statistics(df, columns):
    """Tính thống kê mô tả đầy đủ cho biến định lượng."""
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col]
        mean_value = series.mean()
        median_value = series.median()
        mode_value = safe_mode(series)
        min_value = series.min()
        max_value = series.max()
        range_value = max_value - min_value
        variance_value = series.var()
        std_value = series.std()
        cv_value = safe_cv(mean_value, std_value)
        q1, q3, iqr, lower, upper = iqr_bounds(series)
        rows.append({
            "Variable": col,
            "Count": series.count(),
            "Missing_Count": series.isna().sum(),
            "Mean": mean_value,
            "Median": median_value,
            "Mode": mode_value,
            "Min": min_value,
            "Max": max_value,
            "Range": range_value,
            "Variance": variance_value,
            "Standard_Deviation": std_value,
            "CV_Percent": cv_value,
            "Q1_25_Percentile": q1,
            "Q2_50_Percentile": median_value,
            "Q3_75_Percentile": q3,
            "IQR": iqr,
            "Lower_Bound_IQR": lower,
            "Upper_Bound_IQR": upper,
            "Outlier_Count_IQR": count_outliers(series)
        })
    return pd.DataFrame(rows)


def frequency_table(df, column):
    """Lập bảng tần suất tuyệt đối, tương đối và mode."""
    freq = df[column].value_counts(dropna=False).reset_index()
    freq.columns = [column, "Frequency"]
    freq["Relative_Frequency"] = freq["Frequency"] / len(df)
    freq["Percent"] = freq["Relative_Frequency"] * 100
    freq["Mode_Of_Variable"] = safe_mode(df[column])
    return freq


def save_current_figure(path):
    """Lưu biểu đồ hiện tại."""
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


# =========================================================
# 3. TÌM METADATA VÀ AUDIO FOLD
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
    raise FileNotFoundError("Không tìm thấy UrbanSound8K.csv.")

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
    raise FileNotFoundError("Không tìm thấy thư mục fold1 đến fold10.")


# =========================================================
# 4. ĐỌC METADATA VÀ TẠO ĐƯỜNG DẪN FILE AUDIO
# =========================================================

metadata_df = pd.read_csv(metadata_path)
metadata_df["duration_metadata"] = metadata_df["end"] - metadata_df["start"]

metadata_df["audio_path"] = metadata_df.apply(
    lambda row: str(audio_base_dir / f"fold{int(row['fold'])}" / str(row["slice_file_name"])),
    axis=1
)

metadata_df["file_exists"] = metadata_df["audio_path"].apply(lambda p: Path(p).exists())

save_csv(metadata_df, DATA_DIR / "C3_metadata_with_audio_path.csv")


# =========================================================
# 5. BẢNG PHÂN LOẠI BIẾN / FEATURE AUDIO
# =========================================================

variable_rows = [
    ["slice_file_name", "Tên file âm thanh", "Qualitative", "Nominal", "Identifier"],
    ["fold", "Nhóm fold dùng cho chia dữ liệu", "Qualitative", "Nominal", "Split group"],
    ["class", "Tên lớp âm thanh", "Qualitative", "Nominal", "Target label"],
    ["duration_audio", "Thời lượng thực tế của file âm thanh", "Quantitative", "Continuous", "Audio feature"],
    ["sample_rate", "Tần số lấy mẫu", "Quantitative", "Discrete", "Technical feature"],
    ["rms_mean", "Năng lượng âm thanh trung bình", "Quantitative", "Continuous", "Energy feature"],
    ["zcr_mean", "Zero Crossing Rate trung bình", "Quantitative", "Continuous", "Temporal feature"],
    ["spectral_centroid_mean", "Tần số trung tâm trung bình", "Quantitative", "Continuous", "Spectral feature"],
    ["spectral_bandwidth_mean", "Độ rộng phổ trung bình", "Quantitative", "Continuous", "Spectral feature"],
    ["spectral_rolloff_mean", "Spectral rolloff trung bình", "Quantitative", "Continuous", "Spectral feature"],
    ["mfcc_1_mean ... mfcc_13_mean", "Trung bình các hệ số MFCC", "Quantitative", "Continuous", "Cepstral feature"],
    ["mfcc_1_std ... mfcc_13_std", "Độ lệch chuẩn các hệ số MFCC", "Quantitative", "Continuous", "Cepstral feature"]
]

variable_classification_df = pd.DataFrame(
    variable_rows,
    columns=["Variable", "Meaning", "Variable_Nature", "Subtype", "Role"]
)

save_csv(
    variable_classification_df,
    TABLE_DATASET_DIR / "C3_variable_classification_audio_features.csv"
)


# =========================================================
# 6. TRÍCH XUẤT FEATURE AUDIO
# =========================================================

def extract_audio_features(audio_path):
    """Trích xuất đặc trưng định lượng từ một file WAV."""
    audio_path = Path(audio_path)
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    if y is None or len(y) == 0:
        raise ValueError("Audio rỗng hoặc không đọc được tín hiệu.")

    duration_audio = librosa.get_duration(y=y, sr=sr)
    audio_samples = len(y)

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    feature_dict = {
        "sample_rate": sr,
        "audio_samples": audio_samples,
        "duration_audio": duration_audio,
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "spectral_centroid_mean": float(np.mean(spectral_centroid)),
        "spectral_centroid_std": float(np.std(spectral_centroid)),
        "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
        "spectral_bandwidth_std": float(np.std(spectral_bandwidth)),
        "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
        "spectral_rolloff_std": float(np.std(spectral_rolloff))
    }

    for i in range(N_MFCC):
        feature_dict[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        feature_dict[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))

    return feature_dict


if FEATURE_PATH.exists() and not FORCE_REEXTRACT:
    print(f"Đã có file feature: {FEATURE_PATH}")
    print("Không trích xuất lại vì FORCE_REEXTRACT = False.")
    features_df = pd.read_csv(FEATURE_PATH)
else:
    rows = []
    error_rows = []

    process_df = metadata_df.copy()
    if MAX_FILES_FOR_DEBUG is not None:
        process_df = process_df.head(MAX_FILES_FOR_DEBUG).copy()

    total_files = len(process_df)
    print(f"Bắt đầu trích xuất feature audio cho {total_files} file.")
    print("Quá trình này có thể mất vài phút đến vài chục phút tùy máy.")

    for idx, row in process_df.iterrows():
        if len(rows) % 100 == 0:
            print(f"Đã xử lý {len(rows)}/{total_files} file...")

        base_info = {
            "slice_file_name": row["slice_file_name"],
            "fsID": row["fsID"],
            "start": row["start"],
            "end": row["end"],
            "duration_metadata": row["duration_metadata"],
            "salience": row["salience"],
            "fold": row["fold"],
            "classID": row["classID"],
            "class": row["class"],
            "audio_path": row["audio_path"],
            "file_exists": row["file_exists"]
        }

        try:
            if not row["file_exists"]:
                raise FileNotFoundError(f"Không tìm thấy file: {row['audio_path']}")
            feature_dict = extract_audio_features(row["audio_path"])
            base_info.update(feature_dict)
            base_info["feature_status"] = "ok"
            base_info["feature_error"] = ""
            rows.append(base_info)
        except Exception as exc:
            error_info = base_info.copy()
            error_info["feature_status"] = "error"
            error_info["feature_error"] = str(exc)
            error_rows.append(error_info)
            rows.append(error_info)

    features_df = pd.DataFrame(rows)
    save_csv(features_df, FEATURE_PATH)

    error_df = pd.DataFrame(error_rows)
    save_csv(error_df, TABLE_QUALITY_DIR / "C3_audio_feature_extraction_errors.csv")


# =========================================================
# 7. KIỂM TRA CHẤT LƯỢNG FEATURE
# =========================================================

quality_summary = pd.DataFrame({
    "Item": [
        "Metadata rows",
        "Feature rows",
        "Feature status ok",
        "Feature status error",
        "Unique classes",
        "Unique folds"
    ],
    "Value": [
        len(metadata_df),
        len(features_df),
        int((features_df["feature_status"] == "ok").sum()) if "feature_status" in features_df.columns else np.nan,
        int((features_df["feature_status"] == "error").sum()) if "feature_status" in features_df.columns else np.nan,
        features_df["class"].nunique() if "class" in features_df.columns else np.nan,
        features_df["fold"].nunique() if "fold" in features_df.columns else np.nan
    ]
})

save_csv(quality_summary, TABLE_QUALITY_DIR / "C3_audio_feature_quality_summary.csv")

missing_feature_values = pd.DataFrame({
    "Column": features_df.columns,
    "Missing_Count": features_df.isna().sum().values,
    "Missing_Rate": features_df.isna().sum().values / len(features_df) * 100
}).sort_values(by="Missing_Count", ascending=False)

save_csv(missing_feature_values, TABLE_QUALITY_DIR / "C3_audio_features_missing_values.csv")


# =========================================================
# 8. THỐNG KÊ DATASET-LEVEL VÀ CLASS-LEVEL
# =========================================================

audio_feature_cols = [
    "duration_audio",
    "sample_rate",
    "audio_samples",
    "rms_mean",
    "rms_std",
    "zcr_mean",
    "zcr_std",
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std"
]

audio_feature_cols += [f"mfcc_{i}_mean" for i in range(1, N_MFCC + 1)]
audio_feature_cols += [f"mfcc_{i}_std" for i in range(1, N_MFCC + 1)]

available_feature_cols = [col for col in audio_feature_cols if col in features_df.columns]

dataset_stats_df = descriptive_statistics(features_df, available_feature_cols)
save_csv(dataset_stats_df, TABLE_DATASET_DIR / "C3_dataset_level_audio_feature_statistics.csv")

class_distribution = frequency_table(features_df, "class")
save_csv(class_distribution, TABLE_DATASET_DIR / "C3_class_distribution_from_features.csv")

fold_distribution = frequency_table(features_df, "fold")
save_csv(fold_distribution, TABLE_DATASET_DIR / "C3_fold_distribution_from_features.csv")

class_level_rows = []

for class_name, group in features_df.groupby("class"):
    for col in available_feature_cols:
        series = group[col]
        q1, q3, iqr, lower, upper = iqr_bounds(series)
        class_level_rows.append({
            "Class": class_name,
            "Variable": col,
            "Count": series.count(),
            "Mean": series.mean(),
            "Median": series.median(),
            "Std": series.std(),
            "Min": series.min(),
            "Max": series.max(),
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Outlier_Count_IQR": count_outliers(series)
        })

class_level_stats_df = pd.DataFrame(class_level_rows)
save_csv(class_level_stats_df, TABLE_CLASS_DIR / "C3_class_level_audio_feature_statistics.csv")

class_feature_mean = features_df.groupby("class")[available_feature_cols].mean().reset_index()
save_csv(class_feature_mean, TABLE_CLASS_DIR / "C3_class_feature_mean_table.csv")


# =========================================================
# 9. VẼ BIỂU ĐỒ DATASET-LEVEL VÀ CLASS-LEVEL
# =========================================================

def plot_histogram(df, column, output_path, title):
    """Vẽ histogram cho một biến định lượng."""
    plt.figure(figsize=(8, 5))
    plt.hist(df[column].dropna(), bins=30)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Tần suất")
    save_current_figure(output_path)


def plot_boxplot(df, column, output_path, title):
    """Vẽ boxplot cho một biến định lượng."""
    plt.figure(figsize=(7, 5))
    plt.boxplot(df[column].dropna(), vert=True)
    plt.title(title)
    plt.ylabel(column)
    save_current_figure(output_path)


def plot_bar_count(df, column, output_path, title):
    """Vẽ bar chart tần suất cho biến định tính."""
    counts = df[column].value_counts()
    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Số lượng")
    plt.xticks(rotation=45, ha="right")
    save_current_figure(output_path)


def plot_boxplot_by_class(df, value_col, output_path, title):
    """Vẽ boxplot của một feature theo từng lớp."""
    plot_df = df[["class", value_col]].dropna()
    classes = sorted(plot_df["class"].unique())
    data = [plot_df.loc[plot_df["class"] == c, value_col] for c in classes]
    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=classes)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel(value_col)
    plt.xticks(rotation=45, ha="right")
    save_current_figure(output_path)


def plot_scatter_by_class(df, x_col, y_col, output_path, title):
    """Vẽ scatter plot theo lớp."""
    plt.figure(figsize=(10, 6))
    for class_name, group in df.groupby("class"):
        plt.scatter(group[x_col], group[y_col], alpha=0.5, label=class_name, s=18)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(fontsize=8, ncol=2)
    save_current_figure(output_path)


def plot_correlation_heatmap(df, columns, output_path, title):
    """Vẽ heatmap tương quan feature audio."""
    corr = df[columns].corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Hệ số tương quan")
    plt.title(title)
    plt.xticks(ticks=np.arange(len(corr.columns)), labels=corr.columns, rotation=90)
    plt.yticks(ticks=np.arange(len(corr.index)), labels=corr.index)
    save_current_figure(output_path)


# Dataset-level
plot_bar_count(features_df, "class", FIG_DATASET_DIR / "C3_bar_class_distribution.png", "Phân bố số lượng audio theo lớp")
plot_bar_count(features_df, "fold", FIG_DATASET_DIR / "C3_bar_fold_distribution.png", "Phân bố số lượng audio theo fold")

for col in ["duration_audio", "rms_mean", "zcr_mean", "spectral_centroid_mean", "spectral_bandwidth_mean", "spectral_rolloff_mean"]:
    if col in features_df.columns:
        plot_histogram(features_df, col, FIG_DATASET_DIR / f"C3_hist_{col}.png", f"Histogram feature {col}")
        plot_boxplot(features_df, col, FIG_DATASET_DIR / f"C3_boxplot_{col}.png", f"Boxplot feature {col}")

main_heatmap_cols = [
    col for col in [
        "duration_audio",
        "rms_mean",
        "zcr_mean",
        "spectral_centroid_mean",
        "spectral_bandwidth_mean",
        "spectral_rolloff_mean",
        "mfcc_1_mean",
        "mfcc_2_mean",
        "mfcc_3_mean",
        "mfcc_4_mean",
        "mfcc_5_mean"
    ]
    if col in features_df.columns
]

plot_correlation_heatmap(
    features_df,
    main_heatmap_cols,
    FIG_DATASET_DIR / "C3_heatmap_main_audio_features.png",
    "Ma trận tương quan các đặc trưng âm thanh chính"
)

# Class-level
for col in ["duration_audio", "rms_mean", "zcr_mean", "spectral_centroid_mean", "spectral_bandwidth_mean", "spectral_rolloff_mean"]:
    if col in features_df.columns:
        plot_boxplot_by_class(
            features_df,
            col,
            FIG_CLASS_DIR / f"C3_boxplot_{col}_by_class.png",
            f"Boxplot {col} theo từng lớp âm thanh"
        )

plot_scatter_by_class(
    features_df,
    "rms_mean",
    "spectral_centroid_mean",
    FIG_CLASS_DIR / "C3_scatter_rms_mean_vs_spectral_centroid_by_class.png",
    "Scatter RMS mean và Spectral Centroid theo lớp"
)

plot_scatter_by_class(
    features_df,
    "zcr_mean",
    "spectral_centroid_mean",
    FIG_CLASS_DIR / "C3_scatter_zcr_mean_vs_spectral_centroid_by_class.png",
    "Scatter ZCR mean và Spectral Centroid theo lớp"
)

# Heatmap trung bình MFCC theo class
mfcc_mean_cols = [f"mfcc_{i}_mean" for i in range(1, N_MFCC + 1) if f"mfcc_{i}_mean" in features_df.columns]
mfcc_class_mean = features_df.groupby("class")[mfcc_mean_cols].mean().sort_index()

plt.figure(figsize=(12, 7))
plt.imshow(mfcc_class_mean.values, aspect="auto")
plt.colorbar(label="MFCC mean")
plt.title("Heatmap trung bình MFCC theo lớp âm thanh")
plt.xticks(ticks=np.arange(len(mfcc_class_mean.columns)), labels=mfcc_class_mean.columns, rotation=90)
plt.yticks(ticks=np.arange(len(mfcc_class_mean.index)), labels=mfcc_class_mean.index)
save_current_figure(FIG_CLASS_DIR / "C3_heatmap_mfcc_mean_by_class.png")


# =========================================================
# 10. VẼ WAVEFORM VÀ MEL-SPECTROGRAM MẪU CHO MỖI LỚP
# =========================================================

def plot_waveform_and_spectrogram(audio_path, class_name, output_prefix):
    """Vẽ waveform và mel-spectrogram cho một file audio mẫu."""
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveform mẫu - {class_name}")
    plt.xlabel("Thời gian (giây)")
    plt.ylabel("Biên độ")
    save_current_figure(FIG_WAVEFORM_DIR / f"{output_prefix}_waveform.png")

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel-spectrogram mẫu - {class_name}")
    plt.xlabel("Thời gian")
    plt.ylabel("Mel frequency")
    save_current_figure(FIG_WAVEFORM_DIR / f"{output_prefix}_mel_spectrogram.png")


sample_rows = (
    features_df[features_df["feature_status"] == "ok"]
    .sort_values(["class", "fold", "slice_file_name"])
    .groupby("class")
    .head(1)
)

sample_audio_rows = []

for _, row in sample_rows.iterrows():
    class_name = row["class"]
    safe_class_name = str(class_name).replace(" ", "_").replace("/", "_")
    audio_path = row["audio_path"]

    try:
        plot_waveform_and_spectrogram(audio_path, class_name, f"C3_sample_{safe_class_name}")
        sample_audio_rows.append({
            "Class": class_name,
            "Sample_File": row["slice_file_name"],
            "Fold": row["fold"],
            "Audio_Path": audio_path,
            "Waveform_Figure": f"C3_sample_{safe_class_name}_waveform.png",
            "Mel_Spectrogram_Figure": f"C3_sample_{safe_class_name}_mel_spectrogram.png",
            "Error": ""
        })
    except Exception as exc:
        sample_audio_rows.append({
            "Class": class_name,
            "Sample_File": row["slice_file_name"],
            "Fold": row["fold"],
            "Audio_Path": audio_path,
            "Waveform_Figure": "",
            "Mel_Spectrogram_Figure": "",
            "Error": str(exc)
        })

sample_audio_df = pd.DataFrame(sample_audio_rows)
save_csv(sample_audio_df, TABLE_CLASS_DIR / "C3_sample_audio_figures_by_class.csv")


# =========================================================
# 11. GHI LOG
# =========================================================

with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("C3 UrbanSound8K - Round 4 Audio Feature Extraction and EDA\n")
    f.write("=" * 90 + "\n")
    f.write(f"Run time: {datetime.now()}\n")
    f.write(f"Raw dir: {RAW_DIR}\n")
    f.write(f"Metadata path: {metadata_path}\n")
    f.write(f"Audio base dir: {audio_base_dir}\n")
    f.write(f"Round output directory: {ROUND_DIR}\n\n")

    f.write("METADATA SUMMARY\n")
    f.write("-" * 90 + "\n")
    f.write(f"Metadata rows: {len(metadata_df)}\n")
    f.write(f"Original metadata columns: slice_file_name, fsID, start, end, salience, fold, classID, class\n")
    f.write(f"Existing audio files by metadata: {int(metadata_df['file_exists'].sum())}/{len(metadata_df)}\n\n")

    f.write("FEATURE EXTRACTION SUMMARY\n")
    f.write("-" * 90 + "\n")
    f.write(quality_summary.to_string(index=False))
    f.write("\n\n")

    f.write("CLASS DISTRIBUTION FROM FEATURES\n")
    f.write("-" * 90 + "\n")
    f.write(class_distribution.to_string(index=False))
    f.write("\n\n")

    f.write("DATASET-LEVEL AUDIO FEATURE STATISTICS\n")
    f.write("-" * 90 + "\n")
    selected_stats = dataset_stats_df[
        dataset_stats_df["Variable"].isin([
            "duration_audio",
            "rms_mean",
            "zcr_mean",
            "spectral_centroid_mean",
            "spectral_bandwidth_mean",
            "spectral_rolloff_mean"
        ])
    ]
    f.write(selected_stats.to_string(index=False))
    f.write("\n\n")

    f.write("OUTPUT FILES\n")
    f.write("-" * 90 + "\n")
    f.write(f"Audio features: {FEATURE_PATH}\n")
    f.write(f"Dataset-level tables: {TABLE_DATASET_DIR}\n")
    f.write(f"Class-level tables: {TABLE_CLASS_DIR}\n")
    f.write(f"Quality-check tables: {TABLE_QUALITY_DIR}\n")
    f.write(f"Dataset-level figures: {FIG_DATASET_DIR}\n")
    f.write(f"Class-level figures: {FIG_CLASS_DIR}\n")
    f.write(f"Waveform/spectrogram figures: {FIG_WAVEFORM_DIR}\n")

print("Hoàn thành Vòng 4 - Audio feature extraction và EDA cho C3.")
print(f"Output directory: {ROUND_DIR}")
print(f"Log saved to: {LOG_PATH}")
print("Bạn hãy gửi lại file C3_round4_audio_eda_log.txt sau khi chạy xong.")
