# =========================================================
# VÒNG 3 - EDA CHO DATASET B3: BIKE SHARING DATASET
# Mục tiêu:
# 1. Phân tích dữ liệu chuỗi thời gian Bike Sharing
# 2. Phân loại biến dữ liệu
# 3. Tính thống kê mô tả cho biến định lượng
# 4. Lập bảng tần suất cho biến định tính/ordinal
# 5. Vẽ Line Chart, Histogram, Boxplot, Scatter Plot, Bar Chart, Heatmap
# 6. Kiểm tra thiếu mốc thời gian và xử lý bằng nội suy/forward fill
# 7. Tạo biến biến đổi chuỗi: log-transform, differencing, lag, rolling
# 8. So sánh trước và sau tiền xử lý/biến đổi chuỗi
# =========================================================

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# =========================================================

RAW_DIR = Path("data/raw/B3_bike_sharing")
DAY_PATH = RAW_DIR / "day.csv"
HOUR_PATH = RAW_DIR / "hour.csv"

ROUND_DIR = Path("outputs/round_03_B3_eda")
DATA_DIR = ROUND_DIR / "data"
LOG_DIR = ROUND_DIR / "logs"

TABLE_RAW_DIR = ROUND_DIR / "tables" / "raw"
TABLE_TRANSFORMED_DIR = ROUND_DIR / "tables" / "transformed"
TABLE_COMPARE_DIR = ROUND_DIR / "tables" / "comparison"

FIG_RAW_DIR = ROUND_DIR / "figures" / "raw"
FIG_TRANSFORMED_DIR = ROUND_DIR / "figures" / "transformed"

for folder in [
    DATA_DIR,
    LOG_DIR,
    TABLE_RAW_DIR,
    TABLE_TRANSFORMED_DIR,
    TABLE_COMPARE_DIR,
    FIG_RAW_DIR,
    FIG_TRANSFORMED_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / "B3_round3_eda_log.txt"


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


def save_current_figure(path):
    """Lưu biểu đồ hiện tại."""
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def descriptive_statistics(df, columns):
    """Tính đầy đủ các chỉ số thống kê theo yêu cầu."""
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
    """Lập bảng tần suất tuyệt đối, tương đối và mode cho biến định tính/ordinal."""
    freq = df[column].value_counts(dropna=False).reset_index()
    freq.columns = [column, "Frequency"]
    freq["Relative_Frequency"] = freq["Frequency"] / len(df)
    freq["Percent"] = freq["Relative_Frequency"] * 100
    freq["Mode_Of_Variable"] = safe_mode(df[column])
    return freq


# =========================================================
# 3. ĐỌC DỮ LIỆU
# =========================================================

day_raw = pd.read_csv(DAY_PATH)
hour_raw = pd.read_csv(HOUR_PATH)

day_raw["dteday"] = pd.to_datetime(day_raw["dteday"], errors="coerce")
hour_raw["dteday"] = pd.to_datetime(hour_raw["dteday"], errors="coerce")

# Tạo datetime theo giờ cho hour.csv
hour_raw["datetime"] = hour_raw["dteday"] + pd.to_timedelta(hour_raw["hr"], unit="h")

# Lưu bản copy dữ liệu gốc có datetime để truy vết
save_csv(day_raw, DATA_DIR / "B3_day_raw_with_datetime.csv")
save_csv(hour_raw, DATA_DIR / "B3_hour_raw_with_datetime.csv")


# =========================================================
# 4. PHÂN LOẠI BIẾN
# =========================================================

variable_rows = [
    {
        "Variable": "instant",
        "Meaning": "Mã thứ tự bản ghi",
        "Python_Data_Type": str(hour_raw["instant"].dtype),
        "Variable_Nature": "Qualitative",
        "Subtype": "Identifier",
        "Role": "Không dùng như biến phân tích chính"
    },
    {
        "Variable": "dteday",
        "Meaning": "Ngày quan sát",
        "Python_Data_Type": str(hour_raw["dteday"].dtype),
        "Variable_Nature": "Time",
        "Subtype": "Date",
        "Role": "Trục thời gian"
    },
    {
        "Variable": "datetime",
        "Meaning": "Mốc thời gian theo giờ, tạo từ dteday và hr",
        "Python_Data_Type": str(hour_raw["datetime"].dtype),
        "Variable_Nature": "Time",
        "Subtype": "Datetime",
        "Role": "Trục thời gian cho line chart và time-series split"
    },
    {
        "Variable": "season",
        "Meaning": "Mùa trong năm",
        "Python_Data_Type": str(hour_raw["season"].dtype),
        "Variable_Nature": "Qualitative",
        "Subtype": "Ordinal encoded as integer",
        "Role": "Biến giải thích mùa vụ"
    },
    {
        "Variable": "yr",
        "Meaning": "Năm quan sát, 0 là 2011 và 1 là 2012",
        "Python_Data_Type": str(hour_raw["yr"].dtype),
        "Variable_Nature": "Qualitative",
        "Subtype": "Binary",
        "Role": "Biến thời gian"
    },
    {
        "Variable": "mnth",
        "Meaning": "Tháng trong năm",
        "Python_Data_Type": str(hour_raw["mnth"].dtype),
        "Variable_Nature": "Qualitative",
        "Subtype": "Ordinal",
        "Role": "Biến mùa vụ"
    },
    {
        "Variable": "hr",
        "Meaning": "Giờ trong ngày",
        "Python_Data_Type": str(hour_raw["hr"].dtype),
        "Variable_Nature": "Qualitative",
        "Subtype": "Ordinal",
        "Role": "Biến chu kỳ ngày"
    },
    {
        "Variable": "holiday",
        "Meaning": "Có phải ngày lễ hay không",
        "Python_Data_Type": str(hour_raw["holiday"].dtype),
        "Variable_Nature": "Qualitative",
        "Subtype": "Binary",
        "Role": "Biến lịch"
    },
    {
        "Variable": "weekday",
        "Meaning": "Thứ trong tuần",
        "Python_Data_Type": str(hour_raw["weekday"].dtype),
        "Variable_Nature": "Qualitative",
        "Subtype": "Ordinal/Nominal",
        "Role": "Biến chu kỳ tuần"
    },
    {
        "Variable": "workingday",
        "Meaning": "Có phải ngày làm việc hay không",
        "Python_Data_Type": str(hour_raw["workingday"].dtype),
        "Variable_Nature": "Qualitative",
        "Subtype": "Binary",
        "Role": "Biến lịch"
    },
    {
        "Variable": "weathersit",
        "Meaning": "Tình trạng thời tiết",
        "Python_Data_Type": str(hour_raw["weathersit"].dtype),
        "Variable_Nature": "Qualitative",
        "Subtype": "Ordinal",
        "Role": "Biến thời tiết"
    },
    {
        "Variable": "temp",
        "Meaning": "Nhiệt độ chuẩn hóa",
        "Python_Data_Type": str(hour_raw["temp"].dtype),
        "Variable_Nature": "Quantitative",
        "Subtype": "Continuous",
        "Role": "Biến định lượng phân tích chính"
    },
    {
        "Variable": "atemp",
        "Meaning": "Nhiệt độ cảm nhận chuẩn hóa",
        "Python_Data_Type": str(hour_raw["atemp"].dtype),
        "Variable_Nature": "Quantitative",
        "Subtype": "Continuous",
        "Role": "Biến định lượng phân tích chính"
    },
    {
        "Variable": "hum",
        "Meaning": "Độ ẩm chuẩn hóa",
        "Python_Data_Type": str(hour_raw["hum"].dtype),
        "Variable_Nature": "Quantitative",
        "Subtype": "Continuous",
        "Role": "Biến thời tiết"
    },
    {
        "Variable": "windspeed",
        "Meaning": "Tốc độ gió chuẩn hóa",
        "Python_Data_Type": str(hour_raw["windspeed"].dtype),
        "Variable_Nature": "Quantitative",
        "Subtype": "Continuous",
        "Role": "Biến thời tiết"
    },
    {
        "Variable": "casual",
        "Meaning": "Số lượt thuê xe của người dùng không đăng ký",
        "Python_Data_Type": str(hour_raw["casual"].dtype),
        "Variable_Nature": "Quantitative",
        "Subtype": "Discrete",
        "Role": "Biến định lượng"
    },
    {
        "Variable": "registered",
        "Meaning": "Số lượt thuê xe của người dùng đã đăng ký",
        "Python_Data_Type": str(hour_raw["registered"].dtype),
        "Variable_Nature": "Quantitative",
        "Subtype": "Discrete",
        "Role": "Biến định lượng"
    },
    {
        "Variable": "cnt",
        "Meaning": "Tổng số lượt thuê xe",
        "Python_Data_Type": str(hour_raw["cnt"].dtype),
        "Variable_Nature": "Quantitative",
        "Subtype": "Discrete",
        "Role": "Target chính cho hồi quy/forecasting"
    }
]

variable_classification_df = pd.DataFrame(variable_rows)
save_csv(variable_classification_df, TABLE_RAW_DIR / "B3_variable_classification.csv")


# =========================================================
# 5. THỐNG KÊ MÔ TẢ VÀ BẢNG TẦN SUẤT DỮ LIỆU THÔ
# =========================================================

numeric_cols = [
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "casual",
    "registered",
    "cnt"
]

categorical_cols = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit"
]

raw_desc_df = descriptive_statistics(hour_raw, numeric_cols)
save_csv(raw_desc_df, TABLE_RAW_DIR / "B3_descriptive_statistics_hour_raw.csv")

for col in categorical_cols:
    freq_df = frequency_table(hour_raw, col)
    save_csv(freq_df, TABLE_RAW_DIR / f"B3_frequency_{col}_raw.csv")


# =========================================================
# 6. KIỂM TRA THIẾU MỐC THỜI GIAN THEO GIỜ
# =========================================================

hour_sorted = hour_raw.sort_values("datetime").copy()

full_hour_index = pd.date_range(
    start=hour_sorted["datetime"].min(),
    end=hour_sorted["datetime"].max(),
    freq="h"
)

observed_hour_index = pd.DatetimeIndex(hour_sorted["datetime"])
missing_hours = full_hour_index.difference(observed_hour_index)

missing_time_summary = pd.DataFrame({
    "Item": [
        "Start datetime",
        "End datetime",
        "Expected hourly records",
        "Observed hourly records",
        "Missing hourly records"
    ],
    "Value": [
        str(full_hour_index.min()),
        str(full_hour_index.max()),
        len(full_hour_index),
        len(observed_hour_index),
        len(missing_hours)
    ]
})

save_csv(missing_time_summary, TABLE_RAW_DIR / "B3_missing_time_summary_raw.csv")

missing_hours_df = pd.DataFrame({
    "Missing_Datetime": missing_hours
})

save_csv(missing_hours_df, TABLE_RAW_DIR / "B3_missing_hourly_timestamps_raw.csv")


# =========================================================
# 7. TẠO DỮ LIỆU LIÊN TỤC THEO GIỜ VÀ BIẾN ĐỔI CHUỖI
# =========================================================

# Đưa datetime thành index, reindex để có đủ mốc giờ liên tục
hour_ts = hour_sorted.set_index("datetime").sort_index()
hour_complete = hour_ts.reindex(full_hour_index)
hour_complete.index.name = "datetime"

# Giữ lại datetime thành cột
hour_complete = hour_complete.reset_index()

# Các biến gốc dteday, hr có thể tạo lại từ datetime
hour_complete["dteday"] = hour_complete["datetime"].dt.date
hour_complete["dteday"] = pd.to_datetime(hour_complete["dteday"])
hour_complete["hr"] = hour_complete["datetime"].dt.hour
hour_complete["mnth_from_datetime"] = hour_complete["datetime"].dt.month
hour_complete["weekday_from_datetime"] = hour_complete["datetime"].dt.weekday

# Đánh dấu dòng nào là quan sát gốc và dòng nào là mốc giờ được bổ sung
hour_complete["is_observed_original"] = hour_complete["cnt"].notna().astype(int)

# Nội suy các biến numeric theo thời gian
numeric_to_interpolate = [
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "casual",
    "registered",
    "cnt"
]

for col in numeric_to_interpolate:
    if col in hour_complete.columns:
        hour_complete[f"{col}_interpolated"] = (
            hour_complete[col]
            .interpolate(method="linear")
            .ffill()
            .bfill()
        )

# Với biến categorical/ordinal, dùng forward fill và backward fill
categorical_to_fill = [
    "season",
    "yr",
    "mnth",
    "holiday",
    "weekday",
    "workingday",
    "weathersit"
]

for col in categorical_to_fill:
    if col in hour_complete.columns:
        hour_complete[f"{col}_filled"] = (
            hour_complete[col]
            .ffill()
            .bfill()
        )

# Biến đổi chuỗi cho target cnt
hour_complete["cnt_log1p"] = np.log1p(hour_complete["cnt_interpolated"])
hour_complete["cnt_diff_1"] = hour_complete["cnt_interpolated"].diff(1)

# Tạo lag features
hour_complete["lag_1"] = hour_complete["cnt_interpolated"].shift(1)
hour_complete["lag_24"] = hour_complete["cnt_interpolated"].shift(24)
hour_complete["lag_168"] = hour_complete["cnt_interpolated"].shift(168)

# Tạo rolling features
hour_complete["rolling_mean_24"] = (
    hour_complete["cnt_interpolated"]
    .rolling(window=24, min_periods=1)
    .mean()
)

hour_complete["rolling_std_24"] = (
    hour_complete["cnt_interpolated"]
    .rolling(window=24, min_periods=2)
    .std()
)

hour_complete["rolling_mean_168"] = (
    hour_complete["cnt_interpolated"]
    .rolling(window=168, min_periods=1)
    .mean()
)

# Lưu dữ liệu đã biến đổi
save_csv(hour_complete, DATA_DIR / "B3_hour_transformed_time_features.csv")


# =========================================================
# 8. THỐNG KÊ SAU BIẾN ĐỔI
# =========================================================

transformed_numeric_cols = [
    "temp_interpolated",
    "atemp_interpolated",
    "hum_interpolated",
    "windspeed_interpolated",
    "casual_interpolated",
    "registered_interpolated",
    "cnt_interpolated",
    "cnt_log1p",
    "cnt_diff_1",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
    "rolling_std_24",
    "rolling_mean_168"
]

transformed_desc_df = descriptive_statistics(hour_complete, transformed_numeric_cols)
save_csv(
    transformed_desc_df,
    TABLE_TRANSFORMED_DIR / "B3_descriptive_statistics_transformed.csv"
)

# Bảng so sánh trước/sau cho biến cnt
comparison_rows = []

comparison_pairs = [
    ("cnt", "cnt_interpolated", "Nội suy mốc thời gian thiếu"),
    ("cnt", "cnt_log1p", "Log-transform để giảm lệch phải"),
    ("cnt", "cnt_diff_1", "Differencing bậc 1 để phân tích biến động ngắn hạn")
]

for raw_col, transformed_col, note in comparison_pairs:
    if raw_col not in hour_raw.columns or transformed_col not in hour_complete.columns:
        continue

    raw_series = hour_raw[raw_col]
    transformed_series = hour_complete[transformed_col]

    comparison_rows.append({
        "Raw_Variable": raw_col,
        "Transformed_Variable": transformed_col,
        "Transformation": note,
        "Missing_Before": int(raw_series.isna().sum()),
        "Missing_After": int(transformed_series.isna().sum()),
        "Mean_Before": raw_series.mean(),
        "Mean_After": transformed_series.mean(),
        "Median_Before": raw_series.median(),
        "Median_After": transformed_series.median(),
        "Std_Before": raw_series.std(),
        "Std_After": transformed_series.std(),
        "Outliers_Before": count_outliers(raw_series),
        "Outliers_After": count_outliers(transformed_series)
    })

comparison_df = pd.DataFrame(comparison_rows)
save_csv(
    comparison_df,
    TABLE_COMPARE_DIR / "B3_before_after_time_transformation_summary.csv"
)

# Bảng mô tả các bước biến đổi
transformation_steps_df = pd.DataFrame({
    "Step": [
        "Create datetime",
        "Check missing hourly timestamps",
        "Reindex to full hourly timeline",
        "Linear interpolation for numeric variables",
        "Forward/backward fill for categorical variables",
        "Create log transform",
        "Create differencing",
        "Create lag features",
        "Create rolling features"
    ],
    "Description": [
        "Tạo datetime từ dteday và hr để dùng làm trục thời gian.",
        "So sánh số mốc giờ kỳ vọng và số mốc giờ quan sát được.",
        "Bổ sung các mốc giờ còn thiếu trong toàn bộ khoảng thời gian.",
        "Nội suy tuyến tính cho temp, atemp, hum, windspeed, casual, registered, cnt.",
        "Dùng ffill/bfill cho season, yr, mnth, holiday, weekday, workingday, weathersit.",
        "Tạo cnt_log1p = log(1 + cnt_interpolated) để giảm độ lệch phân phối.",
        "Tạo cnt_diff_1 để xem biến động giữa hai mốc giờ liên tiếp.",
        "Tạo lag_1, lag_24, lag_168 để phục vụ mô hình dự báo.",
        "Tạo rolling_mean_24, rolling_std_24, rolling_mean_168 để mô tả xu hướng ngắn hạn và tuần."
    ]
})

save_csv(
    transformation_steps_df,
    TABLE_COMPARE_DIR / "B3_time_transformation_steps.csv"
)


# =========================================================
# 9. TỔNG HỢP THEO NGÀY ĐỂ PHÂN TÍCH XU HƯỚNG/MÙA VỤ
# =========================================================

daily_from_hour = (
    hour_raw
    .groupby("dteday")
    .agg(
        cnt_sum=("cnt", "sum"),
        cnt_mean=("cnt", "mean"),
        casual_sum=("casual", "sum"),
        registered_sum=("registered", "sum"),
        temp_mean=("temp", "mean"),
        hum_mean=("hum", "mean"),
        windspeed_mean=("windspeed", "mean")
    )
    .reset_index()
)

daily_from_hour["rolling_mean_7"] = (
    daily_from_hour["cnt_sum"]
    .rolling(window=7, min_periods=1)
    .mean()
)

daily_from_hour["rolling_mean_30"] = (
    daily_from_hour["cnt_sum"]
    .rolling(window=30, min_periods=1)
    .mean()
)

save_csv(daily_from_hour, DATA_DIR / "B3_daily_aggregated_from_hour.csv")


# =========================================================
# 10. VẼ BIỂU ĐỒ DỮ LIỆU THÔ
# =========================================================

def plot_line(df, x_col, y_col, output_path, title, xlabel, ylabel):
    """Vẽ line chart."""
    plt.figure(figsize=(12, 5))
    plt.plot(df[x_col], df[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_current_figure(output_path)


def plot_histogram(df, column, output_path, title):
    """Vẽ histogram."""
    plt.figure(figsize=(8, 5))
    plt.hist(df[column].dropna(), bins=30)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Tần suất")
    save_current_figure(output_path)


def plot_boxplot(df, column, output_path, title):
    """Vẽ boxplot cho một biến."""
    plt.figure(figsize=(7, 5))
    plt.boxplot(df[column].dropna(), vert=True)
    plt.title(title)
    plt.ylabel(column)
    save_current_figure(output_path)


def plot_boxplot_by_group(df, group_col, value_col, output_path, title):
    """Vẽ boxplot của biến value_col theo nhóm group_col."""
    plot_df = df[[group_col, value_col]].dropna()
    groups = sorted(plot_df[group_col].unique())
    data = [plot_df.loc[plot_df[group_col] == g, value_col] for g in groups]

    plt.figure(figsize=(9, 5))
    plt.boxplot(data, labels=[str(g) for g in groups])
    plt.title(title)
    plt.xlabel(group_col)
    plt.ylabel(value_col)
    save_current_figure(output_path)


def plot_bar_mean(df, group_col, value_col, output_path, title):
    """Vẽ bar chart giá trị trung bình theo nhóm."""
    grouped = df.groupby(group_col)[value_col].mean().reset_index()

    plt.figure(figsize=(9, 5))
    plt.bar(grouped[group_col].astype(str), grouped[value_col])
    plt.title(title)
    plt.xlabel(group_col)
    plt.ylabel(f"Mean {value_col}")
    plt.xticks(rotation=45, ha="right")
    save_current_figure(output_path)


def plot_scatter(df, x_col, y_col, output_path, title):
    """Vẽ scatter plot giữa hai biến định lượng."""
    plot_df = df[[x_col, y_col]].dropna()

    plt.figure(figsize=(8, 5))
    plt.scatter(plot_df[x_col], plot_df[y_col], alpha=0.5)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    save_current_figure(output_path)


def plot_correlation_heatmap(df, columns, output_path, title):
    """Vẽ heatmap ma trận tương quan bằng matplotlib."""
    corr = df[columns].corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Hệ số tương quan")
    plt.title(title)

    plt.xticks(
        ticks=np.arange(len(corr.columns)),
        labels=corr.columns,
        rotation=90
    )
    plt.yticks(
        ticks=np.arange(len(corr.index)),
        labels=corr.index
    )

    save_current_figure(output_path)


# Line chart bắt buộc cho nhóm B
plot_line(
    hour_raw,
    "datetime",
    "cnt",
    FIG_RAW_DIR / "B3_line_cnt_hourly_raw.png",
    "Line chart số lượt thuê xe theo giờ - dữ liệu thô",
    "Thời gian",
    "cnt"
)

plot_line(
    daily_from_hour,
    "dteday",
    "cnt_sum",
    FIG_RAW_DIR / "B3_line_daily_cnt_sum_raw.png",
    "Line chart tổng lượt thuê xe theo ngày - dữ liệu thô",
    "Ngày",
    "Tổng cnt theo ngày"
)

# Rolling mean để thấy xu hướng
plt.figure(figsize=(12, 5))
plt.plot(daily_from_hour["dteday"], daily_from_hour["cnt_sum"], label="Daily cnt")
plt.plot(daily_from_hour["dteday"], daily_from_hour["rolling_mean_7"], label="Rolling mean 7 ngày")
plt.plot(daily_from_hour["dteday"], daily_from_hour["rolling_mean_30"], label="Rolling mean 30 ngày")
plt.title("Xu hướng tổng lượt thuê xe theo ngày với rolling mean")
plt.xlabel("Ngày")
plt.ylabel("Tổng cnt theo ngày")
plt.legend()
save_current_figure(FIG_RAW_DIR / "B3_daily_cnt_rolling_mean_raw.png")

# Histogram
for col in ["cnt", "casual", "registered", "temp", "hum", "windspeed"]:
    plot_histogram(
        hour_raw,
        col,
        FIG_RAW_DIR / f"B3_hist_{col}_raw.png",
        f"Histogram biến {col} - dữ liệu thô"
    )

# Boxplot
for col in ["cnt", "casual", "registered", "temp", "hum", "windspeed"]:
    plot_boxplot(
        hour_raw,
        col,
        FIG_RAW_DIR / f"B3_boxplot_{col}_raw.png",
        f"Boxplot biến {col} - dữ liệu thô"
    )

# Boxplot theo nhóm
plot_boxplot_by_group(
    hour_raw,
    "season",
    "cnt",
    FIG_RAW_DIR / "B3_boxplot_cnt_by_season_raw.png",
    "Boxplot cnt theo season - dữ liệu thô"
)

plot_boxplot_by_group(
    hour_raw,
    "hr",
    "cnt",
    FIG_RAW_DIR / "B3_boxplot_cnt_by_hour_raw.png",
    "Boxplot cnt theo giờ trong ngày - dữ liệu thô"
)

plot_boxplot_by_group(
    hour_raw,
    "weathersit",
    "cnt",
    FIG_RAW_DIR / "B3_boxplot_cnt_by_weathersit_raw.png",
    "Boxplot cnt theo tình trạng thời tiết - dữ liệu thô"
)

# Bar chart trung bình theo nhóm
plot_bar_mean(
    hour_raw,
    "hr",
    "cnt",
    FIG_RAW_DIR / "B3_bar_mean_cnt_by_hour_raw.png",
    "Trung bình cnt theo giờ"
)

plot_bar_mean(
    hour_raw,
    "weekday",
    "cnt",
    FIG_RAW_DIR / "B3_bar_mean_cnt_by_weekday_raw.png",
    "Trung bình cnt theo thứ trong tuần"
)

plot_bar_mean(
    hour_raw,
    "season",
    "cnt",
    FIG_RAW_DIR / "B3_bar_mean_cnt_by_season_raw.png",
    "Trung bình cnt theo mùa"
)

# Scatter plot
plot_scatter(
    hour_raw,
    "temp",
    "cnt",
    FIG_RAW_DIR / "B3_scatter_temp_cnt_raw.png",
    "Quan hệ giữa temp và cnt - dữ liệu thô"
)

plot_scatter(
    hour_raw,
    "hum",
    "cnt",
    FIG_RAW_DIR / "B3_scatter_hum_cnt_raw.png",
    "Quan hệ giữa hum và cnt - dữ liệu thô"
)

plot_scatter(
    hour_raw,
    "windspeed",
    "cnt",
    FIG_RAW_DIR / "B3_scatter_windspeed_cnt_raw.png",
    "Quan hệ giữa windspeed và cnt - dữ liệu thô"
)

# Heatmap tương quan
heatmap_cols = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "casual",
    "registered",
    "cnt"
]

plot_correlation_heatmap(
    hour_raw,
    heatmap_cols,
    FIG_RAW_DIR / "B3_heatmap_correlation_raw.png",
    "Ma trận tương quan các biến Bike Sharing - dữ liệu thô"
)


# =========================================================
# 11. VẼ BIỂU ĐỒ SAU BIẾN ĐỔI CHUỖI
# =========================================================

plot_line(
    hour_complete,
    "datetime",
    "cnt_interpolated",
    FIG_TRANSFORMED_DIR / "B3_line_cnt_interpolated.png",
    "Line chart cnt sau bổ sung mốc giờ và nội suy",
    "Thời gian",
    "cnt_interpolated"
)

plot_line(
    hour_complete,
    "datetime",
    "cnt_log1p",
    FIG_TRANSFORMED_DIR / "B3_line_cnt_log1p.png",
    "Line chart log1p(cnt) sau biến đổi",
    "Thời gian",
    "cnt_log1p"
)

plot_line(
    hour_complete.dropna(subset=["cnt_diff_1"]),
    "datetime",
    "cnt_diff_1",
    FIG_TRANSFORMED_DIR / "B3_line_cnt_diff_1.png",
    "Line chart sai phân bậc 1 của cnt",
    "Thời gian",
    "cnt_diff_1"
)

plt.figure(figsize=(12, 5))
plt.plot(hour_complete["datetime"], hour_complete["cnt_interpolated"], label="cnt_interpolated", alpha=0.5)
plt.plot(hour_complete["datetime"], hour_complete["rolling_mean_24"], label="rolling_mean_24")
plt.plot(hour_complete["datetime"], hour_complete["rolling_mean_168"], label="rolling_mean_168")
plt.title("cnt sau nội suy và rolling mean 24h/168h")
plt.xlabel("Thời gian")
plt.ylabel("cnt")
plt.legend()
save_current_figure(FIG_TRANSFORMED_DIR / "B3_cnt_interpolated_rolling_features.png")

for col in ["cnt_interpolated", "cnt_log1p", "cnt_diff_1", "rolling_mean_24", "rolling_mean_168"]:
    plot_histogram(
        hour_complete,
        col,
        FIG_TRANSFORMED_DIR / f"B3_hist_{col}.png",
        f"Histogram biến {col} - sau biến đổi"
    )

for col in ["cnt_interpolated", "cnt_log1p", "cnt_diff_1", "rolling_mean_24", "rolling_mean_168"]:
    plot_boxplot(
        hour_complete,
        col,
        FIG_TRANSFORMED_DIR / f"B3_boxplot_{col}.png",
        f"Boxplot biến {col} - sau biến đổi"
    )

transformed_heatmap_cols = [
    "temp_interpolated",
    "atemp_interpolated",
    "hum_interpolated",
    "windspeed_interpolated",
    "casual_interpolated",
    "registered_interpolated",
    "cnt_interpolated",
    "cnt_log1p",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
    "rolling_mean_168"
]

plot_correlation_heatmap(
    hour_complete,
    transformed_heatmap_cols,
    FIG_TRANSFORMED_DIR / "B3_heatmap_correlation_transformed.png",
    "Ma trận tương quan sau biến đổi chuỗi thời gian"
)


# =========================================================
# 12. GHI LOG VÒNG 3
# =========================================================

with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("B3 Bike Sharing Dataset - Round 3 EDA and Time-Series Transformation\n")
    f.write("=" * 80 + "\n")
    f.write(f"Run time: {datetime.now()}\n")
    f.write(f"Raw day path: {DAY_PATH}\n")
    f.write(f"Raw hour path: {HOUR_PATH}\n")
    f.write(f"Round output directory: {ROUND_DIR}\n\n")

    f.write("RAW DATA SUMMARY\n")
    f.write("-" * 80 + "\n")
    f.write(f"day.csv shape: {day_raw.shape[0]} rows, {day_raw.shape[1]} columns\n")
    f.write(f"hour.csv shape: {hour_raw.shape[0]} rows, {hour_raw.shape[1]} columns after datetime creation\n")
    f.write(f"hour.csv datetime range: {hour_raw['datetime'].min()} to {hour_raw['datetime'].max()}\n")
    f.write(f"Missing values in hour.csv: {int(hour_raw.isna().sum().sum())}\n")
    f.write(f"Duplicate rows in hour.csv: {int(hour_raw.duplicated().sum())}\n\n")

    f.write("TIME INDEX CHECK\n")
    f.write("-" * 80 + "\n")
    f.write(missing_time_summary.to_string(index=False))
    f.write("\n\n")

    f.write("TARGET CNT RAW SUMMARY\n")
    f.write("-" * 80 + "\n")
    f.write(hour_raw["cnt"].describe().to_string())
    f.write("\n\n")

    f.write("TIME-SERIES TRANSFORMATION STEPS\n")
    f.write("-" * 80 + "\n")
    f.write(transformation_steps_df.to_string(index=False))
    f.write("\n\n")

    f.write("TRANSFORMED DATA SUMMARY\n")
    f.write("-" * 80 + "\n")
    f.write(f"Transformed hourly shape: {hour_complete.shape[0]} rows, {hour_complete.shape[1]} columns\n")
    f.write(f"Number of rows observed in original data: {int(hour_complete['is_observed_original'].sum())}\n")
    f.write(f"Number of rows added by full hourly reindex: {int((hour_complete['is_observed_original'] == 0).sum())}\n")
    f.write(f"Missing cnt_interpolated after preprocessing: {int(hour_complete['cnt_interpolated'].isna().sum())}\n\n")

    f.write("OUTPUT FILES\n")
    f.write("-" * 80 + "\n")
    f.write(f"Raw tables: {TABLE_RAW_DIR}\n")
    f.write(f"Transformed tables: {TABLE_TRANSFORMED_DIR}\n")
    f.write(f"Comparison tables: {TABLE_COMPARE_DIR}\n")
    f.write(f"Raw figures: {FIG_RAW_DIR}\n")
    f.write(f"Transformed figures: {FIG_TRANSFORMED_DIR}\n")
    f.write(f"Transformed data: {DATA_DIR / 'B3_hour_transformed_time_features.csv'}\n")
    f.write(f"Daily aggregated data: {DATA_DIR / 'B3_daily_aggregated_from_hour.csv'}\n")

print("Hoàn thành Vòng 3 - EDA và biến đổi chuỗi thời gian B3.")
print(f"Output directory: {ROUND_DIR}")
print(f"Log saved to: {LOG_PATH}")
print("Bạn hãy gửi lại file B3_round3_eda_log.txt sau khi chạy xong.")
