# =========================================================
# VÒNG 2 - EDA CHO DATASET A3: CUSTOMER PERSONALITY ANALYSIS
# Mục tiêu:
# 1. Phân loại biến dữ liệu
# 2. Tính thống kê mô tả cho biến định lượng
# 3. Lập bảng tần suất cho biến định tính
# 4. Phát hiện outliers theo IQR
# 5. Vẽ biểu đồ Histogram, Boxplot, Scatter, Bar, Pie, Heatmap
# 6. Tiền xử lý dữ liệu và so sánh trước/sau tiền xử lý
# =========================================================

import os
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

RAW_PATH = Path("data/raw/A3_customer_personality/marketing_campaign.csv")

ROUND_DIR = Path("outputs/round_02_A3_eda")
DATA_DIR = ROUND_DIR / "data"
LOG_DIR = ROUND_DIR / "logs"

TABLE_RAW_DIR = ROUND_DIR / "tables" / "raw"
TABLE_CLEAN_DIR = ROUND_DIR / "tables" / "cleaned"
TABLE_COMPARE_DIR = ROUND_DIR / "tables" / "comparison"

FIG_RAW_DIR = ROUND_DIR / "figures" / "raw"
FIG_CLEAN_DIR = ROUND_DIR / "figures" / "cleaned"

for folder in [
    DATA_DIR,
    LOG_DIR,
    TABLE_RAW_DIR,
    TABLE_CLEAN_DIR,
    TABLE_COMPARE_DIR,
    FIG_RAW_DIR,
    FIG_CLEAN_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / "A3_round2_eda_log.txt"


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


def cap_outliers_iqr(series):
    """
    Làm mịn outliers bằng cách cắt giá trị về lower/upper bound theo IQR.
    Cách này không xóa dòng dữ liệu, phù hợp để so sánh trước/sau tiền xử lý.
    """
    q1, q3, iqr, lower, upper = iqr_bounds(series)

    if pd.isna(lower) or pd.isna(upper):
        return series

    return series.clip(lower=lower, upper=upper)


def save_current_figure(path):
    """Lưu biểu đồ hiện tại."""
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


# =========================================================
# 3. ĐỌC DỮ LIỆU THÔ
# =========================================================

# Dataset này là CSV nhưng phân tách bằng tab
df_raw_original = pd.read_csv(RAW_PATH, sep="\t")

# Giữ bản gốc để truy vết
df_raw_original.to_csv(
    DATA_DIR / "A3_raw_original_copy.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 4. TẠO BIẾN DẪN XUẤT CHO PHÂN TÍCH
# =========================================================

def add_derived_features(df):
    """
    Tạo các biến mới phục vụ EDA, học máy và giải thích kinh doanh.
    Các biến này không thay thế biến gốc mà bổ sung thêm thông tin.
    """
    df = df.copy()

    # Chuyển ngày khách hàng tham gia thành datetime
    df["Dt_Customer_Parsed"] = pd.to_datetime(
        df["Dt_Customer"],
        dayfirst=True,
        errors="coerce"
    )

    # Lấy năm tham chiếu là năm lớn nhất trong dữ liệu khách hàng
    # Cách này hợp lý hơn dùng năm hiện tại vì dataset là dữ liệu lịch sử.
    reference_year = int(df["Dt_Customer_Parsed"].dt.year.max())

    df["Age"] = reference_year - df["Year_Birth"]

    # Số ngày khách hàng đã gắn bó tính từ ngày đăng ký đến ngày đăng ký muộn nhất trong dataset
    max_customer_date = df["Dt_Customer_Parsed"].max()
    df["Customer_Tenure_Days"] = (
        max_customer_date - df["Dt_Customer_Parsed"]
    ).dt.days

    # Tổng chi tiêu trên các nhóm sản phẩm
    spending_columns = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds"
    ]

    df["Total_Spending"] = df[spending_columns].sum(axis=1)

    # Tổng số trẻ em/thanh thiếu niên trong nhà
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

    # Tổng số lần mua hàng qua các kênh
    purchase_columns = [
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases"
    ]

    df["Total_Purchases"] = df[purchase_columns].sum(axis=1)

    # Tổng số chiến dịch đã được khách hàng chấp nhận
    campaign_columns = [
        "AcceptedCmp1",
        "AcceptedCmp2",
        "AcceptedCmp3",
        "AcceptedCmp4",
        "AcceptedCmp5",
        "Response"
    ]

    df["Total_Accepted_Campaigns"] = df[campaign_columns].sum(axis=1)

    return df


df_raw = add_derived_features(df_raw_original)


# =========================================================
# 5. XÁC ĐỊNH CÁC BIẾN PHÂN TÍCH CHÍNH
# =========================================================

# Chọn nhiều hơn 5 biến để báo cáo đầy đủ hơn yêu cầu tối thiểu.
numeric_analysis_cols = [
    "Age",
    "Income",
    "Recency",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "Customer_Tenure_Days",
    "Total_Spending",
    "Total_Children",
    "Total_Purchases",
    "Total_Accepted_Campaigns"
]

categorical_analysis_cols = [
    "Education",
    "Marital_Status",
    "Response",
    "Complain",
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5"
]


# =========================================================
# 6. BẢNG PHÂN LOẠI BIẾN
# =========================================================

variable_rows = [
    {
        "Variable": "ID",
        "Meaning": "Mã định danh khách hàng",
        "Python_Data_Type": str(df_raw["ID"].dtype),
        "Variable_Nature": "Qualitative",
        "Subtype": "Identifier",
        "Role": "Không dùng trực tiếp để phân tích thống kê"
    },
    {
        "Variable": "Year_Birth",
        "Meaning": "Năm sinh của khách hàng",
        "Python_Data_Type": str(df_raw["Year_Birth"].dtype),
        "Variable_Nature": "Quantitative",
        "Subtype": "Discrete",
        "Role": "Biến gốc để tạo Age"
    },
    {
        "Variable": "Age",
        "Meaning": "Tuổi khách hàng tại thời điểm tham chiếu của dataset",
        "Python_Data_Type": str(df_raw["Age"].dtype),
        "Variable_Nature": "Quantitative",
        "Subtype": "Discrete",
        "Role": "Biến phân tích chính"
    },
    {
        "Variable": "Education",
        "Meaning": "Trình độ học vấn",
        "Python_Data_Type": str(df_raw["Education"].dtype),
        "Variable_Nature": "Qualitative",
        "Subtype": "Nominal/Ordinal",
        "Role": "Biến định tính, dùng lập bảng tần suất"
    },
    {
        "Variable": "Marital_Status",
        "Meaning": "Tình trạng hôn nhân",
        "Python_Data_Type": str(df_raw["Marital_Status"].dtype),
        "Variable_Nature": "Qualitative",
        "Subtype": "Nominal",
        "Role": "Biến định tính, dùng lập bảng tần suất"
    },
    {
        "Variable": "Income",
        "Meaning": "Thu nhập hằng năm của khách hàng",
        "Python_Data_Type": str(df_raw["Income"].dtype),
        "Variable_Nature": "Quantitative",
        "Subtype": "Continuous",
        "Role": "Biến phân tích chính, có missing cần xử lý"
    },
    {
        "Variable": "Kidhome",
        "Meaning": "Số trẻ nhỏ trong hộ gia đình",
        "Python_Data_Type": str(df_raw["Kidhome"].dtype),
        "Variable_Nature": "Quantitative",
        "Subtype": "Discrete",
        "Role": "Biến mô tả cấu trúc gia đình"
    },
    {
        "Variable": "Teenhome",
        "Meaning": "Số trẻ vị thành niên trong hộ gia đình",
        "Python_Data_Type": str(df_raw["Teenhome"].dtype),
        "Variable_Nature": "Quantitative",
        "Subtype": "Discrete",
        "Role": "Biến mô tả cấu trúc gia đình"
    },
    {
        "Variable": "Recency",
        "Meaning": "Số ngày từ lần mua gần nhất",
        "Python_Data_Type": str(df_raw["Recency"].dtype),
        "Variable_Nature": "Quantitative",
        "Subtype": "Discrete",
        "Role": "Biến hành vi khách hàng"
    },
    {
        "Variable": "Total_Spending",
        "Meaning": "Tổng chi tiêu trên các nhóm sản phẩm",
        "Python_Data_Type": str(df_raw["Total_Spending"].dtype),
        "Variable_Nature": "Quantitative",
        "Subtype": "Continuous/Discrete",
        "Role": "Biến dẫn xuất quan trọng cho phân tích và clustering"
    },
    {
        "Variable": "Total_Purchases",
        "Meaning": "Tổng số lần mua hàng qua các kênh",
        "Python_Data_Type": str(df_raw["Total_Purchases"].dtype),
        "Variable_Nature": "Quantitative",
        "Subtype": "Discrete",
        "Role": "Biến dẫn xuất mô tả hành vi mua hàng"
    },
    {
        "Variable": "Response",
        "Meaning": "Khách hàng có phản hồi chiến dịch marketing cuối hay không",
        "Python_Data_Type": str(df_raw["Response"].dtype),
        "Variable_Nature": "Qualitative",
        "Subtype": "Binary",
        "Role": "Target chính cho bài toán classification"
    }
]

variable_classification_df = pd.DataFrame(variable_rows)

save_csv(
    variable_classification_df,
    TABLE_RAW_DIR / "A3_variable_classification.csv"
)


# =========================================================
# 7. THỐNG KÊ MÔ TẢ CHO BIẾN ĐỊNH LƯỢNG
# =========================================================

def descriptive_statistics(df, columns):
    """Tính đầy đủ các thông số thống kê theo yêu cầu."""
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


raw_desc_df = descriptive_statistics(df_raw, numeric_analysis_cols)

save_csv(
    raw_desc_df,
    TABLE_RAW_DIR / "A3_descriptive_statistics_raw.csv"
)


# =========================================================
# 8. BẢNG TẦN SUẤT CHO BIẾN ĐỊNH TÍNH
# =========================================================

def frequency_table(df, column):
    """Lập bảng tần suất tuyệt đối, tần suất tương đối và xác định mode."""
    freq = (
        df[column]
        .value_counts(dropna=False)
        .reset_index()
    )

    freq.columns = [column, "Frequency"]
    freq["Relative_Frequency"] = freq["Frequency"] / len(df)
    freq["Percent"] = freq["Relative_Frequency"] * 100

    mode_value = safe_mode(df[column])
    freq["Mode_Of_Variable"] = mode_value

    return freq


for col in categorical_analysis_cols:
    if col in df_raw.columns:
        freq_df = frequency_table(df_raw, col)
        save_csv(
            freq_df,
            TABLE_RAW_DIR / f"A3_frequency_{col}_raw.csv"
        )


# =========================================================
# 9. TÓM TẮT OUTLIERS DỮ LIỆU THÔ
# =========================================================

def outlier_summary(df, columns):
    """Tạo bảng tổng hợp outliers theo IQR."""
    rows = []

    for col in columns:
        if col not in df.columns:
            continue

        series = df[col]
        q1, q3, iqr, lower, upper = iqr_bounds(series)
        outlier_count = count_outliers(series)

        rows.append({
            "Variable": col,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Lower_Bound": lower,
            "Upper_Bound": upper,
            "Outlier_Count": outlier_count,
            "Outlier_Rate_Percent": outlier_count / len(df) * 100
        })

    return pd.DataFrame(rows)


raw_outlier_df = outlier_summary(df_raw, numeric_analysis_cols)

save_csv(
    raw_outlier_df,
    TABLE_RAW_DIR / "A3_outlier_summary_raw.csv"
)


# =========================================================
# 10. BIỂU ĐỒ CHO DỮ LIỆU THÔ
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


def plot_bar(df, column, output_path, title):
    """Vẽ bar chart cho biến định tính."""
    counts = df[column].value_counts(dropna=False)

    plt.figure(figsize=(9, 5))
    counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Số lượng")
    plt.xticks(rotation=45, ha="right")
    save_current_figure(output_path)


def plot_pie(df, column, output_path, title):
    """Vẽ pie chart cho biến định tính có ít nhóm."""
    counts = df[column].value_counts(dropna=False)

    plt.figure(figsize=(7, 7))
    counts.plot(kind="pie", autopct="%1.1f%%", startangle=90)
    plt.title(title)
    plt.ylabel("")
    save_current_figure(output_path)


def plot_scatter(df, x_col, y_col, output_path, title):
    """Vẽ scatter plot giữa hai biến định lượng."""
    plot_df = df[[x_col, y_col]].dropna()

    plt.figure(figsize=(8, 5))
    plt.scatter(plot_df[x_col], plot_df[y_col], alpha=0.6)
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


def plot_missing_values(df, output_path, title):
    """Vẽ biểu đồ số lượng missing values."""
    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    plt.figure(figsize=(8, 5))

    if len(missing_counts) == 0:
        plt.text(
            0.5,
            0.5,
            "Không có giá trị khuyết thiếu",
            ha="center",
            va="center"
        )
        plt.axis("off")
    else:
        missing_counts.plot(kind="bar")
        plt.xlabel("Biến")
        plt.ylabel("Số lượng missing")

    plt.title(title)
    save_current_figure(output_path)


# Biểu đồ missing
plot_missing_values(
    df_raw,
    FIG_RAW_DIR / "A3_missing_values_raw.png",
    "Missing values trong dữ liệu thô A3"
)

# Histogram
for col in ["Age", "Income", "Recency", "Total_Spending", "Total_Purchases"]:
    plot_histogram(
        df_raw,
        col,
        FIG_RAW_DIR / f"A3_hist_{col}_raw.png",
        f"Histogram của biến {col} - dữ liệu thô"
    )

# Boxplot
for col in ["Age", "Income", "Recency", "Total_Spending", "Total_Purchases"]:
    plot_boxplot(
        df_raw,
        col,
        FIG_RAW_DIR / f"A3_boxplot_{col}_raw.png",
        f"Boxplot của biến {col} - dữ liệu thô"
    )

# Bar chart và Pie chart
plot_bar(
    df_raw,
    "Education",
    FIG_RAW_DIR / "A3_bar_Education_raw.png",
    "Phân bố trình độ học vấn - dữ liệu thô"
)

plot_bar(
    df_raw,
    "Marital_Status",
    FIG_RAW_DIR / "A3_bar_Marital_Status_raw.png",
    "Phân bố tình trạng hôn nhân - dữ liệu thô"
)

plot_pie(
    df_raw,
    "Response",
    FIG_RAW_DIR / "A3_pie_Response_raw.png",
    "Tỷ lệ phản hồi chiến dịch marketing - dữ liệu thô"
)

# Scatter plot
plot_scatter(
    df_raw,
    "Income",
    "Total_Spending",
    FIG_RAW_DIR / "A3_scatter_Income_Total_Spending_raw.png",
    "Quan hệ giữa Income và Total_Spending - dữ liệu thô"
)

plot_scatter(
    df_raw,
    "Age",
    "Total_Spending",
    FIG_RAW_DIR / "A3_scatter_Age_Total_Spending_raw.png",
    "Quan hệ giữa Age và Total_Spending - dữ liệu thô"
)

# Heatmap
heatmap_cols = [
    "Age",
    "Income",
    "Recency",
    "MntWines",
    "MntMeatProducts",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "Total_Spending",
    "Total_Purchases",
    "Response"
]

plot_correlation_heatmap(
    df_raw,
    heatmap_cols,
    FIG_RAW_DIR / "A3_heatmap_correlation_raw.png",
    "Ma trận tương quan các biến định lượng - dữ liệu thô"
)


# =========================================================
# 11. TIỀN XỬ LÝ DỮ LIỆU
# =========================================================

df_clean = df_raw.copy()

# 11.1. Điền missing Income bằng median
income_median = df_clean["Income"].median()
df_clean["Income"] = df_clean["Income"].fillna(income_median)

# 11.2. Gộp một số tình trạng hôn nhân quá hiếm thành nhóm Other
rare_marital_values = ["Alone", "Absurd", "YOLO"]
df_clean["Marital_Status_Clean"] = df_clean["Marital_Status"].replace(
    rare_marital_values,
    "Other"
)

# 11.3. Xóa cột hằng số vì không có giá trị phân biệt
constant_cols = [
    col for col in df_clean.columns
    if df_clean[col].nunique(dropna=False) == 1
]

df_clean = df_clean.drop(columns=constant_cols)

# 11.4. Làm mịn outliers bằng IQR cho các biến numeric quan trọng
# Không dùng ID, không dùng biến nhị phân chiến dịch.
cols_to_cap = [
    "Age",
    "Income",
    "Recency",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "Customer_Tenure_Days",
    "Total_Spending",
    "Total_Children",
    "Total_Purchases"
]

for col in cols_to_cap:
    if col in df_clean.columns:
        df_clean[col] = cap_outliers_iqr(df_clean[col])

# 11.5. Tạo file dữ liệu sạch
save_csv(
    df_clean,
    DATA_DIR / "A3_cleaned.csv"
)

# 11.6. Tạo dữ liệu chuẩn hóa Min-Max và Z-score cho các biến numeric
# File này phục vụ clustering/machine learning ở vòng sau.
scaled_df = df_clean.copy()

for col in numeric_analysis_cols:
    if col not in scaled_df.columns:
        continue

    col_mean = scaled_df[col].mean()
    col_std = scaled_df[col].std()
    col_min = scaled_df[col].min()
    col_max = scaled_df[col].max()

    if col_std != 0:
        scaled_df[f"{col}_zscore"] = (scaled_df[col] - col_mean) / col_std
    else:
        scaled_df[f"{col}_zscore"] = 0

    if col_max != col_min:
        scaled_df[f"{col}_minmax"] = (
            scaled_df[col] - col_min
        ) / (col_max - col_min)
    else:
        scaled_df[f"{col}_minmax"] = 0

save_csv(
    scaled_df,
    DATA_DIR / "A3_cleaned_with_scaled_features.csv"
)


# =========================================================
# 12. THỐNG KÊ SAU TIỀN XỬ LÝ
# =========================================================

clean_desc_df = descriptive_statistics(df_clean, numeric_analysis_cols)

save_csv(
    clean_desc_df,
    TABLE_CLEAN_DIR / "A3_descriptive_statistics_cleaned.csv"
)

clean_outlier_df = outlier_summary(df_clean, numeric_analysis_cols)

save_csv(
    clean_outlier_df,
    TABLE_CLEAN_DIR / "A3_outlier_summary_cleaned.csv"
)

for col in ["Education", "Marital_Status_Clean", "Response", "Complain"]:
    if col in df_clean.columns:
        freq_df = frequency_table(df_clean, col)
        save_csv(
            freq_df,
            TABLE_CLEAN_DIR / f"A3_frequency_{col}_cleaned.csv"
        )


# =========================================================
# 13. BẢNG SO SÁNH TRƯỚC VÀ SAU TIỀN XỬ LÝ
# =========================================================

comparison_rows = []

for col in numeric_analysis_cols:
    if col not in df_raw.columns or col not in df_clean.columns:
        continue

    comparison_rows.append({
        "Variable": col,
        "Missing_Before": int(df_raw[col].isna().sum()),
        "Missing_After": int(df_clean[col].isna().sum()),
        "Mean_Before": df_raw[col].mean(),
        "Mean_After": df_clean[col].mean(),
        "Median_Before": df_raw[col].median(),
        "Median_After": df_clean[col].median(),
        "Std_Before": df_raw[col].std(),
        "Std_After": df_clean[col].std(),
        "Outliers_Before": count_outliers(df_raw[col]),
        "Outliers_After": count_outliers(df_clean[col])
    })

comparison_df = pd.DataFrame(comparison_rows)

comparison_df["Mean_Change"] = (
    comparison_df["Mean_After"] - comparison_df["Mean_Before"]
)

comparison_df["Std_Change"] = (
    comparison_df["Std_After"] - comparison_df["Std_Before"]
)

save_csv(
    comparison_df,
    TABLE_COMPARE_DIR / "A3_before_after_preprocessing_summary.csv"
)

# Ghi lại các thao tác tiền xử lý đã thực hiện
preprocessing_steps_df = pd.DataFrame({
    "Step": [
        "Impute missing Income",
        "Create Age",
        "Create Customer_Tenure_Days",
        "Create Total_Spending",
        "Create Total_Children",
        "Create Total_Purchases",
        "Group rare Marital_Status",
        "Drop constant columns",
        "Cap outliers by IQR",
        "Create scaled features"
    ],
    "Description": [
        f"Điền 24 giá trị thiếu của Income bằng median = {income_median}",
        "Tạo Age từ Year_Birth và năm tham chiếu của dataset",
        "Tính số ngày gắn bó của khách hàng từ Dt_Customer",
        "Cộng các biến chi tiêu MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds",
        "Cộng Kidhome và Teenhome",
        "Cộng các biến NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases",
        "Gộp Alone, Absurd, YOLO thành Other",
        f"Xóa các cột hằng số: {constant_cols}",
        "Cắt giá trị outliers về lower/upper bound theo quy tắc IQR",
        "Tạo thêm các biến z-score và min-max cho vòng clustering/machine learning"
    ]
})

save_csv(
    preprocessing_steps_df,
    TABLE_COMPARE_DIR / "A3_preprocessing_steps.csv"
)


# =========================================================
# 14. BIỂU ĐỒ SAU TIỀN XỬ LÝ
# =========================================================

plot_missing_values(
    df_clean,
    FIG_CLEAN_DIR / "A3_missing_values_cleaned.png",
    "Missing values sau tiền xử lý A3"
)

for col in ["Age", "Income", "Recency", "Total_Spending", "Total_Purchases"]:
    plot_histogram(
        df_clean,
        col,
        FIG_CLEAN_DIR / f"A3_hist_{col}_cleaned.png",
        f"Histogram của biến {col} - sau tiền xử lý"
    )

for col in ["Age", "Income", "Recency", "Total_Spending", "Total_Purchases"]:
    plot_boxplot(
        df_clean,
        col,
        FIG_CLEAN_DIR / f"A3_boxplot_{col}_cleaned.png",
        f"Boxplot của biến {col} - sau tiền xử lý"
    )

plot_bar(
    df_clean,
    "Education",
    FIG_CLEAN_DIR / "A3_bar_Education_cleaned.png",
    "Phân bố trình độ học vấn - sau tiền xử lý"
)

plot_bar(
    df_clean,
    "Marital_Status_Clean",
    FIG_CLEAN_DIR / "A3_bar_Marital_Status_Clean_cleaned.png",
    "Phân bố tình trạng hôn nhân - sau tiền xử lý"
)

plot_pie(
    df_clean,
    "Response",
    FIG_CLEAN_DIR / "A3_pie_Response_cleaned.png",
    "Tỷ lệ phản hồi chiến dịch marketing - sau tiền xử lý"
)

plot_scatter(
    df_clean,
    "Income",
    "Total_Spending",
    FIG_CLEAN_DIR / "A3_scatter_Income_Total_Spending_cleaned.png",
    "Quan hệ giữa Income và Total_Spending - sau tiền xử lý"
)

plot_scatter(
    df_clean,
    "Age",
    "Total_Spending",
    FIG_CLEAN_DIR / "A3_scatter_Age_Total_Spending_cleaned.png",
    "Quan hệ giữa Age và Total_Spending - sau tiền xử lý"
)

plot_correlation_heatmap(
    df_clean,
    heatmap_cols,
    FIG_CLEAN_DIR / "A3_heatmap_correlation_cleaned.png",
    "Ma trận tương quan các biến định lượng - sau tiền xử lý"
)


# =========================================================
# 15. GHI LOG VÒNG 2
# =========================================================

with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("A3 Customer Personality Analysis - Round 2 EDA and Preprocessing\n")
    f.write("=" * 70 + "\n")
    f.write(f"Run time: {datetime.now()}\n")
    f.write(f"Raw path: {RAW_PATH}\n")
    f.write(f"Round output directory: {ROUND_DIR}\n\n")

    f.write("RAW DATA SUMMARY\n")
    f.write("-" * 70 + "\n")
    f.write(f"Raw shape after derived features: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns\n")
    f.write(f"Missing Income before preprocessing: {int(df_raw['Income'].isna().sum())}\n")
    f.write(f"Response distribution:\n")
    f.write(df_raw["Response"].value_counts().to_string())
    f.write("\n\n")

    f.write("DERIVED FEATURES\n")
    f.write("-" * 70 + "\n")
    f.write("Created: Age, Customer_Tenure_Days, Total_Spending, Total_Children, Total_Purchases, Total_Accepted_Campaigns\n\n")

    f.write("PREPROCESSING STEPS\n")
    f.write("-" * 70 + "\n")
    f.write(preprocessing_steps_df.to_string(index=False))
    f.write("\n\n")

    f.write("CLEANED DATA SUMMARY\n")
    f.write("-" * 70 + "\n")
    f.write(f"Cleaned shape: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns\n")
    f.write(f"Missing Income after preprocessing: {int(df_clean['Income'].isna().sum())}\n")
    f.write(f"Constant columns removed: {constant_cols}\n\n")

    f.write("OUTPUT FILES\n")
    f.write("-" * 70 + "\n")
    f.write(f"Raw tables: {TABLE_RAW_DIR}\n")
    f.write(f"Cleaned tables: {TABLE_CLEAN_DIR}\n")
    f.write(f"Comparison tables: {TABLE_COMPARE_DIR}\n")
    f.write(f"Raw figures: {FIG_RAW_DIR}\n")
    f.write(f"Cleaned figures: {FIG_CLEAN_DIR}\n")
    f.write(f"Cleaned data: {DATA_DIR / 'A3_cleaned.csv'}\n")

print("Hoàn thành Vòng 2 - EDA và tiền xử lý A3.")
print(f"Output directory: {ROUND_DIR}")
print(f"Log saved to: {LOG_PATH}")
print("Bạn hãy gửi lại file log A3_round2_eda_log.txt sau khi chạy xong.")