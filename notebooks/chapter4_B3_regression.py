from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor


warnings.filterwarnings("ignore")


# =========================================================
# 1. Cấu hình đường dẫn
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "outputs" / "round_03_B3_eda" / "data" / "B3_hour_raw_with_datetime.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "round_05_chapter4_ml"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures" / "B3_regression"

# Hình dùng trong báo cáo đặt vào thư mục ảnh B3 hiện có
REPORT_IMG_DIR = PROJECT_ROOT / "images" / "nhom-b-3"

for folder in [TABLE_DIR, FIG_DIR, REPORT_IMG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. Hàm tiện ích
# =========================================================

def make_onehot_encoder():
    """
    Tạo OneHotEncoder tương thích nhiều phiên bản scikit-learn.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_b3_data() -> pd.DataFrame:
    """
    Đọc dữ liệu B3 và tạo cột datetime nếu cần.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file đầu vào: {INPUT_FILE}\n"
            "Bạn kiểm tra lại file B3_hour_raw_with_datetime.csv đã nằm đúng thư mục chưa."
        )

    df = pd.read_csv(INPUT_FILE)

    if "datetime" not in df.columns:
        if "dteday" in df.columns and "hr" in df.columns:
            df["datetime"] = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
        else:
            raise ValueError("Không tìm thấy cột datetime hoặc cặp cột dteday + hr để tạo datetime.")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


def create_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo đặc trưng chuỗi thời gian cho B3.
    Dữ liệu được sắp xếp theo datetime trước khi tạo lag/rolling.
    """
    df = df.copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    # Nếu thiếu các biến thời gian, tạo lại từ datetime
    if "hr" not in df.columns:
        df["hr"] = df["datetime"].dt.hour
    if "weekday" not in df.columns:
        df["weekday"] = df["datetime"].dt.weekday
    if "mnth" not in df.columns:
        df["mnth"] = df["datetime"].dt.month
    if "yr" not in df.columns:
        df["yr"] = df["datetime"].dt.year

    # Đặc trưng chuỗi thời gian
    df["lag_1"] = df["cnt"].shift(1)
    df["lag_24"] = df["cnt"].shift(24)
    df["lag_168"] = df["cnt"].shift(168)

    # rolling dùng shift(1) để tránh dùng chính giá trị hiện tại khi dự đoán
    df["rolling_mean_24"] = df["cnt"].shift(1).rolling(window=24, min_periods=12).mean()
    df["rolling_std_24"] = df["cnt"].shift(1).rolling(window=24, min_periods=12).std()
    df["rolling_mean_168"] = df["cnt"].shift(1).rolling(window=168, min_periods=48).mean()

    # Một số đặc trưng lịch bổ sung
    df["is_weekend"] = df["weekday"].isin([0, 6]).astype(int)

    # Loại bỏ các dòng đầu bị thiếu do lag/rolling
    df = df.dropna(subset=[
        "lag_1",
        "lag_24",
        "lag_168",
        "rolling_mean_24",
        "rolling_std_24",
        "rolling_mean_168",
        "cnt"
    ]).reset_index(drop=True)

    return df


def build_feature_target(df: pd.DataFrame):
    """
    Tạo X, y cho mô hình hồi quy.
    Không sử dụng casual và registered vì:
    cnt = casual + registered, nếu dùng hai biến này thì rò rỉ trực tiếp target.
    """
    target = "cnt"

    candidate_features = [
        # thời gian / lịch
        "season",
        "yr",
        "mnth",
        "hr",
        "weekday",
        "holiday",
        "workingday",
        "is_weekend",

        # thời tiết
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed",

        # time-series features
        "lag_1",
        "lag_24",
        "lag_168",
        "rolling_mean_24",
        "rolling_std_24",
        "rolling_mean_168",
    ]

    features = [c for c in candidate_features if c in df.columns]

    X = df[features].copy()
    y = df[target].copy()

    return X, y, features


def build_preprocessor(X: pd.DataFrame):
    """
    Tiền xử lý:
    - Numeric: median imputation + StandardScaler
    - Categorical: most_frequent + OneHotEncoder
    """
    categorical_cols = [
        c for c in X.columns
        if c in ["season", "yr", "mnth", "hr", "weekday", "holiday", "workingday", "weathersit", "is_weekend"]
    ]

    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def get_models():
    """
    Các mô hình hồi quy/dự báo dùng cho Dataset B3.
    """
    models = {
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=42,
        ),
    }

    return models


def evaluate_regression(y_true, y_pred):
    """
    Tính các chỉ số đánh giá hồi quy.
    Cách tính RMSE dùng sqrt(MSE) để tương thích với nhiều phiên bản scikit-learn.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }

def plot_actual_vs_predicted(result_df: pd.DataFrame, model_name: str, output_path: Path):
    """
    Vẽ actual vs predicted theo thời gian cho mô hình tốt nhất.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(result_df["datetime"], result_df["actual"], label="Actual", linewidth=1.2)
    ax.plot(result_df["datetime"], result_df["predicted"], label="Predicted", linewidth=1.2)

    ax.set_title(f"B3 Actual vs Predicted - {model_name}")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Bike rental count")
    ax.legend()

    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metrics_comparison(metrics_df: pd.DataFrame, output_path: Path):
    """
    Vẽ biểu đồ so sánh MAE và RMSE giữa các mô hình.
    """
    plot_df = metrics_df.set_index("model")[["MAE", "RMSE"]]

    ax = plot_df.plot(kind="bar", figsize=(10, 6))
    ax.set_title("B3 Regression - Error Metrics Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Error")
    ax.legend(title="Metric")

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_r2_comparison(metrics_df: pd.DataFrame, output_path: Path):
    """
    Vẽ biểu đồ so sánh R2 giữa các mô hình.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(metrics_df["model"], metrics_df["R2"])
    ax.set_title("B3 Regression - R2 Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("R2-score")
    ax.set_ylim(0, 1)

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_residual_histogram(result_df: pd.DataFrame, model_name: str, output_path: Path):
    """
    Vẽ histogram phần dư của mô hình tốt nhất.
    """
    residuals = result_df["actual"] - result_df["predicted"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=40)

    ax.set_title(f"B3 Residual Distribution - {model_name}")
    ax.set_xlabel("Residual = Actual - Predicted")
    ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_train_test_split(df_model: pd.DataFrame, split_idx: int, output_path: Path):
    """
    Vẽ minh họa vùng train/test theo thời gian.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df_model["datetime"], df_model["cnt"], linewidth=0.8, label="cnt")

    split_time = df_model.iloc[split_idx]["datetime"]
    ax.axvline(split_time, linestyle="--", label="Train/Test split")

    ax.set_title("B3 Time-based Train/Test Split")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("cnt")
    ax.legend()

    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 3. Main process
# =========================================================

def main():
    print("=" * 80)
    print("B3 Regression / Forecasting - Chapter 4")
    print("=" * 80)

    df_raw = load_b3_data()

    print(f"Input file: {INPUT_FILE}")
    print(f"Raw shape: {df_raw.shape}")
    print(f"Datetime range: {df_raw['datetime'].min()} -> {df_raw['datetime'].max()}")

    df_model = create_time_series_features(df_raw)

    print(f"Modeling shape after lag/rolling: {df_model.shape}")

    X, y, features = build_feature_target(df_model)

    print(f"Features used ({len(features)}):")
    for f in features:
        print(f"  - {f}")

    # =====================================================
    # Time-based split
    # =====================================================

    split_idx = int(len(df_model) * 0.8)

    train_df = df_model.iloc[:split_idx].copy()
    test_df = df_model.iloc[split_idx:].copy()

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    print("\nTime-based split:")
    print(f"  Train: {train_df['datetime'].min()} -> {train_df['datetime'].max()} | n={len(train_df)}")
    print(f"  Test : {test_df['datetime'].min()} -> {test_df['datetime'].max()} | n={len(test_df)}")

    # Lưu summary train/test
    split_summary = pd.DataFrame([
        {
            "set": "train",
            "start_datetime": train_df["datetime"].min(),
            "end_datetime": train_df["datetime"].max(),
            "n_rows": len(train_df),
            "cnt_mean": train_df["cnt"].mean(),
            "cnt_std": train_df["cnt"].std(),
            "cnt_min": train_df["cnt"].min(),
            "cnt_max": train_df["cnt"].max(),
        },
        {
            "set": "test",
            "start_datetime": test_df["datetime"].min(),
            "end_datetime": test_df["datetime"].max(),
            "n_rows": len(test_df),
            "cnt_mean": test_df["cnt"].mean(),
            "cnt_std": test_df["cnt"].std(),
            "cnt_min": test_df["cnt"].min(),
            "cnt_max": test_df["cnt"].max(),
        }
    ])

    split_summary_path = TABLE_DIR / "B3_CH4_train_test_split_summary.csv"
    split_summary.to_csv(split_summary_path, index=False, encoding="utf-8-sig")

    train_test_fig_report = REPORT_IMG_DIR / "B3_CH4_train_test_split.png"
    train_test_fig_output = FIG_DIR / "B3_CH4_train_test_split.png"

    plot_train_test_split(df_model, split_idx, train_test_fig_report)
    plot_train_test_split(df_model, split_idx, train_test_fig_output)

    # =====================================================
    # Train models
    # =====================================================

    models = get_models()
    all_metrics = []
    prediction_outputs = {}

    for model_name, regressor in models.items():
        print(f"\nHuấn luyện mô hình: {model_name}")

        preprocessor = build_preprocessor(X_train)

        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", regressor),
            ]
        )

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Không để dự đoán âm
        y_pred = np.clip(y_pred, 0, None)

        metrics = evaluate_regression(y_test, y_pred)

        print(f"  MAE : {metrics['MAE']:.3f}")
        print(f"  RMSE: {metrics['RMSE']:.3f}")
        print(f"  R2  : {metrics['R2']:.3f}")

        all_metrics.append({
            "model": model_name,
            **metrics,
        })

        result_df = pd.DataFrame({
            "datetime": test_df["datetime"].values,
            "actual": y_test.values,
            "predicted": y_pred,
            "residual": y_test.values - y_pred,
        })

        prediction_outputs[model_name] = result_df

        pred_path = TABLE_DIR / f"B3_CH4_predictions_{model_name.replace(' ', '_')}.csv"
        result_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    # =====================================================
    # Lưu metrics và chọn mô hình tốt nhất
    # =====================================================

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.sort_values("RMSE").reset_index(drop=True)

    metrics_path = TABLE_DIR / "B3_CH4_regression_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    best_model_name = metrics_df.iloc[0]["model"]
    best_result_df = prediction_outputs[best_model_name]

    best_summary = pd.DataFrame([{
        "best_model": best_model_name,
        "MAE": metrics_df.iloc[0]["MAE"],
        "RMSE": metrics_df.iloc[0]["RMSE"],
        "R2": metrics_df.iloc[0]["R2"],
    }])

    best_summary_path = TABLE_DIR / "B3_CH4_regression_best_model.csv"
    best_summary.to_csv(best_summary_path, index=False, encoding="utf-8-sig")

    print("\nMô hình tốt nhất theo RMSE:")
    print(best_summary)

    # =====================================================
    # Vẽ hình cho báo cáo
    # =====================================================

    actual_pred_fig = REPORT_IMG_DIR / "B3_CH4_actual_vs_predicted_best_model.png"
    actual_pred_fig_output = FIG_DIR / "B3_CH4_actual_vs_predicted_best_model.png"

    metrics_fig = REPORT_IMG_DIR / "B3_CH4_regression_metrics_comparison.png"
    metrics_fig_output = FIG_DIR / "B3_CH4_regression_metrics_comparison.png"

    r2_fig = REPORT_IMG_DIR / "B3_CH4_regression_r2_comparison.png"
    r2_fig_output = FIG_DIR / "B3_CH4_regression_r2_comparison.png"

    residual_fig = REPORT_IMG_DIR / "B3_CH4_residual_histogram_best_model.png"
    residual_fig_output = FIG_DIR / "B3_CH4_residual_histogram_best_model.png"

    plot_actual_vs_predicted(best_result_df, best_model_name, actual_pred_fig)
    plot_actual_vs_predicted(best_result_df, best_model_name, actual_pred_fig_output)

    plot_metrics_comparison(metrics_df, metrics_fig)
    plot_metrics_comparison(metrics_df, metrics_fig_output)

    plot_r2_comparison(metrics_df, r2_fig)
    plot_r2_comparison(metrics_df, r2_fig_output)

    plot_residual_histogram(best_result_df, best_model_name, residual_fig)
    plot_residual_histogram(best_result_df, best_model_name, residual_fig_output)

    print("\n" + "=" * 80)
    print("Hoàn tất B3 Regression / Forecasting")
    print(f"Metrics: {metrics_path}")
    print(f"Best model: {best_summary_path}")
    print(f"Train/Test summary: {split_summary_path}")
    print(f"Actual vs Predicted figure: {actual_pred_fig}")
    print(f"Metrics comparison figure: {metrics_fig}")
    print(f"R2 comparison figure: {r2_fig}")
    print(f"Residual histogram figure: {residual_fig}")
    print("=" * 80)


if __name__ == "__main__":
    main()