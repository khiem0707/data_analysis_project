from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


warnings.filterwarnings("ignore")


# =========================================================
# 1. Cấu hình đường dẫn
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "outputs" / "round_02_A3_eda" / "data" / "A3_cleaned.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "round_05_chapter4_ml"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures" / "A3_classification"

# Hình dùng trong báo cáo đặt trực tiếp vào thư mục ảnh A3 hiện có
REPORT_IMG_DIR = PROJECT_ROOT / "images" / "nhom-a-3"

for folder in [TABLE_DIR, FIG_DIR, REPORT_IMG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. Hàm tiện ích
# =========================================================

def make_onehot_encoder():
    """
    Tạo OneHotEncoder tương thích với nhiều phiên bản scikit-learn.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def safe_filename(name: str) -> str:
    """
    Chuyển tên target/model thành tên file an toàn.
    """
    return (
        str(name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
    )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Tạo pipeline tiền xử lý:
    - Numeric: điền median + StandardScaler
    - Categorical: điền most_frequent + OneHotEncoder
    """
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

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
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def get_models():
    """
    Ba mô hình phân lớp dùng thống nhất cho Dataset A.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf",
            class_weight="balanced",
            random_state=42,
        ),
    }
    return models


def get_feature_matrix(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Tạo X, y cho từng target.

    Lưu ý:
    - Loại bỏ ID, ngày tháng, các cột hằng nếu có.
    - Khi target là Education thì loại bỏ cả Education/Education_Clean để tránh rò rỉ.
    - Khi target là Marital_Status_Clean thì loại bỏ cả Marital_Status và Marital_Status_Clean.
    """
    y = df[target].copy()

    common_drop_cols = [
        "ID",
        "Dt_Customer",
        "Dt_Customer_Parsed",
        "Z_CostContact",
        "Z_Revenue",
    ]

    target_related_drop = [target]

    # =========================================================
    # Chống rò rỉ nhãn khi dự đoán Response
    # =========================================================
    # Response là phản hồi chiến dịch marketing cuối cùng.
    # Các biến AcceptedCmp* và Total_Accepted_Campaigns có thể mang thông tin
    # liên quan trực tiếp đến việc khách hàng đã chấp nhận chiến dịch hay chưa.
    # Vì vậy khi target là Response, loại bỏ toàn bộ các biến chiến dịch để
    # tránh mô hình học từ thông tin rò rỉ.
    if target == "Response":
        campaign_leakage_cols = [
            c for c in df.columns
            if c.startswith("AcceptedCmp")
            or c in ["Total_Accepted_Campaigns"]
        ]
        target_related_drop += campaign_leakage_cols

    if target in ["Education", "Education_Clean"]:
        target_related_drop += ["Education", "Education_Clean"]

    if target in ["Marital_Status", "Marital_Status_Clean"]:
        target_related_drop += ["Marital_Status", "Marital_Status_Clean"]

    # Loại bỏ các cột không tồn tại một cách an toàn
    drop_cols = [c for c in common_drop_cols + target_related_drop if c in df.columns]

    X = df.drop(columns=drop_cols)

    # Loại bỏ các cột toàn missing hoặc chỉ có 1 giá trị duy nhất
    nunique = X.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    X = X.drop(columns=constant_cols)

    # Loại bỏ dòng target bị missing nếu có
    valid_mask = y.notna()
    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()

    return X, y


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Huấn luyện và đánh giá một mô hình.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    return metrics, y_pred


def plot_confusion_matrix(y_test, y_pred, target_name, model_name, output_path):
    """
    Vẽ confusion matrix cho mô hình tốt nhất.
    """
    labels = sorted(pd.Series(y_test).astype(str).unique())

    y_test_str = pd.Series(y_test).astype(str)
    y_pred_str = pd.Series(y_pred).astype(str)

    cm = confusion_matrix(y_test_str, y_pred_str, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=45, values_format="d", colorbar=False)

    ax.set_title(f"Confusion Matrix - {target_name} - {model_name}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_f1_comparison(metrics_df: pd.DataFrame, output_path: Path):
    """
    Vẽ biểu đồ so sánh F1-macro theo target và mô hình.
    """
    pivot_df = metrics_df.pivot(index="target", columns="model", values="f1_macro")

    ax = pivot_df.plot(kind="bar", figsize=(10, 6))
    ax.set_title("A3 Classification - F1-macro Comparison")
    ax.set_xlabel("Target")
    ax.set_ylabel("F1-macro")
    ax.set_ylim(0, 1)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 3. Main process
# =========================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file đầu vào: {INPUT_FILE}\n"
            "Bạn kiểm tra lại file A3_cleaned.csv đã nằm đúng thư mục chưa."
        )

    df = pd.read_csv(INPUT_FILE)

    print("=" * 80)
    print("A3 Classification - Chapter 4")
    print(f"Input file: {INPUT_FILE}")
    print(f"Shape: {df.shape}")
    print("=" * 80)

    # Chọn target.
    # Nếu có Education_Clean thì dùng Education_Clean, nếu không thì dùng Education.
    targets = ["Response"]

    if "Education_Clean" in df.columns:
        targets.append("Education_Clean")
    elif "Education" in df.columns:
        targets.append("Education")

    if "Marital_Status_Clean" in df.columns:
        targets.append("Marital_Status_Clean")
    elif "Marital_Status" in df.columns:
        targets.append("Marital_Status")

    all_metrics = []
    best_models = []
    class_distribution_rows = []

    models = get_models()

    for target in targets:
        print(f"\nĐang xử lý target: {target}")

        X, y = get_feature_matrix(df, target)

        # Lưu phân phối lớp
        class_counts = y.value_counts(dropna=False)
        class_ratios = y.value_counts(normalize=True, dropna=False) * 100

        for cls in class_counts.index:
            class_distribution_rows.append(
                {
                    "target": target,
                    "class": cls,
                    "count": int(class_counts.loc[cls]),
                    "percent": float(class_ratios.loc[cls]),
                }
            )

        # Stratify nếu mỗi lớp có ít nhất 2 mẫu
        stratify_y = y if y.value_counts().min() >= 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify_y,
        )

        print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
        print(f"  Số lớp: {y.nunique()}")

        target_results = []

        for model_name, clf in models.items():
            print(f"  Huấn luyện mô hình: {model_name}")

            preprocessor = build_preprocessor(X_train)

            pipe = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", clf),
                ]
            )

            metrics, y_pred = evaluate_model(pipe, X_train, X_test, y_train, y_test)

            row = {
                "target": target,
                "model": model_name,
                **metrics,
            }

            all_metrics.append(row)
            target_results.append((model_name, pipe, metrics, y_pred))

        # Chọn mô hình tốt nhất theo F1-macro
        best_model_name, best_pipe, best_metrics, best_pred = max(
            target_results,
            key=lambda item: item[2]["f1_macro"],
        )

        best_models.append(
            {
                "target": target,
                "best_model": best_model_name,
                "accuracy": best_metrics["accuracy"],
                "f1_macro": best_metrics["f1_macro"],
                "f1_weighted": best_metrics["f1_weighted"],
            }
        )

        # Lưu confusion matrix của mô hình tốt nhất
        target_safe = safe_filename(target)
        model_safe = safe_filename(best_model_name)

        cm_name = f"A3_CH4_{target_safe}_confusion_matrix_best_model.png"
        cm_output_report = REPORT_IMG_DIR / cm_name
        cm_output_output = FIG_DIR / cm_name

        plot_confusion_matrix(
            y_test=y_test,
            y_pred=best_pred,
            target_name=target,
            model_name=best_model_name,
            output_path=cm_output_report,
        )

        # Copy thêm một bản vào outputs để lưu log
        plot_confusion_matrix(
            y_test=y_test,
            y_pred=best_pred,
            target_name=target,
            model_name=best_model_name,
            output_path=cm_output_output,
        )

        print(f"  Mô hình tốt nhất cho {target}: {best_model_name}")
        print(f"  Đã lưu confusion matrix: {cm_output_report}")

    # =====================================================
    # 4. Xuất bảng kết quả
    # =====================================================

    metrics_df = pd.DataFrame(all_metrics)
    best_df = pd.DataFrame(best_models)
    class_dist_df = pd.DataFrame(class_distribution_rows)

    metrics_path = TABLE_DIR / "A3_CH4_classification_metrics.csv"
    best_path = TABLE_DIR / "A3_CH4_classification_best_models.csv"
    class_dist_path = TABLE_DIR / "A3_CH4_class_distribution.csv"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    best_df.to_csv(best_path, index=False, encoding="utf-8-sig")
    class_dist_df.to_csv(class_dist_path, index=False, encoding="utf-8-sig")

    # Hình so sánh F1
    f1_report_path = REPORT_IMG_DIR / "A3_CH4_classification_f1_comparison.png"
    f1_output_path = FIG_DIR / "A3_CH4_classification_f1_comparison.png"

    plot_f1_comparison(metrics_df, f1_report_path)
    plot_f1_comparison(metrics_df, f1_output_path)

    print("\n" + "=" * 80)
    print("Hoàn tất A3 Classification")
    print(f"Metrics: {metrics_path}")
    print(f"Best models: {best_path}")
    print(f"Class distribution: {class_dist_path}")
    print(f"F1 comparison figure: {f1_report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()