from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


# =========================================================
# 1. Cấu hình đường dẫn
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "outputs" / "round_03_B3_eda" / "data" / "B3_hour_raw_with_datetime.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "round_05_chapter4_ml"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures" / "B3_clustering"

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
    Đọc dữ liệu B3 và tạo cột datetime nếu chưa có.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file đầu vào: {INPUT_FILE}\n"
            "Bạn kiểm tra lại B3_hour_raw_with_datetime.csv đã nằm đúng thư mục chưa."
        )

    df = pd.read_csv(INPUT_FILE)

    if "datetime" not in df.columns:
        if "dteday" in df.columns and "hr" in df.columns:
            df["datetime"] = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
        else:
            raise ValueError("Không tìm thấy datetime hoặc dteday + hr để tạo datetime.")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


def create_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo đặc trưng chuỗi thời gian cho B3.
    Các lag/rolling được tạo từ cnt nhưng chỉ dùng làm feature lịch sử,
    không dùng trực tiếp cnt hiện tại để fit clustering.
    """
    df = df.copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    if "hr" not in df.columns:
        df["hr"] = df["datetime"].dt.hour
    if "weekday" not in df.columns:
        df["weekday"] = df["datetime"].dt.weekday
    if "mnth" not in df.columns:
        df["mnth"] = df["datetime"].dt.month
    if "yr" not in df.columns:
        df["yr"] = df["datetime"].dt.year

    df["is_weekend"] = df["weekday"].isin([0, 6]).astype(int)

    # Lag features
    df["lag_1"] = df["cnt"].shift(1)
    df["lag_24"] = df["cnt"].shift(24)
    df["lag_168"] = df["cnt"].shift(168)

    # Rolling features, shift(1) để tránh dùng chính cnt tại thời điểm hiện tại
    df["rolling_mean_24"] = df["cnt"].shift(1).rolling(window=24, min_periods=12).mean()
    df["rolling_std_24"] = df["cnt"].shift(1).rolling(window=24, min_periods=12).std()
    df["rolling_mean_168"] = df["cnt"].shift(1).rolling(window=168, min_periods=48).mean()

    df = df.dropna(subset=[
        "lag_1",
        "lag_24",
        "lag_168",
        "rolling_mean_24",
        "rolling_std_24",
        "rolling_mean_168",
        "cnt",
    ]).reset_index(drop=True)

    return df


def select_clustering_features(df: pd.DataFrame):
    """
    Chọn feature cho B3 clustering.

    Không dùng trực tiếp cnt, casual, registered để fit clustering.
    cnt chỉ dùng sau đó để đọc vị cụm.
    """
    categorical_features = [
        "season",
        "yr",
        "mnth",
        "hr",
        "weekday",
        "holiday",
        "workingday",
        "weathersit",
        "is_weekend",
    ]

    numeric_features = [
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "lag_1",
        "lag_24",
        "lag_168",
        "rolling_mean_24",
        "rolling_std_24",
        "rolling_mean_168",
    ]

    categorical_features = [c for c in categorical_features if c in df.columns]
    numeric_features = [c for c in numeric_features if c in df.columns]

    features = categorical_features + numeric_features

    X_raw = df[features].copy()

    return X_raw, categorical_features, numeric_features, features


def preprocess_features(X_raw, categorical_features, numeric_features):
    """
    Numeric: median + StandardScaler.
    Categorical: most frequent + OneHotEncoder.
    """
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

    X_scaled = preprocessor.fit_transform(X_raw)

    return X_scaled, preprocessor


def evaluate_clustering(X_scaled, labels):
    """
    Tính các chỉ số clustering.
    Với DBSCAN, bỏ noise label = -1 khi tính metric.
    """
    labels = np.asarray(labels)

    unique_labels = sorted(set(labels))
    n_clusters = len([x for x in unique_labels if x != -1])
    noise_count = int(np.sum(labels == -1))

    metrics = {
        "n_clusters": n_clusters,
        "noise_count": noise_count,
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
    }

    valid_mask = labels != -1
    valid_labels = labels[valid_mask]
    valid_X = X_scaled[valid_mask]

    if len(set(valid_labels)) >= 2 and len(set(valid_labels)) < len(valid_labels):
        metrics["silhouette"] = silhouette_score(valid_X, valid_labels)
        metrics["davies_bouldin"] = davies_bouldin_score(valid_X, valid_labels)
        metrics["calinski_harabasz"] = calinski_harabasz_score(valid_X, valid_labels)

    return metrics


def choose_best_k_by_silhouette(X_scaled, k_min=2, k_max=8):
    """
    Chạy KMeans với nhiều K, chọn K có silhouette cao nhất.
    """
    rows = []

    for k in range(k_min, k_max + 1):
        print(f"  Đang chạy KMeans với K={k}")

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20,
        )

        labels = kmeans.fit_predict(X_scaled)

        inertia = kmeans.inertia_
        sil = silhouette_score(X_scaled, labels)

        rows.append({
            "k": k,
            "inertia": inertia,
            "silhouette": sil,
        })

    result_df = pd.DataFrame(rows)
    best_k = int(result_df.sort_values("silhouette", ascending=False).iloc[0]["k"])

    return best_k, result_df


def sample_for_heavy_models(X_scaled, df_model, max_n=5000, random_state=42):
    """
    Agglomerative và DBSCAN có thể tốn tài nguyên với 17k dòng.
    Dùng sample để đánh giá hai mô hình này.
    KMeans vẫn chạy trên full data.
    """
    n = X_scaled.shape[0]

    if n <= max_n:
        sample_idx = np.arange(n)
    else:
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(np.arange(n), size=max_n, replace=False)
        sample_idx = np.sort(sample_idx)

    X_sample = X_scaled[sample_idx]
    df_sample = df_model.iloc[sample_idx].copy().reset_index(drop=True)

    return X_sample, df_sample, sample_idx


def tune_dbscan(X_sample):
    """
    Thử nhiều eps/min_samples trên sample để chọn DBSCAN.
    """
    rows = []

    eps_values = np.arange(1.0, 8.5, 0.5)
    min_samples_values = [10, 20, 30]

    for eps in eps_values:
        for min_samples in min_samples_values:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_sample)

            metrics = evaluate_clustering(X_sample, labels)

            rows.append({
                "eps": eps,
                "min_samples": min_samples,
                **metrics,
            })

    result_df = pd.DataFrame(rows)

    valid_df = result_df[
        result_df["silhouette"].notna()
        & (result_df["n_clusters"] >= 2)
    ].copy()

    if len(valid_df) == 0:
        best = {
            "eps": 3.0,
            "min_samples": 20,
        }
    else:
        valid_df["noise_ratio"] = valid_df["noise_count"] / len(X_sample)
        valid_df = valid_df[valid_df["noise_ratio"] <= 0.7]

        if len(valid_df) == 0:
            valid_df = result_df[result_df["silhouette"].notna()].copy()

        best_row = valid_df.sort_values("silhouette", ascending=False).iloc[0]

        best = {
            "eps": float(best_row["eps"]),
            "min_samples": int(best_row["min_samples"]),
        }

    return best, result_df


def run_pca(X_scaled):
    """
    PCA 2D để trực quan hóa cụm.
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_

    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
    })

    return pca_df, explained


def create_cluster_profile(df_model: pd.DataFrame, labels, cluster_col: str):
    """
    Tạo profile cụm B3.
    cnt không dùng để fit, nhưng được dùng để đọc vị cụm.
    """
    temp_df = df_model.copy()
    temp_df[cluster_col] = labels

    profile_cols = [
        "cnt",
        "hr",
        "weekday",
        "mnth",
        "season",
        "workingday",
        "holiday",
        "is_weekend",
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "lag_1",
        "lag_24",
        "lag_168",
        "rolling_mean_24",
        "rolling_mean_168",
    ]

    profile_cols = [c for c in profile_cols if c in temp_df.columns]

    profile = temp_df.groupby(cluster_col)[profile_cols].mean().reset_index()

    counts = temp_df[cluster_col].value_counts().sort_index()
    percents = temp_df[cluster_col].value_counts(normalize=True).sort_index() * 100

    profile.insert(1, "count", profile[cluster_col].map(counts))
    profile.insert(2, "percent", profile[cluster_col].map(percents))

    return profile


def create_compact_profile(profile_df: pd.DataFrame, cluster_col: str):
    """
    Rút gọn profile để đưa vào báo cáo.
    """
    keep_cols = [
        cluster_col,
        "count",
        "percent",
        "cnt",
        "hr",
        "weekday",
        "season",
        "workingday",
        "weathersit",
        "temp",
        "hum",
        "windspeed",
        "rolling_mean_24",
        "rolling_mean_168",
    ]

    keep_cols = [c for c in keep_cols if c in profile_df.columns]

    compact = profile_df[keep_cols].copy()

    return compact


def add_auto_interpretation(compact_profile: pd.DataFrame, cluster_col: str):
    """
    Gợi ý diễn giải cụm dựa trên cnt/hr/weather.
    """
    df = compact_profile.copy()

    interpretations = []

    cnt_median = df["cnt"].median() if "cnt" in df.columns else np.nan
    hr_median = df["hr"].median() if "hr" in df.columns else np.nan
    temp_median = df["temp"].median() if "temp" in df.columns else np.nan
    hum_median = df["hum"].median() if "hum" in df.columns else np.nan
    workingday_median = df["workingday"].median() if "workingday" in df.columns else np.nan

    for _, row in df.iterrows():
        phrases = []

        if "cnt" in df.columns:
            if row["cnt"] >= cnt_median:
                phrases.append("nhu cầu thuê xe cao")
            else:
                phrases.append("nhu cầu thuê xe thấp")

        if "hr" in df.columns:
            if row["hr"] < hr_median:
                phrases.append("thời điểm sớm hơn trong ngày")
            else:
                phrases.append("thời điểm muộn hơn trong ngày")

        if "temp" in df.columns:
            if row["temp"] >= temp_median:
                phrases.append("nhiệt độ cao hơn")
            else:
                phrases.append("nhiệt độ thấp hơn")

        if "hum" in df.columns:
            if row["hum"] >= hum_median:
                phrases.append("độ ẩm cao hơn")
            else:
                phrases.append("độ ẩm thấp hơn")

        if "workingday" in df.columns:
            if row["workingday"] >= workingday_median:
                phrases.append("thiên về ngày làm việc")
            else:
                phrases.append("thiên về ngày nghỉ/cuối tuần")

        interpretations.append(", ".join(phrases))

    df["interpretation_suggestion"] = interpretations

    return df


# =========================================================
# 3. Vẽ hình
# =========================================================

def plot_elbow(k_result_df: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(k_result_df["k"], k_result_df["inertia"], marker="o")
    ax.set_title("B3 K-Means Elbow Method")
    ax.set_xlabel("Number of clusters K")
    ax.set_ylabel("Inertia")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_silhouette(k_result_df: pd.DataFrame, best_k: int, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(k_result_df["k"], k_result_df["silhouette"], marker="o")
    ax.axvline(best_k, linestyle="--", label=f"Best K = {best_k}")
    ax.set_title("B3 K-Means Silhouette Score")
    ax.set_xlabel("Number of clusters K")
    ax.set_ylabel("Silhouette score")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pca_clusters(pca_df: pd.DataFrame, labels, title: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        c=labels,
        s=10,
        alpha=0.70,
    )

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    legend = ax.legend(
        *scatter.legend_elements(),
        title="Cluster",
        loc="best"
    )
    ax.add_artist(legend)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cluster_size(labels, output_path: Path):
    labels_series = pd.Series(labels, name="cluster")
    size_df = labels_series.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(size_df.index.astype(str), size_df.values)
    ax.set_title("B3 Cluster Size Distribution")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of observations")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cluster_hour_profile(df_model: pd.DataFrame, labels, output_path: Path):
    """
    Vẽ cnt trung bình theo giờ và cụm.
    """
    temp_df = df_model.copy()
    temp_df["cluster"] = labels

    hour_profile = (
        temp_df
        .groupby(["cluster", "hr"])["cnt"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    for cluster_id in sorted(hour_profile["cluster"].unique()):
        sub = hour_profile[hour_profile["cluster"] == cluster_id]
        ax.plot(sub["hr"], sub["cnt"], marker="o", label=f"Cluster {cluster_id}")

    ax.set_title("B3 Average cnt by Hour and Cluster")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Average cnt")
    ax.legend(title="Cluster")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 4. Main process
# =========================================================

def main():
    print("=" * 80)
    print("B3 Clustering - Chapter 4")
    print("=" * 80)

    df_raw = load_b3_data()

    print(f"Input file: {INPUT_FILE}")
    print(f"Raw shape: {df_raw.shape}")
    print(f"Datetime range: {df_raw['datetime'].min()} -> {df_raw['datetime'].max()}")

    df_model = create_time_series_features(df_raw)

    print(f"Modeling shape after lag/rolling: {df_model.shape}")

    X_raw, categorical_features, numeric_features, features = select_clustering_features(df_model)

    print(f"Features used ({len(features)}):")
    for f in features:
        print(f"  - {f}")

    print("\nCategorical features:")
    print(categorical_features)

    print("\nNumeric features:")
    print(numeric_features)

    X_scaled, preprocessor = preprocess_features(X_raw, categorical_features, numeric_features)

    print(f"\nTransformed feature matrix shape: {X_scaled.shape}")

    # =====================================================
    # KMeans full data
    # =====================================================

    print("\nKMeans K search:")
    best_k, k_result_df = choose_best_k_by_silhouette(X_scaled, k_min=2, k_max=8)

    print(k_result_df)
    print(f"Best K by silhouette: {best_k}")

    k_search_path = TABLE_DIR / "B3_CH4_kmeans_k_search.csv"
    k_result_df.to_csv(k_search_path, index=False, encoding="utf-8-sig")

    kmeans = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=20,
    )
    kmeans_labels = kmeans.fit_predict(X_scaled)

    # =====================================================
    # Agglomerative và DBSCAN trên sample để tránh quá tải
    # =====================================================

    X_sample, df_sample, sample_idx = sample_for_heavy_models(
        X_scaled,
        df_model,
        max_n=5000,
        random_state=42,
    )

    print(f"\nSample size for Agglomerative/DBSCAN: {X_sample.shape[0]}")

    agg = AgglomerativeClustering(n_clusters=best_k)
    agg_labels_sample = agg.fit_predict(X_sample)

    best_dbscan_params, dbscan_search_df = tune_dbscan(X_sample)

    dbscan = DBSCAN(
        eps=best_dbscan_params["eps"],
        min_samples=best_dbscan_params["min_samples"],
    )
    dbscan_labels_sample = dbscan.fit_predict(X_sample)

    dbscan_search_path = TABLE_DIR / "B3_CH4_dbscan_parameter_search_sample.csv"
    dbscan_search_df.to_csv(dbscan_search_path, index=False, encoding="utf-8-sig")

    print("\nBest DBSCAN params on sample:")
    print(best_dbscan_params)

    # =====================================================
    # Metrics summary
    # =====================================================

    metrics_rows = []

    kmeans_metrics = evaluate_clustering(X_scaled, kmeans_labels)
    metrics_rows.append({
        "dataset": "B3",
        "model": "KMeans",
        "data_used": "full",
        **kmeans_metrics,
    })

    agg_metrics = evaluate_clustering(X_sample, agg_labels_sample)
    metrics_rows.append({
        "dataset": "B3",
        "model": "Agglomerative",
        "data_used": "sample_5000",
        **agg_metrics,
    })

    dbscan_metrics = evaluate_clustering(X_sample, dbscan_labels_sample)
    metrics_rows.append({
        "dataset": "B3",
        "model": "DBSCAN",
        "data_used": "sample_5000",
        **dbscan_metrics,
    })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = TABLE_DIR / "B3_CH4_clustering_metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print("\nClustering metrics:")
    print(metrics_df)

    # =====================================================
    # Lưu dữ liệu có nhãn cụm KMeans full
    # =====================================================

    df_result = df_model.copy()
    df_result["B3_CH4_kmeans_cluster"] = kmeans_labels

    labeled_data_path = TABLE_DIR / "B3_CH4_clustering_labeled_data.csv"
    df_result.to_csv(labeled_data_path, index=False, encoding="utf-8-sig")

    # =====================================================
    # Profile cụm
    # =====================================================

    profile = create_cluster_profile(
        df_model=df_result,
        labels=kmeans_labels,
        cluster_col="B3_CH4_kmeans_cluster",
    )

    profile_path = TABLE_DIR / "B3_CH4_kmeans_cluster_profile_full.csv"
    profile.to_csv(profile_path, index=False, encoding="utf-8-sig")

    compact_profile = create_compact_profile(profile, cluster_col="B3_CH4_kmeans_cluster")
    compact_profile = add_auto_interpretation(compact_profile, cluster_col="B3_CH4_kmeans_cluster")

    compact_profile_path = TABLE_DIR / "B3_CH4_kmeans_cluster_profile_compact.csv"
    compact_profile.to_csv(compact_profile_path, index=False, encoding="utf-8-sig")

    print("\nCompact cluster profile:")
    print(compact_profile)

    # =====================================================
    # PCA full data
    # =====================================================

    pca_df, explained = run_pca(X_scaled)

    pca_result = pca_df.copy()
    pca_result["kmeans_cluster"] = kmeans_labels

    pca_path = TABLE_DIR / "B3_CH4_pca_coordinates.csv"
    pca_result.to_csv(pca_path, index=False, encoding="utf-8-sig")

    explained_path = TABLE_DIR / "B3_CH4_pca_explained_variance.csv"
    pd.DataFrame({
        "component": ["PC1", "PC2"],
        "explained_variance_ratio": explained,
    }).to_csv(explained_path, index=False, encoding="utf-8-sig")

    print("\nPCA explained variance:")
    print(f"  PC1: {explained[0]:.4f}")
    print(f"  PC2: {explained[1]:.4f}")
    print(f"  Total: {explained.sum():.4f}")

    # =====================================================
    # Vẽ hình
    # =====================================================

    fig_paths = {
        "elbow_report": REPORT_IMG_DIR / "B3_CH4_kmeans_elbow.png",
        "elbow_output": FIG_DIR / "B3_CH4_kmeans_elbow.png",

        "silhouette_report": REPORT_IMG_DIR / "B3_CH4_kmeans_silhouette.png",
        "silhouette_output": FIG_DIR / "B3_CH4_kmeans_silhouette.png",

        "pca_report": REPORT_IMG_DIR / "B3_CH4_pca_clusters_kmeans.png",
        "pca_output": FIG_DIR / "B3_CH4_pca_clusters_kmeans.png",

        "cluster_size_report": REPORT_IMG_DIR / "B3_CH4_kmeans_cluster_size.png",
        "cluster_size_output": FIG_DIR / "B3_CH4_kmeans_cluster_size.png",

        "hour_profile_report": REPORT_IMG_DIR / "B3_CH4_cluster_hour_profile.png",
        "hour_profile_output": FIG_DIR / "B3_CH4_cluster_hour_profile.png",
    }

    plot_elbow(k_result_df, fig_paths["elbow_report"])
    plot_elbow(k_result_df, fig_paths["elbow_output"])

    plot_silhouette(k_result_df, best_k, fig_paths["silhouette_report"])
    plot_silhouette(k_result_df, best_k, fig_paths["silhouette_output"])

    plot_pca_clusters(
        pca_df,
        kmeans_labels,
        title=f"B3 Time-Weather Clusters by K-Means PCA, K={best_k}",
        output_path=fig_paths["pca_report"],
    )
    plot_pca_clusters(
        pca_df,
        kmeans_labels,
        title=f"B3 Time-Weather Clusters by K-Means PCA, K={best_k}",
        output_path=fig_paths["pca_output"],
    )

    plot_cluster_size(kmeans_labels, fig_paths["cluster_size_report"])
    plot_cluster_size(kmeans_labels, fig_paths["cluster_size_output"])

    plot_cluster_hour_profile(df_result, kmeans_labels, fig_paths["hour_profile_report"])
    plot_cluster_hour_profile(df_result, kmeans_labels, fig_paths["hour_profile_output"])

    print("\n" + "=" * 80)
    print("Hoàn tất B3 Clustering")
    print(f"K search: {k_search_path}")
    print(f"Metrics summary: {metrics_path}")
    print(f"Cluster profile full: {profile_path}")
    print(f"Cluster profile compact: {compact_profile_path}")
    print(f"Labeled data: {labeled_data_path}")
    print(f"Elbow figure: {fig_paths['elbow_report']}")
    print(f"Silhouette figure: {fig_paths['silhouette_report']}")
    print(f"PCA clusters figure: {fig_paths['pca_report']}")
    print(f"Cluster size figure: {fig_paths['cluster_size_report']}")
    print(f"Cluster hour profile figure: {fig_paths['hour_profile_report']}")
    print("=" * 80)


if __name__ == "__main__":
    main()