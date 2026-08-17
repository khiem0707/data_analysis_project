from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


# =========================================================
# 1. Cấu hình đường dẫn
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "outputs" / "round_04_C3_audio_eda" / "data" / "C3_audio_features.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "round_05_chapter4_ml"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures" / "C3_clustering"

# Hình dùng trong báo cáo đặt vào thư mục ảnh C3 hiện có
REPORT_IMG_DIR = PROJECT_ROOT / "images" / "nhom-c-3"

for folder in [TABLE_DIR, FIG_DIR, REPORT_IMG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. Hàm tiện ích
# =========================================================

def load_c3_data() -> pd.DataFrame:
    """
    Đọc dữ liệu đặc trưng âm thanh C3 đã được trích xuất ở Chương 3.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file đầu vào: {INPUT_FILE}\n"
            "Bạn kiểm tra lại C3_audio_features.csv đã nằm đúng thư mục chưa."
        )

    df = pd.read_csv(INPUT_FILE)

    return df


def select_clustering_features(df: pd.DataFrame):
    """
    Chọn feature numeric cho C3 clustering.

    Không dùng:
    - tên file
    - đường dẫn audio
    - class/classID
    - fold
    - trạng thái trích xuất feature
    """
    drop_cols = [
        "slice_file_name",
        "fsID",
        "start",
        "end",
        "salience",
        "fold",
        "classID",
        "class",
        "audio_path",
        "feature_status",
        "feature_error",
        "duration_metadata",
    ]

    candidate_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    numeric_features = candidate_df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

    # Loại các cột không nên dùng nếu còn sót
    numeric_features = [
        c for c in numeric_features
        if not c.lower().endswith("id")
        and c not in ["fold", "classID"]
    ]

    if len(numeric_features) == 0:
        raise ValueError("Không tìm thấy feature numeric phù hợp cho C3 clustering.")

    X_raw = df[numeric_features].copy()

    return X_raw, numeric_features


def preprocess_features(X_raw: pd.DataFrame):
    """
    Impute median + StandardScaler.
    """
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    X_scaled = pipeline.fit_transform(X_raw)

    return X_scaled, pipeline


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


def choose_best_k_by_silhouette(X_scaled, k_min=2, k_max=10):
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
    Agglomerative và DBSCAN có thể tốn tài nguyên.
    Với C3 có 8732 dòng, sample 5000 để chạy nhanh hơn.
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

    eps_values = np.arange(1.0, 10.5, 0.5)
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


def create_cluster_profile(df_model: pd.DataFrame, features: list[str], labels, cluster_col: str):
    """
    Tạo profile cụm theo giá trị trung bình của feature âm thanh.
    """
    temp_df = df_model.copy()
    temp_df[cluster_col] = labels

    profile = temp_df.groupby(cluster_col)[features].mean().reset_index()

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
        "duration_audio",
        "rms_mean",
        "zcr_mean",
        "spectral_centroid_mean",
        "spectral_bandwidth_mean",
        "spectral_rolloff_mean",
        "mfcc_1_mean",
        "mfcc_2_mean",
        "mfcc_3_mean",
    ]

    keep_cols = [c for c in keep_cols if c in profile_df.columns]

    compact = profile_df[keep_cols].copy()

    return compact


def add_auto_interpretation(compact_profile: pd.DataFrame, cluster_col: str):
    """
    Gợi ý diễn giải cụm dựa trên các đặc trưng âm thanh chính.
    """
    df = compact_profile.copy()

    interpretations = []

    rms_col = "rms_mean"
    zcr_col = "zcr_mean"
    centroid_col = "spectral_centroid_mean"
    duration_col = "duration_audio"

    rms_median = df[rms_col].median() if rms_col in df.columns else np.nan
    zcr_median = df[zcr_col].median() if zcr_col in df.columns else np.nan
    centroid_median = df[centroid_col].median() if centroid_col in df.columns else np.nan
    duration_median = df[duration_col].median() if duration_col in df.columns else np.nan

    for _, row in df.iterrows():
        phrases = []

        if rms_col in df.columns:
            if row[rms_col] >= rms_median:
                phrases.append("năng lượng cao")
            else:
                phrases.append("năng lượng thấp")

        if zcr_col in df.columns:
            if row[zcr_col] >= zcr_median:
                phrases.append("dao động nhanh/nhiễu cao")
            else:
                phrases.append("dao động thấp hơn")

        if centroid_col in df.columns:
            if row[centroid_col] >= centroid_median:
                phrases.append("âm sắc sáng/tần số cao")
            else:
                phrases.append("âm sắc trầm hơn")

        if duration_col in df.columns:
            if row[duration_col] >= duration_median:
                phrases.append("thời lượng dài hơn")
            else:
                phrases.append("thời lượng ngắn hơn")

        interpretations.append(", ".join(phrases))

    df["interpretation_suggestion"] = interpretations

    return df


def create_cluster_class_table(df_model: pd.DataFrame, labels, cluster_col: str):
    """
    Tạo bảng đối chiếu cụm với class thật.
    class không được dùng để fit, chỉ dùng để diễn giải sau clustering.
    """
    if "class" not in df_model.columns:
        return None, None

    temp_df = df_model.copy()
    temp_df[cluster_col] = labels

    count_table = pd.crosstab(temp_df[cluster_col], temp_df["class"])
    ratio_table = pd.crosstab(temp_df[cluster_col], temp_df["class"], normalize="index") * 100

    return count_table, ratio_table


# =========================================================
# 3. Vẽ hình
# =========================================================

def plot_elbow(k_result_df: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.lineplot(data=k_result_df, x="k", y="inertia", marker="o", ax=ax, color="#4c72b0", linewidth=2, markersize=8)
    
    ax.set_title("C3 K-Means Elbow Method", fontsize=16, fontweight='bold')
    ax.set_xlabel("Số lượng cụm (K)", fontsize=14)
    ax.set_ylabel("Inertia", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_silhouette(k_result_df: pd.DataFrame, best_k: int, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(data=k_result_df, x="k", y="silhouette", marker="o", ax=ax, color="#dd8452", linewidth=2, markersize=8)
    ax.axvline(best_k, linestyle="--", color="red", label=f"Best K = {best_k}")
    
    ax.set_title("C3 K-Means Silhouette Score", fontsize=16, fontweight='bold')
    ax.set_xlabel("Số lượng cụm (K)", fontsize=14)
    ax.set_ylabel("Silhouette score", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pca_clusters(pca_df: pd.DataFrame, labels, title: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(12, 8))

    # Xử lý màu sắc
    unique_labels = np.unique(labels)
    palette = sns.color_palette("tab10", n_colors=len(unique_labels))
    
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue=labels,
        palette=palette,
        s=30,
        alpha=0.8,
        ax=ax,
        legend="full"
    )

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("Principal Component 1 (PC1)", fontsize=14)
    ax.set_ylabel("Principal Component 2 (PC2)", fontsize=14)

    ax.legend(title="Cụm (Cluster)", title_fontsize=13, fontsize=12, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cluster_size(labels, output_path: Path):
    labels_series = pd.Series(labels, name="cluster")
    size_df = labels_series.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(size_df.index.astype(str), size_df.values)
    ax.set_title("C3 Cluster Size Distribution")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of audio clips")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cluster_class_heatmap(ratio_table: pd.DataFrame, output_path: Path):
    """
    Vẽ heatmap đối chiếu cụm với class bằng matplotlib.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    data = ratio_table.values

    im = ax.imshow(data, aspect="auto")

    ax.set_title("C3 Cluster vs Class Distribution (%)")
    ax.set_xlabel("Audio class")
    ax.set_ylabel("Cluster")

    ax.set_xticks(np.arange(ratio_table.shape[1]))
    ax.set_xticklabels(ratio_table.columns, rotation=45, ha="right")

    ax.set_yticks(np.arange(ratio_table.shape[0]))
    ax.set_yticklabels(ratio_table.index.astype(str))

    # Ghi số phần trăm lên từng ô
    for i in range(ratio_table.shape[0]):
        for j in range(ratio_table.shape[1]):
            value = data[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Percent within cluster")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_profile(compact_profile: pd.DataFrame, cluster_col: str, output_path: Path):
    """
    Vẽ một số feature chính theo cụm.
    """
    feature_cols = [
        "rms_mean",
        "zcr_mean",
        "spectral_centroid_mean",
        "spectral_rolloff_mean",
    ]

    feature_cols = [c for c in feature_cols if c in compact_profile.columns]

    plot_df = compact_profile.set_index(cluster_col)[feature_cols]

    # Chuẩn hóa min-max theo cột để cùng nhìn trên một biểu đồ
    normalized = (plot_df - plot_df.min()) / (plot_df.max() - plot_df.min() + 1e-9)

    ax = normalized.plot(kind="bar", figsize=(10, 6))

    ax.set_title("C3 Normalized Audio Feature Profile by Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Normalized value")
    ax.legend(title="Feature", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 4. Main process
# =========================================================

def main():
    print("=" * 80)
    print("C3 Clustering - Chapter 4")
    print("=" * 80)

    df = load_c3_data()

    print(f"Input file: {INPUT_FILE}")
    print(f"Raw shape: {df.shape}")

    if "class" in df.columns:
        print(f"Number of classes: {df['class'].nunique()}")
        print("Class distribution:")
        print(df["class"].value_counts())

    X_raw, features = select_clustering_features(df)

    print(f"\nFeatures used ({len(features)}):")
    for f in features:
        print(f"  - {f}")

    X_scaled, preprocess_pipeline = preprocess_features(X_raw)

    print(f"\nTransformed feature matrix shape: {X_scaled.shape}")

    # =====================================================
    # KMeans full data
    # =====================================================

    print("\nKMeans K search:")
    best_k, k_result_df = choose_best_k_by_silhouette(X_scaled, k_min=2, k_max=10)

    print(k_result_df)
    print(f"Best K by silhouette: {best_k}")

    k_search_path = TABLE_DIR / "C3_CH4_kmeans_k_search.csv"
    k_result_df.to_csv(k_search_path, index=False, encoding="utf-8-sig")

    kmeans = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=20,
    )
    kmeans_labels = kmeans.fit_predict(X_scaled)

    # =====================================================
    # Agglomerative và DBSCAN trên sample
    # =====================================================

    X_sample, df_sample, sample_idx = sample_for_heavy_models(
        X_scaled,
        df,
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

    dbscan_search_path = TABLE_DIR / "C3_CH4_dbscan_parameter_search_sample.csv"
    dbscan_search_df.to_csv(dbscan_search_path, index=False, encoding="utf-8-sig")

    print("\nBest DBSCAN params on sample:")
    print(best_dbscan_params)

    # =====================================================
    # Metrics summary
    # =====================================================

    metrics_rows = []

    kmeans_metrics = evaluate_clustering(X_scaled, kmeans_labels)
    metrics_rows.append({
        "dataset": "C3",
        "model": "KMeans",
        "data_used": "full",
        **kmeans_metrics,
    })

    agg_metrics = evaluate_clustering(X_sample, agg_labels_sample)
    metrics_rows.append({
        "dataset": "C3",
        "model": "Agglomerative",
        "data_used": "sample_5000",
        **agg_metrics,
    })

    dbscan_metrics = evaluate_clustering(X_sample, dbscan_labels_sample)
    metrics_rows.append({
        "dataset": "C3",
        "model": "DBSCAN",
        "data_used": "sample_5000",
        **dbscan_metrics,
    })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = TABLE_DIR / "C3_CH4_clustering_metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print("\nClustering metrics:")
    print(metrics_df)

    # =====================================================
    # Lưu dữ liệu có nhãn cụm KMeans full
    # =====================================================

    df_result = df.copy()
    df_result["C3_CH4_kmeans_cluster"] = kmeans_labels

    labeled_data_path = TABLE_DIR / "C3_CH4_clustering_labeled_data.csv"
    df_result.to_csv(labeled_data_path, index=False, encoding="utf-8-sig")

    # =====================================================
    # Profile cụm
    # =====================================================

    profile = create_cluster_profile(
        df_model=df_result,
        features=features,
        labels=kmeans_labels,
        cluster_col="C3_CH4_kmeans_cluster",
    )

    profile_path = TABLE_DIR / "C3_CH4_kmeans_cluster_profile_full.csv"
    profile.to_csv(profile_path, index=False, encoding="utf-8-sig")

    compact_profile = create_compact_profile(profile, cluster_col="C3_CH4_kmeans_cluster")
    compact_profile = add_auto_interpretation(compact_profile, cluster_col="C3_CH4_kmeans_cluster")

    compact_profile_path = TABLE_DIR / "C3_CH4_kmeans_cluster_profile_compact.csv"
    compact_profile.to_csv(compact_profile_path, index=False, encoding="utf-8-sig")

    print("\nCompact cluster profile:")
    print(compact_profile)

    # =====================================================
    # Cluster vs Class
    # =====================================================

    count_table, ratio_table = create_cluster_class_table(
        df_model=df_result,
        labels=kmeans_labels,
        cluster_col="C3_CH4_kmeans_cluster",
    )

    if count_table is not None:
        count_table_path = TABLE_DIR / "C3_CH4_cluster_vs_class_count.csv"
        ratio_table_path = TABLE_DIR / "C3_CH4_cluster_vs_class_percent.csv"

        count_table.to_csv(count_table_path, encoding="utf-8-sig")
        ratio_table.to_csv(ratio_table_path, encoding="utf-8-sig")

        print("\nCluster vs class percent:")
        print(ratio_table)

    # =====================================================
    # PCA full data
    # =====================================================

    pca_df, explained = run_pca(X_scaled)

    pca_result = pca_df.copy()
    pca_result["kmeans_cluster"] = kmeans_labels

    if "class" in df.columns:
        pca_result["class"] = df["class"].values

    pca_path = TABLE_DIR / "C3_CH4_pca_coordinates.csv"
    pca_result.to_csv(pca_path, index=False, encoding="utf-8-sig")

    explained_path = TABLE_DIR / "C3_CH4_pca_explained_variance.csv"
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
        "elbow_report": REPORT_IMG_DIR / "C3_CH4_kmeans_elbow.png",
        "elbow_output": FIG_DIR / "C3_CH4_kmeans_elbow.png",

        "silhouette_report": REPORT_IMG_DIR / "C3_CH4_kmeans_silhouette.png",
        "silhouette_output": FIG_DIR / "C3_CH4_kmeans_silhouette.png",

        "pca_report": REPORT_IMG_DIR / "C3_CH4_pca_clusters_kmeans.png",
        "pca_output": FIG_DIR / "C3_CH4_pca_clusters_kmeans.png",

        "cluster_size_report": REPORT_IMG_DIR / "C3_CH4_kmeans_cluster_size.png",
        "cluster_size_output": FIG_DIR / "C3_CH4_kmeans_cluster_size.png",

        "feature_profile_report": REPORT_IMG_DIR / "C3_CH4_cluster_feature_profile.png",
        "feature_profile_output": FIG_DIR / "C3_CH4_cluster_feature_profile.png",

        "cluster_class_heatmap_report": REPORT_IMG_DIR / "C3_CH4_cluster_vs_class_heatmap.png",
        "cluster_class_heatmap_output": FIG_DIR / "C3_CH4_cluster_vs_class_heatmap.png",
    }

    plot_elbow(k_result_df, fig_paths["elbow_report"])
    plot_elbow(k_result_df, fig_paths["elbow_output"])

    plot_silhouette(k_result_df, best_k, fig_paths["silhouette_report"])
    plot_silhouette(k_result_df, best_k, fig_paths["silhouette_output"])

    plot_pca_clusters(
        pca_df,
        kmeans_labels,
        title=f"C3 Audio Clusters by K-Means PCA, K={best_k}",
        output_path=fig_paths["pca_report"],
    )
    plot_pca_clusters(
        pca_df,
        kmeans_labels,
        title=f"C3 Audio Clusters by K-Means PCA, K={best_k}",
        output_path=fig_paths["pca_output"],
    )

    plot_cluster_size(kmeans_labels, fig_paths["cluster_size_report"])
    plot_cluster_size(kmeans_labels, fig_paths["cluster_size_output"])

    plot_feature_profile(compact_profile, "C3_CH4_kmeans_cluster", fig_paths["feature_profile_report"])
    plot_feature_profile(compact_profile, "C3_CH4_kmeans_cluster", fig_paths["feature_profile_output"])

    if count_table is not None:
        plot_cluster_class_heatmap(ratio_table, fig_paths["cluster_class_heatmap_report"])
        plot_cluster_class_heatmap(ratio_table, fig_paths["cluster_class_heatmap_output"])

    print("\n" + "=" * 80)
    print("Hoàn tất C3 Clustering")
    print(f"K search: {k_search_path}")
    print(f"Metrics summary: {metrics_path}")
    print(f"Cluster profile full: {profile_path}")
    print(f"Cluster profile compact: {compact_profile_path}")
    print(f"Labeled data: {labeled_data_path}")
    print(f"Elbow figure: {fig_paths['elbow_report']}")
    print(f"Silhouette figure: {fig_paths['silhouette_report']}")
    print(f"PCA clusters figure: {fig_paths['pca_report']}")
    print(f"Cluster size figure: {fig_paths['cluster_size_report']}")
    print(f"Feature profile figure: {fig_paths['feature_profile_report']}")
    if count_table is not None:
        print(f"Cluster vs class heatmap: {fig_paths['cluster_class_heatmap_report']}")
    print("=" * 80)


if __name__ == "__main__":
    main()