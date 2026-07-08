from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

INPUT_FILE = PROJECT_ROOT / "outputs" / "round_02_A3_eda" / "data" / "A3_cleaned.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "round_05_chapter4_ml"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures" / "A3_clustering"

# Hình dùng trong báo cáo đặt vào thư mục ảnh A3 hiện có
REPORT_IMG_DIR = PROJECT_ROOT / "images" / "nhom-a-3"

for folder in [TABLE_DIR, FIG_DIR, REPORT_IMG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. Hàm tiện ích
# =========================================================

def load_a3_data() -> pd.DataFrame:
    """
    Đọc dữ liệu A3 đã làm sạch.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file đầu vào: {INPUT_FILE}\n"
            "Bạn kiểm tra lại A3_cleaned.csv đã nằm đúng thư mục chưa."
        )

    df = pd.read_csv(INPUT_FILE)

    return df


def select_clustering_features(df: pd.DataFrame):
    """
    Chọn feature cho clustering A3.

    Nguyên tắc:
    - Không dùng ID, ngày tháng, target Response.
    - Không dùng trực tiếp Education/Marital_Status vì đây là biến nhãn/nhân khẩu học dạng category.
    - Ưu tiên các biến numeric mô tả thu nhập, tuổi, chi tiêu, số lần mua, tương tác.
    """
    preferred_features = [
        "Income",
        "Age",
        "Recency",
        "Customer_Tenure_Days",
        "Total_Spending",
        "Total_Children",
        "Total_Purchases",
        "Total_Accepted_Campaigns",

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
    ]

    features = [c for c in preferred_features if c in df.columns]

    if len(features) == 0:
        raise ValueError("Không tìm thấy feature numeric phù hợp cho A3 clustering.")

    X_raw = df[features].copy()

    return X_raw, features


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
    Với DBSCAN, bỏ noise label = -1 khi cần tính silhouette.
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

    # Cần ít nhất 2 cụm và không phải mỗi điểm một cụm
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
    Chạy KMeans nhiều K, chọn K có silhouette cao nhất.
    """
    rows = []

    for k in range(k_min, k_max + 1):
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


def tune_dbscan(X_scaled):
    """
    Thử một số eps cho DBSCAN và chọn cấu hình có silhouette tốt nhất.
    Nếu DBSCAN không tìm được cụm hợp lệ, trả về cấu hình mặc định.
    """
    rows = []

    eps_values = np.arange(0.5, 5.5, 0.5)
    min_samples_values = [5, 10, 15]

    for eps in eps_values:
        for min_samples in min_samples_values:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_scaled)

            metrics = evaluate_clustering(X_scaled, labels)

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
            "eps": 1.5,
            "min_samples": 10,
        }
    else:
        # Ưu tiên silhouette cao, nhưng tránh cấu hình noise quá nhiều
        valid_df["noise_ratio"] = valid_df["noise_count"] / len(X_scaled)
        valid_df = valid_df[valid_df["noise_ratio"] <= 0.6]

        if len(valid_df) == 0:
            valid_df = result_df[result_df["silhouette"].notna()].copy()

        best_row = valid_df.sort_values("silhouette", ascending=False).iloc[0]

        best = {
            "eps": float(best_row["eps"]),
            "min_samples": int(best_row["min_samples"]),
        }

    return best, result_df


def create_cluster_profile(df: pd.DataFrame, features: list[str], labels, cluster_col: str):
    """
    Tạo bảng profile cụm theo giá trị trung bình của feature gốc.
    """
    temp_df = df.copy()
    temp_df[cluster_col] = labels

    profile = temp_df.groupby(cluster_col)[features].mean().reset_index()

    counts = temp_df[cluster_col].value_counts().sort_index()
    percents = temp_df[cluster_col].value_counts(normalize=True).sort_index() * 100

    profile.insert(1, "count", profile[cluster_col].map(counts))
    profile.insert(2, "percent", profile[cluster_col].map(percents))

    return profile


def create_compact_profile(profile_df: pd.DataFrame, cluster_col: str):
    """
    Tạo bảng profile rút gọn cho báo cáo.
    Chỉ giữ một số biến chính dễ diễn giải.
    """
    keep_cols = [
        cluster_col,
        "count",
        "percent",
        "Income",
        "Age",
        "Recency",
        "Total_Spending",
        "Total_Purchases",
        "Total_Children",
        "Customer_Tenure_Days",
    ]

    keep_cols = [c for c in keep_cols if c in profile_df.columns]

    compact = profile_df[keep_cols].copy()

    return compact


def add_auto_interpretation(compact_profile: pd.DataFrame, cluster_col: str):
    """
    Thêm diễn giải tự động dựa trên các biến chính.
    Diễn giải này chỉ là gợi ý ban đầu, khi viết báo cáo có thể chỉnh lại.
    """
    df = compact_profile.copy()

    interpretations = []

    spending_col = "Total_Spending"
    income_col = "Income"
    purchases_col = "Total_Purchases"
    recency_col = "Recency"
    children_col = "Total_Children"

    spending_median = df[spending_col].median() if spending_col in df.columns else np.nan
    income_median = df[income_col].median() if income_col in df.columns else np.nan
    purchases_median = df[purchases_col].median() if purchases_col in df.columns else np.nan
    recency_median = df[recency_col].median() if recency_col in df.columns else np.nan
    children_median = df[children_col].median() if children_col in df.columns else np.nan

    for _, row in df.iterrows():
        phrases = []

        if spending_col in df.columns:
            if row[spending_col] >= spending_median:
                phrases.append("chi tiêu cao")
            else:
                phrases.append("chi tiêu thấp")

        if income_col in df.columns:
            if row[income_col] >= income_median:
                phrases.append("thu nhập cao")
            else:
                phrases.append("thu nhập thấp")

        if purchases_col in df.columns:
            if row[purchases_col] >= purchases_median:
                phrases.append("mua hàng nhiều")
            else:
                phrases.append("mua hàng ít")

        if recency_col in df.columns:
            if row[recency_col] >= recency_median:
                phrases.append("lâu chưa mua lại")
            else:
                phrases.append("gần đây có tương tác")

        if children_col in df.columns:
            if row[children_col] >= children_median:
                phrases.append("nhiều con hơn")
            else:
                phrases.append("ít con hơn")

        interpretations.append(", ".join(phrases))

    df["interpretation_suggestion"] = interpretations

    return df


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


# =========================================================
# 3. Vẽ hình
# =========================================================

def plot_elbow(k_result_df: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(k_result_df["k"], k_result_df["inertia"], marker="o")
    ax.set_title("A3 K-Means Elbow Method")
    ax.set_xlabel("Number of clusters K")
    ax.set_ylabel("Inertia")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_silhouette(k_result_df: pd.DataFrame, best_k: int, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(k_result_df["k"], k_result_df["silhouette"], marker="o")
    ax.axvline(best_k, linestyle="--", label=f"Best K = {best_k}")
    ax.set_title("A3 K-Means Silhouette Score")
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
        s=18,
        alpha=0.75,
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
    ax.set_title("A3 Cluster Size Distribution")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of customers")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 4. Main process
# =========================================================

def main():
    print("=" * 80)
    print("A3 Clustering - Chapter 4")
    print("=" * 80)

    df = load_a3_data()

    print(f"Input file: {INPUT_FILE}")
    print(f"Raw shape: {df.shape}")

    X_raw, features = select_clustering_features(df)

    print(f"Features used ({len(features)}):")
    for f in features:
        print(f"  - {f}")

    X_scaled, preprocess_pipeline = preprocess_features(X_raw)

    # =====================================================
    # KMeans: chọn K bằng silhouette
    # =====================================================

    best_k, k_result_df = choose_best_k_by_silhouette(X_scaled, k_min=2, k_max=8)

    print("\nKMeans K search:")
    print(k_result_df)
    print(f"Best K by silhouette: {best_k}")

    k_search_path = TABLE_DIR / "A3_CH4_kmeans_k_search.csv"
    k_result_df.to_csv(k_search_path, index=False, encoding="utf-8-sig")

    kmeans = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=20,
    )
    kmeans_labels = kmeans.fit_predict(X_scaled)

    # =====================================================
    # Agglomerative
    # =====================================================

    agg = AgglomerativeClustering(n_clusters=best_k)
    agg_labels = agg.fit_predict(X_scaled)

    # =====================================================
    # DBSCAN
    # =====================================================

    best_dbscan_params, dbscan_search_df = tune_dbscan(X_scaled)

    dbscan = DBSCAN(
        eps=best_dbscan_params["eps"],
        min_samples=best_dbscan_params["min_samples"],
    )
    dbscan_labels = dbscan.fit_predict(X_scaled)

    dbscan_search_path = TABLE_DIR / "A3_CH4_dbscan_parameter_search.csv"
    dbscan_search_df.to_csv(dbscan_search_path, index=False, encoding="utf-8-sig")

    print("\nBest DBSCAN params:")
    print(best_dbscan_params)

    # =====================================================
    # Metrics summary
    # =====================================================

    metrics_rows = []

    for model_name, labels in [
        ("KMeans", kmeans_labels),
        ("Agglomerative", agg_labels),
        ("DBSCAN", dbscan_labels),
    ]:
        metrics = evaluate_clustering(X_scaled, labels)

        metrics_rows.append({
            "dataset": "A3",
            "model": model_name,
            **metrics,
        })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = TABLE_DIR / "A3_CH4_clustering_metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print("\nClustering metrics:")
    print(metrics_df)

    # Chọn mô hình chính để diễn giải: KMeans
    df_result = df.copy()
    df_result["A3_CH4_kmeans_cluster"] = kmeans_labels
    df_result["A3_CH4_agg_cluster"] = agg_labels
    df_result["A3_CH4_dbscan_cluster"] = dbscan_labels

    labeled_data_path = TABLE_DIR / "A3_CH4_clustering_labeled_data.csv"
    df_result.to_csv(labeled_data_path, index=False, encoding="utf-8-sig")

    # =====================================================
    # Profile cụm KMeans
    # =====================================================

    profile = create_cluster_profile(
        df=df_result,
        features=features,
        labels=kmeans_labels,
        cluster_col="A3_CH4_kmeans_cluster",
    )

    profile_path = TABLE_DIR / "A3_CH4_kmeans_cluster_profile_full.csv"
    profile.to_csv(profile_path, index=False, encoding="utf-8-sig")

    compact_profile = create_compact_profile(profile, cluster_col="A3_CH4_kmeans_cluster")
    compact_profile = add_auto_interpretation(compact_profile, cluster_col="A3_CH4_kmeans_cluster")

    compact_profile_path = TABLE_DIR / "A3_CH4_kmeans_cluster_profile_compact.csv"
    compact_profile.to_csv(compact_profile_path, index=False, encoding="utf-8-sig")

    print("\nCompact cluster profile:")
    print(compact_profile)

    # =====================================================
    # PCA
    # =====================================================

    pca_df, explained = run_pca(X_scaled)

    pca_result = pca_df.copy()
    pca_result["kmeans_cluster"] = kmeans_labels
    pca_result["agg_cluster"] = agg_labels
    pca_result["dbscan_cluster"] = dbscan_labels

    pca_path = TABLE_DIR / "A3_CH4_pca_coordinates.csv"
    pca_result.to_csv(pca_path, index=False, encoding="utf-8-sig")

    explained_path = TABLE_DIR / "A3_CH4_pca_explained_variance.csv"
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
        "elbow_report": REPORT_IMG_DIR / "A3_CH4_kmeans_elbow.png",
        "elbow_output": FIG_DIR / "A3_CH4_kmeans_elbow.png",

        "silhouette_report": REPORT_IMG_DIR / "A3_CH4_kmeans_silhouette.png",
        "silhouette_output": FIG_DIR / "A3_CH4_kmeans_silhouette.png",

        "pca_report": REPORT_IMG_DIR / "A3_CH4_pca_clusters_kmeans.png",
        "pca_output": FIG_DIR / "A3_CH4_pca_clusters_kmeans.png",

        "cluster_size_report": REPORT_IMG_DIR / "A3_CH4_kmeans_cluster_size.png",
        "cluster_size_output": FIG_DIR / "A3_CH4_kmeans_cluster_size.png",
    }

    plot_elbow(k_result_df, fig_paths["elbow_report"])
    plot_elbow(k_result_df, fig_paths["elbow_output"])

    plot_silhouette(k_result_df, best_k, fig_paths["silhouette_report"])
    plot_silhouette(k_result_df, best_k, fig_paths["silhouette_output"])

    plot_pca_clusters(
        pca_df,
        kmeans_labels,
        title=f"A3 Customer Clusters by K-Means PCA, K={best_k}",
        output_path=fig_paths["pca_report"],
    )
    plot_pca_clusters(
        pca_df,
        kmeans_labels,
        title=f"A3 Customer Clusters by K-Means PCA, K={best_k}",
        output_path=fig_paths["pca_output"],
    )

    plot_cluster_size(kmeans_labels, fig_paths["cluster_size_report"])
    plot_cluster_size(kmeans_labels, fig_paths["cluster_size_output"])

    print("\n" + "=" * 80)
    print("Hoàn tất A3 Clustering")
    print(f"K search: {k_search_path}")
    print(f"Metrics summary: {metrics_path}")
    print(f"Cluster profile full: {profile_path}")
    print(f"Cluster profile compact: {compact_profile_path}")
    print(f"Labeled data: {labeled_data_path}")
    print(f"Elbow figure: {fig_paths['elbow_report']}")
    print(f"Silhouette figure: {fig_paths['silhouette_report']}")
    print(f"PCA clusters figure: {fig_paths['pca_report']}")
    print(f"Cluster size figure: {fig_paths['cluster_size_report']}")
    print("=" * 80)


if __name__ == "__main__":
    main()