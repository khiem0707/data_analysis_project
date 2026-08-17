import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Config paths
A3_PATH = Path("../data/raw/A3_customer_personality/marketing_campaign.csv")
B3_PATH = Path("../data/raw/B3_bike_sharing/day.csv")
OUTPUT_DIR = Path("../outputs/round_03_advanced_plots")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Styling
sns.set_theme(style="whitegrid")

def plot_advanced_comparisons(df, column, title_prefix, output_filename, is_a3=False):
    if is_a3:
        pass
    
    data = df[column].dropna()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"{title_prefix}: So sánh Boxplot với các biểu đồ phân bố nâng cao\n(Biến: {column})", fontsize=16, fontweight='bold', y=0.95)
    
    # 1. Boxplot
    sns.boxplot(y=data, ax=axes[0, 0], color="skyblue", showmeans=True, meanprops={"marker":"X", "markerfacecolor":"red", "markeredgecolor":"red"})
    axes[0, 0].set_title("1. Boxplot (Truyền thống)", fontsize=14)
    axes[0, 0].set_ylabel("Giá trị")
    
    # 2. Violin Plot
    sns.violinplot(y=data, ax=axes[0, 1], color="lightgreen", inner="quartile")
    axes[0, 1].set_title("2. Violin Plot (Mật độ phân bố KDE)", fontsize=14)
    axes[0, 1].set_ylabel("Giá trị")
    
    # 3. Boxen Plot (Letter-value)
    sns.boxenplot(y=data, ax=axes[1, 0], color="coral")
    axes[1, 0].set_title("3. Boxen Plot (Tối ưu cho dữ liệu lớn/outliers)", fontsize=14)
    axes[1, 0].set_ylabel("Giá trị")
    
    # 4. Strip Plot / Swarm Plot (Strip with jitter)
    sns.boxplot(y=data, ax=axes[1, 1], color="lightgray", showfliers=False) # boxplot background
    sns.stripplot(y=data, ax=axes[1, 1], color="purple", alpha=0.4, jitter=True, size=4)
    axes[1, 1].set_title("4. Strip Plot đè trên Boxplot (Từng điểm dữ liệu thực)", fontsize=14)
    axes[1, 1].set_ylabel("Giá trị")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    out_path = OUTPUT_DIR / output_filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path.resolve()}")
    plt.close()

def main():
    print("Reading A3 dataset...")
    df_a3 = pd.read_csv(A3_PATH, sep="\t")
    
    # Filter A3 Income outliers for better visualization just like in preprocessing
    q1 = df_a3["Income"].quantile(0.25)
    q3 = df_a3["Income"].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    df_a3_filtered = df_a3[df_a3["Income"] <= upper]
    
    plot_advanced_comparisons(df_a3_filtered, "Income", "Dataset A3", "A3_advanced_plots_Income.png", is_a3=True)
    
    print("Reading B3 dataset...")
    df_b3 = pd.read_csv(B3_PATH)
    plot_advanced_comparisons(df_b3, "cnt", "Dataset B3", "B3_advanced_plots_cnt.png")

if __name__ == "__main__":
    main()
