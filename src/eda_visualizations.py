import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def create_professional_visualizations(df, feature_cols, X_train, y_train, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")

    # --------------------------------------------------
    # 1. Class distribution
    # --------------------------------------------------
    counts = df['target_label'].value_counts()

    plt.figure(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values)
    plt.title("Class Distribution")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_class_distribution.png", dpi=300)
    plt.close()

    # --------------------------------------------------
    # 2. Feature boxplots (log scaled)
    # --------------------------------------------------
    plt.figure(figsize=(14, 8))
    for i, feature in enumerate(feature_cols):
        plt.subplot(2, 3, i + 1)
        sns.boxplot(
            x=df['target_label'],
            y=np.log1p(df[feature])
        )
        plt.title(feature)
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_feature_boxplots.png", dpi=300)
    plt.close()

    # --------------------------------------------------
    # 3. PCA projection
    # --------------------------------------------------
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y_train,
        cmap="viridis",
        alpha=0.6
    )
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    plt.title("PCA Projection (Training Data)")
    plt.colorbar(scatter, label="Class")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_pca_projection.png", dpi=300)
    plt.close()

    print("✅ EDA visualizations saved")
