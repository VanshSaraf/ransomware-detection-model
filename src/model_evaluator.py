import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

def full_evaluation(results, X_test, y_test, feature_cols, output_dir):
    plots_dir = os.path.join(output_dir, "plots")
    reports_dir = os.path.join(output_dir, "reports")

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    summary = []

    # --------------------------------------------------
    # Collect metrics
    # --------------------------------------------------
    for name, res in results.items():
        summary.append({
            "Model": name,
            "Accuracy": res["accuracy"],
            "Precision": res["precision"],
            "Recall": res["recall"],
            "F1-Score": res["f1"],
            "ROC-AUC": res["roc_auc"],
            "Training Time (s)": res["time"]
        })

    summary_df = pd.DataFrame(summary).sort_values("F1-Score", ascending=False)
    summary_df.to_csv(
        os.path.join(reports_dir, "model_comparison.csv"),
        index=False
    )

    print("\n📊 Model comparison:")
    print(summary_df.round(4))

    # --------------------------------------------------
    # ROC curves
    # --------------------------------------------------
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "04_roc_curves.png"), dpi=300)
    plt.close()

    # --------------------------------------------------
    # Confusion matrices
    # --------------------------------------------------
    for name, res in results.items():
        cm = confusion_matrix(y_test, res["y_pred"])

        plt.figure(figsize=(4, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Benign", "Ransomware"],
            yticklabels=["Benign", "Ransomware"]
        )
        plt.title(f"{name} - Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(
            os.path.join(plots_dir, f"05_cm_{name.replace(' ', '_')}.png"),
            dpi=300
        )
        plt.close()

    print("✅ Evaluation reports & plots saved")
    return summary_df
