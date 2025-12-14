import os
import joblib

def save_best_models(trained_models, results, comparison_df, output_dir):
    """
    Save top-performing models based on F1-score
    """

    os.makedirs(output_dir, exist_ok=True)

    # Take top 3 models
    top_models = comparison_df.head(3)["Model"].tolist()

    print("\n💾 Saving best models:")

    for model_name in top_models:
        if model_name not in trained_models:
            continue

        model = trained_models[model_name]
        result = results[model_name]

        filename = f"best_{model_name.replace(' ', '_').lower()}.pkl"
        filepath = os.path.join(output_dir, filename)

        joblib.dump(model, filepath)

        # Save performance summary
        perf_file = filepath.replace(".pkl", "_metrics.txt")
        with open(perf_file, "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"Precision: {result['precision']:.4f}\n")
            f.write(f"Recall: {result['recall']:.4f}\n")
            f.write(f"F1-Score: {result['f1']:.4f}\n")
            f.write(f"ROC-AUC: {result['roc_auc']:.4f}\n")
            f.write(f"Training Time (s): {result['time']:.2f}\n")

        print(f"   ✅ {model_name} saved → {filepath}")
