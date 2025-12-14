from pathlib import Path

from src.data_loader import load_dataset
from src.preprocessing import preprocess
from src.model_trainer import train_models
from src.eda_visualizations import create_professional_visualizations
from src.model_evaluator import full_evaluation
from src.utils import save_best_models

# ------------------------------------------------------------------
# Paths (Mac-safe, Python3-safe)
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "ransap-5d-features-clean-merged.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"
MODELS_DIR = OUTPUT_DIR / "models"

# ------------------------------------------------------------------
def main():
    # Ensure output folders exist
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1️⃣ Load dataset
    df, feature_cols = load_dataset(str(DATA_PATH))

    # 2️⃣ Preprocess
    X_train, X_test, y_train, y_test, scaler = preprocess(df, feature_cols)

    # 3️⃣ EDA (plots)
    create_professional_visualizations(
        df,
        feature_cols,
        X_train,
        y_train,
        output_dir=str(PLOTS_DIR)
    )

    # 4️⃣ Train models (LogReg, RF, GB, XGB, MLP)
    results, trained_models = train_models(
        X_train, X_test, y_train, y_test
    )

    # 5️⃣ Evaluation (ROC, confusion matrices, CSV reports)
    comparison_df = full_evaluation(
        results,
        X_test,
        y_test,
        feature_cols,
        output_dir=str(OUTPUT_DIR)
    )

    # 6️⃣ Save best models
    save_best_models(
        trained_models,
        results,
        comparison_df,
        output_dir=str(MODELS_DIR)
    )

    print("\n✅ PIPELINE COMPLETED SUCCESSFULLY")
    print(f"📁 Plots saved to: {PLOTS_DIR}")
    print(f"📁 Reports saved to: {REPORTS_DIR}")
    print(f"📁 Models saved to: {MODELS_DIR}")

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
