# 🛡️ Ransomware Detection using Behavioral Machine Learning

A **production-ready ransomware detection system** that identifies malicious processes using **low-level disk I/O behavioral patterns** and **multiple machine learning models**.  
This project converts a research-oriented Kaggle pipeline into a **modular, deployable Python system** with full **EDA, model comparison, evaluation, and reporting**.

---

## 🧠 Problem Statement

Traditional ransomware detection relies on static signatures, which fail against **zero-day and obfuscated attacks**.  
This project detects ransomware using **behavioral I/O patterns** that remain consistent even when malware changes its code.


## 🚀 Key Features

- 🔍 **Behavior-based ransomware detection** (resistant to signature evasion)
- 🧠 **Multiple ML models trained & compared** (Logistic Regression, Random Forest, Gradient Boosting, XGBoost, Neural Network)
- 📊 **Professional EDA & visual analytics**
- 📈 **Advanced evaluation metrics** (ROC-AUC, Precision-Recall, F1, Confusion Matrix)
- 💾 **Automatic saving of best-performing models**
- 🧩 **Clean, modular, production-style architecture**
- ⚙️ **Fully reproducible pipeline**

---

## 🧰 Tech Stack

### Programming & Environment
- **Python 3.x** (tested on **Python 3.13**)
- **macOS (Apple Silicon – MacBook Air M2)**

### Data Science & Machine Learning
- **NumPy**
- **Pandas**
- **Scikit-learn**
- **XGBoost**
- **Joblib**

### Visualization
- **Matplotlib**
- **Seaborn**



---



---

## 📂 Project Structure

```text
ransomware-detection/
│
├── data/
│   └── ransap-5d-features-clean-merged.csv   # Dataset (ignored in git)
│
├── src/
│   ├── data_loader.py          # Dataset loading & labeling
│   ├── preprocessing.py        # Cleaning, scaling, splitting
│   ├── eda_visualizations.py   # EDA & plots
│   ├── model_trainer.py        # Model training & tuning
│   ├── model_evaluator.py      # Evaluation & reports
│   └── utils.py                # Model persistence
│
├── outputs/
│   ├── plots/                  # EDA, ROC, PR, confusion matrices
│   ├── reports/                # CSV performance summaries
│   └── models/                 # Saved best models
│
├── main.py                     # End-to-end pipeline
├── requirements.txt
├── README.md
└── .gitignore
