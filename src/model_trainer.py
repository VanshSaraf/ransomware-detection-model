from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import time

def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=2000),
            "params": {
                "C": [0.1, 1, 10, 100],
                "solver": ["liblinear", "lbfgs"]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(n_jobs=-1),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, None]
            }
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(),
            "params": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1, 0.2]
            }
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(
                eval_metric="logloss",
                n_jobs=-1,
                use_label_encoder=False
            ),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2]
            }
        },
        "Neural Network": {
            "model": MLPClassifier(max_iter=1000),
            "params": {
                "hidden_layer_sizes": [(50,), (100,), (50,25)],
                "alpha": [0.0001, 0.001]
            }
        }
    }

    results = {}
    trained_models = {}

    for name, cfg in models.items():
        print(f"\n🔧 Training {name}...")
        start = time.time()

        search = RandomizedSearchCV(
            cfg["model"],
            cfg["params"],
            n_iter=10,
            cv=3,
            scoring="f1",
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_

        y_pred = best.predict(X_test)
        y_prob = best.predict_proba(X_test)[:,1]

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "time": time.time() - start,
            "model": best,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "best_params": search.best_params_
        }

        trained_models[name] = best

    return results, trained_models
